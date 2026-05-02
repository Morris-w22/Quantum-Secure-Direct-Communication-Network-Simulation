import heapq
from typing import Optional

import numpy as np

# network parameters
MTU = 1500
PROCESS_PER_STEP = 100  # 处理qubit速率
LINK_VELOCITY = 0.3  # km/us
MAX_IP_DATAGRAM = 65536
timestep = 1  # us
protect_time = 10  # us
qubit_read_time = 0.5  # us
cbyte_read_time = 0.08  # us
alpha = 0.2


def split_integer(n, chunk_size):
    num_chunks = (n + chunk_size - 1) // chunk_size
    chunks = np.full(num_chunks, chunk_size)
    chunks[-1] = n - chunk_size * (num_chunks - 1)
    return chunks


class Packet:
    __slots__ = (
        'src', 'dst', 'crt', 'qubits_len', 'cbytes_len', 'belong', 'guard',
        'node_delay', 'trans_delay', 'type', 'etg_lifetime', 'success', 'fail',
        'location', 'next_hop', 'route_history', 'traveled_distance', 'save_qm_qubits',
        'pause_delay'
    )

    def __init__(self, src: int, dst: int, qubits_len: int, cbytes_len: int, session_id: int, guard: int):
        self.src = src
        self.dst = dst
        self.crt = src
        self.qubits_len = qubits_len
        self.cbytes_len = cbytes_len
        self.belong = session_id
        self.guard = guard
        self.node_delay = -100
        self.trans_delay = -100
        self.type = 'ED'
        self.etg_lifetime = 1000
        self.success = False
        self.fail = False
        self.location = 'link'
        self.next_hop = dst
        self.route_history = [src]
        self.traveled_distance = 0.0
        self.save_qm_qubits = 0
        self.pause_delay = 0

    def update_qubits_len(self, distance):
        prob = 10 ** (-alpha * distance / 10)
        self.qubits_len = int(np.random.binomial(self.qubits_len, prob))

    def update_node_delay(self):
        self.node_delay = int((self.cbytes_len * cbyte_read_time + self.qubits_len * qubit_read_time + protect_time * self.guard) / timestep)
        self.guard = max(0, self.guard - 1)


class Node:
    __slots__ = ('id', 'max_node_buffer', 'remain_queue', 'rcv_rate', 'snd_rate', 'queue_dict', 'qm_capacity', 'start_node')

    def __init__(self, node_id: int, max_qm_capacity: int, max_node_buffer: int):
        self.id = node_id
        self.max_node_buffer = max_node_buffer
        self.remain_queue = max_node_buffer
        self.rcv_rate = PROCESS_PER_STEP
        self.snd_rate = 0
        self.queue_dict = {}
        self.qm_capacity = max_qm_capacity
        self.start_node = []

    def snd_packets(self, session_id, guard, src_node, dst_node):
        QTP_header_len = 32 + self.snd_rate * 2
        if QTP_header_len + 16 <= MTU:
            q_payload = np.array([self.snd_rate])
            c_bytes = np.array([QTP_header_len + 16 + 26])
        elif QTP_header_len <= MAX_IP_DATAGRAM:
            MAC_frags = split_integer(QTP_header_len + 16, MTU)
            q_payload = MAC_frags // 2
            q_payload[0] -= 24
            c_bytes = MAC_frags + 26
        else:
            IP_frags = split_integer(QTP_header_len, MAX_IP_DATAGRAM)
            c_bytes = np.array([], dtype=int)
            q_payload = np.array([], dtype=int)
            for IP_frag in IP_frags:
                MAC_frags = split_integer(IP_frag + 16, MTU)
                if len(q_payload) == 0:
                    MAC_frags[0] -= 48
                else:
                    MAC_frags[0] -= 16
                q_payload = np.append(q_payload, MAC_frags // 2)
                c_bytes = np.append(c_bytes, MAC_frags + 26)

        assert len(q_payload) == len(c_bytes)
        assert np.sum(q_payload) == self.snd_rate
        return [Packet(src_node, dst_node, int(q_payload[i]), int(c_bytes[i]), session_id, guard) for i in range(len(q_payload))]

    def update_remain_queue(self):
        self.remain_queue = self.max_node_buffer - sum(item[0] for lst in self.queue_dict.values() for item in lst)


class Session:
    __slots__ = ('id', 'src', 'dst', 'dataflow', 'start_time', 'target_qubits', 'target_reached_time', 'finish_time')

    def __init__(self, src: int, dst: int, id: int, start_time: int = 0, target_qubits: int = 0):
        self.id = id
        self.src = src
        self.dst = dst
        self.dataflow = 0
        self.start_time = int(start_time)
        self.target_qubits = int(target_qubits)
        self.target_reached_time = None
        self.finish_time = None

    @property
    def target_completed(self):
        return self.dataflow >= self.target_qubits

    def injection_open(self, time_now: int) -> bool:
        if time_now < self.start_time:
            return False
        return not self.target_completed

    def mark_target_reached(self, time_now: int):
        if self.target_reached_time is None and self.target_completed:
            self.target_reached_time = int(time_now)
            self.finish_time = int(time_now)


class Link:
    __slots__ = ('nodes', 'capacity', 'hop_distance', 'current_load', 'snd_rate_per_session', 'in_sessions')

    def __init__(self, node1, node2, capacity, distance):
        self.nodes = (node1, node2)
        self.capacity = capacity
        self.hop_distance = distance
        self.current_load = 0
        self.snd_rate_per_session = capacity
        self.in_sessions = 0

    def get_current_load_ratio(self):
        if self.capacity == 0:
            return 1.0
        return min(1.0, self.current_load / self.capacity)


class Network:
    def __init__(self, link_capacity_matrix: np.ndarray, hop_distance_matrix: np.ndarray, max_qm_capacity: np.ndarray, max_node_buffer: np.ndarray):
        self.hop_distances = hop_distance_matrix
        self.capacities = link_capacity_matrix * timestep
        self.nodes_num = self.capacities.shape[0]
        self.max_dist = float(np.max(self.hop_distances[self.hop_distances > 0])) if np.any(self.hop_distances > 0) else 1.0
        self.max_buffer_scale = float(np.max(max_node_buffer)) if np.max(max_node_buffer) > 0 else 1.0

        self.nodes = [Node(i, max_qm_capacity[i], max_node_buffer[i]) for i in range(self.nodes_num)]
        self.sessions = []
        self.links = []
        self.link_dict = {}
        self.adj = [[] for _ in range(self.nodes_num)]
        self.route_switches_dict = {}

        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                if self.capacities[i][j] != 0:
                    link = Link(i, j, self.capacities[i][j], self.hop_distances[i][j])
                    self.links.append(link)
                    self.link_dict[(i, j)] = link
                    self.adj[i].append(j)

        self.active_packets = []
        self.pause_packets = []

        self.active_qubits = 0
        self.etg_fail_qubits = 0
        self.queue_fail_qubits = 0
        self.qm_fail_qubits = 0
        self.success_qubits = 0
        self.time = 0

    def activate_sessions(self):
        for session in self.sessions:
            if session.id not in self.nodes[session.src].start_node:
                self.nodes[session.src].start_node.append(session.id)

    def update_link_states(self):
        for link in self.links:
            link.current_load = 0
        for packet in self.active_packets:
            if packet.location == 'link' and (packet.crt, packet.next_hop) in self.link_dict:
                self.link_dict[(packet.crt, packet.next_hop)].current_load += packet.qubits_len

    def get_current_loads_matrix(self):
        loads = np.zeros((self.nodes_num, self.nodes_num), dtype=float)
        for link in self.links:
            loads[link.nodes[0], link.nodes[1]] = link.get_current_load_ratio()
        return loads

    def get_node_buffers_vector(self):
        return np.array([node.remain_queue for node in self.nodes], dtype=float)

    def route_path(self, src: int, dst: int, current_loads: np.ndarray, node_buffers: np.ndarray):
        if src == dst:
            return [src]

        dist_to = [float('inf')] * self.nodes_num
        prev = [-1] * self.nodes_num
        dist_to[src] = 0.0
        pq = [(0.0, src)]

        while pq:
            cur_dist, u = heapq.heappop(pq)
            if cur_dist != dist_to[u]:
                continue
            if u == dst:
                break
            for v in self.adj[u]:
                dist_norm = self.hop_distances[u, v] / self.max_dist
                congestion = current_loads[u, v]
                buffer_pressure = 1.0 - (node_buffers[v] / self.max_buffer_scale)
                w = 0.55 * dist_norm + 0.30 * congestion + 0.15 * buffer_pressure
                nd = cur_dist + w
                if nd < dist_to[v]:
                    dist_to[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist_to[dst] == float('inf'):
            return None

        path = []
        cur = dst
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path if path and path[0] == src else None

    def allocate_bandwidth_to_sessions(self, current_loads, node_buffers):
        source_next_hops = {}
        counts = {}
        for session in self.sessions:
            if not session.injection_open(self.time):
                continue
            path = self.route_path(session.src, session.dst, current_loads, node_buffers)
            if path is None or len(path) < 2:
                continue
            next_hop = path[1]
            source_next_hops[session.id] = path
            counts[(session.src, next_hop)] = counts.get((session.src, next_hop), 0) + 1

        for link in self.links:
            c = counts.get(link.nodes, 0)
            link.in_sessions = c
            link.snd_rate_per_session = link.capacity if c == 0 else int(link.capacity / c)
        return source_next_hops, counts

    def cleanup(self):
        for node in self.nodes:
            empty_keys = []
            for session_id, q in node.queue_dict.items():
                kept = []
                for qubits, packet in q:
                    if packet.fail:
                        if packet.type == 'MS':
                            self.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                        self.queue_fail_qubits += packet.qubits_len
                    else:
                        kept.append([qubits, packet])
                node.queue_dict[session_id] = kept
                if not kept:
                    empty_keys.append(session_id)
            for k in empty_keys:
                del node.queue_dict[k]

    def step(self):
        self.activate_sessions()

        self.update_link_states()
        current_loads = self.get_current_loads_matrix()
        node_buffers = self.get_node_buffers_vector()
        source_paths, source_link_counts = self.allocate_bandwidth_to_sessions(current_loads, node_buffers)

        # 源节点发送：只对仍在生命周期内的会话注入新包
        for node in self.nodes:
            for session_id in node.start_node:
                session = self.sessions[session_id]
                if not session.injection_open(self.time):
                    continue
                path = source_paths.get(session_id)
                if path is None or len(path) < 2:
                    continue
                next_hop = path[1]
                link = self.link_dict[(node.id, next_hop)]
                share = source_link_counts[(node.id, next_hop)]
                next_node_obj = self.nodes[next_hop]
                queue_sessions = max(1, len(next_node_obj.queue_dict))
                node.snd_rate = int(min(
                    link.snd_rate_per_session,
                    next_node_obj.rcv_rate / queue_sessions,
                    max(0, next_node_obj.remain_queue / max(1, share))
                ))
                if node.snd_rate <= 0:
                    continue
                guard = len(path) - 1
                new_packets = node.snd_packets(session_id, guard, node.id, session.dst)
                hop_distance = self.hop_distances[node.id, next_hop]
                for packet in new_packets:
                    packet.next_hop = next_hop
                    packet.route_history.append(next_hop)
                    packet.trans_delay = int(hop_distance / LINK_VELOCITY) + 1
                self.active_packets.extend(new_packets)

        for node in self.nodes:
            if node.queue_dict:
                for session_id in list(node.queue_dict.keys()):
                    q = node.queue_dict[session_id]
                    new_q = []
                    for qubits, packet in q:
                        assert packet.location == 'queue'
                        packet.etg_lifetime -= 1
                        if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                            packet.fail = True
                            self.etg_fail_qubits += packet.qubits_len
                        else:
                            new_q.append([qubits, packet])
                    node.queue_dict[session_id] = new_q

                n_sessions = max(1, len(node.queue_dict))
                for session_id in list(node.queue_dict.keys()):
                    process_bits = int(node.rcv_rate / n_sessions)
                    q = node.queue_dict[session_id]
                    new_q = []
                    for qubits, packet in q:
                        assert packet.location == 'queue'
                        if process_bits <= 0:
                            new_q.append([qubits, packet])
                        elif qubits <= process_bits:
                            process_bits -= qubits
                            packet.location = 'node'
                            packet.update_node_delay()
                            self.active_packets.append(packet)
                        else:
                            new_q.append([qubits - int(process_bits), packet])
                            process_bits = 0
                    if new_q:
                        node.queue_dict[session_id] = new_q
                    else:
                        del node.queue_dict[session_id]

            node.update_remain_queue()
            assert node.remain_queue >= 0

        next_active_packets = []
        for packet in self.active_packets:
            packet.etg_lifetime -= 1
            if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                packet.fail = True
                self.etg_fail_qubits += packet.qubits_len
                continue

            if packet.location == 'link':
                if packet.trans_delay > 0:
                    packet.trans_delay -= 1
                    next_active_packets.append(packet)
                elif packet.trans_delay == 0:
                    hop_distance = self.hop_distances[packet.crt, packet.next_hop]
                    packet.crt = packet.next_hop
                    packet.traveled_distance += hop_distance
                    packet.update_qubits_len(hop_distance)
                    node = self.nodes[packet.crt]
                    session_id = packet.belong
                    if node.remain_queue > packet.qubits_len:
                        packet.location = 'queue'
                        node.queue_dict.setdefault(session_id, []).append([int(packet.qubits_len), packet])
                        node.update_remain_queue()
                    elif node.remain_queue > 0:
                        packet.location = 'queue'
                        self.queue_fail_qubits += packet.qubits_len - node.remain_queue
                        packet.qubits_len = int(node.remain_queue)
                        node.queue_dict.setdefault(session_id, []).append([int(packet.qubits_len), packet])
                        node.update_remain_queue()
                    else:
                        packet.fail = True
                        self.queue_fail_qubits += packet.qubits_len
                else:
                    raise Exception('packet.trans_delay error')

            elif packet.location == 'node':
                if packet.node_delay > 0:
                    packet.node_delay -= 1
                    next_active_packets.append(packet)
                elif packet.node_delay == 0:
                    if packet.dst == packet.crt:
                        if packet.type == 'ED':
                            node = self.nodes[packet.crt]
                            if node.qm_capacity >= packet.qubits_len:
                                node.qm_capacity -= packet.qubits_len
                                packet.save_qm_qubits = packet.qubits_len
                                packet.pause_delay = int(packet.traveled_distance / LINK_VELOCITY)
                                packet.location = 'pause'
                                self.pause_packets.append(packet)
                            elif node.qm_capacity > 0:
                                self.qm_fail_qubits += packet.qubits_len - node.qm_capacity
                                packet.qubits_len = int(node.qm_capacity)
                                packet.save_qm_qubits = packet.qubits_len
                                node.qm_capacity = 0
                                packet.pause_delay = int(packet.traveled_distance / LINK_VELOCITY)
                                packet.location = 'pause'
                                self.pause_packets.append(packet)
                            else:
                                packet.fail = True
                                self.qm_fail_qubits += packet.qubits_len
                        elif packet.type == 'MS':
                            packet.success = True
                            session_id = packet.belong
                            self.sessions[session_id].dataflow += packet.qubits_len
                            self.sessions[session_id].mark_target_reached(self.time)
                            self.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                            route_key = tuple(packet.route_history)
                            self.route_switches_dict.setdefault(session_id, {})
                            self.route_switches_dict[session_id][route_key] = self.route_switches_dict[session_id].get(route_key, 0) + 1
                    else:
                        path = self.route_path(packet.crt, packet.dst, current_loads, node_buffers)
                        if path is None or len(path) < 2:
                            packet.fail = True
                            self.queue_fail_qubits += packet.qubits_len
                            continue
                        packet.guard = len(path) - 1
                        packet.next_hop = path[1]
                        packet.route_history.append(packet.next_hop)
                        packet.location = 'link'
                        hop_distance = self.hop_distances[packet.crt, packet.next_hop]
                        packet.trans_delay = int(hop_distance / LINK_VELOCITY) + 1
                        next_active_packets.append(packet)
                else:
                    raise Exception('packet.node_delay error')
            else:
                raise Exception('packet.location error')

        self.active_packets = next_active_packets

        next_pause_packets = []
        for packet in self.pause_packets:
            assert packet.location == 'pause'
            packet.etg_lifetime -= 1
            if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                packet.fail = True
                self.etg_fail_qubits += packet.qubits_len
                self.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                continue
            if packet.pause_delay > 0:
                packet.pause_delay -= 1
                next_pause_packets.append(packet)
            elif packet.pause_delay == 0:
                packet.type = 'MS'
                packet.crt = self.sessions[packet.belong].src
                packet.route_history.append(packet.crt)
                packet.traveled_distance = 0.0
                path = self.route_path(packet.crt, packet.dst, current_loads, node_buffers)
                if path is None or len(path) < 2:
                    packet.fail = True
                    self.etg_fail_qubits += packet.qubits_len
                    self.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                    continue
                packet.guard = len(path) - 1
                packet.next_hop = path[1]
                packet.route_history.append(packet.next_hop)
                packet.trans_delay = int(self.hop_distances[packet.crt, packet.next_hop] / LINK_VELOCITY) + 1
                packet.location = 'link'
                self.active_packets.append(packet)
            else:
                raise Exception('packet.pause_delay error')
        self.pause_packets = next_pause_packets

        self.cleanup()
        self.active_qubits = sum(packet.qubits_len for packet in self.active_packets)
        self.success_qubits = sum(session.dataflow for session in self.sessions)
        self.time += 1

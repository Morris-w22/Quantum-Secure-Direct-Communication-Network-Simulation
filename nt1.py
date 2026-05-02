import heapq
import math
from typing import Optional

import numpy as np

# network parameters
MTU = 1500
PROCESS_PER_STEP = 100  # 处理 qubit 速率
LINK_VELOCITY = 0.3  # km/us, 数据面传播速度
CLASSICAL_VELOCITY = 0.2  # km/us, 建链控制面速度
MAX_IP_DATAGRAM = 65536
timestep = 1  # us
protect_time = 10  # us
qubit_read_time = 0.5  # us
cbyte_read_time = 0.08  # us
alpha = 0.2

# circuit parameters
CIRCUIT_REFERENCE_RATE = 40   # 首选预留速率（参考值，不是硬阈值）
CIRCUIT_MIN_RESERVE_RATE = 5  # 对半切分后的最小可接受预留速率
CIRCUIT_RETRY_INTERVAL = 100  # 被阻塞会话的重试周期（us）


def split_integer(n, chunk_size):
    num_chunks = (n + chunk_size - 1) // chunk_size
    chunks = np.full(num_chunks, chunk_size)
    chunks[-1] = n - chunk_size * (num_chunks - 1)
    return chunks


def dijkstra_route(dist_matrix, bw_matrix, src, dst, alpha=0.9, beta=0.1):
    n = dist_matrix.shape[0]
    dist = dist_matrix.copy().astype(float)
    bw = bw_matrix.copy().astype(float)
    bw[bw <= 0] = np.inf

    max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
    finite_bw = bw[bw != np.inf]
    max_bw = np.max(finite_bw) if finite_bw.size > 0 else 1.0

    norm_dist = dist / max_dist
    norm_bw = bw / max_bw

    weight = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if dist[i, j] > 0:
                weight[i, j] = alpha * norm_dist[i, j] + beta * (1.0 / norm_bw[i, j])

    pq = [(0.0, src)]
    dist_to = [float('inf')] * n
    prev = [-1] * n
    dist_to[src] = 0.0

    while pq:
        cur_dist, u = heapq.heappop(pq)
        if u == dst:
            break
        if cur_dist > dist_to[u]:
            continue
        for v in range(n):
            if weight[u, v] == np.inf:
                continue
            nd = cur_dist + weight[u, v]
            if nd < dist_to[v]:
                dist_to[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    path = []
    cur = dst
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if not path or path[0] != src:
        return None
    return path


class Packet:
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
        self.location = 'link'  # link / queue / node / pause
        self.save_qm_qubits = 0
        self.pause_delay = 0

    def update_qubits_len(self, distance):
        prob = 10 ** (-alpha * distance / 10)
        self.qubits_len = int(np.random.binomial(self.qubits_len, prob))

    def update_node_delay(self):
        self.node_delay = int(
            (self.cbytes_len * cbyte_read_time + self.qubits_len * qubit_read_time + protect_time * self.guard) / timestep
        )
        self.guard = max(0, self.guard - 1)


class Node:
    def __init__(self, node_id: int, max_qm_capacity: int, max_node_buffer: int):
        self.id = node_id
        self.max_node_buffer = max_node_buffer
        self.remain_queue = max_node_buffer
        self.rcv_rate = PROCESS_PER_STEP
        self.snd_rate = 0
        self.queue_dict = {}
        self.qm_capacity = max_qm_capacity

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
            q_payload = np.array([], dtype=int)
            c_bytes = np.array([], dtype=int)
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
    def __init__(self, src: int, dst: int, id: int, start_time: int = 0, target_qubits: int = 0):
        self.id = id
        self.src = src
        self.dst = dst
        self.dataflow = 0

        # request lifecycle
        self.start_time = int(start_time)
        self.target_qubits = int(target_qubits)
        self.target_reached_time = None
        self.finish_time = None

        # data plane / control plane
        self.path = None
        self.path_distance = 0.0
        self.reserved_rate = 0
        self.state = 'idle'  # idle / setup / established / blocked / closing / finished
        self.setup_remaining = 0
        self.next_retry_time = self.start_time
        self.last_setup_attempts = 0

    @property
    def established(self):
        return self.state in ('setup', 'established', 'closing')

    @property
    def blocked(self):
        return self.state == 'blocked'

    @property
    def target_completed(self):
        return self.dataflow >= self.target_qubits

    @property
    def ready_to_send(self):
        return self.state == 'established' and not self.target_completed

    def request_active(self, time_now: int) -> bool:
        if time_now < self.start_time:
            return False
        return self.state != 'finished'

    def mark_target_reached(self, time_now: int):
        if self.target_reached_time is None and self.target_completed:
            self.target_reached_time = int(time_now)


class Link:
    def __init__(self, node1, node2, capacity, distance):
        self.nodes = (node1, node2)
        self.capacity = int(capacity)
        self.hop_distance = distance
        self.residual_capacity = int(capacity)
        self.reserved_sessions = []


class Network:
    def __init__(self, link_capacity_matrix: np.ndarray, hop_distance_matrix: np.ndarray, max_qm_capacity: np.ndarray, max_node_buffer: np.ndarray):
        self.hop_distances = hop_distance_matrix
        self.capacities = (link_capacity_matrix * timestep).astype(int)
        self.nodes_num = self.capacities.shape[0]

        self.nodes = [Node(i, max_qm_capacity[i], max_node_buffer[i]) for i in range(self.nodes_num)]
        self.sessions = []
        self.links = []
        self.link_dict = {}

        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                if self.capacities[i, j] != 0:
                    link = Link(i, j, self.capacities[i, j], self.hop_distances[i, j])
                    self.links.append(link)
                    self.link_dict[(i, j)] = link

        self.active_packets = []
        self.pause_packets = []

        self.active_qubits = 0
        self.success_qubits = 0
        self.etg_fail_qubits = 0
        self.queue_fail_qubits = 0
        self.qm_fail_qubits = 0
        self.blocked_sessions = 0
        self.established_sessions = 0
        self.finished_sessions = 0
        self.time = 0

    def get_residual_bw_matrix(self):
        bw = np.zeros_like(self.capacities)
        for (u, v), link in self.link_dict.items():
            bw[u, v] = max(0, link.residual_capacity)
        return bw

    def session_has_live_packets(self, session_id: int) -> bool:
        for packet in self.active_packets:
            if packet.belong == session_id:
                return True
        for packet in self.pause_packets:
            if packet.belong == session_id:
                return True
        for node in self.nodes:
            q = node.queue_dict.get(session_id)
            if q:
                return True
        return False

    def release_circuit(self, session: Session):
        if session.path is not None and session.reserved_rate > 0:
            for u, v in zip(session.path[:-1], session.path[1:]):
                link = self.link_dict[(u, v)]
                link.residual_capacity += session.reserved_rate
                if session.id in link.reserved_sessions:
                    link.reserved_sessions.remove(session.id)
        session.reserved_rate = 0

    def choose_reserved_rate(self, bottleneck: int):
        rate = CIRCUIT_REFERENCE_RATE
        attempts = 1
        while rate > bottleneck and rate > CIRCUIT_MIN_RESERVE_RATE:
            rate = max(rate // 2, CIRCUIT_MIN_RESERVE_RATE)
            attempts += 1
        if rate > bottleneck:
            return 0, attempts
        return int(rate), attempts

    def try_build_one_session(self, session: Session):
        if session.target_completed:
            session.state = 'finished'
            session.finish_time = self.time if session.finish_time is None else session.finish_time
            return

        residual_bw = self.get_residual_bw_matrix()
        path = dijkstra_route(self.hop_distances, residual_bw, session.src, session.dst)
        if path is None or len(path) < 2:
            session.state = 'blocked'
            session.next_retry_time = self.time + CIRCUIT_RETRY_INTERVAL
            return

        path_distance = 0.0
        bottleneck = float('inf')
        for u, v in zip(path[:-1], path[1:]):
            path_distance += self.hop_distances[u, v]
            bottleneck = min(bottleneck, self.link_dict[(u, v)].residual_capacity)

        reserve_rate, attempts = self.choose_reserved_rate(int(bottleneck))
        session.last_setup_attempts = attempts
        session.path = path
        session.path_distance = path_distance

        if reserve_rate <= 0:
            session.state = 'blocked'
            session.next_retry_time = self.time + CIRCUIT_RETRY_INTERVAL
            return

        session.reserved_rate = reserve_rate
        for u, v in zip(path[:-1], path[1:]):
            link = self.link_dict[(u, v)]
            link.residual_capacity -= reserve_rate
            link.reserved_sessions.append(session.id)

        rtt = max(1, int(math.ceil(2 * path_distance / CLASSICAL_VELOCITY)))
        session.setup_remaining = attempts * rtt
        session.state = 'setup'
        session.next_retry_time = self.time

    def process_session_control(self):
        for session in self.sessions:
            if session.state == 'finished':
                continue
            if self.time < session.start_time:
                continue

            # 达到任务量后停止继续注入；已建链会话等待在途分组排空后释放电路
            if session.target_reached_time is not None:
                if session.state == 'setup':
                    self.release_circuit(session)
                    session.state = 'finished'
                    session.finish_time = self.time if session.finish_time is None else session.finish_time
                    continue
                if session.state == 'established':
                    session.state = 'closing'
                elif session.state in ('idle', 'blocked'):
                    session.state = 'finished'
                    session.finish_time = self.time if session.finish_time is None else session.finish_time
                    continue

            if session.state == 'closing':
                if not self.session_has_live_packets(session.id):
                    self.release_circuit(session)
                    session.state = 'finished'
                    session.finish_time = self.time if session.finish_time is None else session.finish_time
                continue

            if session.state == 'setup':
                if session.setup_remaining > 0:
                    session.setup_remaining -= 1
                if session.setup_remaining == 0:
                    if session.target_reached_time is not None:
                        self.release_circuit(session)
                        session.state = 'finished'
                        session.finish_time = self.time if session.finish_time is None else session.finish_time
                    else:
                        session.state = 'established'
                continue

            if session.state in ('idle', 'blocked') and not session.target_completed:
                if self.time >= session.next_retry_time:
                    self.try_build_one_session(session)

        self.established_sessions = sum(1 for s in self.sessions if s.state in ('setup', 'established', 'closing'))
        self.blocked_sessions = sum(1 for s in self.sessions if s.state == 'blocked')
        self.finished_sessions = sum(1 for s in self.sessions if s.state == 'finished')

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
        self.process_session_control()

        # 1. 源节点发送：仅建链完成且尚未完成任务量的会话允许注入新包
        for session in self.sessions:
            if not session.ready_to_send:
                continue
            node = self.nodes[session.src]
            next_node = session.path[1]
            next_node_obj = self.nodes[next_node]
            queue_sessions = max(1, len(next_node_obj.queue_dict))
            node.snd_rate = int(min(
                session.reserved_rate,
                next_node_obj.rcv_rate / queue_sessions,
                next_node_obj.remain_queue
            ))
            if node.snd_rate <= 0:
                continue
            new_packets = node.snd_packets(session.id, len(session.path) - 1, session.src, session.dst)
            hop_distance = self.hop_distances[session.src, next_node]
            for packet in new_packets:
                packet.trans_delay = int(hop_distance / LINK_VELOCITY) + 1
            self.active_packets.extend(new_packets)

        # 2. 更新队列包
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

        # 3. 更新 active packets
        next_active_packets = []
        for packet in self.active_packets:
            packet.etg_lifetime -= 1
            if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                packet.fail = True
                self.etg_fail_qubits += packet.qubits_len
                continue

            session = self.sessions[packet.belong]
            path = session.path

            if packet.location == 'link':
                if packet.trans_delay > 0:
                    packet.trans_delay -= 1
                    next_active_packets.append(packet)
                elif packet.trans_delay == 0:
                    idx = path.index(packet.crt)
                    next_node = path[idx + 1]
                    hop_distance = self.hop_distances[packet.crt, next_node]
                    packet.crt = next_node
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
                                packet.pause_delay = int(session.path_distance / LINK_VELOCITY)
                                packet.location = 'pause'
                                self.pause_packets.append(packet)
                            elif node.qm_capacity > 0:
                                self.qm_fail_qubits += packet.qubits_len - node.qm_capacity
                                packet.qubits_len = int(node.qm_capacity)
                                packet.save_qm_qubits = packet.qubits_len
                                node.qm_capacity = 0
                                packet.pause_delay = int(session.path_distance / LINK_VELOCITY)
                                packet.location = 'pause'
                                self.pause_packets.append(packet)
                            else:
                                packet.fail = True
                                self.qm_fail_qubits += packet.qubits_len
                        elif packet.type == 'MS':
                            packet.success = True
                            session_obj = self.sessions[packet.belong]
                            session_obj.dataflow += packet.qubits_len
                            session_obj.mark_target_reached(self.time)
                            self.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                    else:
                        idx = path.index(packet.crt)
                        next_node = path[idx + 1]
                        packet.location = 'link'
                        hop_distance = self.hop_distances[packet.crt, next_node]
                        packet.trans_delay = int(hop_distance / LINK_VELOCITY) + 1
                        next_active_packets.append(packet)
                else:
                    raise Exception('packet.node_delay error')
            else:
                raise Exception('packet.location error')

        self.active_packets = next_active_packets

        # 4. 更新 pause packets：MS 阶段回源后仍沿固定电路路径发送
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
                session = self.sessions[packet.belong]
                packet.type = 'MS'
                packet.crt = session.src
                packet.guard = len(session.path) - 1
                first_hop = session.path[1]
                packet.trans_delay = int(self.hop_distances[packet.crt, first_hop] / LINK_VELOCITY) + 1
                packet.location = 'link'
                self.active_packets.append(packet)
            else:
                raise Exception('packet.pause_delay error')
        self.pause_packets = next_pause_packets

        self.cleanup()
        self.active_qubits = sum(packet.qubits_len for packet in self.active_packets)
        self.success_qubits = sum(session.dataflow for session in self.sessions)
        self.time += 1

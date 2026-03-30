import numpy as np
import heapq

# network parameters
MTU = 1500
PROCESS_PER_STEP = 100 # 处理qubit速率
max_IP_datagram = 65536
timestep = 1 # us
protect_time = 10 # us
qubit_read_time = 0.5 # us
cbyte_read_time = 0.08 # us
alpha = 0.2

def split_integer(n, chunk_size):
    num_chunks = (n + chunk_size - 1) // chunk_size    
    chunks = np.full(num_chunks, chunk_size)
    chunks[-1] = n - chunk_size * (num_chunks - 1)    
    return chunks

def dijkstra_next_hop_correct(dist_matrix, current_loads, node_buffers, src, dst, link_dict):
    """
    正确的动态路由函数，基于链路字典的Dijkstra算法
    
    Args:
        dist_matrix: 物理距离矩阵，仅用于Dijkstra中的启发
        current_loads: 链路负载率矩阵 shape (n, n) [0,1]
        node_buffers: 节点缓冲向量 shape (n,)
        src: 源节点ID
        dst: 目标节点ID
        link_dict: 真实存在的链路字典 {(from_node, to_node): Link对象}
    
    Returns:
        next_hop: 下一跳节点ID，或None（如果无法路由）
    
    工作原理:
    1. 构建权重矩阵，仅在link_dict中存在的链路上计算权重
    2. 使用Dijkstra算法找最优路径
    3. 返回路径中的第二个节点作为next_hop
    4. 如果无法到达，返回None
    """
    
    if src == dst:
        return None
    
    # ===== 步骤1：构建权重矩阵（仅在真实链路上有值）=====
    n = dist_matrix.shape[0]
    weight = np.full((n, n), np.inf)
    
    # 遍历所有真实存在的链路
    for (from_node, to_node) in link_dict.keys():
        # 只对真实存在且距离矩阵中有值的链路计算权重
        if dist_matrix[from_node][to_node] > 0:
            # 距离部分（归一化）
            max_dist = np.max(dist_matrix[dist_matrix > 0])
            dist_norm = dist_matrix[from_node][to_node] / max_dist if max_dist > 0 else 0
            
            # 拥塞部分：链路负载越高，权重越大
            congestion = current_loads[from_node][to_node]  # [0, 1]
            
            # 缓冲部分：下一跳节点缓冲越满，权重越大
            max_buffer = np.max(node_buffers)
            if max_buffer > 0:
                buffer_pressure = 1.0 - (node_buffers[to_node] / max_buffer)  # [0, 1]
            else:
                buffer_pressure = 0.5
            
            # 综合权重（距离、拥塞、缓冲）
            w = 0.7 * dist_norm + 0.2 * congestion + 0.1 * buffer_pressure
            weight[from_node][to_node] = w
    
    # ===== 步骤2：使用Dijkstra算法计算最短路径 =====
    dist_to = [float('inf')] * n
    prev = [-1] * n
    dist_to[src] = 0
    visited = set()
    
    # 标准Dijkstra实现
    for _ in range(n):
        # 找未访问的距离最小的节点
        u = -1
        for v in range(n):
            if v not in visited and dist_to[v] < float('inf'):
                if u == -1 or dist_to[v] < dist_to[u]:
                    u = v
        
        # 没有可达节点，停止
        if u == -1 or dist_to[u] == float('inf'):
            break
        
        visited.add(u)
        if u == dst:
            break
        
        # 松弛相邻节点（仅通过真实链路）
        for (from_node, to_node) in link_dict.keys():
            if from_node == u and weight[from_node][to_node] != np.inf:
                new_dist = dist_to[u] + weight[from_node][to_node]
                if new_dist < dist_to[to_node]:
                    dist_to[to_node] = new_dist
                    prev[to_node] = u
    
    # ===== 步骤3：检查是否可达 =====
    if dist_to[dst] == float('inf'):
        # 目标节点无法到达
        return None
    
    # ===== 步骤4：还原路径 =====
    path = []
    cur = dst
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    
    # 路径长度必须至少为2（src和dst）
    if len(path) < 2:
        return None
    
    # ===== 步骤5：返回下一跳 =====
    next_hop = path[1]
    
    # 最后验证：确保这条链路真的存在
    if (src, next_hop) not in link_dict:
        print(f"[ERROR] Dijkstra返回了不��在的链路: {src}->{next_hop}")
        return None
    
    return next_hop


class Packet:
    def __init__(self, src: int, dst: int, qubits_len: int, cbytes_len: int, session_id: int, guard):
        self.src = src
        self.dst = dst
        self.crt = src # 当前节点
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
        self.location = "link"
        
        self.next_hop = dst
        self.route_history = [src]

    def update_qubits_len(self, distance):
        prob = 10**(-alpha*distance/10)
        self.qubits_len = int(np.random.binomial(self.qubits_len, prob))
        
    def update_node_delay(self):
        self.node_delay = int((self.cbytes_len*cbyte_read_time + self.qubits_len*qubit_read_time + protect_time*self.guard) / timestep)
        self.guard -= 1


class Node:
    def __init__(self, node_id: int, max_qm_capacity: int, max_node_buffer: int):
        self.id = node_id
        self.max_node_buffer = max_node_buffer
        self.remain_queue = max_node_buffer
        self.rcv_rate = PROCESS_PER_STEP
        self.snd_rate = 0

        self.queue_dict = {}
        self.qm_capacity = max_qm_capacity
        self.start_node = []
    
    def snd_packets(self, session_id, src_node, dst_node):
        QTP_header_len = 32 + self.snd_rate*2
        if QTP_header_len+16 <= MTU:
            q_payload = np.array([self.snd_rate])
            c_bytes = np.array([QTP_header_len + 16 + 26])
        elif QTP_header_len+16 > MTU and QTP_header_len <= max_IP_datagram:
            MAC_frags = split_integer(QTP_header_len+16, MTU)
            q_payload = MAC_frags // 2
            q_payload[0] -= 24
            c_bytes = MAC_frags + 26
        elif QTP_header_len > max_IP_datagram:
            IP_frags = split_integer(QTP_header_len, max_IP_datagram)
            c_bytes = np.array([])
            q_payload = np.array([])
            for IP_frag in IP_frags:
                MAC_frags = split_integer(IP_frag+16, MTU)
                if len(q_payload) == 0:
                    MAC_frags[0] -= 48
                else:
                    MAC_frags[0] -= 16
                q_payload = np.append(q_payload, MAC_frags // 2)
                c_bytes = np.append(c_bytes, MAC_frags + 26)
        
        assert len(q_payload) == len(c_bytes)
        assert np.sum(q_payload) == self.snd_rate
        new_packets = []
        for i in range(len(q_payload)):
            new_packets.append(Packet(src_node, dst_node, int(q_payload[i]), int(c_bytes[i]), session_id, 5))
        return new_packets
        
    def update_queue(self, network):
        if self.queue_dict:
            for session_id in list(self.queue_dict.keys()):
                remove = 0
                for i, zip in enumerate(self.queue_dict[session_id].copy()):
                    packet = zip[1]
                    assert packet.location == "queue"
                    packet.etg_lifetime -= 1
                    if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                        packet.fail = True
                        network.etg_fail_qubits += packet.qubits_len
                        del self.queue_dict[session_id][i-remove]
                        remove += 1            
            
            for session_id in list(self.queue_dict.keys()):
                process_bits = int(self.rcv_rate / max(1, len(self.queue_dict)))
                remove = 0
                for i, zip in enumerate(self.queue_dict[session_id].copy()):                    
                    qubits, packet = zip[0], zip[1]
                    assert packet.location == "queue"                
                    if qubits <= process_bits:
                        process_bits -= qubits 
                        packet.location = "node"
                        network.active_packets.append(packet)
                        packet.update_node_delay()
                        del self.queue_dict[session_id][i-remove]            
                        remove += 1
                    elif process_bits > 0:
                        self.queue_dict[session_id][i-remove][0] -= int(process_bits)
                        break
                    else:
                        break
        
            for session_id in list(self.queue_dict.keys()):
                if self.queue_dict[session_id] == []:
                    del self.queue_dict[session_id]
        
    def update_remain_queue(self):
        self.remain_queue = self.max_node_buffer - sum(
            item[0] for lst in self.queue_dict.values() for item in lst
        )

class Session:
    def __init__(self, src: int, dst: int, start_time, id: int, data_len: int):
        self.id = id
        self.src = src
        self.dst = dst
        self.start_time = start_time
        self.dataflow = data_len
        self.remain_bit = data_len
        self.active = False

class Link:
    def __init__(self, node1, node2, capacity, distance):
        self.nodes = (node1, node2)
        self.capacity = capacity
        self.hop_distance = distance
        self.current_load = 0
        self.snd_rate_per_session = capacity
        self.in_sessions = 0
    
    def get_current_load_ratio(self):
        """获取当前链路使用率 [0, 1]"""
        if self.capacity == 0:
            return 1.0
        ratio = min(1.0, self.current_load / self.capacity)
        return ratio
    

class Network:
    def __init__(self, link_capacity_matrix: np.ndarray, hop_distance_matrix: np.ndarray, max_qm_capacity: np.ndarray, max_node_buffer: np.ndarray):
        self.hop_distances = hop_distance_matrix
        self.capacities = link_capacity_matrix*timestep
        self.nodes_num = self.capacities.shape[0]

        self.nodes = []
        self.sessions = []
        self.links = []
        self.link_dict = {}
        
        for i in range(self.nodes_num):
            self.nodes.append(Node(i, max_qm_capacity[i], max_node_buffer[i]))
        
        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                if self.capacities[i][j] != 0:
                    link = Link(i, j, self.capacities[i][j], self.hop_distances[i][j])
                    self.links.append(link)
                    self.link_dict[(i, j)] = link

        self.active_sessions = []
        self.active_packets = []
        self.pause_packets = []

        self.active_qubits = 0
        self.success_qubits = 0
        self.etg_fail_qubits = 0 # 由于ETG超时丢弃的qubit数量
        self.queue_fail_qubits = 0 # 由于缓冲区已满丢弃的qubit数量
        self.qm_fail_qubits = 0 # 由于量子存储已满丢弃的qubit数量

    def make_sessions(self, info: np.ndarray, qubits_total_len: np.ndarray):
        for i in range(info.shape[0]):
            self.sessions.append(Session(int(info[i][0]), int(info[i][1]), info[i][2], i, qubits_total_len[i]))

    def activate_sessions(self, time):
        """激活会话，并注册源节点"""
        for session in self.sessions:
            if (not session.active) and (time >= session.start_time):
                session.active = True                
                self.active_sessions.append(session)               
                if session.id not in self.nodes[session.src].start_node:
                    self.nodes[session.src].start_node.append(session.id)
    
    def update_link_states(self):
        """更新所有链路的当前负载状态"""
        for link in self.links:
            link.current_load = 0
        
        for packet in self.active_packets:
            if packet.location == "link":
                next_hop = packet.next_hop
                if next_hop is not None and (packet.crt, next_hop) in self.link_dict:
                    link = self.link_dict[(packet.crt, next_hop)]
                    link.current_load += packet.qubits_len
    
    def get_current_loads_matrix(self):
        """获取当前链路负载率矩阵"""
        loads = np.zeros((self.nodes_num, self.nodes_num))
        for link in self.links:
            loads[link.nodes[0]][link.nodes[1]] = link.get_current_load_ratio()
        return loads
    
    def get_node_buffers_vector(self):
        """获取各节点的剩余缓冲容量向量"""
        buffers = np.array([node.remain_queue for node in self.nodes])
        return buffers
    
    def allocate_bandwidth_to_sessions(self):
        """
        为活跃的会话分配链路带宽
        使用动态计算的路径
        """
        for link in self.links:
            link.in_sessions = 0
        
        for session in self.active_sessions:
            current_loads = self.get_current_loads_matrix()
            node_buffers = self.get_node_buffers_vector()
            
            # 为这个会话计算路径（用于带宽分配统计）
            # 这里使用dijkstra_next_hop_correct逐跳计算来估算路径
            path = self._estimate_session_path(session.src, session.dst, current_loads, node_buffers)
            
            if path and len(path) > 1:
                for i in range(len(path) - 1):
                    for link in self.links:
                        if link.nodes == (path[i], path[i+1]):
                            link.in_sessions += 1
                            break
        
        for link in self.links:
            if link.in_sessions == 0:
                link.snd_rate_per_session = link.capacity
            else:
                link.snd_rate_per_session = int(link.capacity / link.in_sessions)
    
    def _estimate_session_path(self, src, dst, current_loads, node_buffers):
        """估算会话路径（通过逐跳next_hop）"""
        path = [src]
        current = src
        visited = {src}
        max_iterations = self.nodes_num
        
        for _ in range(max_iterations):
            if current == dst:
                return path
            
            next_hop = dijkstra_next_hop_correct(self.hop_distances, current_loads, 
                                                 node_buffers, current, dst, self.link_dict)
            if next_hop is None or next_hop in visited:
                # 无法继续，返回部分路径
                break
            
            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
        
        return path if current == dst else None

    def cleanup(self):        
        for packet in self.pause_packets.copy():
            assert packet.fail == False
        for node in self.nodes:
            for session_id in list(node.queue_dict.keys()):
                remove = 0
                zip_lst = node.queue_dict[session_id]
                for i, zip in enumerate(zip_lst):
                    packet = zip[1]
                    if packet.fail == True:
                        del node.queue_dict[session_id][i-remove]
                        remove += 1
                        if packet.type == "MS":
                            self.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                        self.queue_fail_qubits += packet.qubits_len
            
        new_active_sessions = []
        for session in self.active_sessions:
            if session.remain_bit <= 0:
                self.nodes[session.src].start_node.remove(session.id)
            else:
                new_active_sessions.append(session)

        self.active_sessions = new_active_sessions
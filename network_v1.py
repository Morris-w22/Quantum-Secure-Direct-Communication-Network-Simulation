import numpy as np
import heapq

# network parameters
MTU = 1500
MAX_QM_CAPACITY = 100000
PROCESS_PER_STEP = 100 # 处理qubit速率
max_IP_datagram = 65536
timestep = 1 # us
protect_time = 10 # us
qubit_read_time = 0.5 # us
cbyte_read_time = 0.08 # us
alpha = 0.2

def split_integer(n, chunk_size):
    # 计算划分后的数量
    num_chunks = (n + chunk_size - 1) // chunk_size    
    # 创建一个数组，表示每个块的大小
    chunks = np.full(num_chunks, chunk_size)
    # 对最后一块进行调整，确保不超过n
    chunks[-1] = n - chunk_size * (num_chunks - 1)    
    return chunks

def dijkstra_route(dist_matrix, bw_matrix, src, dst, alpha=0.9, beta=0.1):
    n = dist_matrix.shape[0]
    # ===== 归一化 =====
    dist = dist_matrix.copy().astype(float)
    bw = bw_matrix.copy().astype(float)
    # 避免除0
    bw[bw == 0] = np.inf
    # 只对非零边做归一化
    max_dist = np.max(dist)
    max_bw = np.max(bw[bw != np.inf])
    norm_dist = dist / max_dist if max_dist > 0 else dist
    norm_bw = bw / max_bw if max_bw > 0 else bw
    # ===== 构建权重矩阵 =====
    weight = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if dist[i][j] > 0:  # 有边
                w = alpha * norm_dist[i][j] + beta * (1 / norm_bw[i][j])
                weight[i][j] = w
    # ===== Dijkstra =====
    pq = [(0, src)]
    dist_to = [float('inf')] * n
    prev = [-1] * n
    dist_to[src] = 0
    while pq:
        cur_dist, u = heapq.heappop(pq)
        if u == dst:
            break
        if cur_dist > dist_to[u]:
            continue
        for v in range(n):
            if weight[u][v] == np.inf:
                continue
            new_dist = cur_dist + weight[u][v]
            if new_dist < dist_to[v]:
                dist_to[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))
    # ===== 还原路径 =====
    path = []
    cur = dst
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if path[0] != src:
        return None  # 不可达
    return path

class Packet:
    def __init__(self, src: int, dst: int, qubits_len: int, cbytes_len: int, session_id: int, guard):
        self.src = src
        self.dst = dst
        self.crt = src # 当前节点
        self.qubits_len = qubits_len
        self.cbytes_len = cbytes_len
        self.belong = session_id
        self.guard = guard # 还未经过的节点数（用于计算保护时间）
        self.node_delay = -100 # 到达下一节点处理时间消耗（初始为-100，表示未计算）
        self.trans_delay = -100 # 链路时间消耗（初始为-100，表示未计算）
        self.type = 'ED' # 初始为'ED'，表示为纠缠分发包
        self.etg_lifetime = 1000 # 最大纠缠寿命
        self.success = False
        self.fail = False
        self.location = "link" # 初始为"link"，表示在链路上传输("node","queue","pause")

    def update_qubits_len(self, distance):
        prob = 10**(-alpha*distance/10)
        self.qubits_len = int(np.random.binomial(self.qubits_len, prob))
        
    def update_node_delay(self):
        self.node_delay = int((self.cbytes_len*cbyte_read_time + self.qubits_len*qubit_read_time + protect_time*self.guard) / timestep)
        self.guard -= 1
        

class Node:
    def __init__(self, node_id: int, max_node_buffer):
        self.id = node_id
        self.max_node_buffer = max_node_buffer
        self.remain_queue = max_node_buffer
        self.rcv_rate = PROCESS_PER_STEP # 由硬件决定
        self.snd_rate = 0 # 后续计算

        self.queue_dict = {} # 缓冲区(qubit, packet)队列(按会话做区分)
        
        self.qm_capacity = MAX_QM_CAPACITY # 量子存储容量
        self.start_node = [] # 当前节点作为起始节点参与的所有会话
    
    def snd_packets(self, session_id, guard, src_node, dst_node):
        QTP_header_len = 32 + self.snd_rate*2 # QTP头部长度
        if QTP_header_len+16 <= MTU:
            # 不用分片
            q_payload = np.array([self.snd_rate])
            c_bytes = np.array([QTP_header_len + 16 + 26])
        elif QTP_header_len+16 > MTU and QTP_header_len <= max_IP_datagram:
            # MAC分片
            MAC_frags = split_integer(QTP_header_len+16, MTU)
            q_payload = MAC_frags // 2 # 每个片的纠缠对数量
            q_payload[0] -= 24 # 减去IP、QTP头部长度
            c_bytes = MAC_frags + 26 # 每个片的经典部分长度
        elif QTP_header_len > max_IP_datagram:
            # IP分片后MAC分片
            IP_frags = split_integer(QTP_header_len, max_IP_datagram)
            c_bytes = np.array([])
            q_payload = np.array([])
            for IP_frag in IP_frags:
                MAC_frags = split_integer(IP_frag+16, MTU)
                if len(q_payload) == 0:
                    MAC_frags[0] -= 48 # 减去IP、QTP头部长度
                else:
                    MAC_frags[0] -= 16 # 减去IP头部长度
                q_payload = np.append(q_payload, MAC_frags // 2)
                c_bytes = np.append(c_bytes, MAC_frags + 26)
        assert len(q_payload) == len(c_bytes)
        assert np.sum(q_payload) == self.snd_rate
        new_packets = []
        for i in range(len(q_payload)):
            new_packets.append(Packet(src_node, dst_node, int(q_payload[i]), int(c_bytes[i]), session_id, guard))
        return new_packets
        
    def update_queue(self, network):
        if self.queue_dict: # 如果缓冲区不为空
            for session_id in list(self.queue_dict.keys()):
                remove = 0
                for i, zip in enumerate(self.queue_dict[session_id].copy()):
                    packet = zip[1]
                    assert packet.location == "queue"
                    packet.etg_lifetime -= 1
                    if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                        packet.fail = True
                        network.etg_fail_qubits += packet.qubits_len
                        # 删除包
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
                        # packet已被读取
                        packet.location = "node"
                        network.active_packets.append(packet)
                        packet.update_node_delay() # 计算节点时延
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
        self.dataflow = data_len # 数据流总长
        self.remain_bit = data_len # 还需传输的bit
        self.active = False

class Link:
    def __init__(self, node1, node2, capacity, distance):
        self.nodes = (node1, node2) # 约定从node1发向node2
        self.capacity = capacity
        self.hop_distance = distance
        self.in_sessions = 0
        self.snd_rate_per_session = 0

class Network:
    def __init__(self, link_capacity_matrix: np.ndarray, hop_distance_matrix: np.ndarray, max_node_buffer: np.ndarray):
        self.hop_distances = hop_distance_matrix
        self.capacities = link_capacity_matrix*timestep # 每个时间步链路的最大传输比特量
        self.nodes_num = self.capacities.shape[0]

        self.nodes = []
        self.sessions = []
        self.links = []
        # 创建节点类的集合
        for i in range(self.nodes_num):
            self.nodes.append(Node(i, max_node_buffer[i]))
        # 创建链路类
        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                if self.capacities[i][j] != 0:
                    self.links.append(Link(i, j, self.capacities[i][j], self.hop_distances[i][j]))

        self.active_sessions = []
        self.active_packets = [] # 在网络中的包
        self.pause_packets = [] # ED与MS间隔中的包

        # 统计结果参数
        self.active_qubits = 0
        self.success_qubits = 0
        self.etg_fail_qubits = 0
        self.throw_fail_qubits = 0

    def make_sessions(self, info: np.ndarray, qubits_total_len: np.ndarray):
        # 创建会话类
        for i in range(info.shape[0]):
            self.sessions.append(Session(int(info[i][0]), int(info[i][1]), info[i][2], i, qubits_total_len[i]))

    def activate_sessions(self, time):
        for session in self.sessions:
            if (not session.active) and (time >= session.start_time):
                # 激活会话
                session.active = True                
                self.active_sessions.append(session)
                # 添加会话路径
                node_list_path = dijkstra_route(self.hop_distances, self.capacities, session.src, session.dst)
                setattr(session, "path", node_list_path)
                path_distance = 0
                for i in range(len(node_list_path)-1):
                    path_distance += self.hop_distances[node_list_path[i]][node_list_path[i+1]]
                setattr(session, "path_distance", path_distance)
    
    def link_capacity_allocation(self):
        for link in self.links:
            link.in_sessions = 0
        for session in self.active_sessions:
            for i in range(len(session.path)-1):
                for link in self.links:
                    if link.nodes == (session.path[i], session.path[i+1]):
                        link.in_sessions += 1

        for link in self.links:
            if link.in_sessions == 0:
                link.in_sessions = 1
            link.snd_rate_per_session = int(link.capacity / link.in_sessions)
        
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
                        del node.queue_dict[session_id][i-remove] # 删除队列中的纠缠失效包
                        remove += 1
                        if packet.type == "MS":
                            self.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                        self.throw_fail_qubits += packet.qubits_len
            
        new_active_sessions = []
        for session in self.active_sessions:
            if session.remain_bit <= 0:
                # 更新节点所在的会话情况
                self.nodes[session.src].start_node.remove(session.id)
            else:
                new_active_sessions.append(session)

        self.active_sessions = new_active_sessions


import network_v1 as nt
import numpy as np
import matplotlib.pyplot as plt

# input
# 每条链路的量子带宽（单位时间步长能最多传输多少qubit）；逐跳长度
link_capacity_matrix = 10*np.array([
    [0, 10, 0, 0, 6, 3, 0, 0, 0, 0],
    [10, 0, 7, 0, 0, 8, 0, 10, 9, 0],
    [0, 7, 0, 8, 0, 6, 0, 0, 0, 0],
    [0, 0, 8, 0, 7, 3, 0, 0, 0, 6],
    [6, 0, 0, 7, 0, 4, 8, 0, 0, 0],
    [3, 8, 6, 3, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 12],
    [0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 12, 0, 0, 0]
])
L = 5/np.sin(np.pi/5)

hop_distance_matrix = np.array([
    [0, 10, 0, 0, 10, L, 0, 0, 0, 0],
    [10, 0, 10, 0, 0, L, 0, 5, 5, 0],
    [0, 10, 0, 10, 0, L, 0, 0, 0, 0],
    [0, 0, 10, 0, 10, L, 0, 0, 0, 10],
    [10, 0, 0, 10, 0, L, 10, 0, 0, 0],
    [L, L, L, L, L, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 10, 0, 0, 0, 0, 10],
    [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 10, 0, 0, 10, 0, 0, 0]
])

T_STEPS = 15000
SESSIONS_NUM = 20
LINK_VELOCITY = 0.3 #km/us
AVG_SESSION_INTERVAL = 10 #us
INCUM = 192 # 扩频比
max_qm_capacity = np.ones((link_capacity_matrix.shape[0])) * 100000 # 每个节点的量子存储容量
max_node_buffer = np.ones((link_capacity_matrix.shape[0])) * 2000 # 每个节点的最大缓冲区大小
dataflow = np.ones((SESSIONS_NUM)) * 100 # 每个会话需要传输的量子比特数

def sessions_start_time(nodes_num: int, sessions_num: int):
    rng = np.random.default_rng() # 固定的随机数生成器
    start_time_list = np.zeros(sessions_num, dtype=int) # 同时到达
    src_list = np.zeros(sessions_num, dtype=int)
    dst_list = np.zeros_like(src_list)
    for i in range(sessions_num):
        src_list[i], dst_list[i] = rng.choice(nodes_num, size=2, replace=False)
    return np.column_stack((src_list, dst_list, start_time_list))


if __name__ == "__main__":
    # preparation
    print("Constructing network...")
    simulator = nt.Network(link_capacity_matrix, hop_distance_matrix, max_qm_capacity, max_node_buffer)
    # 0. 计算会话的启动时间和信息量
    sessions_info = sessions_start_time(simulator.nodes_num, sessions_num=SESSIONS_NUM)
    simulator.make_sessions(sessions_info, dataflow*INCUM)
    #simulator.make_sessions(np.array([[9, 1, 0], [3, 7, 0], [6, 8, 0], [1, 8, 0], [2, 3, 0], [3, 8, 0], [8, 0, 0], [5, 0, 0], [4, 3, 0], [4, 7, 0]]), dataflow*INCUM)
    
    # iteration
    print("Simulation started.")
    time = 0
    end_time = T_STEPS
    _active_qubits = np.zeros((T_STEPS+1))
    _success_session_qubits = np.zeros((SESSIONS_NUM, T_STEPS+1))

    while time <= T_STEPS:
        # 1. 激活到达的会话，计算会话路径
        simulator.activate_sessions(time)

        # 2. 会话路径添加到节点
        for session in simulator.active_sessions:
            node_path_list = session.path
            for index, node in enumerate(node_path_list):
                if index == 0 and session.id not in simulator.nodes[node].start_node:
                    simulator.nodes[node].start_node.append(session.id)
                    break

        # 3. 更新网络链路带宽对会话的分配
        simulator.link_capacity_allocation()       

        # 4. 存在会话的节点发送packet
        for node in simulator.nodes:
            for session_id in node.start_node:
                # 计算发送速率
                next_node = simulator.sessions[session_id].path[1]
                for link in simulator.links:
                    if link.nodes[0] == node.id and link.nodes[1] == next_node:
                        max_snd_rate = link.snd_rate_per_session
                        break
                dst_node = simulator.nodes[simulator.sessions[session_id].dst]
                queue_sessions = max(1, len(simulator.nodes[next_node].queue_dict))
                node.snd_rate = int(min(max_snd_rate, dst_node.rcv_rate / queue_sessions, simulator.nodes[next_node].remain_queue, simulator.sessions[session_id].remain_bit))
                # 发送packet
                if node.snd_rate > 0:
                    new_packets = node.snd_packets(session_id, len(simulator.sessions[session_id].path)-1, simulator.sessions[session_id].src, simulator.sessions[session_id].dst)
                    # 计算新激活包的传输时延
                    for packet in new_packets:
                        packet.trans_delay = int(simulator.hop_distances[packet.src][packet.dst] / LINK_VELOCITY) + 1
                    simulator.active_packets.extend(new_packets) 

        # 5. 更新节点排队队列
        for node in simulator.nodes:
            node.update_queue(simulator) # 接收排队
            node.update_remain_queue()
            assert node.remain_queue >= 0

        # 6. 更新网络中的packet状态
        for packet in simulator.active_packets.copy():
            # 纠缠时效
            packet.etg_lifetime -= 1
            if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                packet.fail = True
                simulator.etg_fail_qubits += packet.qubits_len
                # 删除包
                simulator.active_packets.remove(packet)
                continue

            if packet.location == "link":
                if packet.trans_delay > 0:
                    # 更新链路包的传输时延
                    packet.trans_delay -= 1
                elif packet.trans_delay == 0 :
                    node_list_path = simulator.sessions[packet.belong].path
                    next_node = node_list_path[node_list_path.index(packet.crt) + 1]
                    hop_distance = simulator.hop_distances[packet.crt][next_node]
                    packet.crt = next_node
                    # 变为队列包，重置                   
                    simulator.active_packets.remove(packet)
                    # 更新有效载荷长度，计算节点时延，并缩短保护时间
                    packet.update_qubits_len(hop_distance)
                    node = simulator.nodes[packet.crt]                   
                    if node.remain_queue > packet.qubits_len: 
                        packet.location = "queue"
                        session_id = packet.belong
                        if session_id not in node.queue_dict:
                            node.queue_dict[session_id] = []
                        node.queue_dict[session_id].append([int(packet.qubits_len), packet])
                        node.update_remain_queue()
                        assert node.remain_queue > 0
                    elif node.remain_queue > 0:
                        # 丢弃部分
                        packet.location = "queue" 
                        session_id = packet.belong
                        if session_id not in node.queue_dict:
                            node.queue_dict[session_id] = []
                        simulator.queue_fail_qubits += packet.qubits_len - node.remain_queue
                        packet.qubits_len = node.remain_queue
                        node.queue_dict[session_id].append([int(packet.qubits_len), packet])
                        node.update_remain_queue()
                        assert node.remain_queue == 0
                    else:
                        # 队列已满，丢弃
                        packet.fail = True
                        simulator.queue_fail_qubits += packet.qubits_len
                        continue
                else:
                    raise Exception("packet.trans_delay error")
            elif packet.location == "node":
                if packet.node_delay > 0:
                    # 更新节点包的处理时延
                    packet.node_delay -= 1
                elif packet.node_delay == 0:
                    if packet.dst == packet.crt:
                        # 已经到达目标节点
                        if packet.type == 'ED':
                            # 量子存储
                            node = simulator.nodes[packet.crt]
                            simulator.active_packets.remove(packet)
                            if node.qm_capacity >= packet.qubits_len:
                                node.qm_capacity -= packet.qubits_len
                                simulator.pause_packets.append(packet)
                            elif node.qm_capacity > 0:
                                # 丢弃部分
                                simulator.qm_fail_qubits += packet.qubits_len - node.qm_capacity
                                packet.qubits_len = node.qm_capacity
                                node.qm_capacity = 0
                                simulator.pause_packets.append(packet)
                            elif node.qm_capacity == 0:
                                # 存储已满，丢弃
                                packet.fail = True
                                simulator.qm_fail_qubits += packet.qubits_len
                                print(f"[警告] 时间{time}：节点{node.id}量子存储已满，丢弃packet")
                                continue
                            else:
                                raise Exception("node.qm_capacity error")
                            setattr(packet, "pause_delay", int(simulator.sessions[packet.belong].path_distance / LINK_VELOCITY))
                            setattr(packet, "save_qm_qubits", packet.qubits_len)
                            # 变为暂停包
                            packet.location = "pause"                           
                        elif packet.type == 'MS':
                            # 传输完成
                            packet.success = True
                            simulator.sessions[packet.belong].remain_bit -= packet.qubits_len*2 # Dense Coding
                            simulator.nodes[packet.dst].qm_capacity += packet.save_qm_qubits # 释放存储器内存
                            simulator.active_packets.remove(packet)
                            # 统计成果
                            simulator.success_qubits += packet.qubits_len
                    else:
                        # 变为链路包，重置
                        packet.location = "link"
                        # 计算传输时延
                        node_list_path = simulator.sessions[packet.belong].path
                        assert packet.crt in node_list_path
                        next_node = node_list_path[node_list_path.index(packet.crt) + 1]
                        hop_distance = simulator.hop_distances[packet.crt][next_node]
                        packet.trans_delay = int(hop_distance / LINK_VELOCITY)
            else:
                raise Exception("packet.location error")
        
        for packet in simulator.pause_packets.copy():
            assert packet.location == "pause"
            packet.etg_lifetime -= 1
            if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                packet.fail = True
                # 删除包
                simulator.etg_fail_qubits += packet.qubits_len
                simulator.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                simulator.pause_packets.remove(packet)
                continue
            if packet.pause_delay > 0:
                # 更新暂停包的时延
                packet.pause_delay -= 1
            elif packet.pause_delay == 0:
                # 重新初始化
                packet.type = 'MS'
                packet.crt = simulator.sessions[packet.belong].src
                packet.guard = len(simulator.sessions[packet.belong].path) - 1
                packet.trans_delay = int(simulator.hop_distances[packet.src][packet.dst] / LINK_VELOCITY)
                packet.location = "link"
                simulator.pause_packets.remove(packet)
                simulator.active_packets.append(packet)
            else:
                raise Exception("packet.pause_delay error")
            
        # 7. 清理完成的会话
        simulator.cleanup()

        # 8. 统计
        simulator.active_qubits = sum(packet.qubits_len for packet in simulator.active_packets)
        _active_qubits[time] = simulator.active_qubits
        for i, session in enumerate(simulator.sessions):
            _success_session_qubits[i, time] = session.dataflow - session.remain_bit
        
        # 展示实时结果
        if time % 100 == 0:
            print(f"Time: {time}")
            print(f"Active Sessions: {len(simulator.active_sessions)}")
            for session in simulator.active_sessions:
                print(f"Session {session.id} progress: {session.src}:{session.dst}, remain bit: {session.remain_bit}, path: {session.path}")
            for node in simulator.nodes:
                print(f"Node {node.id} queue: {sum(item[0] for lst in node.queue_dict.values() for item in lst)}")
            print(f"Active Qubits Number: {int(simulator.active_qubits)}")
            print(f"Success Qubits Number: {int(simulator.success_qubits)}")
            print(f"ETG Fail Qubits Number: {int(simulator.etg_fail_qubits)}")
            print(f"Queue Fail Qubits Number: {int(simulator.queue_fail_qubits)}")
            print(f"QM Fail Qubits Number: {int(simulator.qm_fail_qubits)}")
        
        time += 1
        if not simulator.active_sessions and not simulator.active_packets and not simulator.pause_packets:
            end_time = time
            break
    
    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(_active_qubits[:end_time], 'r--', linewidth=2, label='Active Qubits')
    for i in range(len(simulator.sessions)):
        plt.plot(_success_session_qubits[i, :end_time], linewidth=2, label=f'Success Session Qubits {i}')
    plt.xlabel("Time (us)")
    plt.ylabel("Qubits Number")
    plt.legend()
    plt.show()
    
    print("\n=== 电路交换策略最终统计 ===")
    print(f"Total Time: {end_time}")
    for session in simulator.sessions:
        print(f"Session {session.id}: {session.src}->{session.dst}; Success Qubits: {int(_success_session_qubits[session.id, end_time-1])}")
    print(f"Total Success Qubits: {int(simulator.success_qubits)}")
    print(f"Total Message Bits: {int(simulator.success_qubits*2/INCUM)}")
    print(f"Total ETG Fail Qubits: {int(simulator.etg_fail_qubits)}")
    print(f"Total Queue Fail Qubits: {int(simulator.queue_fail_qubits)}")
    print(f"Total QM Fail Qubits: {int(simulator.qm_fail_qubits)}")

import network_v2 as nt
import numpy as np
import matplotlib.pyplot as plt

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
LINK_VELOCITY = 0.3
INCUM = 192 # 扩频比
max_qm_capacity = np.ones((link_capacity_matrix.shape[0])) * 100000 # 每个节点的量子存储容量
max_node_buffer = np.ones((link_capacity_matrix.shape[0])) * 2000 # 每个节点的最大缓冲区大小
dataflow = np.ones((SESSIONS_NUM)) * 100 # 每个会话需要传输的量子比特数

def sessions_start_time(nodes_num: int, sessions_num: int):
    rng = np.random.default_rng()
    start_time_list = np.zeros(sessions_num, dtype=int) # 同时到达
    src_list = np.zeros(sessions_num, dtype=int)
    dst_list = np.zeros_like(src_list)
    for i in range(sessions_num):
        src_list[i], dst_list[i] = rng.choice(nodes_num, size=2, replace=False)
    return np.column_stack((src_list, dst_list, start_time_list))


if __name__ == "__main__":
    print("Constructing network...")
    simulator = nt.Network(link_capacity_matrix, hop_distance_matrix, max_qm_capacity, max_node_buffer)
    # 0. 计算会话的启动时间和信息量
    sessions_info = sessions_start_time(simulator.nodes_num, sessions_num=SESSIONS_NUM)
    simulator.make_sessions(sessions_info, dataflow*INCUM)
    #simulator.make_sessions(np.array([[9, 1, 0], [3, 7, 0], [6, 8, 0], [1, 8, 0], [2, 3, 0], [3, 8, 0], [8, 0, 0], [5, 0, 0], [4, 3, 0], [4, 7, 0]]), dataflow*INCUM)
    
    print("Simulation started.")
    time = 0
    end_time = T_STEPS
    _active_qubits = np.zeros((T_STEPS+1))
    _success_session_qubits = np.zeros((SESSIONS_NUM, T_STEPS+1))
    route_switches_dict = {}
    
    while time <= T_STEPS:
        # 1. 激活到达的会话
        simulator.activate_sessions(time)
        
        # 2. 更新网络链路状态
        simulator.update_link_states()
        simulator.allocate_bandwidth_to_sessions()

        # 3. 源节点发送packet
        for node in simulator.nodes:
            for session_id in node.start_node:
                session = simulator.sessions[session_id]
                dst_node = simulator.nodes[session.dst]
                
                # 计算到下一跳的最优路由
                current_loads = simulator.get_current_loads_matrix()
                node_buffers = simulator.get_node_buffers_vector()
                
                next_hop = nt.dijkstra_next_hop_correct(
                    simulator.hop_distances, 
                    current_loads, 
                    node_buffers, 
                    node.id, 
                    dst_node.id,
                    simulator.link_dict
                )

                if next_hop is None:
                    print(f"[警告] 时间{time}：节点{node.id}无法路由到{dst_node.id}，跳过本轮发送")
                    continue
                
                link = simulator.link_dict[(node.id, next_hop)]
                max_snd_rate = link.snd_rate_per_session

                # 计算发送速率                
                node.snd_rate = int(min(
                    max_snd_rate, 
                    simulator.nodes[next_hop].remain_queue,
                    session.remain_bit
                ))
            
                if node.snd_rate > 0:
                    new_packets = node.snd_packets(session_id, node.id, dst_node.id)                    
                    for packet in new_packets:
                        packet.next_hop = next_hop
                        packet.route_history.append(next_hop)
                        hop_distance = simulator.hop_distances[packet.crt][packet.next_hop]
                        packet.trans_delay = int(hop_distance / LINK_VELOCITY) + 1                   
                    simulator.active_packets.extend(new_packets)

        # 4. 更新节点排队队列
        for node in simulator.nodes:
            node.update_queue(simulator)
            node.update_remain_queue()
            assert node.remain_queue >= 0

        # 5. 更新网络中的packet状态
        for packet in simulator.active_packets.copy():
            packet.etg_lifetime -= 1
            if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                packet.fail = True
                simulator.etg_fail_qubits += packet.qubits_len
                simulator.active_packets.remove(packet)
                continue

            if packet.location == "link":
                if packet.trans_delay > 0:
                    packet.trans_delay -= 1
                elif packet.trans_delay == 0:
                    hop_distance = simulator.hop_distances[packet.crt][packet.next_hop]
                    packet.crt = packet.next_hop
                    simulator.active_packets.remove(packet)         
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
                        packet.fail = True
                        simulator.queue_fail_qubits += packet.qubits_len
                        continue
                else:
                    raise Exception("packet.trans_delay error")
                    
            elif packet.location == "node":
                if packet.node_delay > 0:
                    packet.node_delay -= 1
                elif packet.node_delay == 0:
                    if packet.dst == packet.crt:
                        if packet.type == 'ED':
                            node = simulator.nodes[packet.crt]
                            simulator.active_packets.remove(packet)
                            if node.qm_capacity >= packet.qubits_len:
                                node.qm_capacity -= packet.qubits_len
                                simulator.pause_packets.append(packet)
                            elif node.qm_capacity > 0:
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
                            setattr(packet, "pause_delay", int(simulator.hop_distances[packet.src][packet.dst] / LINK_VELOCITY))
                            setattr(packet, "save_qm_qubits", packet.qubits_len)
                            packet.location = "pause"
                        elif packet.type == 'MS':
                            packet.success = True
                            session_id = packet.belong
                            simulator.sessions[session_id].remain_bit -= packet.qubits_len*2
                            simulator.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                            simulator.active_packets.remove(packet)
                            simulator.success_qubits += packet.qubits_len
                            # 统计route_switches
                            assert packet.route_history[0] == simulator.sessions[session_id].src
                            assert packet.route_history[-1] == simulator.sessions[session_id].dst
                            route_key = tuple(packet.route_history)
                            if session_id not in route_switches_dict:
                                route_switches_dict[session_id] = {}
                            route_switches_dict[session_id][route_key] = route_switches_dict[session_id].get(route_key, 0) + 1

                    else:
                        packet.location = "link"                        
                        current_loads = simulator.get_current_loads_matrix()
                        node_buffers = simulator.get_node_buffers_vector()
                        
                        old_next_hop = packet.next_hop
                        packet.next_hop = nt.dijkstra_next_hop_correct(
                            simulator.hop_distances, 
                            current_loads,
                            node_buffers, 
                            packet.crt, 
                            packet.dst,
                            simulator.link_dict
                        )
                        
                        if packet.next_hop is None:
                            raise Exception(f"packet {packet.id} at node {packet.crt} cannot find path to destination {packet.dst}")
                                                   
                        packet.route_history.append(packet.next_hop)
                        hop_distance = simulator.hop_distances[packet.crt][packet.next_hop]
                        packet.trans_delay = int(hop_distance / LINK_VELOCITY)
            else:
                raise Exception("packet.location error")
        
        for packet in simulator.pause_packets.copy():
            assert packet.location == "pause"
            packet.etg_lifetime -= 1
            if packet.etg_lifetime == 0 or packet.qubits_len <= 0:
                packet.fail = True
                simulator.etg_fail_qubits += packet.qubits_len
                simulator.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                simulator.pause_packets.remove(packet)
                continue
            if packet.pause_delay > 0:
                packet.pause_delay -= 1
            elif packet.pause_delay == 0:
                packet.type = 'MS'
                packet.crt = simulator.sessions[packet.belong].src
                packet.guard = 5
                # 从源节点开始
                packet.route_history.append(packet.crt)

                current_loads = simulator.get_current_loads_matrix()
                node_buffers = simulator.get_node_buffers_vector()
                
                packet.next_hop = nt.dijkstra_next_hop_correct(
                    simulator.hop_distances, 
                    current_loads,
                    node_buffers, 
                    packet.crt, 
                    packet.dst,
                    simulator.link_dict
                )
                
                if packet.next_hop is None:
                    packet.fail = True
                    simulator.etg_fail_qubits += packet.qubits_len
                    simulator.nodes[packet.dst].qm_capacity += packet.save_qm_qubits
                    simulator.pause_packets.remove(packet)
                    continue
                     
                packet.route_history.append(packet.next_hop)
                packet.trans_delay = int(simulator.hop_distances[packet.src][packet.next_hop] / LINK_VELOCITY)
                packet.location = "link"
                simulator.pause_packets.remove(packet)
                simulator.active_packets.append(packet)
            else:
                raise Exception("packet.pause_delay error")
            
        # 6. 清理完成的会话
        simulator.cleanup()

        # 7. 统计
        simulator.active_qubits = sum(packet.qubits_len for packet in simulator.active_packets)
        _active_qubits[time] = simulator.active_qubits
        for i, session in enumerate(simulator.sessions):
            _success_session_qubits[i, time] = session.dataflow - session.remain_bit
        
        # 展示实时结果
        if time % 100 == 0:
            print(f"\n=== Time: {time} ===")
            print(f"Active Sessions: {len(simulator.active_sessions)}")
            for session in simulator.active_sessions:
                print(f"  Session {session.id}: {session.src}→{session.dst}, remain_bit: {session.remain_bit}")
            for node in simulator.nodes:
                print(f"Node {node.id} queue: {sum(item[0] for lst in node.queue_dict.values() for item in lst)}")
            print(f"Active Qubits: {int(simulator.active_qubits)}")
            print(f"Success Qubits: {int(simulator.success_qubits)}")
            print(f"ETG Fail: {int(simulator.etg_fail_qubits)} \nQueue Fail: {int(simulator.queue_fail_qubits)} \nQM Fail: {int(simulator.qm_fail_qubits)}")
        
        time += 1

        if not simulator.active_sessions and not simulator.active_packets and not simulator.pause_packets:
            end_time = time
            break
    
    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(_active_qubits[:end_time], 'r--', linewidth=2, label='Active Qubits')
    for i in range(len(simulator.sessions)):
        plt.plot(_success_session_qubits[i, :end_time], linewidth=2, label=f'Session {i}')
    plt.xlabel("Time (us)")
    plt.ylabel("Qubits Number")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== 包交换策略最终统计 ===")
    print(f"Total Time: {end_time}")
    for session in simulator.sessions:
        print(f"Session {session.id}: {session.src}→{session.dst}; Success Qubits: {int(_success_session_qubits[session.id, end_time-1])}")
        print(f"\nRoute Switches: {len(route_switches_dict[session.id].values())}; Most Used Route: {max(route_switches_dict[session.id], key=route_switches_dict[session.id].get)}")
    print(f"Total Success Qubits: {int(simulator.success_qubits)}")
    print(f"Total Message Bits: {int(simulator.success_qubits*2/INCUM)}")
    print(f"Total ETG Fail Qubits: {int(simulator.etg_fail_qubits)}")
    print(f"Total Queue Fail Qubits: {int(simulator.queue_fail_qubits)}")
    print(f"Total QM Fail Qubits: {int(simulator.qm_fail_qubits)}")
    print(f"Total Route Switches: {int(sum(len(routes) for routes in route_switches_dict.values()))}")
import csv
import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import nt1
import nt2
from topology.topology3 import topology_info

# ====================== 可调参数 ======================
T_STEPS = 30000
INCUM = 192
CYCLE = 100
MAX_WORKERS = min(CYCLE, os.cpu_count() or 1)
LOG_EVERY = 1000
DELTA_T = 200

ARRIVAL_CHECK_INTERVAL = 20       # 每隔多少个时间步检查一次是否生成新会话
SESSION_ARRIVAL_PROB = 0.02       # 每次检查生成一个新会话的概率
TASK_QUBITS_MIN = 2500            # 单个会话目标任务量下界（成功传输 qubit 数）
TASK_QUBITS_MAX = 6500            # 单个会话目标任务量上界（成功传输 qubit 数）

# ====================================================


def choose_src_dst(nodes_num: int, rng: np.random.Generator):
    src, dst = rng.choice(nodes_num, size=2, replace=False)
    return int(src), int(dst)


def build_workload(topo, index: int):
    """生成随机到达、随机任务量的会话序列。"""
    rng = np.random.default_rng(20260423 + index)
    requests = []
    session_id = 0


    for t in range(0, T_STEPS + 1, ARRIVAL_CHECK_INTERVAL):
        if rng.random() < SESSION_ARRIVAL_PROB:
            src, dst = choose_src_dst(topo.nodes_num, rng)
            target_qubits = int(rng.integers(TASK_QUBITS_MIN, TASK_QUBITS_MAX + 1))
            requests.append({
                'id': session_id,
                'src': src,
                'dst': dst,
                'start_time': int(t),
                'target_qubits': target_qubits,
            })
            session_id += 1
    return requests


def summarize_route_switches(route_switches_dict, limit: int = 20):
    total_paths = sum(len(routes) for routes in route_switches_dict.values())
    sortable = sorted(route_switches_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    lines = []
    for sid, routes in sortable[:limit]:
        route_desc = '; '.join([f'{list(route)} x{count}' for route, count in routes.items()])
        lines.append(f'  Session {sid}: {route_desc}\n')
    return total_paths, lines


def active_request_count_at_time(requests, t: int):
    return sum(1 for req in requests if req['start_time'] <= t)


def compute_total_throughput_ts(total_success_ts: np.ndarray):
    return (total_success_ts[DELTA_T:] - total_success_ts[:-DELTA_T]) / DELTA_T * 1e3


def session_avg_rate_kbps(session, sim_horizon: int):
    if session.target_reached_time is not None:
        end_time = session.target_reached_time
        delivered_qubits = min(session.dataflow, session.target_qubits)
    else:
        end_time = sim_horizon
        delivered_qubits = session.dataflow
    duration = max(1, int(end_time - session.start_time + 1))
    qubits_per_ms = delivered_qubits / duration * 1e3
    return qubits_per_ms * 2 / INCUM


def session_delay_us(session, sim_horizon: int):
    if session.target_reached_time is not None:
        end_time = session.target_reached_time
    else:
        end_time = sim_horizon
    return max(1, int(end_time - session.start_time + 1))


def build_session_rows(requests, circuit_sessions, packet_sessions):
    rows = []
    for req in requests:
        sid = req['id']
        cs = circuit_sessions[sid]
        ps = packet_sessions[sid]
        rows.append({
            'session_id': sid,
            'src': req['src'],
            'dst': req['dst'],
            'start_time_us': req['start_time'],
            'target_qubits': req['target_qubits'],
            'circuit_success_qubits': int(cs.dataflow),
            'packet_success_qubits': int(ps.dataflow),
            'circuit_completed': int(cs.target_reached_time is not None),
            'packet_completed': int(ps.target_reached_time is not None),
            'circuit_delay_us': session_delay_us(cs, T_STEPS),
            'packet_delay_us': session_delay_us(ps, T_STEPS),
            'circuit_target_reached_time_us': '' if cs.target_reached_time is None else int(cs.target_reached_time),
            'packet_target_reached_time_us': '' if ps.target_reached_time is None else int(ps.target_reached_time),
            'circuit_avg_rate_kbps': session_avg_rate_kbps(cs, T_STEPS),
            'packet_avg_rate_kbps': session_avg_rate_kbps(ps, T_STEPS),
        })
    return rows


def run_one_simulation(index: int):
    np.random.seed(20260423 + index)
    topo = topology_info()
    requests = build_workload(topo, index)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'qns_data')
    fig_dir = os.path.join(base_dir, 'qns_figures')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f'QNSlog_v3_{index}.txt')
    csv_file_path = os.path.join(log_dir, f'QNS_session_rates_{index}.csv')
    fig1_path = os.path.join(fig_dir, f'QNS_active_qubits_{index}.pdf')
    fig2_path = os.path.join(fig_dir, f'QNS_total_success_{index}.pdf')
    fig3_path = os.path.join(fig_dir, f'QNS_total_throughput_{index}.pdf')
    fig4_path = os.path.join(fig_dir, f'QNS_session_rates_{index}.pdf')

    now_local = datetime.now().astimezone()
    logs = [f'local: {now_local.isoformat()}\n']

    circuit_simulator = nt1.Network(topo.link_capacity_matrix, topo.hop_distance_matrix, topo.max_qm_capacity, topo.max_node_buffer)
    packet_simulator = nt2.Network(topo.link_capacity_matrix, topo.hop_distance_matrix, topo.max_qm_capacity, topo.max_node_buffer)

    for req in requests:
        circuit_simulator.sessions.append(
            nt1.Session(req['src'], req['dst'], req['id'], start_time=req['start_time'], target_qubits=req['target_qubits'])
        )
        packet_simulator.sessions.append(
            nt2.Session(req['src'], req['dst'], req['id'], start_time=req['start_time'], target_qubits=req['target_qubits'])
        )

    logs.append('================================\n')
    logs.append('Workload info:\n')
    logs.append(f'Total generated requests: {len(requests)}\n')
    logs.append(f'Arrival check interval: {ARRIVAL_CHECK_INTERVAL} us\n')
    logs.append(f'Arrival probability: {SESSION_ARRIVAL_PROB}\n')
    logs.append(f'Task qubits range: [{TASK_QUBITS_MIN}, {TASK_QUBITS_MAX}]\n')
    for req in requests[:20]:
        logs.append(
            f'  Req {req["id"]}: {req["src"]}->{req["dst"]}, start={req["start_time"]}, target={req["target_qubits"]}\n'
        )

    circuit_active_qubits = np.zeros((T_STEPS + 1,))
    packet_active_qubits = np.zeros((T_STEPS + 1,))
    circuit_total_success = np.zeros((T_STEPS + 1,))
    packet_total_success = np.zeros((T_STEPS + 1,))

    begin = time.time()
    for t in range(T_STEPS + 1):
        circuit_simulator.step()
        packet_simulator.step()

        circuit_active_qubits[t] = circuit_simulator.active_qubits
        packet_active_qubits[t] = packet_simulator.active_qubits
        circuit_total_success[t] = circuit_simulator.success_qubits
        packet_total_success[t] = packet_simulator.success_qubits

        if t % LOG_EVERY == 0:
            ready_count = sum(1 for s in circuit_simulator.sessions if getattr(s, 'state', '') == 'established')
            setup_count = sum(1 for s in circuit_simulator.sessions if getattr(s, 'state', '') == 'setup')
            closing_count = sum(1 for s in circuit_simulator.sessions if getattr(s, 'state', '') == 'closing')
            blocked_count = sum(1 for s in circuit_simulator.sessions if getattr(s, 'state', '') == 'blocked')
            arrived_requests = active_request_count_at_time(requests, t)
            circuit_completed = sum(1 for s in circuit_simulator.sessions if getattr(s, 'target_reached_time', None) is not None)
            packet_completed = sum(1 for s in packet_simulator.sessions if getattr(s, 'target_reached_time', None) is not None)

            logs.append('================================\n')
            logs.append(f'Time: {t} us\n')
            logs.append(f'Arrived requests: {arrived_requests}\n')
            logs.append(f'Circuit completed / Packet completed: {circuit_completed}/{packet_completed}\n')
            logs.append(
                f'Circuit states ready/setup/closing/blocked: {ready_count}/{setup_count}/{closing_count}/{blocked_count}\n'
            )
            logs.append(f'Circuit active qubits: {int(circuit_simulator.active_qubits)}, Packet active qubits: {int(packet_simulator.active_qubits)}\n')
            logs.append(f'Circuit success qubits: {int(circuit_simulator.success_qubits)}, Packet success qubits: {int(packet_simulator.success_qubits)}\n')
            logs.append(
                f'Circuit fails(etg/queue/qm): {int(circuit_simulator.etg_fail_qubits)}/'
                f'{int(circuit_simulator.queue_fail_qubits)}/{int(circuit_simulator.qm_fail_qubits)}\n'
            )
            logs.append(
                f'Packet fails(etg/queue/qm): {int(packet_simulator.etg_fail_qubits)}/'
                f'{int(packet_simulator.queue_fail_qubits)}/{int(packet_simulator.qm_fail_qubits)}\n'
            )
    end = time.time()

    circuit_total_throughput_ts = compute_total_throughput_ts(circuit_total_success)
    packet_total_throughput_ts = compute_total_throughput_ts(packet_total_success)

    circuit_total_throughput_kbps = float(circuit_simulator.success_qubits / T_STEPS * 1e3 * 2 / INCUM)
    packet_total_throughput_kbps = float(packet_simulator.success_qubits / T_STEPS * 1e3 * 2 / INCUM)

    session_rows = build_session_rows(requests, circuit_simulator.sessions, packet_simulator.sessions)
    circuit_rates = np.array([row['circuit_avg_rate_kbps'] for row in session_rows], dtype=float)
    packet_rates = np.array([row['packet_avg_rate_kbps'] for row in session_rows], dtype=float)
    circuit_completed = int(sum(row['circuit_completed'] for row in session_rows))
    packet_completed = int(sum(row['packet_completed'] for row in session_rows))

    circuit_mean_rate = float(np.mean(circuit_rates))
    packet_mean_rate = float(np.mean(packet_rates))

    circuit_rate_var = float(np.var(circuit_rates))
    packet_rate_var = float(np.var(packet_rates))

    circuit_rate_std = float(np.std(circuit_rates))
    packet_rate_std = float(np.std(packet_rates))

    circuit_rate_cv = float(circuit_rate_std / max(circuit_mean_rate, 1e-12))
    packet_rate_cv = float(packet_rate_std / max(packet_mean_rate, 1e-12))

    circuit_delays = np.array([row['circuit_delay_us'] for row in session_rows], dtype=float)
    packet_delays = np.array([row['packet_delay_us'] for row in session_rows], dtype=float)

    circuit_delay_p95 = float(np.percentile(circuit_delays, 95))
    packet_delay_p95 = float(np.percentile(packet_delays, 95))

    total_paths, route_lines = summarize_route_switches(packet_simulator.route_switches_dict)

    logs.append('\n=== 最终统计 ===\n')
    logs.append(f'Simulation wall time: {end - begin:.2f} s\n')
    logs.append(f'Total simulation horizon: {T_STEPS} us\n')
    logs.append(f'Total sessions: {len(requests)}\n')
    logs.append(f'Completed sessions (circuit / packet): {circuit_completed} / {packet_completed}\n')
    logs.append(f'Total delivered information: {circuit_simulator.success_qubits * 2 / INCUM:.2f} / {packet_simulator.success_qubits * 2 / INCUM:.2f} bits\n')
    logs.append(f'Overall total throughput: {circuit_total_throughput_kbps:.2f} / {packet_total_throughput_kbps:.2f} kbps\n')
    logs.append(f'Average session rate: {float(np.mean(circuit_rates)):.2f} / {float(np.mean(packet_rates)):.2f} kbps\n')
    logs.append(f'Median session rate: {float(np.median(circuit_rates)):.2f} / {float(np.median(packet_rates)):.2f} kbps\n')
    logs.append(f'Session-rate std: {circuit_rate_std:.2f} / {packet_rate_std:.2f} kbps\n')
    logs.append(f'Session-rate var: {circuit_rate_var:.2f} / {packet_rate_var:.2f} (kbps^2)\n')
    logs.append(f'Session-rate CV: {circuit_rate_cv:.4f} / {packet_rate_cv:.4f}\n')
    logs.append(f'P95 session delay: {circuit_delay_p95:.2f} / {packet_delay_p95:.2f} us\n')
    logs.append(f'Packet route kinds: {int(total_paths)}\n')

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.writelines(logs)

    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(session_rows[0].keys()) if session_rows else [
            'session_id', 'src', 'dst', 'start_time_us', 'target_qubits',
            'circuit_success_qubits', 'packet_success_qubits',
            'circuit_completed', 'packet_completed',
            'circuit_target_reached_time_us', 'packet_target_reached_time_us',
            'circuit_avg_rate_kbps', 'packet_avg_rate_kbps'
        ])
        writer.writeheader()
        if session_rows:
            writer.writerows(session_rows)

    plt.figure(figsize=(10, 6))
    plt.plot(circuit_active_qubits, label='circuit switching')
    plt.plot(packet_active_qubits, label='packet switching')
    plt.xlabel('Time (us)')
    plt.ylabel('Active Qubits')
    plt.title('Active Qubits over Time')
    plt.legend()
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(circuit_total_success, label='circuit switching')
    plt.plot(packet_total_success, label='packet switching')
    plt.xlabel('Time (us)')
    plt.ylabel('Success Qubits')
    plt.title('Total Success Qubits over Time')
    plt.legend()
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    x_tp = np.arange(DELTA_T, T_STEPS + 1)
    plt.plot(x_tp, circuit_total_throughput_ts, label='circuit switching')
    plt.plot(x_tp, packet_total_throughput_ts, label='packet switching')
    plt.xlabel('Time (us)')
    plt.ylabel('Total throughput (qubits/ms)')
    plt.title('Window Total Throughput over Time')
    plt.legend()
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    x_sessions = np.arange(len(session_rows))
    plt.plot(x_sessions, circuit_rates, label='circuit switching')
    plt.plot(x_sessions, packet_rates, label='packet switching')
    plt.xlabel('Session ID')
    plt.ylabel('Average transmission rate (kbps)')
    plt.title('Per-session Average Transmission Rate')
    plt.legend()
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'index': index,
        'circuit_total_throughput_kbps': circuit_total_throughput_kbps,
        'packet_total_throughput_kbps': packet_total_throughput_kbps,
        'circuit_mean_rate_kbps': float(np.mean(circuit_rates)),
        'packet_mean_rate_kbps': float(np.mean(packet_rates)),
        'circuit_rate_std_kbps': circuit_rate_std,
        'packet_rate_std_kbps': packet_rate_std,
        'circuit_rate_var_kbps2': circuit_rate_var,
        'packet_rate_var_kbps2': packet_rate_var,
        'circuit_rate_cv': circuit_rate_cv,
        'packet_rate_cv': packet_rate_cv,
        'total_circuit_bits': circuit_simulator.success_qubits * 2 / INCUM,
        'total_packet_bits': packet_simulator.success_qubits * 2 / INCUM,
        'session_count': len(requests),
        'circuit_completed': circuit_completed,
        'packet_completed': packet_completed,
    }


if __name__ == '__main__':
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_one_simulation, index) for index in range(CYCLE)]
        for fut in tqdm(as_completed(futures), total=CYCLE):
            results.append(fut.result())

    results.sort(key=lambda x: x['index'])
    print('All simulations finished.')
    print('================ Summary ================')
    for item in results:
        print(
            f"Run {item['index']}: total throughput (circuit/packet) = "
            f"{item['circuit_total_throughput_kbps']:.2f}/{item['packet_total_throughput_kbps']:.2f} kbps, "
            f"mean session rate = {item['circuit_mean_rate_kbps']:.2f}/{item['packet_mean_rate_kbps']:.2f} kbps, "
            f"std = {item['circuit_rate_std_kbps']:.2f}/{item['packet_rate_std_kbps']:.2f} kbps, "
            f"cv = {item['circuit_rate_cv']:.4f}/{item['packet_rate_cv']:.4f}, "
            f"completed sessions = {item['circuit_completed']}/{item['packet_completed']} / {item['session_count']}, "
            f"total bits = {item['total_circuit_bits']:.2f}/{item['total_packet_bits']:.2f}"
        )

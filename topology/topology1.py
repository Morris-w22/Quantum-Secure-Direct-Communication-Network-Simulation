import numpy as np

class topology_info:
    def __init__(self):
        self.link_capacity_matrix = 10*np.array([
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
        self.hop_distance_matrix = np.array([
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
        assert self.link_capacity_matrix.shape[0] == self.link_capacity_matrix.shape[1]
        assert self.link_capacity_matrix.shape == self.hop_distance_matrix.shape
        self.nodes_num = self.link_capacity_matrix.shape[0]
        self.max_node_buffer = np.ones((self.nodes_num)) * 2000 # 每个节点的最大缓冲区大小
        self.max_qm_capacity = np.ones((self.nodes_num)) * 500000 # 每个节点的量子存储容量


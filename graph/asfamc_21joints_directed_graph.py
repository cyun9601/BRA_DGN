import sys 
sys.path.append('../')

from typing import Tuple, List
from collections import defaultdict

import numpy as np
import yaml
import argparse
from utils.graph import pop_paris, sort_paris, plus_paris


epsilon = 1e-6

# for node IDs, and reduce index to 0-based
# 바깥 방향으로 작성 
directed_edges = [(i, j) for i, j in [
    (1, 2), (1, 6), (2, 3), (3, 4), (4, 5), 
    (6, 7), (7, 8), (8, 9), (10, 1), (11, 10), 
    (12, 11), (12, 12), (12, 13), (12, 14), (12, 18), 
    (14, 15), (15, 16), (16, 17), (18, 19), (19, 20), 
    (20, 21)
    # Add self loop for Node 21 (the centre) to avoid singular matrices
]]


def normalize_incidence_matrix(im: np.ndarray, full_im: np.ndarray) -> np.ndarray:
    # NOTE:
    # 1. The paper assumes that the Incidence matrix is square,
    #    so that the normalized form A @ (D ** -1) is viable.
    #    However, if the incidence matrix is non-square, then
    #    the above normalization won't work.
    #    For now, move the term (D ** -1) to the front
    # 2. It's not too clear whether the degree matrix of the FULL incidence matrix
    #    should be calculated, or just the target/source IMs.
    #    However, target/source IMs are SINGULAR matrices since not all nodes
    #    have incoming/outgoing edges, but the full IM as described by the paper
    #    is also singular, since ±1 is used for target/source nodes.
    #    For now, we'll stick with adding target/source IMs.
    
    # degree mat은 각 node당 source node와 target node의 수를 대각행렬로 나타냄.
    # 즉, Node에 연결되있는 edge 수
    degree_mat = full_im.sum(-1) * np.eye(len(full_im))
    
    # Since all nodes should have at least some edge, degree matrix is invertible
    # 대각행렬의 inverse matrix이므로, 주대각 원소의 역수  
    inv_degree_mat = np.linalg.inv(degree_mat) 
    
    # normalize 
    return (inv_degree_mat @ im) + epsilon

# index가 0부터 시작하므로 모든 노드 번호 - 1
directed_edges = plus_paris(directed_edges, -1)
num_nodes = len(directed_edges)
# NOTE: for now, let's not add self loops since the paper didn't mention this
# self_loops = [(i, i) for i in range(num_nodes)]

# 이 함수는 node수와 edge 정보를 받아서 Source graph와 target graph를 만든 후 Normalize
def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    # NOTE: For now, we won't consider all possible edges
    # max_edges = int(special.comb(num_nodes, 2))
    max_edges = len(edges) # 25
    source_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    target_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    
    # source_node와 target_node는 DGNN 논문에서 나오는 A^s, A^t와 같음 
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.
        target_graph[target_node, edge_id] = 1.
    
    full_graph = source_graph + target_graph
    
    source_graph = normalize_incidence_matrix(source_graph, full_graph)
    target_graph = normalize_incidence_matrix(target_graph, full_graph)
    
    return source_graph, target_graph

def build_undigraph_adj_matrix(num_nodes, edges: List[Tuple]) -> np.ndarray:
    graph = np.zeros((num_nodes, num_nodes), dtype='float32')
    for (i, j) in edges:
        graph[i, j] = 1
        graph[j, i] = 1
    for i in range(num_nodes):
        graph[i, i] = 1
    return graph

class Graph:
    def __init__(self):
        super().__init__()
        self.num_nodes = num_nodes # 21
        self.edges = directed_edges
        self.source_M, self.target_M = build_digraph_incidence_matrix(self.num_nodes, self.edges)
        self.adj_M = build_undigraph_adj_matrix(self.num_nodes, self.edges)

# TODO:
# Check whether self loop should be added inside the graph
# Check incidence matrix size

'''
if __name__ == "__main__": 
    
    import matplotlib.pyplot as plt
    
    graph = Graph()
    adj_M = graph.adj_M
    plt.imshow(adj_M, cmap='gray')
    plt.show()
'''
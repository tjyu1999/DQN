import numpy as np
from node import Node
from link import Link


class Graph:
    def __init__(self, data):
        self.data = data
        self.nodes = {}
        self.node_adjacent_matrix = []
        self.links = {}
        self.link_adjacent_matrix = []
        self.link_distance_matrix = []
        self.load_nodes()
        self.load_node_adjacent_matrix()
        self.get_links()
        self.get_link_distance_matrix()

    def load_nodes(self):
        self.nodes = {}
        for idx, buffer_size in self.data.node_info.items():
            self.nodes[idx] = Node(int(idx), buffer_size)

    def load_node_adjacent_matrix(self):
        self.node_adjacent_matrix = self.data.node_matrix

    def get_links(self):
        self.links = {}
        node_num = len(self.node_adjacent_matrix)
        start_from_node = {}
        for idx in range(node_num):
            start_from_node[idx] = []
        idx = 0
        for i in range(node_num):
            for j in range(node_num):
                if self.node_adjacent_matrix[i][j] == 0 or i == j:
                    continue
                self.links[idx] = Link(idx, self.nodes[f'{i}'], self.nodes[f'{j}'])
                start_from_node[i].append(idx)
                idx += 1
        edge_num = idx
        self.link_adjacent_matrix = np.zeros([edge_num, edge_num])
        for i in range(edge_num):
            for j in start_from_node[self.links[i].end_node.idx]:
                self.link_adjacent_matrix[i][j] = 1

    def get_link_distance_matrix(self):
        node_num = len(self.nodes)
        edge_num = len(self.links)
        self.link_distance_matrix = np.full([edge_num, node_num], 999)
        for idx in range(edge_num):
            visited_edges = [idx]
            curr_list = [self.links[idx]]
            dist = 1
            self.link_distance_matrix[idx][self.links[idx].start_node.idx] = 0
            while len(curr_list) > 0:
                next_list = []
                for curr_edge in curr_list:
                    self.link_distance_matrix[idx][curr_edge.end_node.idx] = min(self.link_distance_matrix[idx][curr_edge.end_node.idx], dist)
                    for adj_idx in range(edge_num):
                        if self.link_adjacent_matrix[curr_edge.idx][adj_idx] == 1 and adj_idx not in visited_edges:
                            visited_edges.append(adj_idx)
                            next_list.append(self.links[adj_idx])
                curr_list = next_list
                dist += 1
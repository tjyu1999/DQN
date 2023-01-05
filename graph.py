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
        for idx, capacity in self.data.node_info.items():
            self.nodes[idx] = Node(int(idx), capacity)

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
        link_num = idx
        self.link_adjacent_matrix = np.zeros([link_num, link_num])
        for i in range(link_num):
            for j in start_from_node[self.links[i].end_node.idx]:
                self.link_adjacent_matrix[i][j] = 1

    def get_link_distance_matrix(self):
        node_num = len(self.nodes)
        link_num = len(self.links)
        self.link_distance_matrix = np.full([link_num, node_num], 999)
        for idx in range(link_num):
            visited_links = [idx]
            current_list = [self.links[idx]]
            distance = 1
            self.link_distance_matrix[idx][self.links[idx].start_node.idx] = 0
            while len(current_list) > 0:
                next_list = []
                for current_link in current_list:
                    self.link_distance_matrix[idx][current_link.end_node.idx] = min(self.link_distance_matrix[idx][current_link.end_node.idx], distance)
                    for adjacent_idx in range(link_num):
                        if self.link_adjacent_matrix[current_link.idx][adjacent_idx] == 1 and adjacent_idx not in visited_links:
                            visited_links.append(adjacent_idx)
                            next_list.append(self.links[adjacent_idx])
                current_list = next_list
                distance += 1
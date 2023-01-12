import os
import json
import random
import numpy as np
from param import args


class Ladder:
    def __init__(self):
        self.node_matrix = []
        self.node_info = {}
        self.flow_info = {}

    def generate_node_matrix(self, node_num):
        self.node_matrix = np.zeros((node_num, node_num))
        links = []
        idx = 0
        while idx + 1 < node_num:
            links.append([idx, idx + 1])
            idx += 2
        idx = 0
        while idx + 2 < node_num:
            links.append([idx, idx + 2])
            idx += 1
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    self.node_matrix[i, j] = 1
        for link in links:
            self.node_matrix[link[0], link[1]] = 1
            self.node_matrix[link[1], link[0]] = 1

    def generate_node_info(self, node_num):
        for i in range(node_num):
            self.node_info[i] = args.node_capacity[random.randint(0, len(args.node_capacity) - 1)]

    def generate_flow_info(self, flow_idx, node_num, flow_num, flow_len, flow_prd, flow_delay):
        for i in range(flow_num):
            src = random.randint(0, node_num - 1)
            dst = random.randint(0, node_num - 1)
            while src == dst:
                dst = random.randint(0, node_num - 1)
            length = random.randint(flow_len[0], flow_len[1])
            prd = flow_prd[random.randint(0, len(flow_prd) - 1)]
            delay = random.randint(flow_delay[0], flow_delay[1])
            self.flow_info[flow_idx] = [src, dst, length, prd, delay]

    def generate_all_data(self, node_num, flow_num, flow_len, flow_prd, flow_delay):
        self.generate_node_matrix(node_num)
        self.generate_node_info(node_num)
        for num in range(flow_num):
            self.generate_flow_info(num, node_num, flow_num, flow_len, flow_prd, flow_delay)
        np.save('data/node_matrix.npy', self.node_matrix)
        json.dump(self.node_info, open('data/node_info.json', 'w'), indent=4)
        json.dump(self.flow_info, open('data/flow_info.json', 'w'), indent=4)
        print('---------- Saving data in data/ladder')

    def read_file(self):
        self.node_matrix = np.load('data/node_matrix.npy')
        self.node_info = json.load(open('data/node_info.json'))
        self.flow_info = json.load(open('data/flow_info.json'))


def main():
    data = Ladder()
    node_num = 8
    flow_num = 15
    if not os.path.exists('data'):
        os.makedirs('data')
    data.generate_all_data(node_num, flow_num, args.flow_len, args.flow_prd, args.flow_delay)


if __name__ == '__main__':
    main()
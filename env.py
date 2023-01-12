import numpy as np
import torch
from graph import Graph
from param import args


class Env:
    def __init__(self, data):
        self.data = data
        self.graph = Graph(data)
        self.original_src = None
        self.flow_src = None
        self.flow_dst = None
        self.flow_len = None
        self.flow_prd = None
        self.flow_delay = None
        self.visited_node = []
        self.accumulated_delay = 0.0
        self.success_num = 0
        self.total_reward = 0.0

    def flow_info(self, flow_idx):
        self.flow_src, self.flow_dst, self.flow_len, self.flow_prd, self.flow_delay = self.data.flow_info[f'{flow_idx}']
        self.original_src = self.flow_src
        self.visited_node.append(self.flow_src)

    def get_state(self):
        link_num = len(self.graph.links)
        state = np.zeros([link_num, args.state_dim])
        for link in self.graph.links.values():
            if link.start_node.idx == self.original_src:
                link.start_node.is_src_node = 1
            if link.end_node.idx == self.flow_dst:
                link.end_node.is_dst_node = 1
        for link in self.graph.links.values():
            state[link.idx][0] = link.start_node.is_src_node                               # if src or not
            state[link.idx][1] = link.end_node.is_dst_node                                 # if dst or not
            state[link.idx][2] = 1 if link in self.find_valid_link() else 0                # if valid or not
            state[link.idx][3] = self.graph.link_distance_matrix[link.idx][self.flow_dst]  # distance to dst

        return torch.from_numpy(state).float()

    @staticmethod
    def get_reward(done, delay):
        return args.alpha * done - args.beta * delay

    def update(self, action):
        link_idx, offset = action
        self.accumulated_delay += self.calculate_delay(offset)
        self.flow_src = self.graph.links[link_idx].end_node.idx
        if not self.find_valid_link() or not offset or self.accumulated_delay > self.flow_delay:  # successfully scheduled
            done = -1
            reward = -2
            state = self.get_state()
        elif self.flow_src == self.flow_dst:                                                      # unsuccessfully scheduled
            done = 1
            self.success_num += 1
            reward = 1
            state = self.get_state()
        else:                                                                                     # in progress
            done = 0
            self.visited_node.append(self.flow_src)
            reward = 0.1
            state = self.get_state()

        self.total_reward += reward

        return done, reward, state

    @staticmethod
    def calculate_delay(offset):
        return offset[0] if offset else args.slot_num

    def find_valid_link(self):
        valid_link = []
        for link in self.graph.links.values():
            if link.start_node.idx == self.flow_src and link.end_node.idx not in self.visited_node:
                valid_link.append(link.idx)

        return valid_link

    def find_slot(self, link_idx):
        return self.graph.links[link_idx].find_slot(self.flow_len, self.flow_prd)

    def occupy_slot(self, action):
        link_idx, offset = action
        self.graph.links[link_idx].occupy_slot(offset, self.flow_prd)

    def success_rate(self, flow_num):
        return self.success_num / flow_num

    def usage(self):
        available_slot_num = 0
        for link in self.graph.links.values():
            available_slot_num += sum(link.slot_status)

        return 1 - (available_slot_num / (args.slot_num * len(self.graph.links)))

    def mean_reward(self, flow_num):
        return self.total_reward / flow_num

    # transfer to the next state in the same flow
    def renew(self):
        for link in self.graph.links.values():
            link.start_node.is_src_node = 0
            link.end_node.is_dst_node = 0

    # transfer to the next flow
    def refresh(self):
        self.renew()
        self.visited_node.clear()
        self.accumulated_delay = 0.0

    def reset(self):
        self.refresh()
        for link in self.graph.links.values():
            link.slot_status = np.ones(args.slot_num)
        self.success_num = 0
        self.total_reward = 0.0
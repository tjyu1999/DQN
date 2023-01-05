from param import args


class Node:
    def __init__(self, idx, capacity):
        self.idx = idx
        self.capacity = capacity
        self.is_src_node = 0
        self.is_dst_node = 0

    def find_buffer(self):
        pass

    def occupy_buffer(self):
        pass

    def reset(self):
        self.is_src_node = 0
        self.is_dst_node = 0
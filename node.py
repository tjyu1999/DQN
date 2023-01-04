from param import args


class Node:
    def __init__(self, idx, buffer_size):
        self.idx = idx
        self.buffer_size = buffer_size
        self.is_src_node = 0
        self.is_dst_node = 0

    def find_buffer(self):
        pass

    def occupy_buffer(self):
        pass

    def reset(self):
        self.is_src_node = 0
        self.is_dst_node = 0
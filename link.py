import numpy as np
from param import args


class Link:
    def __init__(self, idx, start_node, end_node):
        self.idx = idx
        self.start_node = start_node
        self.end_node = end_node
        self.start_node.is_src_node = 0
        self.end_node.is_dst_node = 0
        self.slot_status = np.ones(args.slot_num)

    def low_degree_method(self, flow_len, flow_prd):
        slot_degree = np.zeros(flow_prd)
        for position in range(flow_prd):
            degree = 0
            for prd in args.flow_prd:
                flag = True
                frames = int(args.slot_num / prd)
                offset = [position + frame * prd for frame in range(frames)]
                for idx in offset:
                    for length in range(flow_len):
                        idx += length
                        if idx + length >= args.slot_num:            # check if the slot is valid
                            flag = False
                            continue
                        else:
                            if self.slot_status[idx + length] == 0:  # check if the slot is available
                                flag = False
                            continue
                if not flag:
                    degree += args.hyper_prd / prd
            slot_degree[position] = degree

        return slot_degree

    # find the slot through the LD method
    def find_slot(self, flow_len, flow_prd):
        slot_degree = self.low_degree_method(flow_len, flow_prd)
        if sum(slot_degree) == 0:
            return []
        else:
            for idx in range(len(slot_degree)):
                if slot_degree[idx] == 0:
                    slot_degree[idx] = 100
        position = np.argmin(slot_degree)
        offset = [position + length for length in range(flow_len)]

        return offset

    def occupy_slot(self, offset, flow_prd):
        frames = int(args.slot_num / flow_prd)
        for frame in range(frames):
            for idx in offset:
                idx += frame * flow_prd
                self.slot_status[idx] -= 1
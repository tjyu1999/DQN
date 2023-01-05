from collections import namedtuple
import random


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def update_priority(self):
        pass

    def __len__(self):
        return len(self.memory)
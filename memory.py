from collections import namedtuple
import random
import torch


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'mask'))


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

    def sample(self, batch_size, device):
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))

        state_batch = torch.FloatTensor(torch.stack(batch.state)).to(device)
        action_batch = torch.LongTensor([batch.action]).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.FloatTensor(torch.stack(batch.next_state)).to(device)
        mask_batch = batch.mask

        return state_batch, action_batch, reward_batch, next_state_batch, mask_batch

    def __len__(self):
        return len(self.memory)
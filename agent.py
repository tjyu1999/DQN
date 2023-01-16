import random
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from env import Env
from memory import Experience, Memory
from model import Model
from param import args


class Agent:
    def __init__(self,
                 device):
        self.device = device
        self.episode = 1
        self.env = Env()
        self.memory = Memory(args.batch_size)
        self.eval_q = Model(gcn_layer_dim=args.gcn_layer_dim,
                            q_layer_dim=args.q_layer_dim,
                            device=device)
        self.target_q = deepcopy(self.eval_q)
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.SGD(self.eval_q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.epsilon = args.init_epsilon

    def select_action(self):
        if random.random() > self.epsilon:
            link_q = self.eval_q()
            link_idx =
        else:

    def store_transition(self, state, action, reward, next_state, is_terminal, mask):
        self.memory.push(state, action, reward, next_state, is_terminal, mask)

    def sample_memory(self):
        transitions = self.memory.sample(args.batch_size)
        batch = Experience(*zip(*transitions))

        state_batch = torch.FloatTensor(torch.stack(batch.state)).to(self.device)
        action_batch = torch.LongTensor([batch.action]).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(torch.stack(batch.next_state)).to(self.device)
        is_terminal_batch = torch.FloatTensor(batch.is_terminal).to(self.device)
        mask_batch = batch.mask

        return state_batch, action_batch, reward_batch, next_state_batch, is_terminal_batch, mask_batch

    def replace_target_network(self):
        if self.episode % args.update_target_network_step:
            self.target_q.load_state_dict(self.eval_q.state_dict())

    def decrement_epsilon(self):
        self.epsilon = min(self.epsilon * args.epsilon_decay, args.min_epsilon)

    def learn(self):
        self.optimizer.zero_grad()
        self.replace_target_network()

        state_batch, action_batch, reward_batch, next_state_batch, is_terminal_batch, mask_batch = self.sample_memory()
        adjacent_matrix = torch.from_numpy(self.env.graph.link_adjacent_matrix).unsqueeze(dim=0).float().to(self.device)

        pred_q = self.eval_q(state)
        target_q =

        loss = self.loss_function(pred_q, target_q)
        loss.backward()
        self.optimizer.step()

        self.episode += 1
        self.decrement_epsilon()
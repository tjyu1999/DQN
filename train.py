import os
import shutil
import time
import datetime
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ladder import Ladder
from env import Env
from memory import Experience, Memory
from model import FeatureExtractor, QNetwork
from param import args


writer = SummaryWriter()


def soft_replacement(eval_net, target_net, tau):
    for eval_param, target_param in zip(eval_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * eval_param.data + (1 - tau) * target_param.data)


def hard_replacement(eval_net, target_net):
    for eval_param, target_param in zip(eval_net.parameters(), target_net.parameters()):
        target_param.data.copy_(eval_param.data)


class Trainer:
    def __init__(self):
        if args.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.data = Ladder()
        self.data.read_file()
        self.env = Env(self.data)
        self.memory = Memory(args.capacity)
        self.feature_extractor = FeatureExtractor(gcn_layer_dim=args.gcn_layer_dim,
                                                  device=self.device)
        self.eval_q = QNetwork(q_layer_dim=args.q_layer_dim,
                               device=self.device)
        self.target_q = deepcopy(self.eval_q)
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.SGD(self.eval_q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.epsilon = args.epsilon
        self.env_record = {'success_rate': [], 'usage': [], 'reward': []}
        self.training_record = []
        self.scalar = []

    def generate_experience(self):
        print(datetime.datetime.now().strftime('\n[%m-%d %H:%M:%S]'),
              '---------- Generating experience')
        self.env.reset()
        flow_indices = [idx for idx in range(len(self.data.flow_info))]
        random.shuffle(flow_indices)

        for flow_idx in flow_indices:
            action_transition = []
            self.env.flow_info(flow_idx)
            state = self.env.get_state()
            while True:
                adjacent_matrix = torch.from_numpy(self.env.graph.link_adjacent_matrix).unsqueeze(dim=0).float().to(self.device)
                embed_state = self.feature_extractor(state.unsqueeze(dim=0).to(self.device).detach(), adjacent_matrix.detach())
                link_q = self.eval_q(embed_state.detach()).reshape(-1)
                # link_q = self.eval_q(state.to(self.device).detach()).reshape(-1)
                link_idx = self.select_action(link_q)
                offset = self.env.find_slot(link_idx)

                # print('flow_idx', flow_idx, 'flow_src', self.env.flow_src, 'flow_dst', self.env.flow_dst, 'link_idx', link_idx, end=' ')

                current_state = state
                done, reward, state = self.env.update([link_idx, offset])
                action_transition.append([link_idx, offset])
                self.memory.push(current_state, link_idx, reward, state)

                # print('next_src', self.env.flow_src)
                # print(link_q)

                if done == 1:                                                  # successfully scheduled
                    for action in action_transition:
                        self.env.occupy_slot(action)
                    break
                elif done == -1:                                               # unsuccessfully scheduled
                    break
                elif done == 0:                                                # in progress
                    self.env.renew()

            self.env.refresh()

        success_rate = self.env.success_rate(len(flow_indices))
        usage = self.env.usage()
        reward = self.env.mean_reward(len(flow_indices))
        self.env_record['success_rate'].append(success_rate)
        self.env_record['usage'].append(usage)
        self.env_record['reward'].append(reward)
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Success Rate: {:.2f}% |'.format(success_rate * 100),
              'Usage: {:.2f}% |'.format(usage * 100),
              'Reward: {:.2f}'.format(reward))

    # select action through epsilon-greedy method
    def select_action(self, link_q):
        if random.random() < self.epsilon:                                                                            # exploration
            link_idx = random.sample(self.env.find_valid_link(), k=1)[0]
        else:                                                                                                         # exploitation
            max_q_idx = torch.argmax(torch.take(link_q.cpu(), torch.LongTensor(self.env.find_valid_link()))).item()
            link_idx = self.env.find_valid_link()[max_q_idx]

        return link_idx

    def train_one_episode(self):
        print(datetime.datetime.now().strftime('\n[%m-%d %H:%M:%S]'),
              '---------- Training memory')
        self.eval_q.train()

        transitions = self.memory.sample(batch_size=args.batch_size)
        batch = Experience(*zip(*transitions))
        state_batch = torch.FloatTensor(torch.stack(batch.state)).to(self.device)
        action_batch = torch.LongTensor([batch.action]).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(torch.stack(batch.next_state)).to(self.device)

        # update eval_q
        self.eval_q.zero_grad()
        adjacent_matrix = torch.from_numpy(self.env.graph.link_adjacent_matrix).unsqueeze(dim=0).float().to(self.device)
        embed_state_batch = self.feature_extractor(state_batch, adjacent_matrix)
        embed_next_state_batch = self.feature_extractor(next_state_batch.detach(), adjacent_matrix.detach())
        current_q = self.eval_q(embed_state_batch).squeeze().gather(1, action_batch).reshape(-1)
        target_q = reward_batch + args.gamma * self.target_q(embed_next_state_batch.detach()).max(1)[0].reshape(-1)

        # current_q = self.eval_q(state_batch).squeeze().gather(1, action_batch).reshape(-1)
        # target_q = reward_batch + args.gamma * self.target_q(next_state_batch.detach()).max(1)[0].reshape(-1)

        loss = self.loss_function(current_q, target_q)
        loss.backward()
        self.optimizer.step()

        self.training_record.append(loss.item())
        self.scalar.append(loss)
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Q Value: {:.4f}/{:.4f} |'.format(torch.mean(current_q), torch.mean(target_q)),
              'Loss: {:.4f} |'.format(loss), end=' ')

    def train(self):
        for episode in range(args.episodes):
            episode += 1
            start_time = time.time()
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Episode: {:04d}'.format(episode))
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Epsilon: {:.2f}'.format(self.epsilon))

            self.generate_experience()
            self.train_one_episode()
            print('Time: {:.2f}s'.format(time.time() - start_time))

            self.write_tensorboard(episode)
            if episode > args.exploration_end_episode and episode % args.epsilon_decay_step == 0:
                self.epsilon *= args.epsilon_decay
            if episode % args.update_target_q_step == 0:
                # soft_replacement(self.eval_q, self.target_q, args.tau)
                # hard_replacement(self.eval_q, self.target_q)
                self.target_q.load_state_dict(self.eval_q.state_dict())
                print(datetime.datetime.now().strftime('\n[%m-%d %H:%M:%S]'),
                      '---------- Copying parameters')
            if episode % args.save_record_step == 0:
                self.save_record(episode)
                print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                      '---------- Saving record')
            print('#' * 70)

    def save_record(self, episode):
        if not os.path.exists('record/env'):
            os.makedirs('record/env')
        for key in self.env_record.keys():
            np.save(f'record/env/{key}_{episode}.npy', self.env_record[key])
        np.save(f'record/loss_{episode}.npy', self.training_record)

    def write_tensorboard(self, episode):
        while len(self.scalar) > args.window_size:
            self.scalar.pop(0)
        if episode >= args.window_size:
            writer.add_scalar('loss', torch.mean(torch.stack(self.scalar)), episode)


writer.close()


def main():
    if os.path.exists('record'):
        shutil.rmtree('record')
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
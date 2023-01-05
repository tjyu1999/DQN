import torch
import torch.nn as nn
import torch.nn.functional as F
from network import GraphCNN


class FeatureExtractor(nn.Module):
    def __init__(self,
                 gcn_layer_dim,
                 device):
        super().__init__()
        self.gcn = GraphCNN(layer_dim=gcn_layer_dim,
                            device=device)

    def forward(self, state, adjacent_matrix):
        embed_state = self.gcn(state, adjacent_matrix)

        return embed_state


class QNetwork(nn.Module):
    def __init__(self,
                 q_layer_dim,
                 device):
        super().__init__()
        self.hid_layer_1 = nn.Linear(q_layer_dim[0], q_layer_dim[1])
        self.hid_layer_2 = nn.Linear(q_layer_dim[1], q_layer_dim[2])
        self.hid_layer_3 = nn.Linear(q_layer_dim[2], q_layer_dim[3])
        self.to(device)

    def forward(self, embed_state):
        out = F.leaky_relu(self.hid_layer_1(embed_state))
        out = F.leaky_relu(self.hid_layer_2(out))
        out = self.hid_layer_3(out)

        return out
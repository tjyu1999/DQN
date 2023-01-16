import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkBatchNorm(nn.Module):
    def __init__(self,
                 dim,
                 device):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(dim)
        self.to(device)

    def forward(self, x):
        out = x.transpose(1, 2).contiguous()
        out = self.batch_norm(out).transpose(1, 2).contiguous()
        return out


class GraphCNNLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 device):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        # nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.zeros_(self.bias)
        self.init_parameters()
        self.to(device)

    def init_parameters(self):
        stdev = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdev, stdev)
        self.bias.data.uniform_(-stdev, stdev)

    def forward(self, x, adjacent_matrix):
        batch_size, _, _ = x.shape
        out = torch.bmm(x, self.weight.repeat(batch_size, 1, 1))
        out = torch.bmm(adjacent_matrix.repeat(batch_size, 1, 1), out) + self.bias

        return out


class GraphCNN(nn.Module):
    def __init__(self,
                 layer_dim,
                 device):
        super().__init__()
        self.gcn_layer_1 = GraphCNNLayer(layer_dim[0], layer_dim[1], device)
        self.gcn_layer_2 = GraphCNNLayer(layer_dim[1], layer_dim[2], device)
        self.to(device)

    def forward(self, x, adjacent_matrix):
        out = F.leaky_relu(self.gcn_layer_1(x, adjacent_matrix))
        out = F.leaky_relu(self.gcn_layer_2(out, adjacent_matrix))

        return out


class QNetwork(nn.Module):
    def __init__(self,
                 layer_dim,
                 device):
        super().__init__()
        self.hid_layer_1 = nn.Linear(layer_dim[0], layer_dim[1])
        self.hid_layer_2 = nn.Linear(layer_dim[1], layer_dim[2])
        self.hid_layer_3 = nn.Linear(layer_dim[2], layer_dim[3])
        self.to(device)

    def forward(self, embed_state):
        out = F.relu(self.hid_layer_1(embed_state))
        out = F.relu(self.hid_layer_2(out))
        out = self.hid_layer_3(out).squeeze()

        return out
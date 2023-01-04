import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphCNNLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 device):
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.biases = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.biases)
        self.to(device)

    def forward(self, x, adjacent_matrix):
        batch_size, _, _ = x.shape
        out = torch.bmm(x, self.weights.repeat(batch_size, 1, 1))
        out = torch.bmm(adjacent_matrix.repeat(batch_size, 1, 1), out) + self.biases

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
import torch.nn as nn
from network import GraphCNN, QNetwork


class Model(nn.Module):
    def __init__(self,
                 gcn_layer_dim,
                 q_layer_dim,
                 device):
        super().__init__()
        self.gcn = GraphCNN(layer_dim=gcn_layer_dim,
                            device=device)
        self.qnet = QNetwork(layer_dim=q_layer_dim,
                             device=device)

    def forward(self, state, adjacent_matrix):
        embed_state = self.gcn(state, adjacent_matrix)
        out = self.qnet(embed_state)

        return out
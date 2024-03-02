import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, SiLU
from torch_geometric.nn import MessagePassing
# from torch_scatter import scatter

class PositionalNNConvLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, nn, edge_dim=4, aggr='mean'):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = hidden_channels
        self.nn = nn
        self.aggregation = aggr

        self.mlp_msg = Sequential(
            Linear(2*hidden_channels + edge_dim + 1, hidden_channels), SiLU(),
            Linear(hidden_channels, hidden_channels)
          )

        self.mlp_upd = Sequential(
            Linear(2*hidden_channels, hidden_channels), SiLU(),
            Linear(hidden_channels, hidden_channels)
          )

    def forward(self, h, pos, edge_index, edge_attr=None):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr, pos=pos)
        return out

    def message(self, h_i, h_j, pos, edge_index, edge_attr):
        # h_i, h_j are the features of the source and target nodes, respectively
        pos_i = pos[edge_index[1]]  # Target node positions
        pos_j = pos[edge_index[0]]  # Source node positions
        rel_pos = pos_i - pos_j  # Relative positions
        rel_pos_norm = (torch.norm(rel_pos, p=2, dim=-1) ** 2).unsqueeze(-1)
        weight = self.nn(torch.cat([edge_attr, rel_pos_norm], dim=-1))
        weight = weight.view(-1, self.in_channels, self.out_channels)
        msg = torch.matmul(h_j.unsqueeze(1), weight).squeeze(1)
        return msg

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggregation)

    def update(self, inputs, h):
        aggr_msg = inputs
        upd_out = torch.cat([h, aggr_msg], dim=-1)
        return self.mlp_upd(upd_out)
    
class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, edge_dim=4, aggr='mean'):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = hidden_channels
        self.aggregation = aggr

        self.mlp_msg = Sequential(
            Linear(2*hidden_channels + edge_dim + 1, hidden_channels), SiLU(),
            Linear(hidden_channels, hidden_channels), SiLU()
          )

        self.mlp_pos = Sequential(
             Linear(hidden_channels, hidden_channels), SiLU(), Linear(hidden_channels, 1))


        self.mlp_upd = Sequential(
            Linear(2*hidden_channels, hidden_channels), SiLU(),
            Linear(hidden_channels, hidden_channels), SiLU()
          )

    def forward(self, h, pos, edge_index, edge_attr=None):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr, pos=pos)
        return out

    def message(self, h_i, h_j, pos, edge_index, edge_attr):
        # h_i, h_j are the features of the source and target nodes, respectively
        pos_i = pos[edge_index[1]]  # Target node positions
        pos_j = pos[edge_index[0]]  # Source node positions
        rel_pos = pos_i - pos_j  # Relative positions
        rel_pos_norm = (torch.norm(rel_pos, p=2, dim=-1)).unsqueeze(-1)
        msg = torch.cat([h_i, h_j, edge_attr, rel_pos_norm], dim=-1)
        msg = self.mlp_msg(msg)
        rel_pos = rel_pos * self.mlp_pos(msg)
        return msg, rel_pos

    def aggregate(self, inputs, index):
        msg, rel_pos = inputs
        aggr_msg = scatter(msg, index, dim=self.node_dim, reduce=self.aggregation)
        aggr_pos = scatter(rel_pos, index, dim=self.node_dim, reduce=self.aggregation)
        return aggr_msg, aggr_pos

    def update(self, inputs, h, pos):
        aggr_msg, aggr_pos = inputs
        upd_out = torch.cat([h, aggr_msg], dim=-1)
        upd_pos = pos + aggr_pos
        return self.mlp_upd(upd_out), upd_pos
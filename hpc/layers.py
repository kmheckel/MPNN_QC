import torch
from torch.nn import Sequential, Linear, SiLU, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class EGNNLayer(MessagePassing):
    """
    Implementation of the EGNN (Equivariant Graph Neural Network) layer.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        edge_dim (int, optional): Dimensionality of edge features. Defaults to 4.
        aggr (str, optional): Aggregation method for message passing. Defaults to 'sum'.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        aggregation (str): Aggregation method for message passing.
        mlp_msg (torch.nn.Sequential): MLP for message computation.
        mlp_upd (torch.nn.Sequential): MLP for node update.

    """

    def __init__(self, in_channels, hidden_channels, edge_dim=4, aggr='mean'):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = hidden_channels
        self.aggregation = aggr

        self.mlp_msg = Sequential(
            Linear(2*hidden_channels + edge_dim + 1, hidden_channels), BatchNorm1d(hidden_channels), SiLU(),
            Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), SiLU()
        )

        self.mlp_upd = Sequential(
            Linear(2*hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), SiLU(),
            Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        Forward pass of the EGNN layer.

        Args:
            x (torch.Tensor): Node features.
            pos (torch.Tensor): Node positions.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor, optional): Edge features. Defaults to None.

        Returns:
            torch.Tensor: Updated node features.

        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)
        return out

    def message(self, x_i, x_j, pos, edge_index, edge_attr):
        """
        Message computation for the EGNN layer.

        Args:
            x_i (torch.Tensor): Source node features.
            x_j (torch.Tensor): Target node features.
            pos (torch.Tensor): Node positions.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge features.

        Returns:
            torch.Tensor: Computed messages.

        """
        pos_i = pos[edge_index[1]]  # Target node positions
        pos_j = pos[edge_index[0]]  # Source node positions
        rel_pos = pos_i - pos_j  # Relative positions
        rel_pos_norm = (torch.norm(rel_pos, p=2, dim=-1) ** 2).unsqueeze(-1)
        msg = torch.cat([x_i, x_j, edge_attr, rel_pos_norm], dim=-1)
        msg = self.mlp_msg(msg)
        return msg

    def aggregate(self, inputs, index):
        """
        Aggregation of messages for the EGNN layer.

        Args:
            inputs (torch.Tensor): Messages.
            index (torch.Tensor): Node indices.

        Returns:
            torch.Tensor: Aggregated messages.

        """
        msg = inputs
        aggr_msg = scatter(msg, index, dim=self.node_dim, reduce=self.aggregation)
        return aggr_msg

    def update(self, inputs, x):
        """
        Node update for the EGNN layer.

        Args:
            inputs (torch.Tensor): Aggregated messages.
            x (torch.Tensor): Node features.

        Returns:
            torch.Tensor: Updated node features.

        """
        aggr_msg = inputs
        upd_out = torch.cat([x, aggr_msg], dim=-1)
        return x + self.mlp_upd(upd_out)

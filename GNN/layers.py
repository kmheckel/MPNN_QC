import torch
from torch.nn import Sequential, Linear, SiLU, BatchNorm1d, Sigmoid
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class EGNNLayer(MessagePassing):
    """
    Equivariant Graph Neural Network layer, implementing message passing with attention and equivariance to node permutations.
    This layer processes node features, node positions, and edge attributes to update node features while considering geometric relations.
    """

    def __init__(self, in_channels, hidden_channels, edge_dim=4, aggr='mean'):
        # Initialize the base MessagePassing class with the aggregation method.
        super().__init__(aggr=aggr)
        
        # Input and output channel sizes.
        self.in_channels = in_channels
        self.out_channels = hidden_channels
        # The aggregation method used for message passing.
        self.aggregation = aggr

        # MLP for message computation combining node features, edge attributes, and geometric information.
        self.mlp_msg = Sequential(
            Linear(2 * hidden_channels + edge_dim + 1, hidden_channels),
            BatchNorm1d(hidden_channels),
            SiLU(),
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            SiLU()
        )
        # MLP for node updates based on aggregated messages and original node features.
        self.mlp_upd = Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            SiLU(),
            Linear(hidden_channels, hidden_channels),
        )
        # MLP for computing attention weights based on message features.
        self.att_mlp = Sequential(
            Linear(hidden_channels, 1),
            Sigmoid()
        )

    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        Propagates node features and edge attributes through the graph to update node features.
        """
        # Call the propagate method from MessagePassing, which internally calls message, aggregate, and update methods.
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)
        return out

    def message(self, x_i, x_j, pos, edge_index, edge_attr):
        """
        Constructs messages for each edge in the graph by combining features of the source and target nodes,
        their relative positions, and edge attributes.
        """
        # Compute relative positions between connected nodes.
        pos_i = pos[edge_index[1]]  # Target node positions
        pos_j = pos[edge_index[0]]  # Source node positions
        rel_pos = pos_i - pos_j  # Relative positions vector
        rel_pos_norm = (torch.norm(rel_pos, p=2, dim=-1) ** 2).unsqueeze(-1)  # Squared Euclidean norm

        # Concatenate node features, edge attributes, and relative position information.
        msg = torch.cat([x_i, x_j, edge_attr, rel_pos_norm], dim=-1)
        # Pass the concatenated vector through an MLP.
        msg = self.mlp_msg(msg)
        # Apply attention to the message.
        msg = msg * self.att_mlp(msg)
        return msg

    def aggregate(self, inputs, index):
        """
        Aggregates messages at target nodes using the specified aggregation method.
        """
        # Aggregate messages using the scatter operation provided by torch_scatter, based on the aggregation method specified.
        aggr_msg = scatter(inputs, index, dim=self.node_dim, reduce=self.aggregation)
        return aggr_msg

    def update(self, inputs, x):
        """
        Updates node features by combining the original features with aggregated messages.
        """
        # Concatenate original node features with aggregated messages.
        upd_out = torch.cat([x, inputs], dim=-1)
        # Apply an MLP to the concatenated vector and add residual connection.
        return x + self.mlp_upd(upd_out)

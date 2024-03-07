import torch
from torch.nn import Linear, Sequential, SiLU, GRU
from layers import EGNNLayer #, PositionalNNConvLayer
from torch_geometric.nn import NNConv, GRUAggregation, Set2Set, GATv2Conv, GatedGraphConv, global_add_pool

class GNN(torch.nn.Module):
    """
    Graph Neural Network (GNN) model.

    Args:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        output_channels (int, optional): Number of output channels. Defaults to 1.
        num_layers (int, optional): Number of GNN layers. Defaults to 4.
        M (int, optional): Number of processing steps for Set2Set aggregation. Defaults to 3.
        edge_feat_dim (int, optional): Dimensionality of edge features. Defaults to 4.
        nn_width_factor (int, optional): Width factor for the MLP used in NNConv. Defaults to 2.
        aggregation (str, optional): Aggregation method. Can be 's2s' (Set2Set) or 'gru' (GRUAggregation). Defaults to 's2s'.
        model_name (str, optional): Name of the GNN model. Can be 'NNConv', 'GAT', or 'GGNN'. Defaults to 'NNConv'.
        args (object, optional): Additional arguments. Defaults to None.
    """

    def __init__(self, input_channels, hidden_channels, output_channels=1,
                 num_layers=4, M=3, edge_feat_dim=4,
                 nn_width_factor=2, aggregation='s2s', model_name='NNConv', args=None):
        super().__init__()
        self.num_layers = num_layers
        self.first_layer = Linear(input_channels, hidden_channels)
        self.args = args
        self.gru = GRU(hidden_channels, hidden_channels)
        self.nl = SiLU()

        # Define the MLP for transforming edge features for NNConv
        if 'nnconv' in model_name.lower():
            self.nn = Sequential(Linear(edge_feat_dim, hidden_channels * nn_width_factor),
                            SiLU(), Linear(hidden_channels * nn_width_factor,
                                            hidden_channels * hidden_channels))

        if 'nnconv' in model_name.lower():
            self.conv = NNConv(hidden_channels, hidden_channels, self.nn)
        elif 'gat' in model_name.lower():
            self.conv = GATv2Conv(hidden_channels, hidden_channels, heads=1,
                                        edge_dim=edge_feat_dim)
        elif 'ggnn' in model_name.lower():
            self.conv = GatedGraphConv(hidden_channels, hidden_channels)
        else:
            raise ValueError(f"Model {model_name} not recognized.")
        
        self.edge_attr_exists = 'edge_attr' in self.conv.forward.__code__.co_varnames
        if aggregation == 's2s':
          self.aggr = Set2Set(in_channels=hidden_channels, processing_steps=M)
          pred_channels = 2 * hidden_channels
        else:
            self.aggr = GRUAggregation(in_channels=hidden_channels, out_channels=hidden_channels)
            pred_channels = hidden_channels

        if args.use_branching and args.predict_all:
            self.out_nets = torch.nn.ModuleList([Sequential(
                Linear(pred_channels, hidden_channels), SiLU(),
                Linear(hidden_channels, 1)
                ) for _ in range(args.output_channels)])
        else:
            self.out = Sequential(
                Linear(pred_channels, hidden_channels), SiLU(),
                Linear(hidden_channels, output_channels)
                )

    def forward(self, data):
        """
        Forward pass of the GNN model.

        Args:
            data (torch_geometric.data.Data): Input data.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.nl(self.first_layer(data.x))
        h = x.unsqueeze(0).contiguous()  # Add singleton dimension for edge features
        # for conv in self.convs:
        for _ in range(self.num_layers):
            if self.edge_attr_exists:
                m = self.nl(self.conv(x, data.edge_index, edge_attr=data.edge_attr))
                x, h = self.gru(m.unsqueeze(0), h.contiguous())
                x = x.squeeze(0)
            else:
                x = self.nl(self.conv(x, data.edge_index))
                x, h = self.gru(x.unsqueeze(0), h.contiguous())
                x = x.squeeze(0)
        x = self.aggr(x, data.batch)
        if not self.args.predict_all or not self.args.use_branching:
            x = self.out(x)
        else:
            x = torch.cat([out_net(x) for out_net in self.out_nets], dim=1)
        return x.squeeze(-1)


class TowerGNN(torch.nn.Module):
    """
    Tower GNN model. For NNConv and EGNN with large hidden
    channel size.

    Args:
        input_channels (int, optional): Number of input channels. Defaults to 11.
        hidden_channels (int, optional): Number of hidden channels. Defaults to 200.
        num_towers (int, optional): Number of towers. Defaults to 8.
        output_channels (int, optional): Number of output channels. Defaults to 1.
        num_layers (int, optional): Number of GNN layers. Defaults to 4.
        M (int, optional): Number of processing steps for Set2Set aggregation. Defaults to 3.
        edge_feat_dim (int, optional): Dimensionality of edge features. Defaults to 4.
        nn_width_factor (int, optional): Width factor for the MLP used in NNConv. Defaults to 2.
        model_name (str, optional): Name of the GNN model. Can be 'nnconv' or 'egnn'. Defaults to 'nnconv'.
        args (object, optional): Additional arguments. Defaults to None.
    """

    def __init__(self, input_channels=11, hidden_channels=200, num_towers=8, output_channels=1,
                 num_layers=4, M=3, edge_feat_dim=4,
                 nn_width_factor=2, model_name='nnconv', args=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_towers = num_towers
        self.tower_dim = hidden_channels // num_towers
        self.args = args
        self.nnconv = 'nnconv' in model_name.lower()
        
        # Initial transformation
        self.first_layer = Linear(input_channels, hidden_channels)
        self.grus = torch.nn.ModuleList([GRU(self.tower_dim, self.tower_dim) for _ in range(num_towers)])
        self.nl = SiLU()
        
        # Edge MLP for NNConv
        if self.nnconv:
            self.nns = torch.nn.ModuleList([Sequential(Linear(edge_feat_dim, self.tower_dim * nn_width_factor),
                                SiLU(), Linear(self.tower_dim * nn_width_factor, self.tower_dim ** 2)) for _ in range(num_towers)])

        # Towers for each convolution layer
        if self.nnconv:
            self.towers = torch.nn.ModuleList([
                NNConv(self.tower_dim, self.tower_dim, self.nns[i]) for i in range(num_towers)
            ])
        elif 'ggnn' in model_name.lower():
            self.towers = torch.nn.ModuleList([
                GatedGraphConv(self.tower_dim, self.tower_dim) for _ in range(num_towers)
            ])
        else:
            raise ValueError(f"Model {model_name} not recognized.")
        # Shared mixing network
        self.mixing_network = Sequential(
            Linear(hidden_channels, hidden_channels), SiLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.aggr = Set2Set(in_channels=hidden_channels, processing_steps=M)
        pred_channels = 2 * hidden_channels
        
        self.out = Sequential(
            Linear(pred_channels, hidden_channels), SiLU(),
            Linear(hidden_channels, output_channels)
            )
    
    def forward(self, data):
        """
        Forward pass of the TowerNNConv model.

        Args:
            data (torch_geometric.data.Data): Input data.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.nl(self.first_layer(data.x))
        # Split the embeddings into towers
        towers_embeddings = x.split(self.tower_dim, dim=1)
        # Process each tower separately
        updated_towers = []
        for i, tower in enumerate(self.towers):
            h = towers_embeddings[i].unsqueeze(0).contiguous()  # Ensure h is contiguous
            for _ in range(self.num_layers):
                if self.nnconv:
                    tower_output = self.nl(tower(towers_embeddings[i], data.edge_index, edge_attr=data.edge_attr))
                else:
                    tower_output = self.nl(tower(towers_embeddings[i], data.edge_index))
                tower_output, h = self.grus[i](tower_output.unsqueeze(0), h.contiguous())
                tower_output = tower_output.squeeze(0)
            updated_towers.append(tower_output)
        
        # Concatenate and mix the outputs from the towers
        mixed_embeddings = torch.cat(updated_towers, dim=1)
        mixed_embeddings = self.mixing_network(mixed_embeddings)
        
        # Aggregate and output
        x = self.aggr(mixed_embeddings, data.batch)
        x = self.out(x)
        return x.squeeze(-1)


class EGNN(torch.nn.Module):
    """
    Equivariant Graph Neural Network (EGNN) model.

    Args:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        output_channels (int, optional): Number of output channels. Defaults to 1.
        num_layers (int, optional): Number of EGNN layers. Defaults to 4.
    """

    def __init__(self, input_channels, hidden_channels, output_channels=1,
                 num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.first_layer = Linear(input_channels, hidden_channels)
        self.nl = SiLU()
        self.predictor1 = Sequential(Linear(hidden_channels, hidden_channels), SiLU(),
                                    Linear(hidden_channels, hidden_channels))
        self.predictor2 = Sequential(Linear(hidden_channels, hidden_channels), SiLU(),
                                    Linear(hidden_channels, output_channels))
        self.convs = torch.nn.ModuleList([EGNNLayer(hidden_channels, hidden_channels) 
                                            for _ in range(num_layers)])

    def forward(self, data):
        """
        Forward pass of the EGNN model.

        Args:
            data (torch_geometric.data.Data): Input data.

        Returns:
            torch.Tensor: Output tensor.
        """
        x, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
        x = self.nl(self.first_layer(data.x))
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        x = self.predictor1(x)
        x = global_add_pool(x, data.batch)
        x = self.predictor2(x)
        return x.squeeze(-1)



######## OLD CODE ########
# class SpatialGNN(torch.nn.Module):
#     def __init__(self, input_channels, hidden_channels, output_channels=1,
#                  num_layers=4, M=3, edge_feat_dim=4,
#                  nn_width_factor=2, aggregation='s2s', model_name='NNConv', args=None):
#         super().__init__()
#         self.num_layers = num_layers
#         self.first_layer = Linear(input_channels, hidden_channels)
#         self.convs = torch.nn.ModuleList()
#         self.nl = SiLU()
#         self.gru = GRU(hidden_channels, hidden_channels)
#         self.nnconv = 'nnconv' in model_name.lower()

#         # Define the MLP for transforming edge features for NNConv
#         if self.nnconv:
#             self.nn = Sequential(Linear(edge_feat_dim+1, hidden_channels * nn_width_factor),
#                             SiLU(), Linear(hidden_channels * nn_width_factor,
#                                             hidden_channels * hidden_channels))

#         if self.nnconv:
#             self.conv = PositionalNNConvLayer(hidden_channels, hidden_channels, self.nn)
#             # self.convs = [PositionalNNConvLayer(hidden_channels, hidden_channels, self.nn)) for _ in range(num_layers)]
#         elif 'egnn' in model_name.lower():
#             # self.conv = EGNNLayer(hidden_channels, hidden_channels)
#             self.predictor1 = Sequential(Linear(hidden_channels, hidden_channels), SiLU(),
#                                         Linear(hidden_channels, hidden_channels))
#             self.predictor2 = Sequential(Linear(hidden_channels, hidden_channels), SiLU(),
#                                         Linear(hidden_channels, output_channels))
#             self.convs = torch.nn.ModuleList([EGNNLayer(hidden_channels, hidden_channels) 
#                                              for _ in range(num_layers)])
#         else:
#             raise ValueError(f"Model {model_name} not recognized.")

#         if aggregation == 's2s':
#           self.aggr = Set2Set(in_channels=hidden_channels, processing_steps=M)
#           self.out = Sequential(
#               Linear(2*hidden_channels, hidden_channels), SiLU(),
#               Linear(hidden_channels, output_channels)
#             )
#         else:
#           self.aggr = GRUAggregation(in_channels=hidden_channels, out_channels=hidden_channels)
#           self.out = Sequential(
#               Linear(hidden_channels, hidden_channels), SiLU(),
#               Linear(hidden_channels, output_channels)
#             )

#     def forward(self, data):
#         x, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
#         x = self.nl(self.first_layer(data.x))
#         h = x.unsqueeze(0)  # Add singleton dimension for edge features

#         # for conv in self.convs:
#         if self.nnconv:
#             for _ in range(self.num_layers):
#                 x = self.nl(self.conv(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr))
#                 x, h = self.gru(x.unsqueeze(0), h)
#                 x = x.squeeze(0)
#             x = self.aggr(x, data.batch)
#             x = self.out(x)
#         else:
#             for conv in self.convs:
#                 x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
#             x = self.predictor1(x)
#             x = global_add_pool(x, data.batch)
#             x = self.predictor2(x)
#         return x.squeeze()
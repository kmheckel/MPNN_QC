import torch
from torch.nn import Linear, Sequential, SiLU
from layers import EGNNLayer, PositionalNNConvLayer
from torch_geometric.nn import NNConv, GRUAggregation, Set2Set, GATv2Conv, GatedGraphConv

class NonSpatialGNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels=1,
                 num_layers=4, M=3, edge_feat_dim=4,
                 nn_width_factor=2, aggregation='s2s', model_name='NNConv', args=None):
        super().__init__()
        self.num_layers = num_layers
        self.first_layer = Linear(input_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.args = args

        # Define the MLP for transforming edge features for NNConv
        if 'nnconv' in model_name.lower():
            self.nn = Sequential(Linear(edge_feat_dim, hidden_channels * nn_width_factor),
                            SiLU(), Linear(hidden_channels * nn_width_factor,
                                            hidden_channels * hidden_channels))

        for _ in range(num_layers):
            if 'nnconv' in model_name.lower():
                self.convs.append(NNConv(hidden_channels, hidden_channels, self.nn))
            elif 'gat' in model_name.lower():
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=1,
                                            edge_dim=edge_feat_dim))
            elif 'ggnn' in model_name.lower():
                self.convs.append(GatedGraphConv(hidden_channels, hidden_channels))
            else:
                raise ValueError(f"Model {model_name} not recognized.")

        if aggregation == 's2s':
          self.aggr = Set2Set(in_channels=hidden_channels, processing_steps=M)
          pred_channels = 2 * hidden_channels
        else:
            self.aggr = GRUAggregation(in_channels=hidden_channels, out_channels=hidden_channels)
            pred_channels = hidden_channels

        # if args.use_branching and args.predict_all:
        #     self.out_nets = torch.nn.ModuleList([Sequential(
        #         Linear(pred_channels, hidden_channels), SiLU(),
        #         Linear(hidden_channels, hidden_channels), SiLU(),
        #         Linear(hidden_channels, 1)
        #         ) for _ in range(args.output_channels)])
        # else:
        self.out = Sequential(
            Linear(pred_channels, hidden_channels), SiLU(),
            Linear(hidden_channels, output_channels)
            )

    def forward(self, data):
        x = self.first_layer(data.x)
        for conv in self.convs:
            if 'edge_attr' in conv.forward.__code__.co_varnames:
                x = x + conv(x, data.edge_index, edge_attr=data.edge_attr)
            else:
                x = x + conv(x, data.edge_index) 
        x = self.aggr(x, data.batch)
        # if not self.args.predict_all or not self.args.use_branching:
        x = self.out(x)
        # else:
        #     x = torch.cat([out_net(x) for out_net in self.out_nets], dim=1)
        # print(x.shape, x.squeeze().shape)
        return x.squeeze()
    
class SpatialGNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels=1,
                 num_layers=4, M=3, edge_feat_dim=4,
                 nn_width_factor=2, aggregation='s2s', model_name='NNConv', args=None):
        super().__init__()
        self.num_layers = num_layers
        self.first_layer = Linear(input_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()

        # Define the MLP for transforming edge features for NNConv
        if 'nnconv' in model_name.lower():
            self.nn = Sequential(Linear(edge_feat_dim+1, hidden_channels * nn_width_factor),
                            SiLU(), Linear(hidden_channels * nn_width_factor,
                                            hidden_channels * hidden_channels))

        for _ in range(num_layers):
            if 'nnconv' in model_name.lower():
                self.convs.append(PositionalNNConvLayer(hidden_channels, hidden_channels, self.nn))
            elif 'egnn' in model_name.lower():
                self.convs.append(EGNNLayer(hidden_channels, hidden_channels))
            else:
                raise ValueError(f"Model {model_name} not recognized.")

        if aggregation == 's2s':
          self.aggr = Set2Set(in_channels=hidden_channels, processing_steps=M)
          self.out = Sequential(
              Linear(2*hidden_channels, hidden_channels), SiLU(),
              Linear(hidden_channels, output_channels)
            )
        else:
          self.aggr = GRUAggregation(in_channels=hidden_channels, out_channels=hidden_channels)
          self.out = Sequential(
              Linear(hidden_channels, hidden_channels), SiLU(),
              Linear(hidden_channels, output_channels)
            )

    def forward(self, data):
        h, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
        h = self.first_layer(h)
        for conv in self.convs:
            h_update, pos_update = conv(h, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
            h = h + h_update
            pos = pos_update
        h = self.aggr(h, data.batch)
        h = self.out(h)
        return h.view(-1)
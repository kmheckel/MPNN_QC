import torch
from torch.nn import Linear, Sequential, SiLU, GRU, BatchNorm1d
from layers import EGNNLayer
from torch_geometric.nn import NNConv, Set2Set, GATv2Conv, global_add_pool

class GNN(torch.nn.Module):
    """
    Implements a Graph Neural Network model supporting various convolutional layers and aggregation methods.
    Can be configured to use different GNN layer types (e.g., NNConv, GATv2Conv) and to predict either single or multiple targets.
    """

    def __init__(self, input_channels, hidden_channels, output_channels=1,
                 num_layers=4, M=3, edge_feat_dim=4,
                 nn_width_factor=2, model_name='NNConv', args=None):
        super().__init__()
        # Number of GNN convolution layers
        self.num_layers = num_layers
        # Initial transformation from input to hidden dimension
        self.first_layer = Linear(input_channels, hidden_channels)
        # Additional arguments for model configuration
        self.args = args
        # GRU for sequential processing across layers
        self.gru = GRU(hidden_channels, hidden_channels)
        # Non-linear activation function
        self.nl = SiLU()

        # Configure NNConv specific components
        if 'nnconv' in model_name.lower():
            # MLP for transforming edge features in NNConv
            self.nn = Sequential(
                Linear(edge_feat_dim, hidden_channels * nn_width_factor),
                BatchNorm1d(hidden_channels * nn_width_factor),
                SiLU(), 
                Linear(hidden_channels * nn_width_factor, hidden_channels * hidden_channels)
            )
            # The NNConv layer itself
            self.conv = NNConv(hidden_channels, hidden_channels, self.nn, aggr='mean')
        elif 'gat' in model_name.lower():
            # GATv2Conv layer for attention-based convolution
            self.conv = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, edge_dim=edge_feat_dim)
        else:
            raise ValueError(f"Model {model_name} not recognized.")
        
        # Set2Set layer for graph-level representation
        self.aggr = Set2Set(in_channels=hidden_channels, processing_steps=M)
        # Determine the size of the output from aggregation
        pred_channels = 2 * hidden_channels

        # Configure output layers for single or multiple target prediction
        if args.use_branching and args.predict_all:
            # Multiple sequential modules for predicting multiple targets
            self.out_nets = torch.nn.ModuleList([
                Sequential(
                    Linear(pred_channels, hidden_channels), BatchNorm1d(hidden_channels), SiLU(),
                    Linear(hidden_channels, 1)
                ) for _ in range(args.output_channels)
            ])
        else:
            # Single sequential module for single target prediction
            self.out = Sequential(
                Linear(pred_channels, hidden_channels), BatchNorm1d(hidden_channels), SiLU(),
                Linear(hidden_channels, output_channels)
            )

    def forward(self, data):
        """
        Forward pass of the GNN model. Processes input graph data and produces output predictions.
        """
        # Apply initial linear transformation and non-linearity
        x = self.nl(self.first_layer(data.x))
        # Prepare hidden state for GRU
        h = x.unsqueeze(0).contiguous()
        # Process each layer
        for _ in range(self.num_layers):
            # Apply GNN convolution and non-linearity
            m = self.nl(self.conv(x, data.edge_index, edge_attr=data.edge_attr))
            # Update the state with GRU
            x, h = self.gru(m.unsqueeze(0), h.contiguous())
            x = x.squeeze(0)
        # Aggregate node features to graph-level representation
        x = self.aggr(x, data.batch)
        # Generate output predictions
        if not self.args.predict_all or not self.args.use_branching:
            x = self.out(x)
        else:
            # Concatenate outputs from multiple target-specific networks
            x = torch.cat([out_net(x) for out_net in self.out_nets], dim=1)
        return x.squeeze(-1)

class TowerGNN(torch.nn.Module):
    """
    Tower GNN model, designed for handling large hidden dimension sizes by splitting them into
    multiple "towers", each processing a subset of the features independently.
    """

    def __init__(self, input_channels=11, hidden_channels=200, num_towers=8, output_channels=1,
                 num_layers=4, M=3, edge_feat_dim=4,
                 nn_width_factor=2, args=None):
        super().__init__()
        # Number of layers and towers
        self.num_layers = num_layers
        self.num_towers = num_towers
        # Compute dimension for each tower
        self.tower_dim = hidden_channels // num_towers
        self.args = args
        
        # Initial linear transformation
        self.first_layer = Linear(input_channels, hidden_channels)
        # GRU for each tower
        self.grus = torch.nn.ModuleList([GRU(self.tower_dim, self.tower_dim) for _ in range(num_towers)])
        # Non-linear activation function
        self.nl = SiLU()
        
        # MLPs for transforming edge features, one per tower
        self.nns = torch.nn.ModuleList([
            Sequential(
                Linear(edge_feat_dim, self.tower_dim * nn_width_factor), 
                BatchNorm1d(self.tower_dim * nn_width_factor),
                SiLU(), 
                Linear(self.tower_dim * nn_width_factor, self.tower_dim ** 2)
            ) for _ in range(num_towers)
        ])

        # NNConv layers for each tower
        self.towers = torch.nn.ModuleList([
            NNConv(self.tower_dim, self.tower_dim, self.nns[i], aggr='mean') for i in range(num_towers)
        ])
        # Mixing network for integrating outputs from all towers
        self.mixing_network = Sequential(
            Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), SiLU(),
            Linear(hidden_channels, hidden_channels)
        )
        # Set2Set for global graph representation
        self.aggr = Set2Set(in_channels=hidden_channels, processing_steps=M)
        pred_channels = 2 * hidden_channels
        
        # Output layer for final prediction
        self.out = Sequential(
            Linear(pred_channels, hidden_channels), BatchNorm1d(hidden_channels), SiLU(),
            Linear(hidden_channels, output_channels)
        )
    
    def forward(self, data):
        """
        Forward pass of the TowerGNN model. Processes input graph data through multiple towers and produces output predictions.
        """
        x = self.nl(self.first_layer(data.x))
        # Split input features for each tower
        towers_embeddings = x.split(self.tower_dim, dim=1)
        updated_towers = []
        for i, tower in enumerate(self.towers):
            # Process each tower
            tower_output = towers_embeddings[i]
            h = tower_output.unsqueeze(0).contiguous()
            for _ in range(self.num_layers):
                tower_output = self.nl(tower(tower_output, data.edge_index, edge_attr=data.edge_attr))
                tower_output, h = self.grus[i](tower_output.unsqueeze(0), h.contiguous())
                tower_output = tower_output.squeeze(0)
            updated_towers.append(tower_output)
        
        # Integrate tower outputs and apply mixing network
        updated_towers = torch.cat(updated_towers, dim=1)
        updated_towers = self.mixing_network(updated_towers)
        # Aggregate to graph-level and predict
        updated_towers = self.aggr(updated_towers, data.batch)
        updated_towers = self.out(updated_towers)
        return updated_towers.squeeze(-1)

class EGNN(torch.nn.Module):
    """
    Implements an Equivariant Graph Neural Network (EGNN) model, preserving equivariance to graph isomorphisms,
    suitable for tasks where node positions are significant.
    """

    def __init__(self, input_channels, hidden_channels, output_channels=1, num_layers=4):
        super().__init__()
        # Number of EGNN layers
        self.num_layers = num_layers
        # Initial transformation from input to hidden dimension
        self.first_layer = Linear(input_channels, hidden_channels)
        # Non-linear activation function
        self.nl = SiLU()
        # Predictors for processing hidden representations
        self.predictor1 = Sequential(
            Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), SiLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.predictor2 = Sequential(
            Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), SiLU(),
            Linear(hidden_channels, output_channels)
        )
        # EGNN layers for processing graph structure and features
        self.convs = torch.nn.ModuleList([EGNNLayer(hidden_channels, hidden_channels) for _ in range(num_layers)])

    def forward(self, data):
        """
        Forward pass of the EGNN model. Processes input graph data, maintaining equivariance, and produces output predictions.
        """
        x, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
        x = self.nl(self.first_layer(x))
        for conv in self.convs:
            # Apply EGNN convolution to update node features
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        x = self.predictor1(x)
        # Global pooling across nodes in a graph
        x = global_add_pool(x, data.batch)
        x = self.predictor2(x)
        return x.squeeze(-1)

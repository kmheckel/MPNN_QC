import torch
import models  # Custom module containing GNN model definitions.
import argparse  # Module for parsing command-line arguments.
from trainer import run_experiment  # Function to run the training and evaluation loop.
from distutils.util import strtobool  # Utility function for converting strings to boolean.
from data import get_data  # Function to prepare and load the dataset.
import copy  # Module for creating shallow copies of objects.
import os.path as osp  # Module providing a way to use operating system dependent functionality.

def str2bool(v):
    """
    Converts a string representation of truth to boolean True or False.

    Args:
        v: A string representation of truth values.

    Returns:
        A boolean representing the truth value of the input string.
    """
    return bool(strtobool(v))

parser = argparse.ArgumentParser(description='Running MLMI4 experiments')

# Define command-line arguments for configuring the experiment.
parser.add_argument('--debugging', type=str2bool, default='False', help="If True uses 1000 samples")
parser.add_argument('--spatial', type=str2bool, default='True', help="Use spatial info?")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=3, help="number of GNN layers")
parser.add_argument('--hidden_channels', type=int, default=200, help="size of hidden node features")
parser.add_argument('--nn_width_factor', type=int, default=2, help="in NNConv, the width of the nn=h_theta")
parser.add_argument('--M', type=int, default=3, help="s2s processing steps")
parser.add_argument('--initial_lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--report_interval', type=int, default=1, help="report interval during training")
parser.add_argument('--num_epochs', type=int, default=120)
parser.add_argument('--patience', type=int, default=14)
parser.add_argument('--aggr', type=str, default='s2s', help="s2s/gru")
parser.add_argument('--target', type=int, default=0, help="target node from 0-11")
parser.add_argument('--predict_all', type=str2bool, default='False', help="Predict all targets?")
parser.add_argument('--use_branching', type=str2bool, default='False', help="Branch after GNN layers to predict all targets")
parser.add_argument('--model_name', type=str, default='NNConv', help="GGNN/NNConv/GAT/EGNN")
parser.add_argument('--num_towers', type=int, default=8, help="Number of towers in the model, 0 for no towers")
parser.add_argument('--pre_trained_path', type=str, default='', help="Path to existing model")
parser.add_argument('--data_split', type=int, default=1000, help="Number of samples in the dataset when debugging")
parser.add_argument('--standardization', type=str, default='std', help="std: y-->(y-mean)/std, mad: y-->(y-mean)/mad")

args = parser.parse_args()

# Ensure that the CUDA cache is cleared to free up memory.
torch.cuda.empty_cache()

# Modify some arguments based on the debugging flag.
if args.debugging:
    print("Running in debugging mode.")
    args.batch_size = 128
    args.num_epochs = 3
    args.patience = 3
    args.num_layers = 2
    args.hidden_channels = 16

# Adjust parameters based on the model name, e.g., for EGNN.
if 'egnn' in args.model_name.lower():
    args.egnn = True
    args.spatial = False
    args.num_towers = 0
else:
    args.egnn = False

# Modify the target variable if predicting all targets is enabled.
if args.predict_all:
    args.target = [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11]

# Set the device to use for computation.
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\n')
print(f'PREDICTING TARGET {args.target}', '\n')
print(args, '\n')

class MyTransform:
    """
    A transformation to apply to each data item in the dataset,
    adjusting the target variable based on the experiment configuration.
    """
    def __call__(self, data):
        data = copy.copy(data)  # Make a shallow copy to avoid modifying the original data.
        data.y = data.y[:, args.target]  # Select the target variable.
        return data

# Apply the custom transformation and split the data into training, validation, and test sets.
loc_transform = MyTransform()
args.train_loader, args.val_loader, args.test_loader = get_data(args, loc_transform)

# Extract input and output channel dimensions from the dataset.
args.output_channels = 1 if not args.predict_all else args.train_loader.dataset[0].y.shape[1]
args.input_channels = args.train_loader.dataset[0].x.shape[1]
args.edge_feat_dim = args.train_loader.dataset[0].edge_attr.shape[1]

print(f"""Input channels: {args.input_channels}, Output channels: {args.output_channels}, 
Edge feature dim: {args.edge_feat_dim}, Hidden channels: {args.hidden_channels}""")

# Initialize the model based on the specified architecture and parameters.
if not args.num_towers:
    if not args.egnn:
        # Initialize a standard GNN model.
        model = models.GNN(args.input_channels, args.hidden_channels,
                           num_layers=args.num_layers, M=args.M,
                           nn_width_factor=args.nn_width_factor,
                           edge_feat_dim=args.edge_feat_dim,
                           model_name=args.model_name,
                           output_channels=args.output_channels, args=args)
    else:
        # Initialize an EGNN model.
        model = models.EGNN(args.input_channels, args.hidden_channels,
                            num_layers=args.num_layers,
                            output_channels=args.output_channels)
else:
    # Initialize a TowerGNN model with multiple towers.
    model = models.TowerGNN(args.input_channels, args.hidden_channels,
                            num_layers=args.num_layers, M=args.M,
                            num_towers=args.num_towers,
                            edge_feat_dim=args.edge_feat_dim,
                            nn_width_factor=args.nn_width_factor,
                            output_channels=args.output_channels, args=args)

# Load a pre-trained model if specified.
if args.pre_trained_path:
    try:
        model.load_state_dict(torch.load(args.pre_trained_path,
                                         map_location=torch.device(args.device)))
        print(f"Loaded pre-trained model from {args.pre_trained_path}")
    except:
        raise Exception(f"Could not load pre-trained model from {args.pre_trained_path}, make sure to initialise the model with the same architecture!")

# Dynamically compile the model for optimization.
args.model_name = type(model).__name__
torch.compile(model, dynamic=True)

# Run the experiment with the configured model and dataset.
run_experiment(model, args)

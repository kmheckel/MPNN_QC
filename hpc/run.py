import torch
import models
import argparse
from trainer import run_experiment
from distutils.util import strtobool
from data import get_data

def str2bool(v):
    return bool(strtobool(v))

parser = argparse.ArgumentParser(description='Running MLMI4 experiments')

# set arguments for training and decoding. 
# parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--debugging', type=str2bool, default='False', help="If True uses 1000 samples")
parser.add_argument('--spatial', type=str2bool, default='False', help="Use spatial info?")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=3, help="number of GNN layers")
parser.add_argument('--hidden_channels', type=int, default=200, help="size of hidden node features")
parser.add_argument('--nn_width_factor', type=int, default=2, help="in NNConv, the width of the nn=h_theta")
parser.add_argument('--M', type=int, default=4, help="s2s processing steps")
parser.add_argument('--initial_lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--report_interval', type=int, default=1, help="report interval during training")
parser.add_argument('--num_epochs', type=int, default=540)
parser.add_argument('--patience', type=int, default=50)
# parser.add_argument('--scheduler', type=str2bool, default='True', help="learning rate scheduler")
parser.add_argument('--aggr', type=str, default='s2s', help="s2s/gru")
parser.add_argument('--target', type=int, default=0, help="target node from 0-11")
parser.add_argument('--predict_all', type=str2bool, default='False', help="Predict all targets?")
parser.add_argument('--use_branching', type=str2bool, default='False', help="Branch after GNN layers to predict all targets")
parser.add_argument('--model_name', type=str, default='NNConv', help="GGNN/NNConv/GAT/EGNN")
parser.add_argument('--num_towers', type=int, default=8, help="Number of towers in the model, 0 for no towers")
args = parser.parse_args()

if args.debugging:
    print("Running in debugging mode.")
    args.batch_size = 64
    args.num_epochs = 200
    args.patience = 50
    args.num_layers = 3
    # args.hidden_channels = 200
    args.initial_lr = 1e-3

if 'egnn' in args.model_name.lower():
    args.egnn = True
    args.spatial = False
    args.num_towers = 0
else:
    args.egnn = False

print(args, '\n')

args.train_loader, args.val_loader, args.test_loader = get_data(args)
# get output channels from train_loader
args.output_channels = 1 if not args.predict_all else args.train_loader.dataset[0].y.shape[1]
args.input_channels = args.train_loader.dataset[0].x.shape[1]
args.edge_feat_dim = 5 if args.spatial else 4
print(f"""Input channels: {args.input_channels}, Output channels: {args.output_channels}, 
Edge feature dim: {args.edge_feat_dim}, Hidden channels: {args.hidden_channels}""")


if not args.num_towers:
    if not args.egnn:
        model = models.GNN(args.input_channels, args.hidden_channels,
                                        num_layers=args.num_layers, M=args.M,
                                        nn_width_factor=args.nn_width_factor,
                                        edge_feat_dim=args.edge_feat_dim,
                                        model_name=args.model_name, aggregation=args.aggr,
                                        output_channels=args.output_channels, args=args)
    else:
        model = models.EGNN(args.input_channels, args.hidden_channels,
                            num_layers=args.num_layers,
                            output_channels=args.output_channels)
else:
    model = models.TowerNNConv(args.input_channels, args.hidden_channels,
                            num_layers=args.num_layers, M=args.M,
                            num_towers=args.num_towers,
                            edge_feat_dim=args.edge_feat_dim,
                            nn_width_factor=args.nn_width_factor,
                            model_name=args.model_name, 
                            output_channels=args.output_channels, args=args)

args.model_name = type(model).__name__
# if torch.__version__ == "2.1.0":
#     model = torch.compile(model)
run_experiment(model, args)


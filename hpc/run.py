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
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--num_layers', type=int, default=7, help="number of GNN layers")
parser.add_argument('--hidden_channels', type=int, default=128, help="size of hidden node features")
parser.add_argument('--nn_width_factor', type=int, default=2, help="in NNConv, the width of the nn=h_theta")
parser.add_argument('--M', type=int, default=8, help="s2s processing steps")
parser.add_argument('--initial_lr', type=float, default=5e-4, help="learning rate")
parser.add_argument('--report_interval', type=int, default=10, help="report interval during training")
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--patience', type=int, default=100)
# parser.add_argument('--scheduler', type=str2bool, default='True', help="learning rate scheduler")
parser.add_argument('--aggr', type=str, default='s2s', help="s2s/gru")
parser.add_argument('--target', type=int, default=0, help="target node from 0-11")
parser.add_argument('--predict_all', type=str2bool, default='False', help="Predict all targets?")
parser.add_argument('--use_branching', type=str2bool, default='False', help="Branch after GNN layers to predict all targets")
parser.add_argument('--model_name', type=str, default='NNConv', help="GGNN/NNConv/GAT/PositionalNNConv")
args = parser.parse_args()

if args.debugging:
    print("Running in debugging mode.")
    args.batch_size = 20
    args.num_epochs = 100
    args.patience = 50
    args.num_layers = 2
    args.hidden_channels = 32

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)

# if torch.cuda.is_available():
#     device = "cuda:0"
# else:
#     device = "cpu"
args.output_channels = 12 if args.predict_all else 1
args.input_channels = 11
print(args, '\n')
# args.device = device

args.train_loader, args.val_loader, args.test_loader = get_data(args)
if args.spatial:
    model = models.SpatialGNN(args.input_channels, args.hidden_channels,
                              num_layers=args.num_layers, M=args.M,
                              nn_width_factor=args.nn_width_factor,
                              model_name=args.model_name, aggregation=args.aggr,
                              output_channels=args.output_channels)
else:
    model = models.NonSpatialGNN(args.input_channels, args.hidden_channels,
                                    num_layers=args.num_layers, M=args.M,
                                    nn_width_factor=args.nn_width_factor,
                                    model_name=args.model_name, aggregation=args.aggr,
                                    output_channels=args.output_channels)

args.model_name = type(model).__name__
if torch.__version__ == "2.1.0": # How to use this?
    model = torch.compile(model)
run_experiment(model, args)


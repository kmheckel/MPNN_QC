from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops
import torch
import copy
import os.path as osp

class Complete:
    """
    A transformation that ensures every node in the graph is connected to every other node,
    potentially altering edge attributes to reflect this full connectivity.
    """
    def __call__(self, data):
        data = copy.copy(data)  # Make a shallow copy of the data to avoid modifying the original data.
        device = data.edge_index.device  # Get the device (CPU/GPU) of the edge indices.

        # Create a fully connected edge_index tensor.
        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)  # Stack row and col to form the edge_index tensor.

        # Handle edge attributes, if present.
        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]  # Calculate index for existing edges.
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes  # Size of the new edge_attr tensor.
            edge_attr = data.edge_attr.new_zeros(size)  # Create a zero-filled tensor for new edge attributes.
            edge_attr[idx] = data.edge_attr  # Assign existing edge attributes to their respective positions.

        # Remove self-loops from the fully connected graph.
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr  # Update the data with the new edge attributes.
        data.edge_index = edge_index  # Update the data with the new edge index.
        return data  # Return the transformed data.

def get_data(args, loc_transform):
    """
    Prepares the dataset by applying transformations, standardizing targets, and splitting into train, validation, and test sets.

    Args:
        args: A namespace or dictionary containing arguments for data preparation and model training.
        loc_transform: A transformation to be applied to the location data of the dataset.

    Returns:
        Three DataLoader instances for the training, validation, and test datasets.
    """
    # Determine if GGNN is part of the model name and set transformations accordingly.
    ggnn = 'ggnn' in args.model_name
    if args.spatial:
        transform = T.Compose([loc_transform, Complete(), T.Distance(norm=False)])
    else:
        transform = T.Compose([loc_transform, Complete()])
    
    # Load the QM9 dataset with the specified transformations.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    dataset = QM9(path, transform=transform).shuffle()
    
    # Standardize the targets based on the method specified in args.
    mean = dataset.data.y[20000:].mean(dim=0, keepdim=True)
    if args.standardization == 'std':
        spread = dataset.data.y[20000:].std(dim=0, keepdim=True)
    # MAD is used in the EGNN paper, but other implementations use Z-score as above
    elif args.standardization == 'mad':
        ma = torch.abs(dataset.data.y[20000:] - mean)
        spread = ma.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Standardization method {args.standardization} not recognized.")
    dataset.data.y = (dataset.data.y - mean) / spread
    
    # Set the spread and mean for target normalization.
    args.spread = spread[:, args.target].to(args.device)
    args.mean = mean[:, args.target].to(args.device)
    
    # Split the dataset into training, validation, and test sets based on whether debugging is enabled.
    if not args.debugging:
        test_dataset = dataset[:10000]
        val_dataset = dataset[10000:20000]
        train_dataset = dataset[20000:]
    else:
        test_dataset = dataset[:args.data_split]
        val_dataset = dataset[args.data_split:2*args.data_split]
        train_dataset = dataset[2*args.data_split:3*args.data_split]
    
    # Create DataLoaders for each dataset split.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)
    
    print(f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")
    return train_loader, val_loader, test_loader

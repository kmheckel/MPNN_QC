from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops
import torch
import copy
import os.path as osp


class Complete:
    def __call__(self, data):
        data = copy.copy(data)
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index
        return data

def get_data(args, loc_transform):
    ggnn = 'ggnn' in args.model_name
    if args.spatial:
        transform = T.Compose([loc_transform, Complete(), T.Distance(norm=False)])
    else:
        transform = T.Compose([loc_transform, Complete()])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    dataset = QM9(path, transform=transform).shuffle()
    mean = dataset.data.y[20000:].mean(dim=0, keepdim=True)
    if args.standardization == 'std':
        spread = dataset.data.y[20000:].std(dim=0, keepdim=True)
    elif args.standardization == 'mad':
        ma = torch.abs(dataset.data.y[20000:] - mean)
        spread = ma.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Standardization method {args.standardization} not recognized.")
    # original
    # dataset.data.y = (dataset.data.y - mean) / spread
    args.spread = spread[:, args.target].to(args.device)
    args.mean = mean[:, args.target].to(args.device)
    
    if not args.debugging:
        test_dataset = dataset[:10000]
        val_dataset = dataset[10000:20000]
        train_dataset = dataset[20000:]
    else:
        test_dataset = dataset[:args.data_split]
        val_dataset = dataset[args.data_split:2*args.data_split]
        train_dataset = dataset[2*args.data_split:3*args.data_split]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)
    print(f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")
    return train_loader, val_loader, test_loader

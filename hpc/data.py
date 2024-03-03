from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

def get_data(args):
    dataset = QM9(root='./tmp/QM9')
    # print(dataset.y == dataset._data.y)
    # raise
    if args.predict_all:
        # get first 12 targets
        dataset.data.y = dataset.data.y[:, :12]
        # standardize targets
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        args.std = std
        # print(f"Standard deviation of target: {args.std}")
        # raise
        args.mean = mean
    else:
        # get only the target we want
        dataset.data.y = dataset.data.y[:, args.target]
        # standardize targets
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        args.std = std.item()
        args.mean = mean.item()
    
    if not args.debugging:
        test_dataset = dataset[:10000]
        val_dataset = dataset[10000:20000]
        train_dataset = dataset[20000:]
    else:
        test_dataset = dataset[:500]
        val_dataset = dataset[500:1000]
        train_dataset = dataset[1000:1500]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")
    return train_loader, val_loader, test_loader

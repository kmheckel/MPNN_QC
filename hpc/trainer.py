import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import pickle

def train(model, args):
    model.train()
    loss_all = 0

    for data in args.train_loader:
        data = data.to(args.device)
        args.optimizer.zero_grad()
        y_pred = model(data)
        loss = F.mse_loss(y_pred, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        args.optimizer.step()
    return loss_all / len(args.train_loader.dataset)

def eval(model, loader, args):
    model.eval()
    error = 0
    for data in loader:
        data = data.to(args.device)
        with torch.no_grad():
            y_pred = model(data)
            error += (y_pred * args.std - data.y * args.std).abs().sum().item()
    return error / len(loader.dataset) / args.output_channels

def run_experiment(model, args):
    print(f"Running experiment for {args.model_name}, training on {len(args.train_loader.dataset)} samples for {args.num_epochs} epochs.")

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {args.device}.')

    print("\nModel architecture:")
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    model = model.to(args.device)

    args.optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr,
                                      weight_decay=1e-16)

    # cosine learning rate annealing
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.optimizer, args.num_epochs,
    #                                                        eta_min=args.initial_lr * 0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(args.optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.00001)

    print("\nStart training:")
    best_val_error = float('inf')
    perf_per_epoch = []
    t = time.time()
    patience_counter = 0
    for epoch in range(1, args.num_epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(model, args)
        val_error = eval(model, args.val_loader, args)

        if val_error < best_val_error:
            best_val_error = val_error
            test_error = eval(model, args.test_loader, args)  # Evaluate model on test set if validation metric improves
            patience_counter = 0  # Reset patience counter

            current_date = datetime.now().strftime('%Y-%m-%d_%H-%M')
            model_save_path = f'models/{current_date}/{args.model_name}_{model.num_layers}layers.pt'
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        # Print performance every 10 epochs
        if epoch % args.report_interval == 0:
            print(f'Epoch: {epoch:03d}, LR: {lr:.7f}, Loss: {loss:.7f}, Val Error: {val_error:.7f}, Test Error: {test_error:.7f}')

        scheduler.step(val_error)
        perf_per_epoch.append((test_error, val_error, epoch, args.model_name))

        if patience_counter >= args.patience:
            print(f"Stopping early due to no improvement in validation error for {args.patience} epochs.")
            break

    t = time.time() - t
    train_time = t / 60
    print(f"\nDone! Training took {train_time:.2f} minutes. Best validation error: {best_val_error:.7f}, corresponding test error: {test_error:.7f}.")
    print(f"Best model saved to: {model_save_path}")
    # create logs directory if it doesn't exist
    os.makedirs(f'results/{current_date}', exist_ok=True)
    # save everything to a pickle file
    with open(f'results/{current_date}/{args.model_name}_{model.num_layers}layers.pkl', 'wb') as f:
        pickle.dump((best_val_error, test_error, train_time, perf_per_epoch), f)
    print(f"Training data saved to: results/{current_date}/{args.model_name}_{model.num_layers}layers.pkl")

    return best_val_error, test_error, train_time, perf_per_epoch
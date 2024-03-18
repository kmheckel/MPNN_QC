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
        # paper's original loss is MSE!
        # loss = F.mse_loss(y_pred, data.y)
        # if using the EGNN direction
        loss = F.l1_loss(y_pred, (data.y - args.mean) / args.spread)
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
            # error += (y_pred * args.spread - data.y * args.spread).abs().sum().item()
            # if using the EGNN direction
            error += (y_pred * args.spread + args.mean - data.y).abs().sum().item()
    return error / len(loader.dataset) / args.output_channels

def eval_multi_pred(model, loader, args):
    model.eval()
    error = torch.zeros(args.output_channels).to(args.device)
    for data in loader:
        data = data.to(args.device)
        with torch.no_grad():
            y_pred = model(data)
            # error += (y_pred * args.spread - data.y * args.spread).abs().sum(axis=0)
            # if using the EGNN direction
            error += (y_pred * args.spread + args.mean - data.y).abs().sum(axis=0)
    return error / len(loader.dataset)

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
    if not args.debugging:
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_save_path = f'models/{current_date}/{args.model_name}_'
        results_save_path = f'results/{current_date}_{args.model_name}.pkl'
        os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
        print(f"Training data being saved to: {results_save_path}")
        print(f"Model being saved to: models/{current_date}/")

    args.optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
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

        val_improvement = val_error < best_val_error
        if val_improvement:
            best_val_error = val_error
            test_error = eval(model, args.test_loader, args)  # Evaluate model on test set if validation metric improves
            patience_counter = 0  # Reset patience counter
            if not args.debugging:
                save_path = model_save_path+f'epoch_{epoch}.pt'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % args.report_interval == 0:
            print(f'Epoch: {epoch:03d}, LR: {lr:.7f}, Loss: {loss:.7f}, Val Error: {val_error:.7f}, Test Error: {test_error:.7f}')

        scheduler.step(val_error)
        perf_per_epoch.append((test_error, val_error, epoch, args.model_name))
        if val_improvement and not args.debugging:
            with open(results_save_path, 'wb') as f:
                pickle.dump((best_val_error, test_error, perf_per_epoch), f)

        if patience_counter >= args.patience:
            print(f"Stopping early due to no improvement in validation error for {args.patience} epochs.")
            break

    if args.predict_all and not args.debugging:
        model.load_state_dict(torch.load(save_path))
        multi_test_error = eval_multi_pred(model, args.test_loader, args)
        print(f"Final test MAE per target for the best model: {multi_test_error.tolist()}")

    t = time.time() - t
    train_time = t / 60
    print(f"\nDone! Training took {train_time:.2f} minutes. Best validation error: {best_val_error:.7f}, corresponding test error: {test_error:.7f}.")
    if not args.debugging:
        print(f"Best model saved to: {save_path}")
        with open(results_save_path, 'wb') as f:
            pickle.dump((best_val_error, test_error, train_time, perf_per_epoch), f)
        print(f"Training data saved to: {results_save_path}")
    return best_val_error, test_error, train_time, perf_per_epoch
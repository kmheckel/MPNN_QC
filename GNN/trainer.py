import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import pickle

def train(model, args):
    """
    Trains the model on the provided dataset.

    Args:
        model: The neural network model to be trained.
        args: A namespace or dictionary containing training parameters and data loaders.

    Returns:
        The average loss over all training data.
    """
    model.train()  # Set the model to training mode.
    loss_all = 0  # Initialize total loss.
    for data in args.train_loader:  # Loop through the training data.
        data = data.to(args.device)  # Move data to the specified device (CPU or GPU).
        args.optimizer.zero_grad()  # Zero the gradients to prevent accumulation.
        y_pred = model(data)  # Forward pass: compute the predicted values.
        loss = F.mse_loss(y_pred, data.y)  # Calculate the mean squared error loss.
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters.
        loss_all += loss.item() * data.num_graphs  # Update total loss.
        args.optimizer.step()  # Perform a single optimization step.
    return loss_all / len(args.train_loader.dataset)  # Return average loss.

def eval(model, loader, args):
    """
    Evaluates the model's performance on a given dataset.

    Args:
        model: The neural network model to be evaluated.
        loader: The data loader for the dataset to evaluate.
        args: A namespace or dictionary containing evaluation parameters.

    Returns:
        The average absolute error over all evaluation data.
    """
    model.eval()  # Set the model to evaluation mode.
    error = 0  # Initialize error.
    for data in loader:  # Loop through the data.
        data = data.to(args.device)  # Move data to the specified device.
        with torch.no_grad():  # Disable gradient calculation.
            y_pred = model(data)  # Forward pass: compute the predicted values.
            error += (y_pred * args.spread - data.y * args.spread).abs().sum().item()  # Calculate and update total error.
    return error / len(loader.dataset) / args.output_channels  # Return average error per output channel.

def eval_multi_pred(model, loader, args):
    """
    Evaluates the model's performance on a given dataset, returning errors for multiple predictions.

    Args:
        model: The neural network model to be evaluated.
        loader: The data loader for the dataset to evaluate.
        args: A namespace or dictionary containing evaluation parameters.

    Returns:
        A tensor of errors for each output channel.
    """
    model.eval()  # Set the model to evaluation mode.
    error = torch.zeros(args.output_channels).to(args.device)  # Initialize errors tensor.
    for data in loader:  # Loop through the data.
        data = data.to(args.device)  # Move data to the specified device.
        with torch.no_grad():  # Disable gradient calculation.
            y_pred = model(data)  # Forward pass: compute the predicted values.
            error += (y_pred * args.spread - data.y * args.spread).abs().sum(axis=0)  # Calculate and update errors for each output channel.
    return error / len(loader.dataset)  # Return average errors for each channel.

def run_experiment(model, args):
    """
    Runs the entire training and evaluation experiment, including setup, training, evaluation,
    and saving model and results.

    Args:
        model: The neural network model to be trained and evaluated.
        args: A namespace or dictionary containing all necessary parameters and data loaders.

    Returns:
        A tuple containing the best validation error, corresponding test error, total training time, and performance per epoch.
    """
    print(f"Running experiment for {args.model_name}, training on {len(args.train_loader.dataset)} samples for {args.num_epochs} epochs.")
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device for training.
    print(f'Training on {args.device}.')
    print("\nModel architecture:")
    print(model)
    total_param = 0  # Initialize parameter counter.
    for param in model.parameters():  # Loop through model parameters.
        total_param += np.prod(list(param.data.size()))  # Calculate total number of parameters.
    print(f'Total parameters: {total_param}')
    model = model.to(args.device)  # Move model to the specified device.
    if not args.debugging:  # Check if not in debugging mode to save models and results.
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Get current date and time.
        model_save_path = f'models/{current_date}/{args.model_name}_'  # Set path for saving model.
        results_save_path = f'results/{current_date}_{args.model_name}.pkl'  # Set path for saving results.
        os.makedirs(os.path.dirname(results_save_path), exist_ok=True)  # Ensure the directory exists.
        print(f"Training data being saved to: {results_save_path}")
        print(f"Model being saved to: models/{current_date}/")

    # Setup optimizer and learning rate scheduler.
    args.optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(args.optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.00001)
    print("\nStart training:")
    best_val_error = float('inf')  # Initialize best validation error.
    perf_per_epoch = []  # Initialize list to store performance per epoch.
    t = time.time()  # Start timing the experiment.
    patience_counter = 0  # Initialize patience counter.
    for epoch in range(1, args.num_epochs + 1):  # Main training loop.
        lr = scheduler.optimizer.param_groups[0]['lr']  # Get current learning rate.
        loss = train(model, args)  # Train the model.
        val_error = eval(model, args.val_loader, args)  # Evaluate on validation set.
        val_improvement = val_error < best_val_error  # Check for improvement.
        if val_improvement:
            best_val_error = val_error  # Update best validation error.
            test_error = eval(model, args.test_loader, args)  # Evaluate on test set.
            patience_counter = 0  # Reset patience counter.
            if not args.debugging:  # Save model if not debugging.
                save_path = model_save_path+f'epoch_{epoch}.pt'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1  # Increment patience counter.

        if epoch % args.report_interval == 0:  # Report progress periodically.
            print(f'Epoch: {epoch:03d}, LR: {lr:.7f}, Loss: {loss:.7f}, Val Error: {val_error:.7f}, Test Error: {test_error:.7f}')

        scheduler.step(val_error)  # Adjust learning rate based on validation error.
        perf_per_epoch.append((test_error, val_error, epoch, args.model_name))  # Store performance per epoch.
        if val_improvement and not args.debugging:  # Save results if not debugging.
            with open(results_save_path, 'wb') as f:
                pickle.dump((best_val_error, test_error, perf_per_epoch), f)

        if patience_counter >= args.patience:  # Check for early stopping.
            print(f"Stopping early due to no improvement in validation error for {args.patience} epochs.")
            break

    if args.predict_all and not args.debugging:  # Perform final evaluation on all predictions if specified.
        model.load_state_dict(torch.load(save_path))
        multi_test_error = eval_multi_pred(model, args.test_loader, args)
        print(f"Final test MAE per target for the best model: {multi_test_error.tolist()}")

    t = time.time() - t  # Calculate total training time.
    train_time = t / 60  # Convert to minutes.
    print(f"\nDone! Training took {train_time:.2f} minutes. Best validation error: {best_val_error:.7f}, corresponding test error: {test_error:.7f}.")
    if not args.debugging:  # Save final results if not debugging.
        print(f"Best model saved to: {save_path}")
        with open(results_save_path, 'wb') as f:
            pickle.dump((best_val_error, test_error, train_time, perf_per_epoch), f)
        print(f"Training data saved to: {results_save_path}")
    return best_val_error, test_error, train_time, perf_per_epoch

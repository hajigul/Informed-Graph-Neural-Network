# train.py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


###########################################################################################################

def calculate_ranking_metrics(model, loader):  # Function to calculate ranking metrics for model evaluation
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    rankings, labels = [], []  # Initialize lists to store predicted rankings and true labels

    with torch.no_grad():  # Disable gradient calculation for faster evaluation
        for batch in loader:  # Iterate over batches from the loader
            out = model(batch)  # Run the model to get predictions
            tail_indices = batch.edge_index[1]  # Extract the tail indices (target entities)
            scores = out[tail_indices].cpu().numpy()  # Get prediction scores for tail entities and move to CPU
            labels.extend(batch.edge_attr.cpu().numpy())  # Extract true labels (relations) and move to CPU

            for score, label in zip(scores, labels):  # Iterate through scores and labels
                rankings.append(score)  # Append each score array for metric calculation

    rankings, labels = np.array(rankings), np.array(labels)  # Convert lists to NumPy arrays for metric calculations
    mrr = np.mean([1.0 / (np.argsort(score)[::-1].tolist().index(label) + 1) for score, label in zip(rankings, labels)])  # Calculate Mean Reciprocal Rank (MRR)
    hits_at_1 = np.mean([label in np.argsort(score)[::-1][:1] for score, label in zip(rankings, labels)])  # Calculate Hits@1
    hits_at_3 = np.mean([label in np.argsort(score)[::-1][:3] for score, label in zip(rankings, labels)])  # Calculate Hits@3
    hits_at_5 = np.mean([label in np.argsort(score)[::-1][:5] for score, label in zip(rankings, labels)])  # Calculate Hits@5
    hits_at_10 = np.mean([label in np.argsort(score)[::-1][:10] for score, label in zip(rankings, labels)])  # Calculate Hits@10

    return mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10  # Return calculated metrics

###########################################################################################################

def train(model, train_loader, valid_loader, test_loader, epochs, lr, checkpoint_path, loss_file_path, test_results_file_path):  # Function to train the model
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Define the optimizer using Adam with specified learning rate and weight decay
    start_epoch = 1  # Initialize the starting epoch

    if os.path.exists(checkpoint_path):  # Check if a checkpoint file exists and load it
        checkpoint = torch.load(checkpoint_path)  # Load checkpoint data
        model.load_state_dict(checkpoint['model_state_dict'])  # Load model state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        start_epoch = checkpoint['epoch'] + 1  # Set the starting epoch to the next one
        print(f"Resuming training from epoch {start_epoch}")  # Print status

    with open(loss_file_path, "a") as loss_file, open(test_results_file_path, "a") as test_results_file:  # Open files for logging training and test results
        for epoch in range(start_epoch, epochs + 1):  # Loop through each epoch
            model.train()  # Set the model to training mode
            total_train_loss = 0  # Initialize total training loss

            for batch in train_loader:  # Iterate over each batch in the training loader
                optimizer.zero_grad()  # Zero the parameter gradients
                out = model(batch)  # Forward pass to get predictions
                tail_indices = batch.edge_index[1]  # Extract indices of the tail entities
                edge_pred = out[tail_indices]  # Get predictions for the tail entities
                target = batch.edge_attr  # Set the target labels (relations)

                min_size = min(edge_pred.size(0), target.size(0))  # Ensure edge_pred and target have the same size
                edge_pred, target = edge_pred[:min_size], target[:min_size]  # Adjust size if needed

                loss = F.cross_entropy(edge_pred, target)  # Calculate cross-entropy loss
                loss.backward()  # Backpropagate the loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to avoid explosion
                optimizer.step()  # Update model weights
                total_train_loss += loss.item()  # Accumulate training loss

            avg_train_loss = total_train_loss / len(train_loader)  # Calculate average training loss for the epoch

            model.eval()  # Set the model to evaluation mode for validation phase
            total_valid_loss = 0  # Initialize total validation loss

            with torch.no_grad():  # Disable gradient calculation for validation
                for batch in valid_loader:  # Iterate over each batch in the validation loader
                    out = model(batch)  # Forward pass
                    tail_indices = batch.edge_index[1]  # Extract tail entity indices
                    edge_pred = out[tail_indices]  # Get predictions for tail entities
                    target = batch.edge_attr  # Set the target labels

                    if edge_pred.size(0) != target.size(0):  # Ensure edge_pred and target have the same size
                        min_size = min(edge_pred.size(0), target.size(0))  # Adjust size if needed
                        edge_pred, target = edge_pred[:min_size], target[:min_size]  # Adjust edge_pred and target size

                    valid_loss = F.cross_entropy(edge_pred, target)  # Calculate cross-entropy loss
                    total_valid_loss += valid_loss.item()  # Accumulate validation loss

            avg_valid_loss = total_valid_loss / len(valid_loader)  # Calculate average validation loss for the epoch

            mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10 = calculate_ranking_metrics(model, test_loader)  # Calculate test metrics using the test loader

            loss_file.write(f"Epoch {epoch}, Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_valid_loss:.2f}\n")  # Log training and validation losses
            test_results_file.write(  # Log test metrics
                f"Epoch {epoch}, MRR: {mrr:.2f}, Hits@1: {hits_at_1:.2f}, Hits@3: {hits_at_3:.2f}, Hits@5: {hits_at_5:.2f}, Hits@10: {hits_at_10:.2f}\n"
            )

            print(f"Epoch {epoch}/{epochs}, Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_valid_loss:.2f}")  # Print training and validation loss
            print(f"Test Metrics - MRR: {mrr:.2f}, Hits@1: {hits_at_1:.2f}, Hits@3: {hits_at_3:.2f}, Hits@5: {hits_at_5:.2f}, Hits@10: {hits_at_10:.2f}")  # Print test metrics

            torch.save({  # Save a checkpoint after each epoch
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Save model state
                'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")  # Print checkpoint status

###########################################################################################################


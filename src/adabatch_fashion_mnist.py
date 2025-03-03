import torch
import csv 
import os
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from model import ResNet14, BasicBlock  # assuming your model is in model.py

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper function to save results to a CSV file
def save_results_to_csv(static_results, adabatch_results, epochs, folder="results", filename=None):
    # Generate a default filename with timestamp if none provided
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_results_{timestamp}.csv"

    # Combine filename and folder
    filename = os.path.join(folder, filename)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['epoch', 'method', 'batch_size', 'learning_rate', 'train_loss',
                         'test_accuracy', 'epoch_time'])
        
        # Estimate learning rates for static training
        static_lr = [0.1 * (0.25 ** (epoch // 20)) for epoch in range(epochs)]
        
        # Write static results
        for epoch in range(epochs):
            writer.writerow([
                epoch + 1,
                'static',
                static_results['batch_sizes'][epoch],
                static_lr[epoch],
                static_results['train_losses'][epoch],
                static_results['test_accuracies'][epoch],
                static_results['training_times'][epoch]
            ])
            
        # Write AdaBatch results
        # Note: For AdaBatch, you'll need to track learning rates during training
        for epoch in range(epochs):
            writer.writerow([
                epoch + 1,
                'adabatch',
                adabatch_results['batch_sizes'][epoch],
                adabatch_results.get('learning_rates', [0] * epochs)[epoch],  # Add this to your result dictionary
                adabatch_results['train_losses'][epoch],
                adabatch_results['test_accuracies'][epoch],
                adabatch_results['training_times'][epoch]
            ])
    
    print(f"Results saved to {filename}")
    return filename


# Data loading and preprocessing
def load_fashion_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dataset, test_dataset, train_loader, test_loader


# Train with static batch size, this is the training function to compare with AdaBatch
# Note: Momentum=0.9 and weight decay=5e-4 are the default values used in the original AdaBatch paper
# Learning rate should be 0.1 to match the paper
def train_static(model, train_loader, test_loader, batch_size, epochs, lr, momentum=0.9, weight_decay=5e-4):
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    # From section 4.2 of paper: Base learning rate is α = 0.1, and is decay 
    # by a factor of 0.25 every 20 epochs.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 40, 60, 80],
        gamma=0.25)

    train_losses = []
    test_accuracies = []
    batch_sizes = [batch_size] * epochs
    training_times = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        # Inputs are the images, targets are the labels
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            pred = model(inputs)
            loss = criterion(pred, targets)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        epoch_time = time.time() - start_time
        training_times.append(epoch_time)

        # Evaluate model for this epoch
        model.eval()
        # Counters to track accuracy
        correct = 0
        total = 0
        # No need to compute gradients during evaluation, this saves memory
        with torch.no_grad():
            # Iterate over the test data for validation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # Compute the models predictions
                outputs = model(inputs)
                # outputs.max(1) returns the maximum value and the predicted class index
                _, predicted = outputs.max(1)
                total += targets.size(0) # Increment total by the number of labels in the batch
                # Add all the correct predictions to the counter
                correct += predicted.eq(targets).sum().item()

        # Epoch statistics
        train_loss = running_loss / len(train_loader)
        test_accuracy = 100. * correct / total

        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch+1}/{epochs} | Batch Size: {batch_size} | '
              f'Loss: {train_loss:.4f} | Test Acc: {test_accuracy:.2f}% | '
              f'Time: {epoch_time:.2f}s')

        # Update the learning rate scheduler
        scheduler.step()

    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'batch_sizes': batch_sizes,
        'training_times': training_times
    }

# Train with AdaBatch
# Note: Batch_increase_factor=2 and batch_increase_interval=20 are the 
# default values used in the original AdaBatch paper
# init_lr (learning rate) should be 0.1 to match the paper.
def train_adabatch(model, train_dataset, test_dataset, init_batch_size, epochs,
                   init_lr, batch_increase_interval=20, batch_increase_factor=2,
                   momentum=0.9, weight_decay=5e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
    
    train_losses = []
    test_accuracies = []
    batch_sizes = []
    training_times = []
    learning_rates = []
    
    current_batch_size = init_batch_size
    current_lr = init_lr
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, 
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=current_batch_size, 
                             shuffle=False, num_workers=2)
    
    # Add linear warmup schedule for first 5 epochs
    warmup_epochs = 5
    if warmup_epochs > 0: 
        target_lr_after_warmup = init_lr * (init_batch_size / 128) # 128 is baseline batch size
        # Find step is of warmup
        warmup_lr_step = (target_lr_after_warmup - init_lr) / warmup_epochs
        current_lr = init_lr

    for epoch in range(epochs):
        # Apply warmup schedule for first 5 epochs, as per AdaBatch paper
        if epoch < warmup_epochs:
            current_lr += warmup_lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr 
            print(f"Epoch {epoch+1}: Warmup LR: {current_lr:.6f}")
        # check if we need to increase batch size, should happen every batch_increase_interval epochs
        if epoch > 0 and epoch % batch_increase_interval == 0:
            old_batch_size = current_batch_size
            current_batch_size = min(current_batch_size * batch_increase_factor, 8192)  # Cap at a reasonable size
            
            # Scale learning rate to maintain α/r ratio
            # In AdaBatch, when batch size r increases by factor β, 
            # learning rate α is scaled by factor α̃ = α/β
            current_lr = current_lr * 0.5 # Hardcoded β = 2 for now

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Create new data loader with updated batch size
            train_loader = DataLoader(train_dataset, batch_size=current_batch_size, 
                                     shuffle=True, num_workers=2)
            
            print(f"Epoch {epoch+1}: Increased batch size to {current_batch_size}, "
                  f"adjusted lr to {current_lr:.6f}")
        
        batch_sizes.append(current_batch_size)
        learning_rates.append(current_lr)
        
        # Training phase
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_time = time.time() - start_time
        training_times.append(epoch_time)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss / len(train_loader)
        test_accuracy = 100. * correct / total
        
        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} | Batch Size: {current_batch_size} | '
              f'LR: {current_lr:.6f} | Loss: {train_loss:.4f} | '
              f'Test Acc: {test_accuracy:.2f}% | Time: {epoch_time:.2f}s')
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'batch_sizes': batch_sizes,
        'training_times': training_times
    }

# Visualization function
def plot_comparison(static_results, adabatch_results, epochs):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    axs[0, 0].plot(range(1, epochs+1), static_results['train_losses'], label='Static Batch Size')
    axs[0, 0].plot(range(1, epochs+1), adabatch_results['train_losses'], label='AdaBatch')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Training Loss')
    axs[0, 0].set_title('Training Loss Comparison')
    axs[0, 0].legend()
    
    # Plot test accuracy
    axs[0, 1].plot(range(1, epochs+1), static_results['test_accuracies'], label='Static Batch Size')
    axs[0, 1].plot(range(1, epochs+1), adabatch_results['test_accuracies'], label='AdaBatch')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Test Accuracy (%)')
    axs[0, 1].set_title('Test Accuracy Comparison')
    axs[0, 1].legend()
    
    # Plot batch size changes for AdaBatch
    axs[1, 0].plot(range(1, epochs+1), adabatch_results['batch_sizes'])
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Batch Size')
    axs[1, 0].set_title('AdaBatch - Batch Size Evolution')
    
    # Plot training time per epoch
    axs[1, 1].plot(range(1, epochs+1), static_results['training_times'], label='Static Batch Size')
    axs[1, 1].plot(range(1, epochs+1), adabatch_results['training_times'], label='AdaBatch')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Training Time (s)')
    axs[1, 1].set_title('Training Time per Epoch')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('adabatch_vs_static.png')
    plt.show()

# Main execution
def main():
    # Setup
    # Note: These are the default values used in the original AdaBatch paper
    epochs = 100
    static_batch_size = 128
    adabatch_init_batch = 128
    learning_rate = 0.1
    
    # Load data
    train_dataset, test_dataset, static_train_loader, static_test_loader = load_fashion_mnist(static_batch_size)
    
    # Static batch size training
    static_model = ResNet14().to(device)
    print("Training with static batch size...")
    static_results = train_static(static_model, static_train_loader, static_test_loader,
                                static_batch_size, epochs, learning_rate)
    
    # AdaBatch training
    adabatch_model = ResNet14().to(device)
    print("\nTraining with AdaBatch...")
    adabatch_results = train_adabatch(adabatch_model, train_dataset, test_dataset, 
                                     adabatch_init_batch, epochs, learning_rate)
    
    # Visualize results

    plot_comparison(static_results, adabatch_results, epochs)
    csv_file = save_results_to_csv(static_results, adabatch_results, epochs)
    # Print overall speedup

    # Save models for future use
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name_static = f"results/static_model_{timestamp}.pth"
    file_name_adabatch = f"results/adabatch_model_{timestamp}.pth"
    torch.save(adabatch_model.state_dict(), file_name_adabatch)
    torch.save(static_model.state_dict(), file_name_static)

    static_total_time = sum(static_results['training_times'])
    adabatch_total_time = sum(adabatch_results['training_times'])
    speedup = static_total_time / adabatch_total_time
    
    print(f"\nTotal training time (Static): {static_total_time:.2f}s")
    print(f"Total training time (AdaBatch): {adabatch_total_time:.2f}s")
    print(f"Overall speedup: {speedup:.2f}x")
    
    # Print final accuracy
    print(f"Final test accuracy (Static): {static_results['test_accuracies'][-1]:.2f}%")
    print(f"Final test accuracy (AdaBatch): {adabatch_results['test_accuracies'][-1]:.2f}%")
    print(f"Accuracy difference: {adabatch_results['test_accuracies'][-1] - static_results['test_accuracies'][-1]:.2f}%")

if __name__ == "__main__":
    print(f"Using device: {device}")
    main()
    # Note: The paper says that speedups on a single GPU are modest, higher batch sizes up to 4096 are best
    # sutied for multi-GPU training.
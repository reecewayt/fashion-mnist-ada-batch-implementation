"""
Fashion MNIST dataset exploration which I used to get familiar with the dataset and PyTorch's DataLoader.
The dataset is downloaded from torchvision.datasets.FashionMNIST and transformed to tensors. This script
demonstrates how to load the Fashion MNIST dataset, explore its properties.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Define transforms for the training and test sets
# Transforms are maniuplations that are applied to the data before it is fed to the model
# Here we are converting the images to tensors and normalizing the pixel values
# Source: https://pytorch.org/vision/stable/transforms.html
transform = transforms.Compose([ # Composes several transforms together, i'm just using tesnor transform for now
    # Trensors are specialized data structures that are very similar to NumPy arrays.
    # They can be used on GPUs to accelerate computing.
    transforms.ToTensor()  # Tensor converts (PIL) images to tensors and scales pixel intensities to the range [0, 1]
])

# Download of training and test datasets
# We use transform to convert PIL images (Python Image Library) to tensors and normalize the pixel values; 
# the raw images of the MNIST dataset are image objects of 28x28 grayscale so we need to convert them to tensors
# and normalize the pixel values.
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Print dataset information first
print("------------Dataset Information------------")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes}")

# Define the class names for the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Count examples per class
class_counts = {i: 0 for i in range(len(class_names))}
for _, label in train_dataset:
    class_counts[label] += 1

print("\nExamples per class in training dataset:")
# Create a table-like format with consistent width
print("-" * 40)
print(f"{'Class':<20} | {'Count':>15}")
print("-" * 40)
for i, count in class_counts.items():
    print(f"{class_names[i]:<20} | {count:>15}")
print("-" * 40)

# Example of a static batch size
"""
Here we are using a batch size of 64. The data loader is an iterable object, and each element in the iterator
will return a batch of 64 features and labels. 
Source: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Display batch information
for batch_images, batch_labels in test_loader:
    print(f"\n------------Batch Information------------")
    print(f"Batch Images Tensor Shape:")
    print(f"  • Batch Size: {batch_images.shape[0]} images")
    print(f"  • Channels:   {batch_images.shape[1]} (grayscale)")
    print(f"  • Height:     {batch_images.shape[2]} pixels")
    print(f"  • Width:      {batch_images.shape[3]} pixels")
    print(f"\nBatch Labels Tensor:")
    print(f"  • Shape:      {batch_labels.shape} (one label per image)")
    print(f"  • Data Type:  {batch_labels.dtype} (integer class indices)")
    
    # Print some sample labels with their class names
    print(f"\nFirst 5 labels in this batch:")
    for i in range(5):
        print(f"  • Image {i+1}: Class {batch_labels[i].item()} ({class_names[batch_labels[i].item()]})")
    
    # Print statistics about pixel values
    print(f"\nPixel Value Statistics:")
    print(f"  • Min: {batch_images.min().item():.2f}")
    print(f"  • Max: {batch_images.max().item():.2f}")
    print(f"  • Mean: {batch_images.mean().item():.2f}")
    
    break  # Only process the first batch

# Get a batch of images for visualization
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# Show a batch of images
print("\n------------Image Visualization------------")
plt.figure(figsize=(12, 6))
imshow(torchvision.utils.make_grid(images[:16]))
plt.title('Fashion MNIST Sample Images')
plt.savefig('./results/fashion_mnist_samples.png')
print("Sample images saved to ./results/fashion_mnist_samples.png")

# Show sample image shape
print(f"Sample image shape: {images[0].shape}")
print(f"Min pixel value: {images.min()}")
print(f"Max pixel value: {images.max()}")

# Calculate mean and std of the training dataset for proper normalization
print("\n------------Dataset Statistics------------")
def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

print("Calculating mean and std of the training dataset...")
mean, std = get_mean_std(train_loader)
print(f"Dataset mean: {mean}")
print(f"Dataset std: {std}")
print("These values can be used for normalizing the dataset in your model training.")
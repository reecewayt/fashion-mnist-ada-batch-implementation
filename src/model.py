"""
This model uses the ResNet architecture but leverages PyTorch's ability to customize the model for the Fashion MNIST dataset.
The model is trained on the Fashion MNIST dataset, which is a dataset of 28x28 grayscale images of 
10 different classes of clothing items.

Sources: 
  - https://www.run.ai/guides/deep-learning-for-computer-vision/pytorch-resnet
  - https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
  - https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
  - https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
  - https://github.com/JiahongChen/resnet-pytorch
  - Claude.AI as a code assist tool
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

"""
BasicBlock: A basic block for the ResNet model
  - Consists of two convolutional layers with batch normalization
  - Plus a shortcut connection
"""
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    ### Define forward pass, or how the data will pass through the network
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16

        # Initial convolution layer, this sets input channels to 1 for grayscale images
        # The output channels are set to 16 to reduce number of filters in the first layer
        # The original ResNet paper uses 64 filters in the first layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64, num_classes)  # 10 classes in Fashion MNIST

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)

        # Dynamically determine the input size for the linear layer
        if self.linear is None:
            print(f"Shape before flattening: {out.shape}")
            flat_features = out.size(1) * out.size(2) * out.size(3)
            self.linear = nn.Linear(flat_features, 10)  # 10 classes in Fashion MNIST
            self.linear = self.linear.to(device)
            print(f"Linear layer initialized with input size: {flat_features}")

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


"""
ResNet-14 model adapted for Fashion MNIST
 - 1 input channel (grayscale images)
 - 10 output classes (clothing categories)
 - 1 initial convolutional layer
 - 6 basic blocks (each block has 2 convolutional layers + shortcut connection)
 - 1 linear layer for classification
 - Total 14 layers

"""
def ResNet14():
    # Similar to CIFAR ResNet-20 but adapted for Fashion MNIST
    # Each layer group has 2 basic blocks, resulting in 14 layers total
    return ResNet(BasicBlock, [2, 2, 2])

# Small test to check that the model is defined properly
if __name__ == '__main__':
    model = ResNet14().to(device)
    print(model)
    
    # Move input tensor to the correct device
    input_tensor = torch.randn(1, 1, 28, 28).to(device)  # Fashion MNIST image size
    
    # Run a forward pass to properly initialize the model
    output = model(input_tensor)
    print(f"Model output shape: {output.shape}")
   
    # Print the updated model after the linear layer is initialized
    print("\nModel after initialization:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

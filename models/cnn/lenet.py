"""
LeNet-5 Implementation for MNIST

Classic CNN architecture for digit classification.
Simple enough for quick benchmarking but representative of convolutional layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 architecture (LeCun et al., 1998)
    
    Architecture:
        Conv(1->6, 5x5) -> ReLU -> MaxPool(2x2) ->
        Conv(6->16, 5x5) -> ReLU -> MaxPool(2x2) ->
        Flatten -> FC(256->120) -> ReLU ->
        FC(120->84) -> ReLU -> FC(84->10)
    
    Total parameters: ~60K
    """
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        """
        Initialize LeNet-5.
        
        Args:
            num_classes: Number of output classes (default: 10 for MNIST)
            in_channels: Number of input channels (default: 1 for grayscale)
        """
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Conv layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x


def create_lenet(num_classes: int = 10, in_channels: int = 1, pretrained: bool = False) -> LeNet5:
    """
    Create a LeNet-5 model.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to load pretrained weights (not implemented)
        
    Returns:
        LeNet5 model
    """
    model = LeNet5(num_classes=num_classes, in_channels=in_channels)
    
    if pretrained:
        raise NotImplementedError("Pretrained weights not available for LeNet-5")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_lenet()
    print(f"Model: {model.__class__.__name__}")
    
    from models.utils import get_model_info
    info = get_model_info(model)
    print(f"Parameters: {info['total_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    
    # Test forward pass
    x = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

"""
ResNet-18 Implementation for CIFAR-10

Modern CNN architecture with residual connections.
Tests operator fusion and memory optimization in different compilation strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List


class BasicBlock(nn.Module):
    """
    Basic ResNet block with two 3x3 convolutions and a skip connection.
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initialize basic block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution (default: 1)
        """
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture for CIFAR-10.
    
    Modified from original ResNet to work with 32x32 images.
    """
    
    def __init__(self, block: Type[BasicBlock], num_blocks: List[int], 
                 num_classes: int = 10, in_channels: int = 3):
        """
        Initialize ResNet.
        
        Args:
            block: Block type (BasicBlock)
            num_blocks: Number of blocks in each layer
            num_classes: Number of output classes (default: 10 for CIFAR-10)
            in_channels: Number of input channels (default: 3 for RGB)
        """
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution (smaller kernel for CIFAR-10)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final fully connected layer
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block: Type[BasicBlock], out_channels: int, 
                    num_blocks: int, stride: int) -> nn.Sequential:
        """Create a ResNet layer with multiple blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


def ResNet18(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """
    Create ResNet-18 model.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        
    Returns:
        ResNet-18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, 
                  in_channels=in_channels)


def create_resnet18(num_classes: int = 10, in_channels: int = 3, 
                    pretrained: bool = False) -> ResNet:
    """
    Create a ResNet-18 model.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to load pretrained weights (not implemented)
        
    Returns:
        ResNet-18 model
    """
    model = ResNet18(num_classes=num_classes, in_channels=in_channels)
    
    if pretrained:
        raise NotImplementedError("Pretrained weights not available for CIFAR-10 ResNet-18")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_resnet18()
    print(f"Model: {model.__class__.__name__}")
    
    import sys
    sys.path.append('../..')
    from models.utils import get_model_info
    info = get_model_info(model)
    print(f"Parameters: {info['total_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

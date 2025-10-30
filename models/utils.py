"""
Utility functions shared across model implementations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model information
    """
    total_params = count_parameters(model)
    
    info = {
        "total_parameters": total_params,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "non_trainable_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "model_size_mb": total_params * 4 / (1024 ** 2),  # Assuming float32
    }
    
    return info


def save_checkpoint(model: nn.Module, optimizer: Any, epoch: int, loss: float, filepath: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model: nn.Module, optimizer: Any, filepath: str, device: str = 'cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        epoch, loss
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    return epoch, loss

"""
Shared utilities for model conversion and validation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """Load a model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_model_from_checkpoint(model_class, checkpoint_path: str, device: str = 'cpu', **model_kwargs):
    """
    Load a model from checkpoint with its state dict.
    
    Args:
        model_class: The model class to instantiate
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        model: The loaded model in eval mode
        checkpoint: The full checkpoint dict (for metadata)
    """
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Instantiate model
    model = model_class(**model_kwargs).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def validate_outputs(pytorch_output: torch.Tensor, 
                    converted_output: torch.Tensor,
                    rtol: float = 1e-4,
                    atol: float = 1e-5) -> Tuple[bool, float]:
    """
    Validate that converted model output matches PyTorch output.
    
    Args:
        pytorch_output: Output from original PyTorch model
        converted_output: Output from converted model
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        (is_valid, max_diff): Whether outputs match and maximum difference
    """
    # Convert to numpy for comparison
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.detach().cpu().numpy()
    if isinstance(converted_output, torch.Tensor):
        converted_output = converted_output.detach().cpu().numpy()
    
    # Calculate maximum absolute difference
    max_diff = np.abs(pytorch_output - converted_output).max()
    
    # Check if outputs are close
    is_valid = np.allclose(pytorch_output, converted_output, rtol=rtol, atol=atol)
    
    return is_valid, max_diff


def create_output_directory(output_path: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

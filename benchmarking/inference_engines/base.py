"""
Base inference engine interface.

All framework-specific engines must implement this interface for
consistent benchmarking across different compilation strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import numpy as np


class InferenceEngine(ABC):
    """
    Abstract base class for framework-specific inference engines.
    
    Each engine is responsible for:
    - Loading models from checkpoints
    - Running inference with proper synchronization
    - Tracking compilation overhead
    - Managing device placement
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize inference engine.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = None
        self.compilation_time_ms = 0.0
        self._is_loaded = False
        
    @abstractmethod
    def load_model(self, model_name: str, checkpoint_path: str, 
                   model_config: Dict[str, Any]) -> None:
        """
        Load model from checkpoint.
        
        Args:
            model_name: Name of the model (e.g., 'lenet', 'resnet18')
            checkpoint_path: Path to model checkpoint
            model_config: Configuration dict containing model kwargs and metadata
        """
        pass
    
    @abstractmethod
    def warmup(self, num_iterations: int = 100, batch_size: int = 1) -> None:
        """
        Perform warmup inferences to stabilize performance.
        
        Args:
            num_iterations: Number of warmup iterations
            batch_size: Batch size for warmup
        """
        pass
    
    @abstractmethod
    def infer(self, input_tensor: Any) -> Any:
        """
        Run single inference.
        
        Args:
            input_tensor: Input data (framework-specific format)
            
        Returns:
            Output tensor
        """
        pass
    
    def get_compilation_time(self) -> float:
        """
        Get compilation time in milliseconds.
        
        Returns:
            Compilation time in ms (0 for eager mode)
        """
        return self.compilation_time_ms
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def create_dummy_input(self, model_name: str, batch_size: int = 1) -> Any:
        """
        Create dummy input tensor for the given model and batch size.
        
        Args:
            model_name: Name of the model
            batch_size: Batch size for input
            
        Returns:
            Dummy input tensor
        """
        # Model input shapes
        input_shapes = {
            'lenet': (batch_size, 1, 28, 28),
            'resnet18': (batch_size, 3, 32, 32),
            'lstm': (batch_size, 256),  # sequence length 256
            'gru': (batch_size, 35),    # sequence length 35
        }
        
        if model_name not in input_shapes:
            raise ValueError(f"Unknown model: {model_name}")
        
        shape = input_shapes[model_name]
        
        # Create appropriate input type
        if model_name in ['lstm', 'gru']:
            # RNN models expect integer tokens
            vocab_sizes = {'lstm': 25000, 'gru': 29573}
            return torch.randint(0, vocab_sizes[model_name], shape, device=self.device)
        else:
            # CNN models expect float images
            return torch.randn(shape, device=self.device)
    
    def synchronize(self) -> None:
        """Synchronize CUDA operations if on GPU."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
    
    def reset_memory_stats(self) -> None:
        """Reset CUDA memory statistics if on GPU."""
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
    
    def get_peak_memory_mb(self) -> float:
        """
        Get peak GPU memory usage in MB.
        
        Returns:
            Peak memory in MB (0 if CPU)
        """
        if self.device == 'cuda':
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, loaded={self._is_loaded})"


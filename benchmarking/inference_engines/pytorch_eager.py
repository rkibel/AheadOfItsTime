"""PyTorch eager mode inference engine."""

import sys
import time
import torch
from pathlib import Path
from typing import Any, Dict

from .base import InferenceEngine

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.cnn.lenet import LeNet5
from models.cnn.resnet import ResNet18
from models.rnn.lstm_sentiment import LSTMSentimentClassifier
from models.rnn.gru_lm import GRULanguageModel
from conversion.utils import load_model_from_checkpoint


class PyTorchEagerEngine(InferenceEngine):
    """
    PyTorch eager mode inference engine.
    
    This represents the baseline dynamic execution mode with no compilation.
    """
    
    MODEL_CLASSES = {
        'lenet': LeNet5,
        'resnet18': ResNet18,
        'lstm': LSTMSentimentClassifier,
        'gru': GRULanguageModel,
    }
    
    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        self.model_name = None
        
    def load_model(self, model_name: str, checkpoint_path: str, 
                   model_config: Dict[str, Any]) -> None:
        """Load PyTorch model in eager mode."""
        print(f"Loading {model_name} in PyTorch eager mode...")
        
        if model_name not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        model_class = self.MODEL_CLASSES[model_name]
        
        # Load model from checkpoint
        start_time = time.time()
        self.model, checkpoint = load_model_from_checkpoint(
            model_class=model_class,
            checkpoint_path=checkpoint_path,
            device=self.device,
            **model_config['kwargs']
        )
        load_time = (time.time() - start_time) * 1000
        
        self.model.eval()
        self._is_loaded = True
        self.compilation_time_ms = 0.0  # No compilation in eager mode
        
        print(f"  ✓ Model loaded in {load_time:.2f}ms (epoch {checkpoint.get('epoch', 'unknown')})")
    
    def warmup(self, num_iterations: int = 100, batch_size: int = 1) -> None:
        """Perform warmup inferences."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"  Warming up ({num_iterations} iterations, batch_size={batch_size})...")
        dummy_input = self.create_dummy_input(self.model_name, batch_size)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        
        self.synchronize()
        print("  ✓ Warmup complete")
    
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run single inference in eager mode."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Handle tuple outputs from RNNs
        if isinstance(output, tuple):
            output = output[0]
        
        return output


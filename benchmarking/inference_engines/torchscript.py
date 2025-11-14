"""TorchScript inference engine."""

import sys
import time
import torch
from pathlib import Path
from typing import Any, Dict

from .base import InferenceEngine


class TorchScriptEngine(InferenceEngine):
    """
    TorchScript JIT inference engine.
    
    Loads pre-compiled TorchScript models for optimized execution.
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        self.model_name = None
        
    def load_model(self, model_name: str, checkpoint_path: str, 
                   model_config: Dict[str, Any]) -> None:
        """Load TorchScript model."""
        print(f"Loading {model_name} TorchScript model...")
        
        self.model_name = model_name
        
        # Load TorchScript model
        start_time = time.time()
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        load_time = (time.time() - start_time) * 1000
        
        self.model.eval()
        self._is_loaded = True
        
        # TorchScript models are pre-compiled, so compilation happened during conversion
        # We track the load time which includes deserialization
        self.compilation_time_ms = load_time
        
        print(f"  ✓ TorchScript model loaded in {load_time:.2f}ms")
    
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
        """Run single inference with TorchScript."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Handle tuple outputs from RNNs
        if isinstance(output, tuple):
            output = output[0]
        
        return output


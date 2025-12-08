"""TorchScript inference engine with TRUE JIT compilation (compile at runtime)."""

import sys
import time
import torch
from pathlib import Path
from typing import Any, Dict

from .base import InferenceEngine


class TorchScriptJITEngine(InferenceEngine):
    """
    TorchScript JIT inference engine - compiles at runtime.
    
    This loads the original PyTorch model and traces it during load_model(),
    capturing the TRUE JIT compilation time.
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        self.model_name = None
        
    def load_model(self, model_name: str, checkpoint_path: str, 
                   model_config: Dict[str, Any]) -> None:
        """Load PyTorch model and JIT compile it."""
        print(f"Loading {model_name} PyTorch model and JIT compiling...")
        
        self.model_name = model_name
        
        # Import model utilities
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from models import get_model_class
        from conversion.utils import load_model_from_checkpoint
        
        # Load original PyTorch model
        model_class = get_model_class(model_name)
        pytorch_model, _ = load_model_from_checkpoint(
            model_class,
            checkpoint_path,
            device=self.device,
            **model_config.get('kwargs', {})
        )
        pytorch_model.eval()
        
        # Create example input for tracing
        dummy_input = self.create_dummy_input(model_name, batch_size=1)
        
        # JIT compile (trace) the model - THIS IS THE REAL COMPILATION
        print(f"  JIT compiling model...")
        start_time = time.time()
        
        with torch.no_grad():
            self.model = torch.jit.trace(pytorch_model, dummy_input)
        
        compilation_time = (time.time() - start_time) * 1000
        self._is_loaded = True
        
        # This is the REAL compilation time
        self.compilation_time_ms = compilation_time
        
        print(f"  ✓ Model JIT compiled in {compilation_time:.2f}ms")
    
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

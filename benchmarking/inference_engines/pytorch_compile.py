"""PyTorch 2.0 torch.compile inference engine."""

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


class PyTorchCompileEngine(InferenceEngine):
    """
    PyTorch 2.0 torch.compile inference engine.
    
    Uses torch.compile for JIT compilation with various backends.
    Represents the hybrid AOT/JIT approach in PyTorch 2.0.
    """
    
    MODEL_CLASSES = {
        'lenet': LeNet5,
        'resnet18': ResNet18,
        'lstm': LSTMSentimentClassifier,
        'gru': GRULanguageModel,
    }
    
    def __init__(self, device: str = 'cuda', backend: str = 'inductor', 
                 mode: str = 'default'):
        """
        Initialize PyTorch 2.0 compile engine.
        
        Args:
            device: Device to run on
            backend: Compilation backend ('inductor', 'aot_eager', 'cudagraphs')
            mode: Optimization mode ('default', 'reduce-overhead', 'max-autotune')
        """
        super().__init__(device)
        self.backend = backend
        self.mode = mode
        self.model_name = None
        
    def load_model(self, model_name: str, checkpoint_path: str, 
                   model_config: Dict[str, Any]) -> None:
        """Load and compile PyTorch model with torch.compile."""
        print(f"Loading {model_name} with torch.compile (backend={self.backend}, mode={self.mode})...")
        
        if model_name not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        model_class = self.MODEL_CLASSES[model_name]
        
        # Load base model
        start_time = time.time()
        base_model, checkpoint = load_model_from_checkpoint(
            model_class=model_class,
            checkpoint_path=checkpoint_path,
            device=self.device,
            **model_config['kwargs']
        )
        base_model.eval()
        load_time = (time.time() - start_time) * 1000
        
        # Apply torch.compile
        compile_start = time.time()
        try:
            self.model = torch.compile(
                base_model,
                backend=self.backend,
                mode=self.mode
            )
            compile_time = (time.time() - compile_start) * 1000
        except Exception as e:
            print(f"  ⚠ torch.compile failed: {e}")
            print(f"  Falling back to eager mode")
            self.model = base_model
            compile_time = 0
        
        self._is_loaded = True
        
        # Note: Actual compilation happens lazily on first inference
        # We'll measure true compilation time during warmup
        self.compilation_time_ms = compile_time
        
        print(f"  ✓ Model loaded in {load_time:.2f}ms, compile setup in {compile_time:.2f}ms")
        print(f"  Note: Actual compilation will occur during first inference")
    
    def warmup(self, num_iterations: int = 100, batch_size: int = 1) -> None:
        """
        Perform warmup inferences.
        
        For torch.compile, the first inference triggers actual compilation,
        so we track that separately.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"  Warming up ({num_iterations} iterations, batch_size={batch_size})...")
        dummy_input = self.create_dummy_input(self.model_name, batch_size)
        
        # First inference triggers compilation
        self.synchronize()
        first_inference_start = time.time()
        with torch.no_grad():
            _ = self.model(dummy_input)
        self.synchronize()
        first_inference_time = (time.time() - first_inference_start) * 1000
        
        # Update compilation time with actual first inference cost
        self.compilation_time_ms = first_inference_time
        print(f"  First inference (compilation): {first_inference_time:.2f}ms")
        
        # Continue warmup
        with torch.no_grad():
            for _ in range(num_iterations - 1):
                _ = self.model(dummy_input)
        
        self.synchronize()
        print("  ✓ Warmup complete")
    
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run single inference with compiled model."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Handle tuple outputs from RNNs
        if isinstance(output, tuple):
            output = output[0]
        
        return output


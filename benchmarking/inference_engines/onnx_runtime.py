"""ONNX Runtime inference engine."""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict

from .base import InferenceEngine

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class ONNXRuntimeEngine(InferenceEngine):
    """
    ONNX Runtime inference engine.
    
    Supports CPU and CUDA execution providers with graph-level optimizations.
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        self.model_name = None
        self.session = None
        self.input_name = None
        self.output_name = None
        
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("onnxruntime not available. Install with: pip install onnxruntime-gpu")
        
    def load_model(self, model_name: str, checkpoint_path: str, 
                   model_config: Dict[str, Any]) -> None:
        """Load ONNX model with appropriate execution provider."""
        print(f"Loading {model_name} ONNX model...")
        
        self.model_name = model_name
        
        # Configure execution providers
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        start_time = time.time()
        self.session = ort.InferenceSession(
            checkpoint_path,
            sess_options=sess_options,
            providers=providers
        )
        load_time = (time.time() - start_time) * 1000
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self._is_loaded = True
        self.compilation_time_ms = load_time
        
        actual_providers = self.session.get_providers()
        print(f"  ✓ ONNX model loaded in {load_time:.2f}ms")
        print(f"  Execution providers: {actual_providers}")
    
    def warmup(self, num_iterations: int = 100, batch_size: int = 1) -> None:
        """Perform warmup inferences."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"  Warming up ({num_iterations} iterations, batch_size={batch_size})...")
        dummy_input = self.create_dummy_input(self.model_name, batch_size)
        dummy_input_np = dummy_input.cpu().numpy()
        
        for _ in range(num_iterations):
            _ = self.session.run([self.output_name], {self.input_name: dummy_input_np})
        
        self.synchronize()
        print("  ✓ Warmup complete")
    
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run single inference with ONNX Runtime."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert to numpy
        input_np = input_tensor.cpu().numpy()
        
        # Run inference
        output_np = self.session.run(
            [self.output_name],
            {self.input_name: input_np}
        )[0]
        
        # Convert back to torch tensor
        return torch.from_numpy(output_np).to(self.device)


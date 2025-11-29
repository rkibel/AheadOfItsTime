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
        
        # Optimization cache
        self.io_binding = None
        self.cached_input_ptr = None
        self.cached_input_shape = None
        self.cached_output_tensor = None
        
        self.dtype_map = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.int64: np.int64,
            torch.int32: np.int32,
            torch.bool: np.bool_,
        }
        
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("onnxruntime not available. Install with: pip install onnxruntime-gpu")
        
    def load_model(self, model_name: str, checkpoint_path: str, 
                   model_config: Dict[str, Any]) -> None:
        """Load ONNX model with appropriate execution provider."""
        print(f"Loading {model_name} ONNX model...")
        
        self.model_name = model_name
        
        # Configure execution providers
        if self.device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'arena_extend_strategy': 'kSameAsRequested',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
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
        
        # Use IOBinding for CUDA to avoid CPU-GPU data transfer overhead
        if self.device == 'cuda':
            return self._infer_cuda(input_tensor)
        
        # CPU fallback
        return self._infer_cpu(input_tensor)

    def _infer_cuda(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference using IOBinding on CUDA."""
        # Ensure tensor is contiguous
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
            
        # Initialize IOBinding if needed
        if self.io_binding is None:
            self.io_binding = self.session.io_binding()
            
        # Check if we need to re-bind input (pointer or shape changed)
        current_ptr = input_tensor.data_ptr()
        current_shape = tuple(input_tensor.shape)
        
        # If input shape changed, we must invalidate output cache because output shape likely changed
        if current_shape != self.cached_input_shape:
            self.cached_output_tensor = None
        
        if current_ptr != self.cached_input_ptr or current_shape != self.cached_input_shape:
            numpy_dtype = self.dtype_map.get(input_tensor.dtype)
            if numpy_dtype is None:
                # Fallback to CPU if dtype not supported
                return self._infer_cpu(input_tensor)
                
            # Bind input directly from GPU memory
            self.io_binding.bind_input(
                name=self.input_name,
                device_type='cuda',
                device_id=0,
                element_type=numpy_dtype,
                shape=current_shape,
                buffer_ptr=current_ptr,
            )
            
            # Update input cache
            self.cached_input_ptr = current_ptr
            self.cached_input_shape = current_shape
            
        # Handle output binding
        if self.cached_output_tensor is None:
            # Slow path: Let ORT allocate output
            # Bind output to CUDA device (no buffer)
            self.io_binding.bind_output(self.output_name, 'cuda', 0)
            
            # Run inference
            self.session.run_with_iobinding(self.io_binding)
            
            # Get output as DLPack
            ort_output = self.io_binding.get_outputs()[0]
            
            try:
                from torch.utils.dlpack import from_dlpack
                output_tensor = from_dlpack(ort_output.to_dlpack())
                
                # Allocate our own buffer for next time (must be on same device)
                self.cached_output_tensor = torch.empty_like(output_tensor)
                
                # Bind this new buffer for FUTURE runs
                numpy_dtype = self.dtype_map.get(self.cached_output_tensor.dtype)
                self.io_binding.bind_output(
                    name=self.output_name,
                    device_type='cuda',
                    device_id=0,
                    element_type=numpy_dtype,
                    shape=tuple(self.cached_output_tensor.shape),
                    buffer_ptr=self.cached_output_tensor.data_ptr()
                )
                
                return output_tensor
            except Exception:
                return self._infer_cpu(input_tensor)
        else:
            # Fast path: Output is already bound to self.cached_output_tensor
            self.session.run_with_iobinding(self.io_binding)
            return self.cached_output_tensor

    def _infer_cpu(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on CPU (with data copy)."""
        # Convert to numpy
        input_np = input_tensor.cpu().numpy()
        
        # Run inference
        output_np = self.session.run(
            [self.output_name],
            {self.input_name: input_np}
        )[0]
        
        # Convert back to torch tensor
        return torch.from_numpy(output_np).to(self.device)


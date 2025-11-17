"""
TensorRT inference engine implementation.

Provides high-performance inference using NVIDIA TensorRT engines.
"""

import time
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict

from .base import InferenceEngine

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT or PyCUDA not available. TensorRT engine will not work.")


class TensorRTEngine(InferenceEngine):
    """
    TensorRT inference engine.
    
    Loads pre-built TensorRT engine files and runs inference using PyCUDA.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize TensorRT engine.
        
        Args:
            device: Device to run inference on (must be 'cuda')
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT and PyCUDA are required for TensorRTEngine")
        
        if device != 'cuda':
            raise ValueError("TensorRT only supports CUDA devices")
        
        super().__init__(device)
        self.runtime = None
        self.context = None
        self.engine = None
        self.stream = None
        self.input_name = None
        self.output_names = []
        self.input_shape = None
        self.output_shapes = []
        self.d_input = None
        self.d_outputs = []
        self.h_input = None
        self.h_outputs = []
        
    def load_model(self, model_name: str, checkpoint_path: str, 
                   model_config: Dict[str, Any]) -> None:
        """
        Load TensorRT engine from file.
        
        Args:
            model_name: Name of the model
            checkpoint_path: Path to TensorRT engine file
            model_config: Configuration dict containing model metadata
        """
        start_time = time.time()
        
        engine_path = Path(checkpoint_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output information
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(shape)
            else:
                self.output_names.append(name)
                self.output_shapes.append(tuple(shape))
        
        # Handle dynamic shapes by using example input
        example_input = model_config.get('example_input')
        if example_input is not None:
            if hasattr(example_input, 'numpy'):
                example_input = example_input.cpu().numpy()
            
            # Check if we need to set dynamic shape
            if -1 in self.input_shape:
                actual_shape = tuple(example_input.shape)
                self.input_shape = actual_shape
                self.context.set_input_shape(self.input_name, actual_shape)
                
                # Update output shapes based on context
                self.output_shapes = []
                for name in self.output_names:
                    shape = tuple(self.context.get_tensor_shape(name))
                    # Fix any remaining dynamic dimensions
                    shape = tuple(
                        actual_shape[i] if shape[i] == -1 and i < len(actual_shape) 
                        else 1 if shape[i] == -1 
                        else shape[i] 
                        for i in range(len(shape))
                    )
                    self.output_shapes.append(shape)
        
        # Allocate device buffers
        self._allocate_buffers()
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for name, d_out in zip(self.output_names, self.d_outputs):
            self.context.set_tensor_address(name, int(d_out))
        
        self.model = self.engine  # For compatibility with base class
        self._is_loaded = True
        
        # Record compilation time (engine is pre-compiled)
        self.compilation_time_ms = (time.time() - start_time) * 1000
        
    def _allocate_buffers(self) -> None:
        """Allocate host and device buffers for inputs and outputs."""
        # Input buffer
        input_size = int(np.prod([max(d, 1) for d in self.input_shape]))
        self.h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        
        # Output buffers
        self.h_outputs = []
        self.d_outputs = []
        for shape in self.output_shapes:
            output_size = int(np.prod([max(d, 1) for d in shape]))
            h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
            d_output = cuda.mem_alloc(h_output.nbytes)
            self.h_outputs.append(h_output)
            self.d_outputs.append(d_output)
    
    def _reallocate_buffers(self) -> None:
        """Reallocate buffers when input shape changes (e.g., different batch size)."""
        # Free old buffers
        if self.d_input is not None:
            self.d_input.free()
        for d_out in self.d_outputs:
            if d_out is not None:
                d_out.free()
        
        # Allocate new buffers with updated shapes
        input_size = int(np.prod([max(d, 1) for d in self.input_shape]))
        self.h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        
        self.h_outputs = []
        self.d_outputs = []
        for shape in self.output_shapes:
            output_size = int(np.prod([max(d, 1) for d in shape]))
            h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
            d_output = cuda.mem_alloc(h_output.nbytes)
            self.h_outputs.append(h_output)
            self.d_outputs.append(d_output)
        
        # Update tensor addresses in context
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for name, d_out in zip(self.output_names, self.d_outputs):
            self.context.set_tensor_address(name, int(d_out))
    
    def warmup(self, num_iterations: int = 100, batch_size: int = 1) -> None:
        """
        Perform warmup inferences.
        
        Args:
            num_iterations: Number of warmup iterations
            batch_size: Batch size for warmup
        """
        if not self._is_loaded:
            raise RuntimeError("Model must be loaded before warmup")
        
        # Create dummy input with correct batch size
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # Warmup iterations
        for _ in range(num_iterations):
            self._run_inference(dummy_input)
    
    def infer(self, input_tensor: Any) -> Any:
        """
        Run single inference.
        
        Args:
            input_tensor: Input data (numpy array or torch tensor)
            
        Returns:
            Output tensor (numpy array)
        """
        if not self._is_loaded:
            raise RuntimeError("Model must be loaded before inference")
        
        # Convert to numpy if needed
        if isinstance(input_tensor, torch.Tensor):
            input_data = input_tensor.cpu().numpy()
        else:
            input_data = input_tensor
        
        # Handle integer inputs (for RNN models)
        if input_data.dtype in [np.int32, np.int64]:
            input_data = input_data.astype(np.float32)
        
        return self._run_inference(input_data)
    
    def _run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        Internal inference execution.
        
        Args:
            input_data: Input numpy array
            
        Returns:
            Output numpy array
        """
        # Check if input shape has changed (e.g., different batch size)
        actual_shape = tuple(input_data.shape)
        if actual_shape != self.input_shape:
            # Update shape context for dynamic batch size
            self.input_shape = actual_shape
            self.context.set_input_shape(self.input_name, actual_shape)
            
            # Update output shapes based on new input shape
            self.output_shapes = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                self.output_shapes.append(shape)
            
            # Reallocate buffers with new sizes
            self._reallocate_buffers()
        
        # Copy input to host buffer
        np.copyto(self.h_input, input_data.ravel())
        
        # Transfer to device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Execute inference
        success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not success:
            raise RuntimeError("TensorRT inference execution failed")
        
        # Transfer outputs back to host
        cuda.memcpy_dtoh_async(self.h_outputs[0], self.d_outputs[0], self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Reshape output
        output = self.h_outputs[0].reshape(self.output_shapes[0])
        
        return output
    
    def __del__(self):
        """Cleanup CUDA resources."""
        # Free device memory
        if hasattr(self, 'd_input') and self.d_input is not None:
            self.d_input.free()
        
        if hasattr(self, 'd_outputs'):
            for d_out in self.d_outputs:
                if d_out is not None:
                    d_out.free()
        
        # Destroy context and engine
        if hasattr(self, 'context') and self.context is not None:
            del self.context
        
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine
        
        if hasattr(self, 'runtime') and self.runtime is not None:
            del self.runtime

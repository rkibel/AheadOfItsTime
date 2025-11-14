"""
Memory profiler for measuring GPU memory consumption.

Measures:
- Peak memory allocation
- Average memory during inference
- Memory per sample
"""

import torch
import numpy as np
from typing import Dict, Any, List
from ..inference_engines.base import InferenceEngine


class MemoryProfiler:
    """
    Profile GPU memory usage during inference.
    
    Uses PyTorch's CUDA memory tracking for accurate measurements.
    """
    
    def __init__(self, engine: InferenceEngine):
        """
        Initialize memory profiler.
        
        Args:
            engine: Inference engine to profile
        """
        self.engine = engine
        
        if engine.device != 'cuda':
            print("    ⚠ Memory profiling only available for CUDA devices")
        
    def profile(self, model_name: str, batch_size: int, 
                num_iterations: int = 100) -> Dict[str, Any]:
        """
        Profile memory usage.
        
        Args:
            model_name: Name of the model being profiled
            batch_size: Batch size for inference
            num_iterations: Number of inferences to measure
            
        Returns:
            Dictionary with memory statistics
        """
        if self.engine.device != 'cuda':
            return {
                'peak_mb': 0.0,
                'average_mb': 0.0,
                'per_sample_mb': 0.0,
                'note': 'Memory profiling only available on CUDA'
            }
        
        print(f"    Profiling memory ({num_iterations} iterations, batch_size={batch_size})...")
        
        # Create input
        dummy_input = self.engine.create_dummy_input(model_name, batch_size)
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Warmup to stabilize memory allocation
        with torch.no_grad():
            for _ in range(10):
                _ = self.engine.infer(dummy_input)
        
        self.engine.synchronize()
        
        # Reset again before measurement
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory during inference
        memory_samples = []
        for _ in range(num_iterations):
            _ = self.engine.infer(dummy_input)
            
            # Sample current memory
            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_samples.append(current_memory)
        
        self.engine.synchronize()
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Calculate statistics
        average_memory = np.mean(memory_samples)
        per_sample_memory = average_memory / batch_size if batch_size > 0 else 0.0
        
        results = {
            'peak_mb': float(peak_memory),
            'average_mb': float(average_memory),
            'per_sample_mb': float(per_sample_memory),
            'std_mb': float(np.std(memory_samples)),
            'num_iterations': num_iterations,
        }
        
        print(f"      Peak: {peak_memory:.2f}MB, Average: {average_memory:.2f}MB")
        
        return results
    
    def profile_batch_sizes(self, model_name: str, 
                           batch_sizes: List[int],
                           num_iterations: int = 100) -> Dict[int, Dict[str, Any]]:
        """
        Profile memory for multiple batch sizes.
        
        Args:
            model_name: Name of the model
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per batch size
            
        Returns:
            Dictionary mapping batch_size -> memory results
        """
        results = {}
        for batch_size in batch_sizes:
            # Reset memory between batch sizes
            if self.engine.device == 'cuda':
                torch.cuda.empty_cache()
            
            results[batch_size] = self.profile(model_name, batch_size, num_iterations)
        return results


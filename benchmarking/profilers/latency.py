"""
Latency profiler for measuring inference time.

Measures:
- Median, P95, P99 latency
- Cold start time (first inference)
- Warm inference statistics
- Throughput (samples/sec)
"""

import time
import torch
import numpy as np
from typing import Dict, Any, List
from ..inference_engines.base import InferenceEngine


class LatencyProfiler:
    """
    Profile inference latency with CUDA synchronization.
    
    Uses CUDA events for precise GPU timing when available.
    """
    
    def __init__(self, engine: InferenceEngine, use_cuda_events: bool = True):
        """
        Initialize latency profiler.
        
        Args:
            engine: Inference engine to profile
            use_cuda_events: Use CUDA events for timing (more accurate on GPU)
        """
        self.engine = engine
        self.use_cuda_events = use_cuda_events and engine.device == 'cuda'
        
    def profile(self, model_name: str, batch_size: int, 
                num_iterations: int = 1000) -> Dict[str, Any]:
        """
        Profile inference latency.
        
        Args:
            model_name: Name of the model being profiled
            batch_size: Batch size for inference
            num_iterations: Number of inferences to measure
            
        Returns:
            Dictionary with latency statistics
        """
        print(f"    Profiling latency ({num_iterations} iterations, batch_size={batch_size})...")
        
        # Create input
        dummy_input = self.engine.create_dummy_input(model_name, batch_size)
        
        # Measure cold start (first inference after warmup)
        self.engine.synchronize()
        cold_start_time = self._measure_single_inference(dummy_input)
        
        # Measure warm inferences
        latencies_ms = []
        for _ in range(num_iterations):
            latency = self._measure_single_inference(dummy_input)
            latencies_ms.append(latency)
        
        # Calculate statistics
        latencies_array = np.array(latencies_ms)
        median = np.median(latencies_array)
        p95 = np.percentile(latencies_array, 95)
        p99 = np.percentile(latencies_array, 99)
        mean = np.mean(latencies_array)
        std = np.std(latencies_array)
        
        # Calculate throughput (samples per second)
        throughput = (batch_size * 1000.0) / median  # samples/sec
        
        results = {
            'cold_start_ms': float(cold_start_time),
            'median_ms': float(median),
            'mean_ms': float(mean),
            'std_ms': float(std),
            'p95_ms': float(p95),
            'p99_ms': float(p99),
            'min_ms': float(np.min(latencies_array)),
            'max_ms': float(np.max(latencies_array)),
            'throughput_samples_per_sec': float(throughput),
            'num_iterations': num_iterations,
        }
        
        print(f"      Median: {median:.3f}ms, P95: {p95:.3f}ms, P99: {p99:.3f}ms")
        print(f"      Throughput: {throughput:.1f} samples/sec")
        
        return results
    
    def _measure_single_inference(self, input_tensor: Any) -> float:
        """
        Measure single inference time in milliseconds.
        
        Args:
            input_tensor: Input for inference
            
        Returns:
            Latency in milliseconds
        """
        if self.use_cuda_events:
            # Use CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = self.engine.infer(input_tensor)
            end_event.record()
            
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event)
        else:
            # Use CPU timing
            self.engine.synchronize()
            start_time = time.perf_counter()
            _ = self.engine.infer(input_tensor)
            self.engine.synchronize()
            end_time = time.perf_counter()
            
            return (end_time - start_time) * 1000.0
    
    def profile_batch_sizes(self, model_name: str, 
                           batch_sizes: List[int],
                           num_iterations: int = 1000) -> Dict[int, Dict[str, Any]]:
        """
        Profile multiple batch sizes.
        
        Args:
            model_name: Name of the model
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per batch size
            
        Returns:
            Dictionary mapping batch_size -> latency results
        """
        results = {}
        for batch_size in batch_sizes:
            results[batch_size] = self.profile(model_name, batch_size, num_iterations)
        return results


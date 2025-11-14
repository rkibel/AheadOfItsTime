"""
Compilation profiler for measuring compile-time overhead.

Measures:
- Compilation time
- Amortization point (number of inferences to break even)
- First inference overhead
"""

from typing import Dict, Any
from ..inference_engines.base import InferenceEngine


class CompilationProfiler:
    """
    Profile compilation overhead and amortization.
    
    Analyzes the cost of compilation and when it pays off.
    """
    
    def __init__(self, engine: InferenceEngine, baseline_latency_ms: float = None):
        """
        Initialize compilation profiler.
        
        Args:
            engine: Inference engine to profile
            baseline_latency_ms: Baseline inference latency (e.g., from eager mode)
                               for amortization calculation
        """
        self.engine = engine
        self.baseline_latency_ms = baseline_latency_ms
        
    def profile(self) -> Dict[str, Any]:
        """
        Profile compilation overhead.
        
        Returns:
            Dictionary with compilation statistics
        """
        compilation_time_ms = self.engine.get_compilation_time()
        
        results = {
            'compilation_time_ms': float(compilation_time_ms),
        }
        
        # Calculate amortization if baseline provided
        if self.baseline_latency_ms is not None and self.baseline_latency_ms > 0:
            # Assume we have compiled model latency from latency profiler
            # For now, we'll calculate this in the runner where we have both latencies
            results['baseline_latency_ms'] = float(self.baseline_latency_ms)
        
        print(f"    Compilation time: {compilation_time_ms:.2f}ms")
        
        return results
    
    @staticmethod
    def calculate_amortization(compilation_time_ms: float, 
                               baseline_latency_ms: float,
                               optimized_latency_ms: float) -> Dict[str, Any]:
        """
        Calculate amortization point.
        
        The amortization point is the number of inferences needed for the
        compiled model to break even compared to the baseline.
        
        Args:
            compilation_time_ms: Time spent on compilation
            baseline_latency_ms: Baseline (eager) inference latency
            optimized_latency_ms: Optimized (compiled) inference latency
            
        Returns:
            Dictionary with amortization analysis
        """
        if baseline_latency_ms <= optimized_latency_ms:
            # No speedup, compilation doesn't pay off
            return {
                'amortization_samples': float('inf'),
                'speedup': 1.0,
                'is_beneficial': False,
                'note': 'Compiled model is not faster than baseline'
            }
        
        # Time saved per inference
        time_saved_per_inference = baseline_latency_ms - optimized_latency_ms
        
        # Number of inferences to break even
        amortization_samples = compilation_time_ms / time_saved_per_inference
        
        # Speedup ratio
        speedup = baseline_latency_ms / optimized_latency_ms
        
        return {
            'amortization_samples': float(amortization_samples),
            'speedup': float(speedup),
            'time_saved_per_inference_ms': float(time_saved_per_inference),
            'is_beneficial': True,
            'note': f'Break even after {amortization_samples:.0f} inferences'
        }


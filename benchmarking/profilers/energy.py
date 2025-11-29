"""
Energy profiler for measuring GPU power consumption.

Measures:
- Average power usage (Watts)
- Total energy consumption (Joules)
- Energy efficiency (Inferences/Joule)
"""

import time
import threading
import torch
import numpy as np
from typing import Dict, Any, List
from ..inference_engines.base import InferenceEngine

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class EnergyProfiler:
    """
    Profile GPU energy consumption during inference.
    
    Uses NVIDIA Management Library (NVML) to query GPU power usage.
    """
    
    def __init__(self, engine: InferenceEngine, sampling_interval: float = 0.01):
        """
        Initialize energy profiler.
        
        Args:
            engine: Inference engine to profile
            sampling_interval: Time between power samples in seconds
        """
        self.engine = engine
        self.sampling_interval = sampling_interval
        self.device_handle = None
        
        if engine.device == 'cuda' and HAS_NVML:
            try:
                pynvml.nvmlInit()
                # Assume using the first visible device
                # In a real multi-GPU setup, we might need to map torch device ID to NVML index
                self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"    ⚠ Warning: Could not initialize NVML: {e}")
        elif engine.device != 'cuda':
            print("    ⚠ Energy profiling only available for CUDA devices")
        elif not HAS_NVML:
            print("    ⚠ nvidia-ml-py not installed, energy profiling disabled")
            
    def profile(self, model_name: str, batch_size: int, 
                num_iterations: int = 100) -> Dict[str, Any]:
        """
        Profile energy usage.
        
        Args:
            model_name: Name of the model being profiled
            batch_size: Batch size for inference
            num_iterations: Number of inferences to measure
            
        Returns:
            Dictionary with energy statistics
        """
        if not self.device_handle:
            return {
                'avg_power_watts': 0.0,
                'total_energy_joules': 0.0,
                'inferences_per_joule': 0.0,
                'note': 'Energy profiling not available'
            }
        
        print(f"    Profiling energy ({num_iterations} iterations, batch_size={batch_size})...")
        
        # Create input
        dummy_input = self.engine.create_dummy_input(model_name, batch_size)
        
        # Warmup
        for _ in range(10):
            _ = self.engine.infer(dummy_input)
        self.engine.synchronize()
        
        # Sampling thread
        stop_event = threading.Event()
        power_readings = []
        
        def sample_power():
            while not stop_event.is_set():
                try:
                    # nvmlDeviceGetPowerUsage returns milliwatts
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.device_handle)
                    power_readings.append(power_mw / 1000.0)
                except Exception:
                    pass
                time.sleep(self.sampling_interval)
        
        # Start sampling
        sampler_thread = threading.Thread(target=sample_power)
        sampler_thread.start()
        
        # Run inference
        start_time = time.time()
        for _ in range(num_iterations):
            _ = self.engine.infer(dummy_input)
        self.engine.synchronize()
        end_time = time.time()
        
        # Stop sampling
        stop_event.set()
        sampler_thread.join()
        
        # Calculate statistics
        duration = end_time - start_time
        avg_power = np.mean(power_readings) if power_readings else 0.0
        total_energy = avg_power * duration # Joules = Watts * Seconds
        
        total_samples = batch_size * num_iterations
        inferences_per_joule = total_samples / total_energy if total_energy > 0 else 0.0
        
        results = {
            'avg_power_watts': float(avg_power),
            'total_energy_joules': float(total_energy),
            'inferences_per_joule': float(inferences_per_joule),
            'duration_seconds': float(duration),
            'num_iterations': num_iterations,
            'num_samples': len(power_readings)
        }
        
        print(f"      Avg Power: {avg_power:.2f}W, Efficiency: {inferences_per_joule:.1f} inf/J")
        
        return results
    
    def __del__(self):
        """Cleanup NVML."""
        if HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

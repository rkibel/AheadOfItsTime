"""
Main benchmark runner.

Orchestrates benchmarking across models, frameworks, and batch sizes.
Aggregates results into structured JSON format.
"""

import argparse
import json
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.config import load_config, MODEL_CONFIGS, get_checkpoint_path
from benchmarking.inference_engines import ENGINE_REGISTRY
from benchmarking.profilers import LatencyProfiler, MemoryProfiler, CompilationProfiler, EnergyProfiler


class BenchmarkRunner:
    """Main benchmark orchestrator."""
    
    def __init__(self, config_path: str):
        """
        Initialize benchmark runner.
        
        Args:
            config_path: Path to benchmark configuration YAML
        """
        self.config = load_config(config_path)
        self.results = {
            'metadata': self._collect_metadata(),
            'benchmarks': []
        }
        self.baseline_latencies = {}  # Store eager mode latencies for comparison
        
    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect system metadata."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config_name': self.config.name,
            'config_description': self.config.description,
            'device': self.config.device,
        }
        
        # GPU information
        if self.config.device == 'cuda' and torch.cuda.is_available():
            metadata['gpu'] = torch.cuda.get_device_name(0)
            metadata['cuda_version'] = torch.version.cuda
            metadata['cudnn_version'] = torch.backends.cudnn.version()
        
        # Framework versions
        metadata['frameworks'] = {
            'pytorch': torch.__version__,
        }
        
        try:
            import onnxruntime as ort
            metadata['frameworks']['onnxruntime'] = ort.__version__
        except ImportError:
            pass
        
        return metadata
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Returns:
            Dictionary with all benchmark results
        """
        print("="*80)
        print(f"Starting Benchmark: {self.config.name}")
        print("="*80)
        print(f"Models: {self.config.models}")
        print(f"Frameworks: {self.config.frameworks}")
        print(f"Batch sizes: {self.config.batch_sizes}")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Device: {self.config.device}")
        print("="*80)
        print()
        
        # Iterate through all combinations
        for model_name in self.config.models:
            print(f"\n{'='*80}")
            print(f"Benchmarking Model: {model_name.upper()}")
            print(f"{'='*80}")
            
            for framework in self.config.frameworks:
                print(f"\n{'-'*80}")
                print(f"Framework: {framework}")
                print(f"{'-'*80}")
                
                try:
                    self._benchmark_model_framework(model_name, framework)
                except Exception as e:
                    print(f"\n✗ Error benchmarking {model_name} with {framework}: {e}")
                    print(f"  Skipping this combination...")
                    continue
        
        # Save results
        self._save_results()
        
        print(f"\n{'='*80}")
        print("Benchmark Complete!")
        print(f"{'='*80}")
        print(f"Results saved to: {self.config.output_dir}")
        
        return self.results
    
    def _benchmark_model_framework(self, model_name: str, framework: str) -> None:
        """
        Benchmark a single model-framework combination.
        
        Args:
            model_name: Name of the model
            framework: Framework identifier
        """
        # Get checkpoint path
        checkpoint_path = get_checkpoint_path(model_name, framework)
        
        if not checkpoint_path.exists():
            print(f"  ⚠ Checkpoint not found: {checkpoint_path}")
            print(f"  Skipping {framework} for {model_name}")
            return
        
        # Initialize engine
        engine_class = ENGINE_REGISTRY[framework]
        engine = engine_class(device=self.config.device)
        
        # Load model
        model_config = MODEL_CONFIGS[model_name]
        engine.load_model(model_name, str(checkpoint_path), model_config)
        
        # Warmup
        engine.warmup(
            num_iterations=self.config.warmup_iterations,
            batch_size=self.config.batch_sizes[0]
        )
        
        # Benchmark each batch size
        for batch_size in self.config.batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            
            result = {
                'model': model_name,
                'framework': framework,
                'batch_size': batch_size,
            }
            
            # Profile latency
            if self.config.profile_latency:
                latency_profiler = LatencyProfiler(engine)
                latency_results = latency_profiler.profile(
                    model_name, batch_size, self.config.num_iterations
                )
                result['latency'] = latency_results
                result['throughput'] = {
                    'samples_per_sec': latency_results['throughput_samples_per_sec']
                }
                
                # Store baseline latency for amortization calculation
                if framework == 'pytorch-eager':
                    key = f"{model_name}_{batch_size}"
                    self.baseline_latencies[key] = latency_results['median_ms']
            
            # Profile memory
            if self.config.profile_memory:
                memory_profiler = MemoryProfiler(engine)
                memory_results = memory_profiler.profile(
                    model_name, batch_size, num_iterations=100
                )
                result['memory'] = memory_results
            
            # Profile compilation
            if self.config.profile_compilation:
                compilation_profiler = CompilationProfiler(engine)
                compilation_results = compilation_profiler.profile()
                result['compilation'] = compilation_results
                
                # Calculate amortization if baseline available
                key = f"{model_name}_{batch_size}"
                if key in self.baseline_latencies and 'latency' in result:
                    baseline = self.baseline_latencies[key]
                    optimized = result['latency']['median_ms']
                    amortization = CompilationProfiler.calculate_amortization(
                        compilation_results['compilation_time_ms'],
                        baseline,
                        optimized
                    )
                    result['compilation']['amortization'] = amortization
            
            # Profile energy
            if self.config.profile_energy:
                energy_profiler = EnergyProfiler(engine)
                energy_results = energy_profiler.profile(
                    model_name, batch_size, self.config.num_iterations
                )
                result['energy'] = energy_results

            self.results['benchmarks'].append(result)
        
        # Cleanup
        del engine
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
    
    def _save_results(self) -> None:
        """Save benchmark results to JSON file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Also save as latest
        latest_file = output_dir / "benchmark_results_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(self.results, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run benchmarks for AOT vs JIT compilation strategies'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to benchmark configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    runner = BenchmarkRunner(args.config)
    runner.run()


if __name__ == '__main__':
    main()


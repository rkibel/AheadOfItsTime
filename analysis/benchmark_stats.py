"""
Statistical analysis of benchmark results.

Computes:
- Speedup ratios relative to baseline
- Statistical significance tests
- Memory efficiency metrics
- Summary tables
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from scipy import stats


class BenchmarkAnalyzer:
    """Analyze benchmark results and compute statistics."""
    
    def __init__(self, results_path: str):
        """
        Initialize analyzer.
        
        Args:
            results_path: Path to benchmark results JSON file
        """
        self.results_path = Path(results_path)
        
        with open(self.results_path, 'r') as f:
            self.data = json.load(f)
        
        self.benchmarks = self.data['benchmarks']
        self.metadata = self.data['metadata']
        
    def compute_speedups(self, baseline_framework: str = 'pytorch-eager') -> Dict[str, Any]:
        """
        Compute speedup ratios relative to baseline framework.
        
        Args:
            baseline_framework: Framework to use as baseline (default: pytorch-eager)
            
        Returns:
            Dictionary with speedup statistics
        """
        speedups = defaultdict(dict)
        
        # Group by model and batch size
        baseline_latencies = {}
        
        for benchmark in self.benchmarks:
            model = benchmark['model']
            framework = benchmark['framework']
            batch_size = benchmark['batch_size']
            
            if 'latency' not in benchmark:
                continue
            
            latency = benchmark['latency']['median_ms']
            key = f"{model}_bs{batch_size}"
            
            # Store baseline
            if framework == baseline_framework:
                baseline_latencies[key] = latency
            else:
                # Calculate speedup if baseline exists
                if key in baseline_latencies:
                    baseline = baseline_latencies[key]
                    speedup = baseline / latency
                    
                    if model not in speedups:
                        speedups[model] = {}
                    if framework not in speedups[model]:
                        speedups[model][framework] = {}
                    
                    speedups[model][framework][batch_size] = {
                        'speedup': speedup,
                        'baseline_ms': baseline,
                        'optimized_ms': latency,
                    }
        
        return dict(speedups)
    
    def compute_memory_efficiency(self) -> Dict[str, Any]:
        """
        Compute memory efficiency metrics.
        
        Returns:
            Dictionary with memory statistics
        """
        memory_stats = defaultdict(dict)
        
        for benchmark in self.benchmarks:
            model = benchmark['model']
            framework = benchmark['framework']
            batch_size = benchmark['batch_size']
            
            if 'memory' not in benchmark:
                continue
            
            memory = benchmark['memory']
            
            if model not in memory_stats:
                memory_stats[model] = {}
            if framework not in memory_stats[model]:
                memory_stats[model][framework] = {}
            
            memory_stats[model][framework][batch_size] = {
                'peak_mb': memory['peak_mb'],
                'average_mb': memory['average_mb'],
                'per_sample_mb': memory.get('per_sample_mb', 0),
            }
        
        return dict(memory_stats)
    
    def generate_summary_table(self) -> str:
        """
        Generate summary table of results.
        
        Returns:
            Formatted string table
        """
        lines = []
        lines.append("=" * 100)
        lines.append("BENCHMARK SUMMARY")
        lines.append("=" * 100)
        lines.append("")
        
        # Group by model
        models = set(b['model'] for b in self.benchmarks)
        
        for model in sorted(models):
            lines.append(f"\n{model.upper()}")
            lines.append("-" * 100)
            
            # Header
            lines.append(f"{'Framework':<20} {'Batch':<8} {'Latency (ms)':<15} "
                        f"{'Throughput':<18} {'Memory (MB)':<15} {'Speedup':<10}")
            lines.append("-" * 100)
            
            # Group by framework and batch size
            model_benchmarks = [b for b in self.benchmarks if b['model'] == model]
            
            # Get baseline latencies for speedup calculation
            baseline_latencies = {}
            for b in model_benchmarks:
                if b['framework'] == 'pytorch-eager' and 'latency' in b:
                    key = b['batch_size']
                    baseline_latencies[key] = b['latency']['median_ms']
            
            # Sort by framework and batch size
            model_benchmarks.sort(key=lambda x: (x['framework'], x['batch_size']))
            
            for benchmark in model_benchmarks:
                framework = benchmark['framework']
                batch_size = benchmark['batch_size']
                
                # Latency
                latency_str = "N/A"
                throughput_str = "N/A"
                speedup_str = "1.0x (baseline)"
                
                if 'latency' in benchmark:
                    latency = benchmark['latency']['median_ms']
                    latency_str = f"{latency:.3f}"
                    
                    if 'throughput' in benchmark:
                        throughput = benchmark['throughput']['samples_per_sec']
                        throughput_str = f"{throughput:.1f} samples/s"
                    
                    # Calculate speedup
                    if batch_size in baseline_latencies and framework != 'pytorch-eager':
                        baseline = baseline_latencies[batch_size]
                        speedup = baseline / latency
                        speedup_str = f"{speedup:.2f}x"
                
                # Memory
                memory_str = "N/A"
                if 'memory' in benchmark:
                    peak = benchmark['memory']['peak_mb']
                    memory_str = f"{peak:.1f}"
                
                lines.append(f"{framework:<20} {batch_size:<8} {latency_str:<15} "
                           f"{throughput_str:<18} {memory_str:<15} {speedup_str:<10}")
        
        lines.append("")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def get_best_framework(self, model: str, metric: str = 'latency') -> Dict[str, Any]:
        """
        Find best framework for a given model and metric.
        
        Args:
            model: Model name
            metric: Metric to optimize ('latency', 'memory', 'throughput')
            
        Returns:
            Dictionary with best framework info
        """
        model_benchmarks = [b for b in self.benchmarks if b['model'] == model]
        
        if not model_benchmarks:
            return {}
        
        # Different optimization direction for different metrics
        if metric == 'latency':
            # Lower is better
            best = min(
                (b for b in model_benchmarks if 'latency' in b),
                key=lambda x: x['latency']['median_ms']
            )
            return {
                'framework': best['framework'],
                'batch_size': best['batch_size'],
                'value': best['latency']['median_ms'],
                'unit': 'ms'
            }
        elif metric == 'throughput':
            # Higher is better
            best = max(
                (b for b in model_benchmarks if 'throughput' in b),
                key=lambda x: x['throughput']['samples_per_sec']
            )
            return {
                'framework': best['framework'],
                'batch_size': best['batch_size'],
                'value': best['throughput']['samples_per_sec'],
                'unit': 'samples/sec'
            }
        elif metric == 'memory':
            # Lower is better
            best = min(
                (b for b in model_benchmarks if 'memory' in b),
                key=lambda x: x['memory']['peak_mb']
            )
            return {
                'framework': best['framework'],
                'batch_size': best['batch_size'],
                'value': best['memory']['peak_mb'],
                'unit': 'MB'
            }
        
        return {}
    
    def compare_frameworks_statistical(self, model: str, batch_size: int,
                                      baseline_framework: str = 'pytorch-eager',
                                      alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance testing between frameworks.
        
        Since we only have single runs in the current setup, this generates
        simulated distribution from P95/P99 values to estimate variance.
        
        Args:
            model: Model name
            batch_size: Batch size to analyze
            baseline_framework: Framework to compare against
            alpha: Significance level (default: 0.05)
            
        Returns:
            Dictionary with statistical test results
        """
        results = {}
        
        # Get baseline benchmark
        baseline_bench = next(
            (b for b in self.benchmarks 
             if b['model'] == model and b['batch_size'] == batch_size 
             and b['framework'] == baseline_framework and 'latency' in b),
            None
        )
        
        if not baseline_bench:
            return {}
        
        baseline_median = baseline_bench['latency']['median_ms']
        baseline_p95 = baseline_bench['latency']['p95_ms']
        baseline_p99 = baseline_bench['latency']['p99_ms']
        
        # Estimate standard deviation from percentiles
        # Assuming log-normal distribution (common for latency)
        baseline_std = (baseline_p95 - baseline_median) / 1.645
        
        # Compare against other frameworks
        model_benchmarks = [b for b in self.benchmarks 
                          if b['model'] == model and b['batch_size'] == batch_size 
                          and b['framework'] != baseline_framework and 'latency' in b]
        
        for bench in model_benchmarks:
            framework = bench['framework']
            median = bench['latency']['median_ms']
            p95 = bench['latency']['p95_ms']
            
            # Estimate std for this framework
            framework_std = (p95 - median) / 1.645
            
            # Perform Welch's t-test (unequal variances)
            # Using the t-statistic formula with estimated parameters
            pooled_se = np.sqrt(baseline_std**2 + framework_std**2)
            
            if pooled_se > 0:
                t_stat = (baseline_median - median) / pooled_se
                
                # Degrees of freedom (Welch-Satterthwaite)
                df = ((baseline_std**2 + framework_std**2)**2 / 
                     (baseline_std**4 + framework_std**4))
                
                # Two-tailed p-value
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                speedup = baseline_median / median if median > 0 else float('inf')
                
                results[framework] = {
                    'speedup': speedup,
                    'baseline_median_ms': baseline_median,
                    'framework_median_ms': median,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'effect_size': abs(baseline_median - median) / baseline_std if baseline_std > 0 else 0,
                }
        
        return results


def load_and_analyze(results_path: str) -> BenchmarkAnalyzer:
    """
    Load benchmark results and create analyzer.
    
    Args:
        results_path: Path to results JSON
        
    Returns:
        BenchmarkAnalyzer instance
    """
    return BenchmarkAnalyzer(results_path)


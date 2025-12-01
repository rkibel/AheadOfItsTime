"""
Visualization tools for benchmark results.

Creates plots for:
- Latency comparisons across frameworks
- Throughput comparisons
- Memory consumption
- Compilation time vs amortization
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class BenchmarkVisualizer:
    """Create visualizations from benchmark results."""
    
    def __init__(self, results_path: str, output_dir: str = None):
        """
        Initialize visualizer.
        
        Args:
            results_path: Path to benchmark results JSON
            output_dir: Directory to save plots (default: same as results)
        """
        self.results_path = Path(results_path)
        
        with open(self.results_path, 'r') as f:
            self.data = json.load(f)
        
        self.benchmarks = self.data['benchmarks']
        self.metadata = self.data['metadata']
        
        if output_dir is None:
            output_dir = self.results_path.parent / 'plots'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_latency_comparison(self, model: str = None) -> None:
        """
        Plot latency comparison across frameworks.
        
        Args:
            model: Specific model to plot (None = all models)
        """
        models = [model] if model else list(set(b['model'] for b in self.benchmarks))
        
        for model_name in models:
            model_benchmarks = [b for b in self.benchmarks 
                              if b['model'] == model_name and 'latency' in b]
            
            if not model_benchmarks:
                continue
            
            # Organize data
            frameworks = sorted(set(b['framework'] for b in model_benchmarks))
            batch_sizes = sorted(set(b['batch_size'] for b in model_benchmarks))
            
            fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5*len(batch_sizes), 5))
            if len(batch_sizes) == 1:
                axes = [axes]
            
            for idx, batch_size in enumerate(batch_sizes):
                ax = axes[idx]
                
                # Get data for this batch size
                latencies = []
                labels = []
                
                for framework in frameworks:
                    matching = [b for b in model_benchmarks 
                              if b['framework'] == framework and b['batch_size'] == batch_size]
                    if matching:
                        latencies.append(matching[0]['latency']['median_ms'])
                        labels.append(framework)
                
                # Create bar plot
                x_pos = np.arange(len(labels))
                bars = ax.bar(x_pos, latencies)
                
                # Color baseline differently
                for i, label in enumerate(labels):
                    if label == 'pytorch-eager':
                        bars[i].set_color('gray')
                        bars[i].set_alpha(0.7)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Latency (ms)')
                ax.set_title(f'Batch Size = {batch_size}')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, latencies)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.2f}',
                           ha='center', va='bottom', fontsize=9)
            
            plt.suptitle(f'Inference Latency Comparison - {model_name.upper()}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.output_dir / f'latency_{model_name}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
    
    def plot_throughput_comparison(self, model: str = None) -> None:
        """
        Plot throughput comparison across frameworks.
        
        Args:
            model: Specific model to plot (None = all models)
        """
        models = [model] if model else list(set(b['model'] for b in self.benchmarks))
        
        for model_name in models:
            model_benchmarks = [b for b in self.benchmarks 
                              if b['model'] == model_name and 'throughput' in b]
            
            if not model_benchmarks:
                continue
            
            # Organize data by framework and batch size
            data = defaultdict(dict)
            for b in model_benchmarks:
                framework = b['framework']
                batch_size = b['batch_size']
                throughput = b['throughput']['samples_per_sec']
                data[framework][batch_size] = throughput
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            batch_sizes = sorted(set(b['batch_size'] for b in model_benchmarks))
            frameworks = sorted(data.keys())
            
            x = np.arange(len(batch_sizes))
            width = 0.8 / len(frameworks)
            
            for i, framework in enumerate(frameworks):
                throughputs = [data[framework].get(bs, 0) for bs in batch_sizes]
                offset = (i - len(frameworks)/2 + 0.5) * width
                bars = ax.bar(x + offset, throughputs, width, label=framework)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (samples/sec)')
            ax.set_title(f'Throughput Comparison - {model_name.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(batch_sizes)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            output_file = self.output_dir / f'throughput_{model_name}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
    
    def plot_memory_comparison(self, model: str = None) -> None:
        """
        Plot memory consumption across frameworks.
        
        Args:
            model: Specific model to plot (None = all models)
        """
        models = [model] if model else list(set(b['model'] for b in self.benchmarks))
        
        for model_name in models:
            model_benchmarks = [b for b in self.benchmarks 
                              if b['model'] == model_name and 'memory' in b]
            
            if not model_benchmarks:
                continue
            
            # Organize data
            data = defaultdict(dict)
            for b in model_benchmarks:
                framework = b['framework']
                batch_size = b['batch_size']
                peak_memory = b['memory']['peak_mb']
                data[framework][batch_size] = peak_memory
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            batch_sizes = sorted(set(b['batch_size'] for b in model_benchmarks))
            frameworks = sorted(data.keys())
            
            # Use different marker styles and line styles for better distinction
            markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
            linestyles = ['-', '--', '-.', ':']
            
            for idx, framework in enumerate(frameworks):
                memory_values = [data[framework].get(bs, 0) for bs in batch_sizes]
                marker = markers[idx % len(markers)]
                linestyle = linestyles[idx % len(linestyles)]
                ax.plot(batch_sizes, memory_values, marker=marker, linestyle=linestyle, 
                       label=framework, linewidth=2, markersize=8)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Peak Memory (MB)')
            ax.set_title(f'Memory Consumption - {model_name.upper()}')
            ax.legend(loc='best')
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            
            output_file = self.output_dir / f'memory_{model_name}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()

    def plot_energy_efficiency(self, model: str = None) -> None:
        """
        Plot energy efficiency comparison across frameworks.
        
        Args:
            model: Specific model to plot (None = all models)
        """
        models = [model] if model else list(set(b['model'] for b in self.benchmarks))
        
        for model_name in models:
            model_benchmarks = [b for b in self.benchmarks 
                              if b['model'] == model_name and 'energy' in b]
            
            if not model_benchmarks:
                continue
            
            # Organize data by framework and batch size
            data = defaultdict(dict)
            for b in model_benchmarks:
                framework = b['framework']
                batch_size = b['batch_size']
                efficiency = b['energy']['inferences_per_joule']
                data[framework][batch_size] = efficiency
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            batch_sizes = sorted(set(b['batch_size'] for b in model_benchmarks))
            frameworks = sorted(data.keys())
            
            x = np.arange(len(batch_sizes))
            width = 0.8 / len(frameworks)
            
            for i, framework in enumerate(frameworks):
                efficiencies = [data[framework].get(bs, 0) for bs in batch_sizes]
                offset = (i - len(frameworks)/2 + 0.5) * width
                bars = ax.bar(x + offset, efficiencies, width, label=framework)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Efficiency (Inferences/Joule)')
            ax.set_title(f'Energy Efficiency - {model_name.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(batch_sizes)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            output_file = self.output_dir / f'energy_{model_name}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
    
    def plot_speedup_heatmap(self, baseline: str = 'pytorch-eager') -> None:
        """
        Create heatmap of speedups relative to baseline.
        
        Args:
            baseline: Baseline framework for comparison
        """
        # Collect speedup data
        models = sorted(set(b['model'] for b in self.benchmarks))
        frameworks = sorted(set(b['framework'] for b in self.benchmarks 
                               if b['framework'] != baseline))
        
        if not frameworks:
            return
        
        # Use batch_size=1 for comparison
        batch_size = 1
        
        # Build speedup matrix
        speedup_matrix = np.zeros((len(models), len(frameworks)))
        
        for i, model in enumerate(models):
            # Get baseline latency
            baseline_bench = [b for b in self.benchmarks 
                            if b['model'] == model and b['framework'] == baseline 
                            and b['batch_size'] == batch_size and 'latency' in b]
            
            if not baseline_bench:
                continue
            
            baseline_latency = baseline_bench[0]['latency']['median_ms']
            
            for j, framework in enumerate(frameworks):
                # Get framework latency
                framework_bench = [b for b in self.benchmarks 
                                 if b['model'] == model and b['framework'] == framework 
                                 and b['batch_size'] == batch_size and 'latency' in b]
                
                if framework_bench:
                    framework_latency = framework_bench[0]['latency']['median_ms']
                    speedup = baseline_latency / framework_latency
                    speedup_matrix[i, j] = speedup
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(len(frameworks)*1.5, len(models)*1.2))
        
        im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.0)
        
        # Set ticks
        ax.set_xticks(np.arange(len(frameworks)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(frameworks, rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Speedup', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(frameworks)):
                value = speedup_matrix[i, j]
                if value > 0:
                    text = ax.text(j, i, f'{value:.2f}x',
                                 ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title(f'Speedup Heatmap (Batch Size = {batch_size}, Baseline = {baseline})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'speedup_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_batch_summary_heatmaps(self, batch_size: int) -> None:
        """
        Create heatmaps for throughput, energy, memory, and speedup at a specific batch size.
        
        Args:
            batch_size: Batch size to visualize
        """
        models = sorted(set(b['model'] for b in self.benchmarks))
        frameworks = sorted(set(b['framework'] for b in self.benchmarks))
        baseline = 'pytorch-eager'
        
        # Metrics to plot
        metrics = [
            ('throughput', 'samples_per_sec', 'Throughput (samples/sec)', 'Greens'),
            ('energy', 'inferences_per_joule', 'Energy Efficiency (inf/J)', 'Greens'),
            ('memory', 'peak_mb', 'Peak Memory (MB)', 'Reds_r')
        ]
        
        # Plot standard metrics
        for metric_key, value_key, title, colormap in metrics:
            # Build data matrix
            data_matrix = np.zeros((len(models), len(frameworks)))
            
            for i, model in enumerate(models):
                for j, framework in enumerate(frameworks):
                    bench = [b for b in self.benchmarks 
                            if b['model'] == model and b['framework'] == framework 
                            and b['batch_size'] == batch_size and metric_key in b]
                    
                    if bench:
                        data_matrix[i, j] = bench[0][metric_key][value_key]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(len(frameworks)*1.5, len(models)*1.2))
            
            im = ax.imshow(data_matrix, cmap=colormap, aspect='auto')
            
            # Set ticks
            ax.set_xticks(np.arange(len(frameworks)))
            ax.set_yticks(np.arange(len(models)))
            ax.set_xticklabels(frameworks, rotation=45, ha='right')
            ax.set_yticklabels(models)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(title, rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(frameworks)):
                    value = data_matrix[i, j]
                    if value > 0:
                        # Format based on metric
                        if metric_key == 'memory':
                            text_val = f'{value:.0f}'
                        elif metric_key == 'energy':
                            text_val = f'{value:.1f}'
                        else:
                            text_val = f'{value:.0f}'
                        ax.text(j, i, text_val,
                               ha="center", va="center", color="black", fontsize=9)
            
            ax.set_title(f'{title} (Batch Size = {batch_size})',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.output_dir / f'{metric_key}_batch{batch_size}_heatmap.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
        
        # Plot speedup heatmap for this batch size
        frameworks_no_baseline = sorted(set(b['framework'] for b in self.benchmarks 
                                           if b['framework'] != baseline))
        
        if frameworks_no_baseline:
            speedup_matrix = np.zeros((len(models), len(frameworks_no_baseline)))
            
            for i, model in enumerate(models):
                # Get baseline latency
                baseline_bench = [b for b in self.benchmarks 
                                if b['model'] == model and b['framework'] == baseline 
                                and b['batch_size'] == batch_size and 'latency' in b]
                
                if baseline_bench:
                    baseline_latency = baseline_bench[0]['latency']['median_ms']
                    
                    for j, framework in enumerate(frameworks_no_baseline):
                        # Get framework latency
                        framework_bench = [b for b in self.benchmarks 
                                         if b['model'] == model and b['framework'] == framework 
                                         and b['batch_size'] == batch_size and 'latency' in b]
                        
                        if framework_bench:
                            framework_latency = framework_bench[0]['latency']['median_ms']
                            speedup = baseline_latency / framework_latency
                            speedup_matrix[i, j] = speedup
            
            # Create speedup heatmap
            fig, ax = plt.subplots(figsize=(len(frameworks_no_baseline)*1.5, len(models)*1.2))
            
            im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.0)
            
            # Set ticks
            ax.set_xticks(np.arange(len(frameworks_no_baseline)))
            ax.set_yticks(np.arange(len(models)))
            ax.set_xticklabels(frameworks_no_baseline, rotation=45, ha='right')
            ax.set_yticklabels(models)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Speedup', rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(frameworks_no_baseline)):
                    value = speedup_matrix[i, j]
                    if value > 0:
                        ax.text(j, i, f'{value:.2f}x',
                               ha="center", va="center", color="black", fontsize=10)
            
            ax.set_title(f'Speedup Heatmap (Batch Size = {batch_size}, Baseline = {baseline})',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.output_dir / f'speedup_batch{batch_size}_heatmap.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
    
    def generate_all_plots(self) -> None:
        """Generate all visualizations."""
        print(f"\nGenerating visualizations...")
        print(f"Output directory: {self.output_dir}")
        print()
        
        models = list(set(b['model'] for b in self.benchmarks))
        
        for model in models:
            print(f"Creating plots for {model}...")
            self.plot_latency_comparison(model)
            self.plot_throughput_comparison(model)
            self.plot_memory_comparison(model)
            self.plot_energy_efficiency(model)
        
        print("\nCreating speedup heatmap...")
        self.plot_speedup_heatmap()
        
        print("\nCreating batch size 1 summary heatmaps...")
        self.plot_batch_summary_heatmaps(batch_size=1)
        
        print("\nCreating batch size 128 summary heatmaps...")
        self.plot_batch_summary_heatmaps(batch_size=128)
        
        print(f"\n✓ All visualizations saved to {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations from benchmark results'
    )
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to benchmark results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for plots (default: results_dir/plots)'
    )
    
    args = parser.parse_args()
    
    visualizer = BenchmarkVisualizer(args.results, args.output)
    visualizer.generate_all_plots()


if __name__ == '__main__':
    main()


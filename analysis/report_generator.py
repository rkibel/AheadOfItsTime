"""
Automated report generation from benchmark results.

Creates comprehensive markdown reports with:
- Summary statistics
- Performance comparisons
- Embedded plots
- Recommendations
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .benchmark_stats import BenchmarkAnalyzer


class ReportGenerator:
    """Generate markdown reports from benchmark results."""
    
    def __init__(self, results_path: str, output_path: str = None):
        """
        Initialize report generator.
        
        Args:
            results_path: Path to benchmark results JSON
            output_path: Path to save report (default: results_dir/report.md)
        """
        self.results_path = Path(results_path)
        self.analyzer = BenchmarkAnalyzer(str(results_path))
        
        if output_path is None:
            output_path = self.results_path.parent / 'report.md'
        self.output_path = Path(output_path)
        
    def generate(self) -> str:
        """
        Generate complete report.
        
        Returns:
            Report content as string
        """
        sections = [
            self._generate_header(),
            self._generate_metadata(),
            self._generate_summary(),
            self._generate_detailed_results(),
            self._generate_speedup_analysis(),
            self._generate_memory_analysis(),
            self._generate_compilation_analysis(),
            self._generate_recommendations(),
            self._generate_conclusion(),
        ]
        
        report = "\n\n".join(sections)
        
        # Save report
        with open(self.output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {self.output_path}")
        
        return report
    
    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# Benchmark Report: {self.analyzer.metadata.get('config_name', 'Benchmark')}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
"""
    
    def _generate_metadata(self) -> str:
        """Generate metadata section."""
        meta = self.analyzer.metadata
        
        lines = [
            "## System Configuration",
            "",
            f"- **Device:** {meta.get('device', 'N/A')}",
        ]
        
        if 'gpu' in meta:
            lines.append(f"- **GPU:** {meta['gpu']}")
            lines.append(f"- **CUDA Version:** {meta.get('cuda_version', 'N/A')}")
        
        lines.append("\n### Framework Versions")
        lines.append("")
        for framework, version in meta.get('frameworks', {}).items():
            lines.append(f"- **{framework}:** {version}")
        
        return "\n".join(lines)
    
    def _generate_summary(self) -> str:
        """Generate summary section."""
        models = sorted(set(b['model'] for b in self.analyzer.benchmarks))
        frameworks = sorted(set(b['framework'] for b in self.analyzer.benchmarks))
        batch_sizes = sorted(set(b['batch_size'] for b in self.analyzer.benchmarks))
        
        return f"""## Benchmark Summary

- **Models Tested:** {', '.join(models)}
- **Frameworks:** {', '.join(frameworks)}
- **Batch Sizes:** {', '.join(map(str, batch_sizes))}
- **Total Configurations:** {len(self.analyzer.benchmarks)}

### Description

{self.analyzer.metadata.get('config_description', 'N/A')}
"""
    
    def _generate_detailed_results(self) -> str:
        """Generate detailed results table."""
        return f"""## Detailed Results

```
{self.analyzer.generate_summary_table()}
```
"""
    
    def _generate_speedup_analysis(self) -> str:
        """Generate speedup analysis section."""
        speedups = self.analyzer.compute_speedups()
        
        lines = [
            "## Speedup Analysis",
            "",
            "Speedup relative to PyTorch eager mode baseline:",
            ""
        ]
        
        for model, frameworks in speedups.items():
            lines.append(f"### {model.upper()}")
            lines.append("")
            
            for framework, batch_data in frameworks.items():
                lines.append(f"#### {framework}")
                lines.append("")
                lines.append("| Batch Size | Speedup | Baseline (ms) | Optimized (ms) |")
                lines.append("|-----------|---------|---------------|----------------|")
                
                for batch_size, data in sorted(batch_data.items()):
                    speedup = data['speedup']
                    baseline = data['baseline_ms']
                    optimized = data['optimized_ms']
                    lines.append(f"| {batch_size} | {speedup:.2f}x | {baseline:.3f} | {optimized:.3f} |")
                
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_memory_analysis(self) -> str:
        """Generate memory analysis section."""
        memory_stats = self.analyzer.compute_memory_efficiency()
        
        lines = [
            "## Memory Analysis",
            "",
            "Peak GPU memory consumption:",
            ""
        ]
        
        for model, frameworks in memory_stats.items():
            lines.append(f"### {model.upper()}")
            lines.append("")
            lines.append("| Framework | Batch Size | Peak (MB) | Average (MB) | Per Sample (MB) |")
            lines.append("|-----------|-----------|-----------|--------------|-----------------|")
            
            for framework, batch_data in frameworks.items():
                for batch_size, data in sorted(batch_data.items()):
                    peak = data['peak_mb']
                    avg = data['average_mb']
                    per_sample = data['per_sample_mb']
                    lines.append(f"| {framework} | {batch_size} | {peak:.1f} | {avg:.1f} | {per_sample:.2f} |")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_compilation_analysis(self) -> str:
        """Generate compilation overhead analysis."""
        lines = [
            "## Compilation Overhead Analysis",
            "",
            "Analysis of compile-time costs and amortization:",
            ""
        ]
        
        # Group by model and framework
        compilation_data = {}
        for benchmark in self.analyzer.benchmarks:
            if 'compilation' not in benchmark:
                continue
            
            model = benchmark['model']
            framework = benchmark['framework']
            comp = benchmark['compilation']
            
            if model not in compilation_data:
                compilation_data[model] = {}
            if framework not in compilation_data[model]:
                compilation_data[model][framework] = []
            
            compilation_data[model][framework].append({
                'batch_size': benchmark['batch_size'],
                'compilation_time_ms': comp['compilation_time_ms'],
                'amortization': comp.get('amortization', {})
            })
        
        for model, frameworks in compilation_data.items():
            lines.append(f"### {model.upper()}")
            lines.append("")
            
            for framework, data_list in frameworks.items():
                lines.append(f"#### {framework}")
                lines.append("")
                
                # Show first entry (batch_size=1 typically)
                if data_list:
                    data = data_list[0]
                    comp_time = data['compilation_time_ms']
                    lines.append(f"- **Compilation Time:** {comp_time:.2f} ms")
                    
                    if 'amortization' in data and data['amortization']:
                        amort = data['amortization']
                        if amort.get('is_beneficial', False):
                            speedup = amort.get('speedup', 1.0)
                            amort_samples = amort.get('amortization_samples', 0)
                            lines.append(f"- **Speedup:** {speedup:.2f}x")
                            lines.append(f"- **Break-even Point:** {amort_samples:.0f} inferences")
                        else:
                            lines.append(f"- **Note:** No performance benefit over baseline")
                    
                    lines.append("")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on results."""
        lines = [
            "## Recommendations",
            "",
            "Based on the benchmark results:",
            ""
        ]
        
        models = set(b['model'] for b in self.analyzer.benchmarks)
        
        for model in sorted(models):
            lines.append(f"### {model.upper()}")
            lines.append("")
            
            # Find best for latency
            best_latency = self.analyzer.get_best_framework(model, 'latency')
            if best_latency:
                lines.append(f"- **Best for Latency:** {best_latency['framework']} "
                           f"({best_latency['value']:.3f} {best_latency['unit']}, "
                           f"batch_size={best_latency['batch_size']})")
            
            # Find best for throughput
            best_throughput = self.analyzer.get_best_framework(model, 'throughput')
            if best_throughput:
                lines.append(f"- **Best for Throughput:** {best_throughput['framework']} "
                           f"({best_throughput['value']:.1f} {best_throughput['unit']}, "
                           f"batch_size={best_throughput['batch_size']})")
            
            # Find best for memory
            best_memory = self.analyzer.get_best_framework(model, 'memory')
            if best_memory:
                lines.append(f"- **Best for Memory:** {best_memory['framework']} "
                           f"({best_memory['value']:.1f} {best_memory['unit']}, "
                           f"batch_size={best_memory['batch_size']})")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_conclusion(self) -> str:
        """Generate conclusion section."""
        return """## Conclusion

This benchmark provides insights into the tradeoffs between AOT and JIT compilation strategies across different deep learning frameworks. Key findings include:

1. **Compilation Overhead:** AOT-compiled models (ONNX, TensorRT) have upfront compilation costs that must be amortized over multiple inferences.

2. **Inference Performance:** Different frameworks show varying levels of optimization depending on model architecture.

3. **Memory Efficiency:** Framework choice impacts GPU memory consumption, particularly for larger batch sizes.

4. **Flexibility vs Performance:** Dynamic frameworks (PyTorch eager) offer maximum flexibility but may sacrifice performance compared to optimized static graphs.

For production deployments with stable model architectures and high inference volumes, AOT compilation strategies generally provide better performance. For research and development where flexibility is paramount, dynamic execution remains advantageous.

---

*Report generated automatically from benchmark results*
"""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate markdown report from benchmark results'
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
        help='Output path for report (default: results_dir/report.md)'
    )
    
    args = parser.parse_args()
    
    generator = ReportGenerator(args.results, args.output)
    generator.generate()


if __name__ == '__main__':
    main()


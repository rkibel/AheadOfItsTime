"""
Benchmarking infrastructure for comparing AOT vs JIT compilation strategies.

This package provides tools to measure:
- Inference latency (median, P95, P99)
- Memory consumption (peak, average)
- Compile-time overhead
- Throughput across different batch sizes

Supports multiple frameworks:
- PyTorch eager mode
- TorchScript (JIT)
- PyTorch 2.0 torch.compile
- ONNX Runtime
"""

__version__ = "0.1.0"


# Benchmarking Harness

Comprehensive benchmarking infrastructure for comparing AOT vs JIT compilation strategies across deep learning frameworks.

## Overview

This benchmarking harness measures:
- **Inference Latency**: Median, P95, P99 percentiles
- **Memory Consumption**: Peak and average GPU memory usage
- **Compile-time Overhead**: Time to compile/optimize models
- **Throughput**: Samples per second at different batch sizes

## Supported Frameworks

1. **PyTorch Eager**: Baseline dynamic execution
2. **TorchScript**: JIT-compiled PyTorch models
3. **PyTorch 2.0 Compile**: torch.compile with inductor backend
4. **ONNX Runtime**: AOT-compiled ONNX models with graph optimization
5. **TensorRT**: NVIDIA TensorRT optimized engines (aggressive AOT)

## Quick Start

### 1. Run Quick Test

Test the pipeline with a small configuration:

```bash
python benchmarking/runner.py --config benchmarking/configs/quick_test.yaml
```

### 2. Run Full Benchmark

Execute comprehensive benchmarks:

```bash
python benchmarking/runner.py --config benchmarking/configs/full_benchmark.yaml
```

### 3. Generate Visualizations

Create plots from results:

```bash
python analysis/visualize.py --results benchmarking/results/quick_test/benchmark_results_latest.json
```

### 4. Generate Report

Create markdown report:

```bash
python analysis/report_generator.py --results benchmarking/results/quick_test/benchmark_results_latest.json
```

## Configuration

Benchmarks are configured using YAML files in `benchmarking/configs/`. Example:

```yaml
name: "My Benchmark"
models:
  - lenet
  - resnet18
frameworks:
  - pytorch-eager
  - torchscript
  - onnx
batch_sizes: [1, 8, 32, 128]
num_iterations: 1000
device: cuda
```

## Notes

- **CUDA Synchronization**: All timing uses proper CUDA synchronization for accurate GPU measurements
- **Warmup**: 100 warmup iterations before measurement to stabilize GPU clocks
- **Memory Reset**: Memory statistics are reset between measurements
- **Error Handling**: Unsupported model-framework combinations are gracefully skipped



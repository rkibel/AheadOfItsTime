# AheadOfItsTime

**Exploring AOT vs JIT Compilation Strategies in Deep Learning Frameworks**

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Background](#background)
   - [Static Computation Graphs](#static-computation-graphs)
   - [Dynamic Computation Graphs](#dynamic-computation-graphs)
   - [Ahead-of-Time (AOT) Compilation](#ahead-of-time-aot-compilation)
   - [Just-in-Time (JIT) Compilation](#just-in-time-jit-compilation)
   - [Hybrid Approaches](#hybrid-approaches)
4. [Project Objectives](#project-objectives)
5. [Technical Approach](#technical-approach)
6. [Architecture](#architecture)
7. [Timeline](#timeline)
8. [Usage](#usage)
9. [Results &amp; Analysis](#results--analysis)
10. [Future Work](#future-work)
11. [Contributors](#contributors)

---

## Overview

**AheadOfItsTime** is a benchmarking and analysis project that investigates the performance and flexibility tradeoffs between Ahead-of-Time (AOT) and Just-in-Time (JIT) compilation strategies in modern deep learning frameworks. Through systematic experimentation with CNNs on computer vision datasets (MNIST, CIFAR-10) and RNNs on sequential/NLP datasets (IMDB, WikiText-2), we quantify how different compilation approaches impact:

- **Inference latency** (end-to-end prediction time)
- **Memory consumption** (peak and average usage)
- **Compile-time overhead** (graph optimization and compilation cost)
- **Flexibility constraints** (dynamic input shapes, control flow)

This project addresses a critical concern in production ML systems: balancing runtime efficiency with development flexibility as frameworks increasingly adopt hybrid compilation strategies.

---

## Motivation

The deep learning ecosystem is witnessing a convergence of traditionally distinct approaches:

- **Static frameworks** (TensorFlow 1.x, ONNX Runtime, TensorRT) offer superior performance through aggressive graph-level optimizations but sacrifice runtime adaptability
- **Dynamic frameworks** (PyTorch eager mode, TensorFlow 2.x eager) provide intuitive debugging and flexible control flow but face performance penalties
- **Hybrid approaches** (PyTorch 2.0 with `torch.compile`, TensorFlow's AutoGraph, JAX) attempt to bridge this gap

Understanding these tradeoffs is essential for:

1. **ML Engineers** deploying models in production environments with strict latency/throughput requirements
2. **Researchers** iterating on novel architectures requiring dynamic behavior
3. **Framework developers** designing next-generation compilation systems
4. **Organizations** making informed technology stack decisions

Despite the importance of this topic, there exists a lack of comprehensive, reproducible benchmarks that systematically compare these approaches across realistic workloads.

---

## Background

### Static Computation Graphs

- **Definition**: The complete computational graph is defined before execution (graph definition phase separate from execution phase)
- **Examples**: TensorFlow 1.x Sessions, ONNX models, TensorRT engines
- **Advantages**:
  - Whole-graph optimization (operator fusion, memory planning, constant folding)
  - Efficient deployment on specialized hardware (GPUs, TPUs, edge devices)
  - Predictable memory usage
- **Disadvantages**:
  - Fixed input dimensions (or limited dynamic shapes)
  - Difficult to implement data-dependent control flow
  - Harder debugging (computation separated from definition)

### Dynamic Computation Graphs

- **Definition**: The graph is constructed on-the-fly during execution (define-by-run)
- **Examples**: PyTorch eager mode, TensorFlow 2.x eager execution
- **Advantages**:
  - Pythonic control flow (if/else, loops with dynamic conditions)
  - Easy debugging with standard tools
  - Natural handling of variable-length sequences
- **Disadvantages**:
  - Overhead from Python interpreter and dispatch logic
  - Limited scope for cross-operation optimization
  - Higher memory fragmentation

### Ahead-of-Time (AOT) Compilation

- **Process**: Model is fully compiled before any inference occurs
- **Characteristics**:
  - One-time upfront compilation cost
  - Generates optimized binary/intermediate representation
  - No runtime compilation overhead
  - Examples: ONNX Runtime with graph optimization, TensorRT engine building
- **Use Cases**: Production serving with fixed model architecture, embedded/edge deployment

### Just-in-Time (JIT) Compilation

- **Process**: Code is compiled during runtime, often with progressive optimization
- **Characteristics**:
  - Lazy compilation (compile as needed)
  - Can adapt to runtime information (actual tensor shapes, data types)
  - Amortized compilation cost over multiple runs
  - Examples: PyTorch TorchScript JIT, JAX JIT compilation, TensorFlow AutoGraph
- **Use Cases**: Research environments, dynamic models, warm-up acceptable scenarios

### Hybrid Approaches

Modern frameworks increasingly blur these boundaries:

- **PyTorch 2.0's `torch.compile`**: Uses TorchDynamo to capture graphs and TorchInductor for AOT compilation, with JIT fallback
- **TensorFlow AutoGraph**: Converts Python control flow to graph operations
- **JAX**: Traces functions into XLA HLO, compiles AOT, but with easy recompilation

---

## Project Objectives

### Primary Goals

1. **Quantify Performance Tradeoffs**

   - Measure inference latency across different compilation strategies
   - Profile memory consumption patterns
   - Analyze compile-time overhead and its amortization
2. **Evaluate Flexibility Constraints**

   - Test dynamic input shape handling
   - Assess control flow implementation complexity
   - Document API usability differences
3. **Provide Reproducible Benchmarks**

   - Create standardized benchmarking harness
   - Document experimental methodology
   - Enable community validation and extension

---

## Technical Approach

### Frameworks & Runtimes Under Evaluation

1. **PyTorch Ecosystem**
   - PyTorch eager mode (baseline dynamic)
   - TorchScript (JIT and traced)
   - PyTorch 2.0 `torch.compile` (hybrid)

2. **ONNX Runtime**
   - Standard ONNX Runtime (AOT with graph optimization)
   - ONNX Runtime with different execution providers (CPU, CUDA)

3. **TensorRT**
   - TensorRT optimized engines (aggressive AOT)

4. **TensorFlow Ecosystem** (stretch goal)
   - TensorFlow 2.x eager (baseline dynamic)
   - TensorFlow SavedModel + graph optimization (AOT)
   - TensorFlow Lite (AOT for mobile/edge)

### Model Architectures

#### CNN Benchmarks

1. **LeNet-5** (MNIST)
   - Simple architecture: Conv → Pool → Conv → Pool → FC
   - ~60K parameters
   - Tests basic convolutional optimization

2. **ResNet-18** (CIFAR-10)
   - Modern architecture with skip connections
   - ~11M parameters
   - Tests operator fusion, memory optimization

#### RNN Benchmarks

1. **LSTM Sentiment Classifier** (IMDB Reviews)
   - Binary sentiment classification (positive/negative reviews)
   - Bidirectional LSTM with embedding layer
   - Variable sequence lengths (up to 512 tokens)
   - ~2M parameters
   - Tests recurrent operator optimization and variable-length handling
   - Highlights dynamic unrolling challenges

2. **GRU Language Model** (WikiText-2)
   - Word-level language modeling on Wikipedia text
   - Stacked GRU architecture (2-3 layers)
   - ~5M parameters
   - Tests sequential dependencies and long-range context modeling
   - Exposes compilation overhead for multi-layer RNNs

### Datasets

**CNN Datasets:**
- **MNIST**: 60K training, 10K test images (28x28 grayscale)
- **CIFAR-10**: 50K training, 10K test images (32x32 RGB)

**RNN Datasets:**
- **IMDB Reviews**: 25K training, 25K test movie reviews (sentiment classification)
- **WikiText-2**: Wikipedia text corpus for language modeling (~2M tokens total, with train/validation/test splits)

All datasets are well-established benchmarks, eliminating confounds from data complexity while representing canonical use cases for each architecture type.

### Metrics

#### Performance Metrics

1. **Inference Latency**
   - Median, P95, P99 latency over 1000 inferences
   - Batch sizes: 1, 8, 32, 128
   - Cold start vs warm inference

2. **Throughput**
   - Images/second (CNNs) or sequences/second (RNNs) for different batch sizes
   - GPU utilization percentage

3. **Memory Usage**
   - Peak memory allocation
   - Average memory consumption
   - Memory fragmentation (via profiler)

4. **Compilation Overhead**
   - Time to compile/optimize model
   - First inference time (cold start)
   - Amortization point (# inferences to break even)

#### Flexibility Metrics

1. **Dynamic Shape Support**
   - Can the model handle variable batch sizes without recompilation?
   - Performance impact of dynamic shapes

2. **Control Flow**
   - Ease of implementing conditional branches
   - Performance cost of dynamic control flow

### Experimental Setup

- **Hardware**: Document CPU/GPU specifications
- **Trials**: 5 runs per configuration with different random seeds
- **Warm-up**: 100 inferences before measurement to stabilize GPU clocks
- **Monitoring**: NVIDIA nsys/nvprof for GPU profiling, `memory_profiler` for Python memory

---

## Architecture

### Project Structure

```
AheadOfItsTime/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── environment.yml             # Conda environment (optional)
│
├── data/                       # Dataset storage
│   ├── mnist/
│   ├── cifar10/
│   ├── imdb/
│   ├── wikitext/              # WikiText-2
│   └── download_datasets.py
│
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── cnn/
│   │   ├── lenet.py           # LeNet-5 implementation
│   │   └── resnet.py          # ResNet-18 implementation
│   ├── rnn/
│   │   ├── lstm_sentiment.py  # LSTM sentiment classifier
│   │   └── gru_lm.py          # GRU language model
│   └── utils.py               # Shared utilities
│
├── training/                   # Training scripts
│   ├── train_mnist_lenet.py
│   ├── train_cifar_resnet.py
│   ├── train_imdb_lstm.py
│   ├── train_wikitext_gru.py
│   └── config/                # Training configurations
│
├── conversion/                 # Framework conversion scripts
│   ├── to_torchscript.py
│   ├── to_onnx.py
│   ├── to_tensorrt.py
│   ├── to_tensorflow.py       # Stretch goal
│   └── utils.py
│
├── benchmarking/              # Benchmarking infrastructure
│   ├── __init__.py
│   ├── runner.py              # Main benchmark runner
│   ├── profilers/
│   │   ├── latency.py
│   │   ├── memory.py
│   │   └── compilation.py
│   ├── configs/               # Benchmark configurations
│   └── results/               # Benchmark outputs
│
├── analysis/                  # Analysis and visualization
│   ├── visualize.py           # Plotting scripts
│   ├── statistics.py          # Statistical analysis
│   ├── report_generator.py    # Automated report creation
│   └── notebooks/             # Jupyter notebooks for exploration
│
├── experiments/               # Experiment definitions
│   ├── dynamic_shapes.py
│   ├── control_flow.py
│   └── configs/
│
├── checkpoints/               # Saved model weights
│   ├── pytorch/
│   ├── onnx/
│   ├── tensorrt/
│   └── tensorflow/
│
├── results/                   # Experimental results
│   ├── benchmarks/            # Raw benchmark data
│   ├── plots/                 # Generated visualizations
│   └── reports/               # Analysis documents
│
├── tests/                     # Unit and integration tests
│   ├── test_models.py
│   ├── test_conversion.py
│   └── test_benchmarking.py
│
└── docs/                      # Additional documentation
    ├── METHODOLOGY.md         # Detailed experimental methodology
    ├── INSTALLATION.md        # Setup instructions
    ├── API.md                 # Code API documentation
    ├── FINDINGS.md            # Results and analysis
    ├── resources.md           # Additional resources
```

### Key Components

#### 1. Model Zoo (`models/`)
- Clean, documented implementations of each architecture
- Framework-agnostic design (define once, convert to others)
- Consistent API for inference across all models

#### 2. Conversion Pipeline (`conversion/`)
- Automated scripts to convert PyTorch models to other formats
- Validation that converted models produce identical outputs
- Optimization flags and configuration templates

#### 3. Benchmarking Harness (`benchmarking/`)
- Pluggable profiler architecture
- Unified interface for different frameworks
- Structured output format (JSON schema for results)

#### 4. Analysis Suite (`analysis/`)
- Automated visualization generation
- Statistical significance testing
- Comparative analysis tools

---

## Timeline

### Week 1: Foundation
- [x] Setup, data pipeline (CNNs + RNNs), LeNet/ResNet in PyTorch
- [x] LSTM sentiment & GRU language models, text preprocessing
- [x] Add background resources documentation
- [x] Train all models, validate accuracy, checkpoint saving

### Week 2: Multi-Framework
- PyTorch Ecosystem
  - [x] TorchScript (JIT and traced)
  - [ ] PyTorch 2.0 `torch.compile` (hybrid)
- ONNX Runtime
  - [x] Standard ONNX Runtime (AOT with graph optimization)
  - [x] ONNX Runtime with different execution providers (CPU, CUDA)
- TensorRT
  - [ ] TensorRT optimized engines (aggressive AOT)
- TensorFlow Ecosystem (stretch goal)
  - [ ] TensorFlow SavedModel + graph optimization (AOT)
  - [ ] TensorFlow Lite (AOT for mobile/edge)

### Week 3: Benchmarking
- Build benchmarking harness
- Profiling integration, dynamic shape experiments
- Run comprehensive benchmarks

### Week 4: Analysis
- Data analysis and visualization
- Report writing, findings documentation
- Presentation prep, final polish

---

## Usage

### Training Models

```bash
# Train LeNet on MNIST
python training/train_mnist_lenet.py \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --save-path checkpoints/pytorch/lenet_mnist.pth

# Train ResNet-18 on CIFAR-10
python training/train_cifar_resnet.py \
    --epochs 50 \
    --batch-size 256 \
    --augmentation \
    --save-path checkpoints/pytorch/resnet18_cifar10.pth

# Train LSTM on IMDB sentiment
python training/train_imdb_lstm.py \
    --epochs 20 \
    --batch-size 64 \
    --embedding-dim 128 \
    --hidden-dim 256 \
    --save-path checkpoints/pytorch/lstm_imdb.pth

# Train GRU language model on WikiText-2
python training/train_wikitext_gru.py \
    --epochs 40 \
    --batch-size 32 \
    --embedding-dim 200 \
    --hidden-dim 200 \
    --save-path checkpoints/pytorch/gru_wikitext.pth
```

### Converting Models

```bash
# Convert to TorchScript
python conversion/to_torchscript.py \
    --model lenet \
    --checkpoint checkpoints/pytorch/lenet_mnist.pth \
    --mode trace \
    --output checkpoints/torchscript/lenet_traced.pt

# Convert to ONNX
python conversion/to_onnx.py \
    --model resnet18 \
    --checkpoint checkpoints/pytorch/resnet18_cifar10.pth \
    --opset 13 \
    --output checkpoints/onnx/resnet18.onnx

# Convert to TensorFlow
python conversion/to_tensorflow.py \
    --model lenet \
    --checkpoint checkpoints/pytorch/lenet_mnist.pth \
    --output checkpoints/tensorflow/lenet_savedmodel/
```

### Running Benchmarks

```bash
# Run comprehensive benchmark suite
python benchmarking/runner.py \
    --config benchmarking/configs/full_benchmark.yaml \
    --output results/benchmarks/run_001/

# Quick benchmark (single model, single framework)
python benchmarking/runner.py \
    --model lenet \
    --framework pytorch-eager \
    --batch-sizes 1,8,32 \
    --iterations 1000

# Profile memory usage
python benchmarking/runner.py \
    --config benchmarking/configs/memory_profile.yaml \
    --profile-memory

# Dynamic shape experiment
python experiments/dynamic_shapes.py \
    --model resnet18 \
    --frameworks pytorch-eager,torchscript,onnx
```

### Generating Reports

```bash
# Create visualizations
python analysis/visualize.py \
    --results results/benchmarks/ \
    --output results/plots/

# Generate comprehensive report
python analysis/report_generator.py \
    --results results/benchmarks/ \
    --output results/reports/final_report.pdf
```

---

## Results & Analysis

*(This section will be populated as experiments complete)*

---

## Future Work

---

## Contributors

- **Ron Kibel**
- **Shane Dirksen**

---

MIT License - see [LICENSE](LICENSE) file for details.
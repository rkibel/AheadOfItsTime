# Resources

## Compilation Fundamentals

### PyTorch 2.x: torch.compile intro
https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

Shows how Dynamo captures graphs and how Inductor compiles them. Useful for understanding hybrid compilation approaches.

### PyTorch 2.x compiler reference
https://docs.pytorch.org/docs/stable/torch.compiler.html

Reference documentation for understanding what gets compiled in PyTorch 2.x.

### TensorFlow: using tf.function and graphs
https://www.tensorflow.org/guide/function

Demonstrates the "dynamic-by-default, compile when asked" approach in TensorFlow.

### JAX JIT with XLA
https://docs.jax.dev/en/latest/jit-compilation.html

Clear JIT compilation example demonstrating the "define, trace, compile, run" workflow.

### OpenXLA / XLA compiler page
https://openxla.org/xla

Reference for AOT backends that many frameworks rely on.

---

## Model Conversion & Cross-Runtime

### Export PyTorch to ONNX (official)
https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html

Official guide for converting PyTorch models to ONNX format.

### ONNX Runtime performance guide
https://onnxruntime.ai/docs/performance/

Documentation for running ONNX Runtime with different execution providers.

### TensorFlow SavedModel guide
https://www.tensorflow.org/guide/saved_model

Guide for converting and serving TensorFlow models.

### NVIDIA TensorRT developer guide
https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html

Documentation for building optimized TensorRT engines (stretch goal).

### TensorFlow Lite / edge optimization
https://ai.google.dev/edge/litert/models/model_optimization

Guide for AOT compilation targeting mobile and edge devices.

---

## Benchmarking & Profiling Tools

### PyTorch performance tuning guide
https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

Best practices for optimizing PyTorch performance.

### NVIDIA Nsight Systems user guide
https://docs.nvidia.com/nsight-systems/UserGuide/index.html

Tool for profiling latency and GPU utilization.

### Nsight Compute profiling guide
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

Tool for kernel-level performance analysis.

---

## Foundational Papers: Static vs Dynamic Graphs

### TensorFlow: "A System for Large-Scale Machine Learning," OSDI 2016
https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf

Canonical reference for static computation graphs.

### PyTorch: "An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019
https://arxiv.org/pdf/1912.01703

Canonical reference for dynamic computation graphs.

### Chainer: "A Next-Generation Open Source Framework for Deep Learning"
https://learningsys.org/papers/LearningSys_2015_paper_33.pdf

Early define-by-run framework demonstrating the evolution of dynamic graphs.

### TensorFlow Fold blog
https://research.google/blog/announcing-tensorflow-fold-deep-learning-with-dynamic-computation-graphs/

Demonstrates implementing dynamic computation on top of a static engine.

---

## Compiler Research & Optimization

### TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
https://arxiv.org/abs/1802.04799

Key reference for operator fusion and cross-platform portability.

### Glow: Graph Lowering Compiler Techniques for Neural Networks
https://research.facebook.com/publications/glow-graph-lowering-compiler-techniques-for-neural-networks/

Alternative compiler approach comparable to TensorRT and TVM.

### XLA: Optimizing Compiler for ML
https://openxla.org/xla/tf2xla

Reference for XLA-style AOT compilation techniques.

---

## Additional Resources

### ONNX tutorials repo
https://github.com/onnx/tutorials

### PyTorch TorchScript (now saying to use torch.export)
https://docs.pytorch.org/docs/stable/jit.html


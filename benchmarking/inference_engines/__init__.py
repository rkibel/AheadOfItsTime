"""Inference engine implementations for different frameworks."""

from .base import InferenceEngine
from .pytorch_eager import PyTorchEagerEngine
from .torchscript import TorchScriptEngine
from .pytorch_compile import PyTorchCompileEngine
from .onnx_runtime import ONNXRuntimeEngine
from .tensorflow import TensorFlowEngine

# Registry of available engines
ENGINE_REGISTRY = {
    'pytorch-eager': PyTorchEagerEngine,
    'torchscript': TorchScriptEngine,
    'pytorch-compile': PyTorchCompileEngine,
    'onnx': ONNXRuntimeEngine,
    'tensorflow': TensorFlowEngine,
}

__all__ = [
    'InferenceEngine',
    'PyTorchEagerEngine',
    'TorchScriptEngine',
    'PyTorchCompileEngine',
    'ONNXRuntimeEngine',
    'TensorFlowEngine',
    'ENGINE_REGISTRY',
]


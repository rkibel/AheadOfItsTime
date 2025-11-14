"""
Centralized model configurations for the entire project.

This module provides a single source of truth for model specifications,
used by training, conversion, and benchmarking scripts.
"""

import torch
from pathlib import Path
from typing import Dict, Any

# Import model classes
from .cnn.lenet import LeNet5
from .cnn.resnet import ResNet18
from .rnn.lstm_sentiment import LSTMSentimentClassifier
from .rnn.gru_lm import GRULanguageModel


# Model class registry
MODEL_CLASSES = {
    'lenet': LeNet5,
    'resnet18': ResNet18,
    'lstm': LSTMSentimentClassifier,
    'gru': GRULanguageModel,
}


# Complete model configurations
MODEL_CONFIGS = {
    'lenet': {
        'class': LeNet5,
        'class_name': 'LeNet5',
        'kwargs': {
            'num_classes': 10,
            'in_channels': 1
        },
        'input_shape': (1, 28, 28),  # (C, H, W)
        'example_input': torch.randn(1, 1, 28, 28),
        'dataset': 'MNIST',
        'task': 'image_classification',
        'checkpoints': {
            'pytorch': 'checkpoints/pytorch/lenet_mnist.pth',
            'torchscript': 'checkpoints/torchscript/lenet_mnist.pt',
            'onnx': 'checkpoints/onnx/lenet_mnist.onnx',
            'tensorflow': 'checkpoints/tensorflow/lenet_savedmodel/',
            'tensorrt': 'checkpoints/tensorrt/lenet_mnist.engine',
        },
        'input_names': ['input'],
        'output_names': ['output'],
    },
    'resnet18': {
        'class': ResNet18,
        'class_name': 'ResNet18',
        'kwargs': {
            'num_classes': 10
        },
        'input_shape': (3, 32, 32),  # (C, H, W)
        'example_input': torch.randn(1, 3, 32, 32),
        'dataset': 'CIFAR-10',
        'task': 'image_classification',
        'checkpoints': {
            'pytorch': 'checkpoints/pytorch/resnet18_cifar10.pth',
            'torchscript': 'checkpoints/torchscript/resnet18_cifar10.pt',
            'onnx': 'checkpoints/onnx/resnet18_cifar10.onnx',
            'tensorflow': 'checkpoints/tensorflow/resnet18_savedmodel/',
            'tensorrt': 'checkpoints/tensorrt/resnet18_cifar10.engine',
        },
        'input_names': ['input'],
        'output_names': ['output'],
    },
    'lstm': {
        'class': LSTMSentimentClassifier,
        'class_name': 'LSTMSentimentClassifier',
        'kwargs': {
            'vocab_size': 25000,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.5
        },
        'input_shape': (256,),  # (sequence_length,)
        'example_input': torch.randint(0, 25000, (1, 256)),
        'dataset': 'IMDB',
        'task': 'sentiment_classification',
        'vocab_size': 25000,
        'checkpoints': {
            'pytorch': 'checkpoints/pytorch/lstm_imdb.pth',
            'torchscript': 'checkpoints/torchscript/lstm_imdb.pt',
            'onnx': 'checkpoints/onnx/lstm_imdb.onnx',
            'tensorflow': 'checkpoints/tensorflow/lstm_savedmodel/',
            'tensorrt': 'checkpoints/tensorrt/lstm_imdb.engine',
        },
        'input_names': ['input'],
        'output_names': ['output'],
    },
    'gru': {
        'class': GRULanguageModel,
        'class_name': 'GRULanguageModel',
        'kwargs': {
            'vocab_size': 29573,
            'embedding_dim': 200,
            'hidden_dim': 200,
            'num_layers': 2,
            'dropout': 0.2
        },
        'input_shape': (35,),  # (sequence_length,)
        'example_input': torch.randint(0, 29573, (32, 35)),
        'dataset': 'WikiText-2',
        'task': 'language_modeling',
        'vocab_size': 29573,
        'checkpoints': {
            'pytorch': 'checkpoints/pytorch/gru_wikitext.pth',
            'torchscript': 'checkpoints/torchscript/gru_wikitext.pt',
            'onnx': 'checkpoints/onnx/gru_wikitext.onnx',
            'tensorflow': 'checkpoints/tensorflow/gru_savedmodel/',
            'tensorrt': 'checkpoints/tensorrt/gru_wikitext.engine',
        },
        'input_names': ['input'],
        'output_names': ['output'],
    }
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model ('lenet', 'resnet18', 'lstm', 'gru')
        
    Returns:
        Model configuration dictionary
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def get_model_class(model_name: str):
    """
    Get the model class for instantiation.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class
    """
    config = get_model_config(model_name)
    return config['class']


def get_checkpoint_path(model_name: str, framework: str, 
                       project_root: Path = None) -> Path:
    """
    Get checkpoint path for a model and framework.
    
    Args:
        model_name: Name of the model
        framework: Framework name ('pytorch', 'torchscript', 'onnx', etc.)
        project_root: Project root directory (default: auto-detect)
        
    Returns:
        Path to checkpoint
    """
    if project_root is None:
        # Auto-detect project root (assumes this file is in models/)
        project_root = Path(__file__).parent.parent
    
    config = get_model_config(model_name)
    
    # Map benchmark framework names to checkpoint types
    framework_mapping = {
        'pytorch-eager': 'pytorch',
        'pytorch-compile': 'pytorch',
        'torchscript': 'torchscript',
        'onnx': 'onnx',
        'tensorflow': 'tensorflow',
        'tensorrt': 'tensorrt',
    }
    
    checkpoint_type = framework_mapping.get(framework, framework)
    
    if checkpoint_type not in config['checkpoints']:
        raise ValueError(f"No checkpoint defined for {model_name} + {framework}")
    
    checkpoint_path = config['checkpoints'][checkpoint_type]
    return project_root / checkpoint_path


def is_rnn_model(model_name: str) -> bool:
    """Check if model is an RNN-based architecture."""
    return model_name in ['lstm', 'gru']


def is_cnn_model(model_name: str) -> bool:
    """Check if model is a CNN-based architecture."""
    return model_name in ['lenet', 'resnet18']


def get_example_input(model_name: str, batch_size: int = 1, 
                      device: str = 'cpu') -> torch.Tensor:
    """
    Create example input tensor for a model.
    
    Args:
        model_name: Name of the model
        batch_size: Batch size for input
        device: Device to create tensor on
        
    Returns:
        Example input tensor
    """
    config = get_model_config(model_name)
    input_shape = config['input_shape']
    
    if is_rnn_model(model_name):
        # RNN models expect integer tokens
        vocab_size = config['vocab_size']
        shape = (batch_size,) + input_shape
        return torch.randint(0, vocab_size, shape, device=device)
    else:
        # CNN models expect float images
        shape = (batch_size,) + input_shape
        return torch.randn(shape, device=device)


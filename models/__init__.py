"""
AheadOfItsTime Model Zoo

This module contains implementations of CNN and RNN architectures
for benchmarking AOT vs JIT compilation strategies.
"""

from .cnn import LeNet5, ResNet18
from .rnn import LSTMSentimentClassifier, GRULanguageModel
from .config import (
    MODEL_CONFIGS,
    MODEL_CLASSES,
    get_model_config,
    get_model_class,
    get_checkpoint_path,
    get_example_input,
    is_rnn_model,
    is_cnn_model,
)

__version__ = "0.1.0"

__all__ = [
    # Model classes
    'LeNet5',
    'ResNet18',
    'LSTMSentimentClassifier',
    'GRULanguageModel',
    # Configuration
    'MODEL_CONFIGS',
    'MODEL_CLASSES',
    'get_model_config',
    'get_model_class',
    'get_checkpoint_path',
    'get_example_input',
    'is_rnn_model',
    'is_cnn_model',
]

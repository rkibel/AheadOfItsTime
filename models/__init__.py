"""
AheadOfItsTime Model Zoo

This module contains implementations of CNN and RNN architectures
for benchmarking AOT vs JIT compilation strategies.
"""

from .cnn import LeNet5, ResNet18
from .rnn import LSTMSentimentClassifier, GRULanguageModel

__version__ = "0.1.0"

__all__ = [
    'LeNet5',
    'ResNet18',
    'LSTMSentimentClassifier',
    'GRULanguageModel',
]

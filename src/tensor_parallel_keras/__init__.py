"""
Tensor Parallel implementation for Keras 3.0
"""

from .tensor_parallel_keras import TensorParallelKeras
from .backend import get_backend
from .losses import VocabParallelCrossEntropy

__version__ = "0.1.0"
__all__ = ["TensorParallelKeras", "get_backend", "VocabParallelCrossEntropy"] 
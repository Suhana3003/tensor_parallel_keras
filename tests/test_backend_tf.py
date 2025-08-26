import os
import sys

import pytest

# Ensure src is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

from tensor_parallel_keras import get_backend, VocabParallelCrossEntropy


@pytest.mark.skipif(tf is None, reason="TensorFlow not available")
def test_tf_backend_vocab_parallel_ce_shapes():
    be = get_backend("tensorflow")

    seq, batch, vocab = 3, 2, 8
    logits = tf.random.normal((seq, batch, vocab), dtype=tf.float32)
    target = tf.random.uniform((seq, batch), minval=0, maxval=vocab, dtype=tf.int32)

    exp_logits, loss = VocabParallelCrossEntropy.forward(logits, target, backend_name="tensorflow")

    assert exp_logits.shape == logits.shape
    assert loss.shape == target.shape
    # Probabilities sum to ~1 on last dim
    probs_sum = tf.reduce_sum(exp_logits, axis=-1)
    assert tf.reduce_all(tf.math.abs(probs_sum - 1.0) < 1e-4)
import os
import sys

import pytest

# Ensure src is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None

from tensor_parallel_keras import get_backend, VocabParallelCrossEntropy


@pytest.mark.skipif(jax is None, reason="JAX not available")
def test_jax_backend_vocab_parallel_ce_shapes():
    be = get_backend("jax")

    seq, batch, vocab = 3, 2, 8
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (seq, batch, vocab))
    target = jax.random.randint(key, (seq, batch), 0, vocab)

    exp_logits, loss = VocabParallelCrossEntropy.forward(logits, target, backend_name="jax")

    assert exp_logits.shape == logits.shape
    assert loss.shape == target.shape
    probs_sum = jnp.sum(exp_logits, axis=-1)
    assert jnp.allclose(probs_sum, jnp.ones_like(probs_sum), atol=1e-4)
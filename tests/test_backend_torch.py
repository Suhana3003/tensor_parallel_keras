import os
import sys

import pytest

# Ensure src is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from tensor_parallel_keras import get_backend, VocabParallelCrossEntropy


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_torch_backend_vocab_parallel_ce_matches_torch():
    be = get_backend("torch")

    seq, batch, vocab = 3, 2, 8
    logits = torch.randn(seq, batch, vocab, dtype=torch.float32)
    target = torch.randint(0, vocab, (seq, batch), dtype=torch.long)

    # Our backend loss (world_size=1 path)
    exp_logits, loss = VocabParallelCrossEntropy.forward(logits, target, backend_name="torch")

    # Reference: -log softmax at target
    log_probs = torch.log_softmax(logits, dim=-1)
    ref = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    assert loss.shape == ref.shape
    assert torch.all(torch.isfinite(loss))
    assert torch.allclose(loss, ref, atol=1e-5)

    # exp_logits should be probabilities
    probs_sum = exp_logits.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)
from typing import Any, Tuple

from .backend import get_backend
from .vocab_utils import VocabUtility


class VocabParallelCrossEntropy:
    @staticmethod
    def forward(vocab_parallel_logits: Any, target: Any, backend_name: str = "torch") -> Any:
        be = get_backend(backend_name)
        ops = be.ops
        dist = be.dist

        # Cast/compute logits max per shard, then global max across shards
        logits = vocab_parallel_logits
        _, logits_max_local = ops.max(logits, axis=-1)
        # Fallback: for TF/JAX max returned (values, None); recompute values directly
        logits_max_vals = ops.max(logits, axis=-1)[0] if isinstance(logits_max_local, type(None)) else logits_max_local
        logits_max = dist.all_reduce(logits_max_vals, op="max") if hasattr(dist, 'all_reduce') else logits_max_vals

        # Subtract max for numerical stability
        logits = logits - logits_max[..., None]

        # Get per-partition vocab range
        part_size = logits.shape[-1]
        rank = getattr(dist, 'rank', lambda: 0)()
        world = getattr(dist, 'world_size', lambda: 1)()
        vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(part_size, rank, world)

        # Mask targets outside this shard
        target_mask = (target < vocab_start) | (target >= vocab_end)
        masked_target = (target - vocab_start)
        masked_target = masked_target * (~target_mask)

        # Gather predicted logits
        rows = ops.arange(masked_target.reshape(-1).shape[0], device=be.device(logits))
        logits2d = ops.view2d(logits, rows.shape[0], part_size)
        pred1d = ops.gather_1d(logits2d, rows, masked_target.reshape(-1))
        predicted_logits = pred1d.reshape(target.shape)
        predicted_logits = predicted_logits * (~target_mask)

        # Softmax pieces
        exp_logits = ops.exp(logits)
        sum_exp = ops.sum(exp_logits, axis=-1)

        # Cross entropy
        loss = ops.log(sum_exp) - predicted_logits

        # Normalize exp in place if possible
        # Some backends may not be in-place; keep functional semantics
        exp_logits = exp_logits / sum_exp[..., None]

        # Aggregate across shards
        predicted_logits = dist.all_reduce(predicted_logits, op="sum")
        sum_exp = dist.all_reduce(sum_exp, op="sum")

        return exp_logits, loss
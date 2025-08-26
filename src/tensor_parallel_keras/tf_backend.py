from typing import Any, List, Optional

import tensorflow as tf

from .backend import Backend, TensorOps, CollectiveOps


class _TFOps:
    def exp(self, x: Any) -> Any:
        return tf.exp(x)

    def log(self, x: Any) -> Any:
        return tf.math.log(x)

    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        return tf.reduce_sum(x, axis=axis)

    def max(self, x: Any, axis: int) -> Any:
        values = tf.reduce_max(x, axis=axis)
        return values, None

    def concatenate(self, xs: List[Any], axis: int) -> Any:
        return tf.concat(xs, axis=axis)

    def arange(self, n: int, device: Any) -> Any:
        return tf.range(n)

    def view2d(self, x: Any, rows: int, cols: int) -> Any:
        return tf.reshape(x, (rows, cols))

    def gather_1d(self, x2d: Any, row_indices: Any, col_indices_1d: Any) -> Any:
        rows = tf.gather(x2d, row_indices)
        cols = tf.gather(tf.transpose(rows), col_indices_1d)
        return tf.transpose(cols)


class _TFDist:
    def __init__(self) -> None:
        self._strategy = tf.distribute.get_strategy()

    def all_reduce(self, x: Any, op: str = "sum") -> Any:
        if self._strategy is None:
            return x
        reduce_op = tf.distribute.ReduceOp.SUM if op == "sum" else tf.distribute.ReduceOp.SUM
        return self._strategy.reduce(reduce_op, x, axis=None)

    def all_gather(self, x: Any, axis: int) -> Any:
        if self._strategy is None or not hasattr(self._strategy, 'experimental_all_gather'):
            return x
        return self._strategy.experimental_all_gather(x, axis=axis)

    def broadcast(self, x: Any, root: int = 0) -> Any:
        return x

    def world_size(self) -> int:
        try:
            return self._strategy.num_replicas_in_sync
        except Exception:
            return 1

    def rank(self) -> int:
        return 0


class TFBackend:
    def __init__(self) -> None:
        self.ops: TensorOps = _TFOps()
        self.dist: CollectiveOps = _TFDist()

    def device(self, x: Any) -> Any:
        return None
from typing import Any, List, Optional

import jax
import jax.numpy as jnp

from .backend import Backend, TensorOps, CollectiveOps


class _JAXOps:
    def exp(self, x: Any) -> Any:
        return jnp.exp(x)

    def log(self, x: Any) -> Any:
        return jnp.log(x)

    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        return jnp.sum(x, axis=axis)

    def max(self, x: Any, axis: int) -> Any:
        values = jnp.max(x, axis=axis)
        return values, None

    def concatenate(self, xs: List[Any], axis: int) -> Any:
        return jnp.concatenate(xs, axis=axis)

    def arange(self, n: int, device: Any) -> Any:
        return jnp.arange(n)

    def view2d(self, x: Any, rows: int, cols: int) -> Any:
        return jnp.reshape(x, (rows, cols))

    def gather_1d(self, x2d: Any, row_indices: Any, col_indices_1d: Any) -> Any:
        return x2d[row_indices, col_indices_1d]


class _JAXDist:
    def all_reduce(self, x: Any, op: str = "sum") -> Any:
        if jax.device_count() <= 1:
            return x
        if op == "sum":
            return jax.lax.psum(x, axis_name='i')
        return x

    def all_gather(self, x: Any, axis: int) -> Any:
        if jax.device_count() <= 1:
            return x
        gathered = jax.lax.all_gather(x, 'i', axis=axis)
        return gathered

    def broadcast(self, x: Any, root: int = 0) -> Any:
        return x

    def world_size(self) -> int:
        return jax.device_count()

    def rank(self) -> int:
        return jax.process_index()


class JAXBackend:
    def __init__(self) -> None:
        self.ops: TensorOps = _JAXOps()
        self.dist: CollectiveOps = _JAXDist()

    def device(self, x: Any) -> Any:
        return None
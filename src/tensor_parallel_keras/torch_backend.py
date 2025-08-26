from typing import Any, List, Optional

import torch

from .backend import Backend, TensorOps, CollectiveOps


class _TorchOps:
    def exp(self, x: Any) -> Any:
        return torch.exp(x)

    def log(self, x: Any) -> Any:
        return torch.log(x)

    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        return torch.sum(x, dim=axis) if axis is not None else torch.sum(x)

    def max(self, x: Any, axis: int) -> Any:
        return torch.max(x, dim=axis)

    def concatenate(self, xs: List[Any], axis: int) -> Any:
        return torch.cat(xs, dim=axis)

    def arange(self, n: int, device: Any) -> Any:
        return torch.arange(0, n, device=device)

    def view2d(self, x: Any, rows: int, cols: int) -> Any:
        return x.view(rows, cols)

    def gather_1d(self, x2d: Any, row_indices: Any, col_indices_1d: Any) -> Any:
        return x2d[row_indices, col_indices_1d]


class _TorchDist:
    def __init__(self) -> None:
        self._group = torch.distributed.group.WORLD if torch.distributed.is_available() else None

    def all_reduce(self, x: Any, op: str = "sum") -> Any:
        if self._group is None or not torch.distributed.is_initialized():
            return x
        reduce_op = torch.distributed.ReduceOp.SUM if op == "sum" else torch.distributed.ReduceOp.MAX
        torch.distributed.all_reduce(x, op=reduce_op, group=self._group)
        return x

    def all_gather(self, x: Any, axis: int) -> Any:
        if self._group is None or not torch.distributed.is_initialized():
            return x
        world = torch.distributed.get_world_size(group=self._group)
        out = [torch.empty_like(x) for _ in range(world)]
        torch.distributed.all_gather(out, x, group=self._group)
        return torch.cat(out, dim=axis)

    def broadcast(self, x: Any, root: int = 0) -> Any:
        if self._group is None or not torch.distributed.is_initialized():
            return x
        ranks = torch.distributed.get_process_group_ranks(group=self._group)
        torch.distributed.broadcast(x, src=ranks[root] if hasattr(ranks, '__getitem__') else root, group=self._group)
        return x

    def world_size(self) -> int:
        return torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1

    def rank(self) -> int:
        return torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0


class TorchBackend:
    def __init__(self) -> None:
        self.ops: TensorOps = _TorchOps()
        self.dist: CollectiveOps = _TorchDist()

    def device(self, x: Any) -> Any:
        return getattr(x, 'device', None)
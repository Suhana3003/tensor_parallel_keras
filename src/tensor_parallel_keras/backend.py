from typing import Protocol, Any, List, Optional


class TensorOps(Protocol):
    def exp(self, x: Any) -> Any: ...
    def log(self, x: Any) -> Any: ...
    def sum(self, x: Any, axis: Optional[int] = None) -> Any: ...
    def max(self, x: Any, axis: int) -> Any: ...
    def concatenate(self, xs: List[Any], axis: int) -> Any: ...
    def arange(self, n: int, device: Any) -> Any: ...
    def view2d(self, x: Any, rows: int, cols: int) -> Any: ...
    def gather_1d(self, x2d: Any, row_indices: Any, col_indices_1d: Any) -> Any: ...


class CollectiveOps(Protocol):
    def all_reduce(self, x: Any, op: str = "sum") -> Any: ...
    def all_gather(self, x: Any, axis: int) -> Any: ...
    def broadcast(self, x: Any, root: int = 0) -> Any: ...
    def world_size(self) -> int: ...
    def rank(self) -> int: ...


class Backend(Protocol):
    ops: TensorOps
    dist: CollectiveOps
    def device(self, x: Any) -> Any: ...


_BACKENDS = {}

def register_backend(name: str, backend: Backend) -> None:
    _BACKENDS[name] = backend


def get_backend(name: str) -> Backend:
    if name in _BACKENDS:
        return _BACKENDS[name]
    # Lazy import to avoid heavy deps until requested
    if name == "torch":
        from .torch_backend import TorchBackend
        b = TorchBackend()
        register_backend(name, b)
        return b
    if name == "tensorflow":
        from .tf_backend import TFBackend
        b = TFBackend()
        register_backend(name, b)
        return b
    if name == "jax":
        from .jax_backend import JAXBackend
        b = JAXBackend()
        register_backend(name, b)
        return b
    raise ValueError(f"Unknown backend: {name}")
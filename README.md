## Tensor Parallel for Keras 3.0 (Slimmed)

Minimal, focused implementation of a `TensorParallelKeras` wrapper with parameter sharding utilities for Keras 3.0.

### Install
```bash
pip install -e .
```

### Quick start
```python
import keras
from tensor_parallel_keras import TensorParallelKeras

inputs = keras.Input(shape=(100,))
x = keras.layers.Dense(128, activation="relu")(inputs)
outputs = keras.layers.Dense(10, activation="softmax")(x)
base = keras.Model(inputs, outputs)

tp = TensorParallelKeras(base)  # auto-detects single CPU device here
tp.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
```

### What’s included
- `TensorParallelKeras` minimal wrapper and exports
- Parameter sharding primitives and simple config utilities

### What was removed
- Large test suite, duplicate docs, and sample data
- Heavy optional backends from default dependencies (still compatible if installed)

MIT License.
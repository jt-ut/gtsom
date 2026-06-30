# gtsom

**General Topology Self-Organizing Map** — a Python package for training Self-Organizing Maps (SOMs) on arbitrary output space geometries, including hexagonal grids, rectangular grids, and data-driven Delaunay triangulations.

## Overview

A Self-Organizing Map projects high-dimensional data onto a low-dimensional output lattice while preserving topological structure — nearby neurons learn to respond to similar inputs. GTSOM extends the classical SOM by allowing any graph structure as the output space, defined either by a regular grid or by the geometry of the data itself.

Key features:

- Hexagonal and rectangular grid topologies (`GTSOM.from_grid`)
- Data-driven Delaunay triangulation topology (`GTSOM.from_data`)
- Exponential neighborhood bandwidth annealing
- Parallel prototype updates via [numba](https://numba.readthedocs.io) (optional)
- Per-epoch learning history including MQE, per-prototype MQE, and BMU change rate
- Built-in lattice visualization with multiple coloring modes

## Installation

```bash
pip install git+https://github.com/jt-ut/gtsom.git
```

`gtsom` depends on [vqlp](https://github.com/jt-ut/vqlp) for vector quantization and recall. **`vqlp` is not published on PyPI** and must be installed manually from GitHub *before* installing `gtsom`:

```bash
pip install git+https://github.com/jt-ut/vqlp.git
pip install git+https://github.com/jt-ut/gtsom.git
```

`vqlp` itself depends on [faiss](https://github.com/facebookresearch/faiss) (`faiss-cpu` on PyPI). If `vqlp`'s installation fails on faiss, install `faiss-cpu` for your platform first — note that prebuilt wheels are not available for all platforms (e.g. some Apple Silicon configurations), in which case consult the [faiss installation instructions](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

For parallel prototype updates (recommended for large datasets):

```bash
pip install "git+https://github.com/jt-ut/gtsom.git[parallel]"
```

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_blobs
from gtsom import GTSOM

# Generate some data
X, y = make_blobs(n_samples=1000, n_features=2, centers=5,
                  cluster_std=0.8, random_state=42)
X = X.astype(np.float32)

# Configure the learning schedule
som = GTSOM(
    rho_0=3.0,              # initial neighborhood bandwidth
    rho_f=1.0,              # final neighborhood bandwidth
    halflife_epochs=50,
    n_jobs=-1,              # use all CPU cores (requires numba)
    random_state=42,
)

# Initialize a 10x10 hexagonal SOM
som.from_grid(
    X,
    shape=(10, 10),
    coord_init='hex',
    W_init='random',
    labels=y,
)

# Train
som.fit(X, n_epochs=100, labels=y)

# Visualize
fig = som.plot(color_by='labels')
fig.save('som.png', dpi=150)

# Map data to output coordinates
coords = som.transform(X)   # (N, 2) array of lattice positions
```

## Documentation

Full documentation and a worked tutorial on MNIST handwritten digits is available at:

**[https://jt-ut.github.io/gtsom](https://jt-ut.github.io/gtsom)**

## Citation

A paper describing GTSOM is currently under review. Citation information will be added here upon publication.

## License

MIT License. See [LICENSE](LICENSE) for details.

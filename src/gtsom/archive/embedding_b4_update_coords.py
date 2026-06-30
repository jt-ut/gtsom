import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


__all__ = ["Embedding"]


class Embedding:
    """
    Low-dimensional embedding of SOM prototype positions, including
    graph topology and pairwise distances.

    Stores the output-space coordinates of M prototypes, a sparse adjacency
    matrix defining their topological connections, and a dense pairwise
    distance matrix derived from that adjacency.

    Attributes
    ----------
    coords : np.ndarray, shape (M, dim)
        Low-dimensional positions of the prototypes. Mutable — can be
        updated externally (e.g. by a tSNE/UMAP step), followed by a
        call to compute_topo() to rebuild adjacency and distances.
    dim : int
        Dimensionality of the embedding space (2 or 3). Inferred from
        coords.
    adjacency : scipy.sparse.csr_matrix, shape (M, M)
        Sparse symmetric adjacency matrix. Entry (i, j) = 1 if prototypes
        i and j are graph neighbors, 0 otherwise. Populated by
        compute_topo().
    adj_metric : str
        Algorithm used to construct the adjacency. One of:
        'delaunay', 'gabriel', 'grid_rect', 'grid_hex'.
    dist : np.ndarray, shape (M, M), dtype float32
        Dense pairwise distance matrix. Populated by compute_topo().
    dist_metric : str
        Distance type used to populate dist. Currently supported:
        'geodesic' (unweighted hop-count shortest path).
    M : int (property)
        Number of prototypes.
    """

    SUPPORTED_ADJ_METRICS = {"delaunay", "gabriel", "grid_rect", "grid_hex"}
    SUPPORTED_DIST_METRICS = {"geodesic"}

    def __init__(self, coords, adj_metric, dist_metric="geodesic", _shape=None):
        """
        Base constructor. Stores coords and metric specs, then calls
        compute_topo() to build adjacency and dist.

        Prefer the classmethods (from_delaunay, from_grid) over calling
        this directly.

        Parameters
        ----------
        coords : array-like, shape (M, dim)
            Low-dimensional prototype positions. dim must be 2 or 3.
        adj_metric : str
            Algorithm to use for adjacency construction.
        dist_metric : str, default 'geodesic'
            Distance type for computing dist.
        _shape : tuple of int, optional
            Grid shape — stored internally by from_grid() so that
            compute_topo() can reconstruct exact grid adjacency.
        """
        self.coords = np.asarray(coords, dtype=float)
        self.adj_metric = adj_metric
        self.dist_metric = dist_metric
        self._shape = _shape  # only set for grid embeddings

        self.adjacency = None
        self.dist = None

        self._validate()
        self.compute_topo()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        if self.coords.ndim != 2:
            raise ValueError(
                f"coords must be a 2D array of shape (M, dim), "
                f"got shape {self.coords.shape}"
            )
        if self.dim not in (2, 3):
            raise ValueError(
                f"dim must be 2 or 3, got {self.dim}"
            )
        if self.adj_metric not in self.SUPPORTED_ADJ_METRICS:
            raise ValueError(
                f"adj_metric '{self.adj_metric}' not supported. "
                f"Choose from {self.SUPPORTED_ADJ_METRICS}."
            )
        if self.dist_metric not in self.SUPPORTED_DIST_METRICS:
            raise ValueError(
                f"dist_metric '{self.dist_metric}' not supported. "
                f"Choose from {self.SUPPORTED_DIST_METRICS}."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def M(self):
        """Number of prototypes."""
        return self.coords.shape[0]

    @property
    def dim(self):
        """Dimensionality of the embedding space (2 or 3)."""
        return self.coords.shape[1]

    # ------------------------------------------------------------------
    # Topology recomputation
    # ------------------------------------------------------------------

    def compute_topo(self):
        """
        Recompute adjacency from current coords (using adj_metric), then
        recompute dist (using dist_metric).

        Call this after updating coords externally (e.g. after a tSNE/UMAP
        coordinate update step). For grid-based embeddings this is called
        once at construction and rarely thereafter.

        Returns
        -------
        self : Embedding
            Returns self for method chaining.
        """
        if self.adj_metric in ("delaunay", "grid_hex"):
            # Delaunay correctly recovers hex grid 6-connectivity and
            # handles arbitrary/random coord layouts.
            self.adjacency = _delaunay_adjacency(self.coords)

        elif self.adj_metric == "grid_rect":
            # Delaunay on a regular grid is ambiguous for square cells:
            # it picks one diagonal per square, missing the other half.
            # Use an explicit 8-connected grid adjacency instead.
            if self._shape is None:
                raise RuntimeError(
                    "grid_rect adjacency requires _shape to be set. "
                    "Use Embedding.from_grid() to construct rect grid embeddings."
                )
            self.adjacency = _rect_grid_adjacency(self._shape)

        elif self.adj_metric == "gabriel":
            self.adjacency = _gabriel_adjacency(self.coords)

        self.dist = self._compute_dist()
        return self

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def _compute_dist(self):
        """
        Compute the pairwise distance matrix from the current adjacency,
        according to dist_metric.

        Returns
        -------
        dist : np.ndarray, shape (M, M), dtype float32
        """
        if self.dist_metric == "geodesic":
            return self._compute_geodesic_dist()

    def _compute_geodesic_dist(self):
        """
        Compute unweighted (hop-count) shortest-path distances between
        all prototype pairs using scipy's Dijkstra implementation.

        Returns
        -------
        dist : np.ndarray, shape (M, M), dtype float32
        """
        dist = shortest_path(
            self.adjacency,
            method="D",
            directed=False,
            unweighted=True,
        )
        return dist.astype(np.float32)

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_delaunay(cls, coords, dist_metric="geodesic"):
        """
        Construct an Embedding from the Delaunay triangulation of coords.

        Parameters
        ----------
        coords : array-like, shape (M, dim)
            Low-dimensional prototype positions. dim must be 2 or 3.
        dist_metric : str, default 'geodesic'
            Distance type for computing dist.

        Returns
        -------
        Embedding
        """
        return cls(coords, adj_metric="delaunay", dist_metric=dist_metric)

    @classmethod
    def from_grid(cls, shape, kind="hex", dist_metric="geodesic"):
        """
        Construct an Embedding from a regular grid.

        Coordinates are generated analytically from the grid shape and
        type. For hexagonal grids, adjacency is derived via Delaunay
        triangulation (which correctly recovers 6-connectivity in 2D and
        richer inter-layer connectivity in 3D). For rectangular grids,
        an explicit 8-connected adjacency is constructed directly to
        avoid the diagonal ambiguity of Delaunay on regular square grids.

        Parameters
        ----------
        shape : tuple of int
            Grid dimensions. (nrows, ncols) for 2D grids, or
            (nrows, ncols, ndepth) for 3D grids.
        kind : str, default 'hex'
            Grid type: 'hex' for hexagonal, 'rect' for rectangular.
        dist_metric : str, default 'geodesic'
            Distance type for computing dist.

        Returns
        -------
        Embedding
        """
        if kind == "hex":
            coords = _hex_grid_coords(shape)
            return cls(
                coords, adj_metric="grid_hex",
                dist_metric=dist_metric, _shape=shape,
            )
        elif kind == "rect":
            coords = _rect_grid_coords(shape)
            return cls(
                coords, adj_metric="grid_rect",
                dist_metric=dist_metric, _shape=shape,
            )
        else:
            raise ValueError(f"kind must be 'hex' or 'rect', got '{kind}'")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"Embedding(M={self.M}, dim={self.dim}, "
            f"adj_metric='{self.adj_metric}', dist_metric='{self.dist_metric}')"
        )


# ----------------------------------------------------------------------
# Module-level helper functions (private)
# ----------------------------------------------------------------------

def _delaunay_adjacency(coords):
    """
    Build a sparse symmetric adjacency matrix from the Delaunay
    triangulation of coords.

    Works for both 2D (triangles, 3 vertices per simplex) and 3D
    (tetrahedra, 4 vertices per simplex). All pairs of vertices within
    each simplex become edges.

    Parameters
    ----------
    coords : np.ndarray, shape (M, dim)

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix, shape (M, M)
    """
    M = coords.shape[0]
    dim = coords.shape[1]
    tri = Delaunay(coords)

    simplices = tri.simplices          # shape (n_simplices, dim+1)
    n_verts = dim + 1

    pair_indices = np.array(
        [(a, b) for a in range(n_verts) for b in range(a + 1, n_verts)]
    )

    pairs = np.concatenate(
        [simplices[:, p] for p in pair_indices], axis=0
    ).reshape(-1, 2)

    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)

    rows = pairs[:, 0]
    cols = pairs[:, 1]
    data = np.ones(len(rows), dtype=np.uint8)

    adjacency = csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([rows, cols]),
          np.concatenate([cols, rows]))),
        shape=(M, M),
    )
    return adjacency


def _rect_grid_adjacency(shape):
    """
    Build an explicit 8-connected adjacency matrix for a rectangular grid.

    Connects each node to all 8 neighbors (cardinal + diagonal). This
    avoids the diagonal ambiguity that arises from Delaunay triangulation
    of regular square grids.

    Parameters
    ----------
    shape : tuple of int
        (nrows, ncols) for 2D or (nrows, ncols, ndepth) for 3D.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix, shape (M, M)
    """
    if len(shape) == 2:
        nrows, ncols = shape
        M = nrows * ncols
        node = lambda i, j: i * ncols + j

        rows, cols = [], []
        # All offsets for 8-connectivity (di, dj) with di>=0 to avoid duplicates
        offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
        for i in range(nrows):
            for j in range(ncols):
                n = node(i, j)
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nrows and 0 <= nj < ncols:
                        m = node(ni, nj)
                        rows += [n, m]
                        cols += [m, n]

    elif len(shape) == 3:
        nrows, ncols, ndepth = shape
        M = nrows * ncols * ndepth
        node = lambda i, j, k: (i * ncols + j) * ndepth + k

        rows, cols = [], []
        # All offsets for 26-connectivity in 3D, upper triangle only
        offsets = [
            (di, dj, dk)
            for di in range(-1, 2)
            for dj in range(-1, 2)
            for dk in range(-1, 2)
            if (di, dj, dk) > (0, 0, 0)  # upper triangle
        ]
        for i in range(nrows):
            for j in range(ncols):
                for k in range(ndepth):
                    n = node(i, j, k)
                    for di, dj, dk in offsets:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < nrows and 0 <= nj < ncols and 0 <= nk < ndepth:
                            m = node(ni, nj, nk)
                            rows += [n, m]
                            cols += [m, n]
    else:
        raise ValueError(f"shape must be a 2- or 3-tuple, got length {len(shape)}")

    data = np.ones(len(rows), dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(M, M))


def _gabriel_adjacency(coords):
    """
    Build a sparse symmetric adjacency matrix from the Gabriel graph
    of coords.

    The Gabriel graph is a subgraph of the Delaunay triangulation:
    edge (i, j) is retained only if no other point lies within the
    open ball whose diameter is the segment from coords[i] to coords[j].

    Uses Delaunay edges as candidates (Gabriel ⊆ Delaunay), then
    filters via the Gabriel criterion using vectorized numpy operations.

    Parameters
    ----------
    coords : np.ndarray, shape (M, dim)

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix, shape (M, M)
    """
    delaunay_adj = _delaunay_adjacency(coords)
    delaunay_adj_coo = delaunay_adj.tocoo()

    mask = delaunay_adj_coo.row < delaunay_adj_coo.col
    rows = delaunay_adj_coo.row[mask]
    cols = delaunay_adj_coo.col[mask]

    M = coords.shape[0]
    keep_rows, keep_cols = [], []

    for i, j in zip(rows, cols):
        mid = (coords[i] + coords[j]) / 2.0
        r = np.linalg.norm(coords[i] - coords[j]) / 2.0

        other_mask = np.ones(M, dtype=bool)
        other_mask[i] = False
        other_mask[j] = False
        dists = np.linalg.norm(coords[other_mask] - mid, axis=1)

        if not np.any(dists < r):
            keep_rows.append(i)
            keep_cols.append(j)

    if len(keep_rows) == 0:
        return csr_matrix((M, M), dtype=np.uint8)

    keep_rows = np.array(keep_rows)
    keep_cols = np.array(keep_cols)
    data = np.ones(len(keep_rows), dtype=np.uint8)

    adjacency = csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([keep_rows, keep_cols]),
          np.concatenate([keep_cols, keep_rows]))),
        shape=(M, M),
    )
    return adjacency


def _rect_grid_coords(shape):
    """
    Generate node coordinates for a rectangular grid.

    Parameters
    ----------
    shape : tuple of int
        (nrows, ncols) for 2D or (nrows, ncols, ndepth) for 3D.

    Returns
    -------
    coords : np.ndarray, shape (M, dim)
    """
    if len(shape) == 2:
        nrows, ncols = shape
        r, c = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
        coords = np.stack([c.ravel(), r.ravel()], axis=1).astype(float)

    elif len(shape) == 3:
        nrows, ncols, ndepth = shape
        r, c, d = np.meshgrid(
            np.arange(nrows), np.arange(ncols), np.arange(ndepth), indexing="ij"
        )
        coords = np.stack([c.ravel(), r.ravel(), d.ravel()], axis=1).astype(float)

    else:
        raise ValueError(f"shape must be a 2- or 3-tuple, got length {len(shape)}")

    return coords


def _hex_grid_coords(shape):
    """
    Generate node coordinates for a hexagonal grid.

    In 2D: odd rows are offset by 0.5 in x, and rows are spaced by
    sqrt(3)/2 in y, giving a standard hexagonal tiling.

    In 3D: 2D hex layers are stacked along the z-axis with unit spacing.
    Delaunay triangulation of the resulting coords naturally captures
    inter-layer connectivity.

    Parameters
    ----------
    shape : tuple of int
        (nrows, ncols) for 2D or (nrows, ncols, ndepth) for 3D.

    Returns
    -------
    coords : np.ndarray, shape (M, dim)
    """
    if len(shape) == 2:
        nrows, ncols = shape
        row_spacing = np.sqrt(3) / 2
        coords = []
        for i in range(nrows):
            x_offset = 0.5 if i % 2 == 1 else 0.0
            for j in range(ncols):
                coords.append([j + x_offset, i * row_spacing])
        return np.array(coords, dtype=float)

    elif len(shape) == 3:
        nrows, ncols, ndepth = shape
        row_spacing = np.sqrt(3) / 2
        coords = []
        for k in range(ndepth):
            for i in range(nrows):
                x_offset = 0.5 if i % 2 == 1 else 0.0
                for j in range(ncols):
                    coords.append([j + x_offset, i * row_spacing, float(k)])
        return np.array(coords, dtype=float)

    else:
        raise ValueError(f"shape must be a 2- or 3-tuple, got length {len(shape)}")
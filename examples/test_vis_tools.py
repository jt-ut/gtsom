"""
vis_tools_example.py
--------------------
Demonstrates vis_embedding_discrete and vis_embedding_continuous
using sklearn's make_blobs to generate labeled 2-D point clouds.

Run from the root of the gtsom repo:
    python vis_tools_example.py
"""

import numpy as np
from scipy.spatial import Delaunay
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

from gtsom import vis_embedding_discrete, vis_embedding_continuous
from gtsom.vis_tools import parse_ctab, build_ctab

# ---------------------------------------------------------------------------
# 1.  Synthetic data  -------------------------------------------------------
# ---------------------------------------------------------------------------
# 300 points in 5 labeled clusters, embedded in 10-D, then projected to 2-D
# with PCA so the layout looks like a typical UMAP / t-SNE embedding.

N_CLUSTERS = 5
N_SAMPLES  = 400
SEED       = 42

X_high, labels = make_blobs(
    n_samples=N_SAMPLES,
    n_features=10,
    centers=N_CLUSTERS,
    cluster_std=1.8,
    random_state=SEED,
)

# Project to 2-D for a plausible embedding layout
pca = PCA(n_components=2, random_state=SEED)
X2  = pca.fit_transform(X_high)
x, y = X2[:, 0], X2[:, 1]

# String cluster labels  (e.g. "C0", "C1", …)
label_names = np.array([f"C{i}" for i in labels])

# A continuous variable: distance from origin (used in the continuous plot)
z_continuous = np.sqrt(x**2 + y**2)

# Per-point size weights: larger points near the cluster centres
# (just for demonstrating point_size_wts)
from sklearn.preprocessing import minmax_scale
size_wts = 1.0 - minmax_scale(z_continuous)   # high weight near origin

# ---------------------------------------------------------------------------
# 2.  Color table for discrete plot  ----------------------------------------
# ---------------------------------------------------------------------------
PALETTE = {
    "C0": "#E63946",   # red
    "C1": "#457B9D",   # steel blue
    "C2": "#2A9D8F",   # teal
    "C3": "#E9C46A",   # golden yellow
    "C4": "#9B5DE5",   # purple
}
ctab = parse_ctab(list(PALETTE.keys()), list(PALETTE.values()))

# ---------------------------------------------------------------------------
# 3.  Plot 1 — discrete, no legend  ----------------------------------------
# ---------------------------------------------------------------------------
g1 = vis_embedding_discrete(
    x, y,
    z=label_names,
    ctab=ctab,
    point_size=3,
    xlab="PC 1", ylab="PC 2",
    title="Discrete embedding — no legend",
    subtitle="my subtitle", caption = "my caption", 
    legend_pos="none",
)
print("Plot 1 ready (discrete, no legend)")

# ---------------------------------------------------------------------------
# 4.  Plot 2 — discrete, legend on the right  --------------------------------
# ---------------------------------------------------------------------------
g2 = vis_embedding_discrete(
    x, y,
    z=label_names,
    ctab=ctab,
    point_size=3,
    xlab="PC 1", ylab="PC 2",
    title="Discrete embedding — right legend",
    legend_pos="right",
    legend_title="Cluster",
    legend_nlines=1,      # 1 column in the legend
    legend_key_size=8,
)
print("Plot 2 ready (discrete, right legend)")

# ---------------------------------------------------------------------------
# 5.  Plot 3 — discrete, legend on the bottom  --------------------------------
# ---------------------------------------------------------------------------
g3 = vis_embedding_discrete(
    x, y,
    z=label_names,
    ctab=ctab,
    point_size=3,
    xlab="PC 1", ylab="PC 2",
    title="Discrete embedding — bottom legend",
    legend_pos="bottom",
    legend_title="Cluster",
    legend_nlines=1,      # 1 row in the legend
    legend_key_size=8,
)
print("Plot 3 ready (discrete, bottom legend)")

# ---------------------------------------------------------------------------
# 6.  Plot 4 — discrete, variable point sizes  --------------------------------
# ---------------------------------------------------------------------------
g4 = vis_embedding_discrete(
    x, y,
    z=label_names,
    ctab=ctab,
    point_size=5,                          # max size
    point_size_wts=size_wts,              # per-point weights in [0, 1]
    xlab="PC 1", ylab="PC 2",
    title="Discrete embedding — variable point size",
    legend_pos="right",
    legend_title="Cluster",
)
print("Plot 4 ready (discrete, variable sizes)")

# ---------------------------------------------------------------------------
# 7.  Plot 5 — continuous, no legend  ----------------------------------------
# ---------------------------------------------------------------------------
g5 = vis_embedding_continuous(
    x, y, z_continuous,
    cmap="viridis",
    point_size=3,
    xlab="PC 1", ylab="PC 2",
    title="Continuous embedding — no legend",
    legend_pos="none",
)
print("Plot 5 ready (continuous, no legend)")

# ---------------------------------------------------------------------------
# 8.  Plot 6 — continuous, right colorbar  ------------------------------------
# ---------------------------------------------------------------------------
g6 = vis_embedding_continuous(
    x, y, z_continuous,
    cmap="viridis",
    point_size=3,
    xlab="PC 1", ylab="PC 2",
    title="Continuous embedding — right colorbar",
    legend_pos="right",
    legend_title="Dist.\nfrom origin",
)
print("Plot 6 ready (continuous, right colorbar)")

# ---------------------------------------------------------------------------
# 9.  Plot 7 — continuous, alternative colormap  ------------------------------
# ---------------------------------------------------------------------------
g7 = vis_embedding_continuous(
    x, y, z_continuous,
    cmap="plasma",
    point_size=3,
    xlab="PC 1", ylab="PC 2",
    title="Continuous embedding — plasma colormap",
    legend_pos="right",
    legend_title="Dist.\nfrom origin",
)
print("Plot 7 ready (continuous, plasma colormap)")

# ---------------------------------------------------------------------------
# 10.  Plot 8 — discrete, no grid  --------------------------------------------
# ---------------------------------------------------------------------------
g8 = vis_embedding_discrete(
    x, y,
    z=label_names,
    ctab=ctab,
    point_size=3,
    xlab="PC 1", ylab="PC 2",
    title="Discrete embedding — no grid",
    legend_pos="right",
    legend_title="Cluster",
    show_grid=False,
)
print("Plot 8 ready (discrete, no grid)")


# ---------------------------------------------------------------------------
# 11.  Graph data — Delaunay triangulation on a subset  ----------------------
# ---------------------------------------------------------------------------
# Use 80 points so the graph is visible without being overwhelming.
# Delaunay triangulation gives a natural spatial connectivity: each point is
# connected to its nearest neighbors in a way that covers the convex hull
# without long crossing edges.

N_GRAPH = 80
rng = np.random.default_rng(SEED)
idx = rng.choice(len(x), size=N_GRAPH, replace=False)
xg, yg = x[idx], y[idx]
zg_discrete   = label_names[idx]
zg_continuous = z_continuous[idx]

# Build Delaunay triangulation and convert to adjacency matrix
tri = Delaunay(np.column_stack([xg, yg]))
adj = np.zeros((N_GRAPH, N_GRAPH), dtype=int)
for simplex in tri.simplices:
    for i in range(3):
        for j in range(3):
            if i != j:
                adj[simplex[i], simplex[j]] = 1

# ---------------------------------------------------------------------------
# 12.  Plot 9 — discrete with graph edges  ------------------------------------
# ---------------------------------------------------------------------------
g9 = vis_embedding_discrete(
    xg, yg,
    z=zg_discrete,
    ctab=ctab,
    point_size=3,
    graph=adj,
    edge_size=0.3,
    edge_color="#BBBBBB",
    xlab="PC 1", ylab="PC 2",
    title="Discrete embedding — Delaunay graph",
    legend_pos="right",
    legend_title="Cluster",
)
print("Plot 9 ready (discrete, graph edges)")

# ---------------------------------------------------------------------------
# 13.  Plot 10 — continuous with graph edges  ----------------------------------
# ---------------------------------------------------------------------------
g10 = vis_embedding_continuous(
    xg, yg, zg_continuous,
    cmap="viridis",
    point_size=3,
    graph=adj,
    edge_size=0.3,
    edge_color="#CCCCCC",
    xlab="PC 1", ylab="PC 2",
    title="Continuous embedding — Delaunay graph",
    legend_pos="right",
    legend_title="Dist.\nfrom origin",
)
print("Plot 10 ready (continuous, graph edges)")

# ---------------------------------------------------------------------------
# 14.  Plot 11 — auto color table (no ctab provided)  -------------------------
# ---------------------------------------------------------------------------
# Demonstrates build_ctab being called internally when ctab=None.
# Colors are assigned automatically from Tab10.
g11 = vis_embedding_discrete(
    x, y,
    z=label_names,           # ctab omitted intentionally
    point_size=3,
    xlab="PC 1", ylab="PC 2",
    title="Discrete embedding — auto color table",
    legend_pos="right",
    legend_title="Cluster",
)
print("Plot 11 ready (discrete, auto ctab)")

# ---------------------------------------------------------------------------
# 15.  Display all plots  -----------------------------------------------------
# ---------------------------------------------------------------------------
# Each call to .draw() or printing the object opens a window.
# If running interactively in a notebook, just evaluate the variable.
# In a script, use .save() or explicitly draw:

plots = {
    "g1_discrete_no_legend":     g1,
    "g2_discrete_right_legend":  g2,
    "g3_discrete_bottom_legend": g3,
    "g4_discrete_var_size":      g4,
    "g5_continuous_no_legend":   g5,
    "g6_continuous_right_cbar":  g6,
    "g7_continuous_plasma":      g7,
    "g8_discrete_no_grid":       g8,
    "g9_discrete_graph":         g9,
    "g10_continuous_graph":      g10,
    "g11_discrete_auto_ctab":    g11,
}

# Option A — save all to PNG files
SAVE = True
if SAVE:
    for name, g in plots.items():
        fname = f"{name}.pdf"
        g.save(fname, dpi=150, width=6, height=5)
        print(f"  Saved {fname}")

# Option B — draw one at a time interactively
# print(g2)
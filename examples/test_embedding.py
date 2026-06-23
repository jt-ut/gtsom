import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gtsom import Embedding, NeighborKernel


def plot_embedding(ax, emb, title):
    """Plot prototype coords and adjacency edges for an Embedding."""
    coords = emb.coords
    adj_coo = emb.adjacency.tocoo()

    # Draw edges
    for i, j in zip(adj_coo.row, adj_coo.col):
        if i < j:
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="steelblue", linewidth=0.8, alpha=0.6, zorder=1,
            )

    # Draw nodes on top
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=30, color="white", edgecolors="steelblue",
        linewidths=1.0, zorder=2,
    )

    ax.set_title(
        f"{title}\nM={emb.M}, edges={emb.adjacency.nnz // 2}, "
        f"max_dist={emb.dist.max():.0f}",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.axis("off")


def plot_kernel(ax, emb, nbr_w, proto_idx, rho, title):
    """
    Plot neighborhood influence of a single prototype.

    Draws the graph edges in light gray, then overlays all prototypes
    colored and sized by their influence weight from proto_idx.
    The focal prototype is marked with a red ring.
    """
    coords = emb.coords
    adj_coo = emb.adjacency.tocoo()

    # Draw edges in background
    for i, j in zip(adj_coo.row, adj_coo.col):
        if i < j:
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="lightgray", linewidth=0.6, zorder=1,
            )

    # Extract full influence vector for proto_idx (dense, zeros for inactive)
    row = nbr_w.getrow(proto_idx)
    influence = np.zeros(emb.M)
    influence[row.indices] = row.data

    # All prototypes: color and size by influence weight
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=influence,
        s=influence * 120 + 8,   # size scales with influence, minimum 8
        cmap="YlOrRd",
        vmin=0, vmax=1,
        edgecolors="gray", linewidths=0.4,
        zorder=2,
    )

    # Highlight focal prototype
    ax.scatter(
        coords[proto_idx, 0], coords[proto_idx, 1],
        s=80, facecolors="none", edgecolors="red",
        linewidths=2.0, zorder=3,
    )

    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="nbr_w")

    n_active = row.nnz
    ax.set_title(
        f"{title}\nproto={proto_idx}, rho={rho:.2f}, "
        f"active neighbors={n_active}/{emb.M}",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.axis("off")


# -----------------------------------------------------------------------
# Build embeddings
# -----------------------------------------------------------------------
emb_rect = Embedding.from_grid((10, 8),  kind="rect")
emb_hex  = Embedding.from_grid((10, 12), kind="hex")

rng = np.random.default_rng(42)
coords_random = rng.uniform(0, 10, size=(100, 2))
emb_rand = Embedding.from_delaunay(coords_random)

# -----------------------------------------------------------------------
# Figure 1: Embedding adjacency graphs
# -----------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

plot_embedding(axes1[0], emb_rect, "Rectangular grid (10×8)")
plot_embedding(axes1[1], emb_hex,  "Hexagonal grid (10×12)")
plot_embedding(axes1[2], emb_rand, "Random Delaunay (N=100)")

plt.suptitle("Embedding adjacency graphs", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("test_embedding.png", dpi=150, bbox_inches="tight")
print("Saved test_embedding.png")

# -----------------------------------------------------------------------
# Figure 2: NeighborKernel visualization on hex grid
# -----------------------------------------------------------------------
kernel = NeighborKernel(kind="exp", h_min=0.1)

# rho = 1/4 of max(width, height) of the hex grid coords
hex_coords = emb_hex.coords
map_span = max(
    hex_coords[:, 0].max() - hex_coords[:, 0].min(),
    hex_coords[:, 1].max() - hex_coords[:, 1].min(),
)
rho = map_span / 4.0
print(f"\nHex grid span: {map_span:.2f}, rho: {rho:.2f}")

nbr_w = kernel.compute(emb_hex.dist, rho=rho)
print(f"nbr_w nnz: {nbr_w.nnz}, avg active per proto: {nbr_w.nnz / emb_hex.M:.1f}")

# Pick a prototype near the center of the hex grid
center = hex_coords.mean(axis=0)
dists_to_center = np.linalg.norm(hex_coords - center, axis=1)
proto_idx = int(np.argmin(dists_to_center))
print(f"Focal prototype: {proto_idx}, coords: {hex_coords[proto_idx]}")

# Three rho values: narrow, chosen, broad
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

for ax, rho_factor, label in zip(
    axes2,
    [1/8, 1/4, 1/2],
    ["rho = span/8 (narrow)", "rho = span/4 (default)", "rho = span/2 (broad)"],
):
    rho_i = map_span * rho_factor
    nbr_w_i = kernel.compute(emb_hex.dist, rho=rho_i)
    plot_kernel(ax, emb_hex, nbr_w_i, proto_idx, rho_i, label)

plt.suptitle(
    f"NeighborKernel (exp, h_min=0.1) — hex grid (10×12), proto={proto_idx}",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("test_kernel.png", dpi=150, bbox_inches="tight")
print("Saved test_kernel.png")

# -----------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------
print()
for name, emb in [("Rect", emb_rect), ("Hex", emb_hex), ("Random", emb_rand)]:
    print(f"{name}: {emb}")
    print(f"  edges:              {emb.adjacency.nnz // 2}")
    print(f"  dist max:           {emb.dist.max():.1f}")
    print(f"  dist mean (off-diag): {emb.dist[emb.dist > 0].mean():.2f}")
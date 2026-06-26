"""
test_MNIST.py — GTSOM showcase on MNIST digits

Workflow:
  1. Load MNIST test set (N=10,000, d=784), scale to [0,1]
  2. PCA reduction to 50 components
  3. Initialise 20x20 hex GTSOM, W_init='random'
  4. Fit (200 epochs, rho_0=10, rho_f=1, halflife_epochs=100, n_jobs=-1)
  5. Plot learning curves, SOM lattice (all three color_by modes),
     and a sample of prototype images back-projected to pixel space

Run from the project root:
    python test_MNIST.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from gtsom import GTSOM

OUTDIR   = os.path.dirname(os.path.abspath(__file__))
SEED     = 42
N        = 10_000
N_PCS    = 50
SHAPE    = (20, 20)

def savefig(fig, name):
    """Save a matplotlib Figure to OUTDIR."""
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")

def savegg(g, name, width=7, height=6):
    """Save a plotnine ggplot object to OUTDIR."""
    path = os.path.join(OUTDIR, name)
    g.save(path, dpi=150, width=width, height=height)
    print(f"  Saved: {path}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ------------------------------------------------------------------
# 1. Load MNIST
# ------------------------------------------------------------------
section("1. Loading MNIST")

print("  Fetching MNIST from OpenML (may download on first run)...")
t0 = time.perf_counter()
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_raw = mnist.data.astype(np.float32)     # (70000, 784)
y_raw = mnist.target.astype(int)          # (70000,)
print(f"  Full dataset: {X_raw.shape}, labels: {np.unique(y_raw)}")
print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

# Use the standard test split (last 10,000 samples) — balanced across digits
X_raw = X_raw[-N:]
y     = y_raw[-N:]
print(f"  Using last {N} samples: {X_raw.shape}")
print(f"  Label counts: { {d: (y==d).sum() for d in range(10)} }")

# ------------------------------------------------------------------
# 2. Preprocess: scale to [0,1] then PCA to 50 components
# ------------------------------------------------------------------
section("2. Preprocessing")

# Global scale to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
print(f"  Scaled to [0,1]: min={X_scaled.min():.3f}, max={X_scaled.max():.3f}")

# PCA reduction
print(f"  Fitting PCA ({N_PCS} components)...")
t0 = time.perf_counter()
pca = PCA(n_components=N_PCS, svd_solver='randomized', random_state=SEED)
X = pca.fit_transform(X_scaled).astype(np.float32)
print(f"  PCA done in {time.perf_counter() - t0:.1f}s")
print(f"  Reduced: {X_scaled.shape} -> {X.shape}")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Scree plot
fig_scree, ax = plt.subplots(figsize=(7, 3))
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
ax.plot(range(1, N_PCS + 1), cumvar, 'o-', ms=3, lw=1.5, color='steelblue')
ax.axhline(85, color='dimgrey', lw=0.8, linestyle='--', label='85% variance')
ax.axhline(95, color='coral',   lw=0.8, linestyle='--', label='95% variance')
ax.set_xlabel('Number of PCs')
ax.set_ylabel('Cumulative variance explained (%)')
ax.set_title('MNIST PCA — Cumulative Variance Explained')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig_scree, 'mnist_pca_scree.png')

# ------------------------------------------------------------------
# 3. Initialise GTSOM
# ------------------------------------------------------------------
section("3. Initialising GTSOM (20x20 hex grid)")

t0 = time.perf_counter()
som = GTSOM(
    rho_0=10.0,
    rho_f=0.5,
    halflife_epochs=100,
    n_jobs=-1,
    nbr_topo_alpha_0=0.5,
    nbr_topo_alpha_f=1.0,
    random_state=SEED,
    compute_dr_metrics=True,
    proto_topo='CONN_STK',
)
som.from_grid(
    X,
    shape=SHAPE,
    coord_init='hex',
    W_init='random',
    labels=y,
)
print(f"  Init done in {time.perf_counter() - t0:.1f}s")
print(f"  {som}")
print(f"  Empty RFs at init: {(som.recaller.RFSize == 0).sum()} / {som.M}")

# Plot initial state
fig_init = som.plot(
    color_by='labels',
    title='GTSOM — MNIST',
    subtitle='Initial state (age=0)',
)
savegg(fig_init, 'mnist_som_init.png')

# ------------------------------------------------------------------
# 4. Fit
# ------------------------------------------------------------------
section("4. Fit (200 epochs)")

print(f"  Schedule: {som.rho_schedule}")
print(f"  rho at epoch  0: {som.rho_schedule(0):.4f}")
print(f"  rho at epoch 50: {som.rho_schedule(50):.4f}")
print(f"  rho at epoch 99: {som.rho_schedule(99):.4f}")
print()

t0 = time.perf_counter()
som.fit(X, n_epochs=200, labels=y, verbose=True, plot_every=20)
wall = time.perf_counter() - t0

print(f"\n  Fit complete in {wall:.1f}s  (train_time={som.train_time:.1f}s)")
print(f"  {som}")
print(f"  Empty RFs after fit: {(som.recaller.RFSize == 0).sum()} / {som.M}")

# ------------------------------------------------------------------
# 5. Learning curves
# ------------------------------------------------------------------
section("5. Learning curves")

ages_arr   = np.array([s['age']    for s in som.learn_history_])
mqes_arr   = np.array([s['mqe']    for s in som.learn_history_])
delbmu_arr = np.array([s['delBMU'] for s in som.learn_history_])

fig_curves, (ax_mqe, ax_del) = plt.subplots(1, 2, figsize=(13, 4))

ax_mqe.plot(ages_arr, mqes_arr, 'o-', lw=1.5, ms=3, color='steelblue')
ax_mqe.axvline(0, color='dimgrey', lw=0.8, linestyle='--', label='Init (age=0)')
ax_mqe.set_xlabel('Age (epochs)')
ax_mqe.set_ylabel('Global MQE')
ax_mqe.set_title('SOM Learning\nGlobal MQE')
ax_mqe.legend(fontsize=8)
ax_mqe.grid(True, alpha=0.3)

ax_del.plot(ages_arr, delbmu_arr, 'o-', lw=1.5, ms=3, color='darkorange')
ax_del.axvline(0, color='dimgrey', lw=0.8, linestyle='--', label='Init (age=0)')
ax_del.set_xlabel('Age (epochs)')
ax_del.set_ylabel('delBMU')
ax_del.set_title('SOM Learning\nProportion of changed BMUs')
ax_del.set_ylim(-0.05, 1.05)
ax_del.legend(fontsize=8)
ax_del.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig_curves, 'mnist_learning_curves.png')

# ------------------------------------------------------------------
# 6. SOM lattice plots — all three color_by modes
# ------------------------------------------------------------------
section("6. SOM lattice plots")

for color_by, fname in [
    ('labels',  'mnist_som_labels.png'),
    ('mqe',     'mnist_som_mqe.png'),
    ('rfsize',  'mnist_som_rfsize.png'),
]:
    print(f"  color_by='{color_by}'")
    g = som.plot(
        color_by=color_by,
        title='GTSOM — MNIST (20×20 hex)',
        subtitle=f'age={som.age}, color_by={color_by!r}',
    )
    savegg(g, fname)

# ------------------------------------------------------------------
# 7. Prototype images — back-project W to pixel space
# ------------------------------------------------------------------
section("7. Prototype images (W back-projected to pixels)")

# W is in PCA score space (50-D); inverse_transform back to pixel space
W_pixels = pca.inverse_transform(som.W)         # (M, 784)
W_pixels = scaler.inverse_transform(W_pixels)   # undo [0,1] scaling
W_pixels = np.clip(W_pixels, 0, 255).astype(np.uint8)

# Arrange prototypes on their hex grid — use embed.coords to place each
# prototype image at the right lattice position
coords = som.embed.coords   # (M, 2) float

# Determine grid extent and tile size
x_vals = np.unique(np.round(coords[:, 0], 3))
y_vals = np.unique(np.round(coords[:, 1], 3))
tile   = 28   # pixels per prototype image

# Build canvas sized to the lattice
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
x_range = x_max - x_min if x_max > x_min else 1.0
y_range = y_max - y_min if y_max > y_min else 1.0

cols = SHAPE[1]
rows = SHAPE[0]
canvas_w = int(cols * tile * 1.15)   # slight padding for hex offset
canvas_h = int(rows * tile * 1.25)
canvas   = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

for m in range(som.M):
    cx = coords[m, 0]
    cy = coords[m, 1]
    # Map lattice coords to canvas pixel position
    px = int((cx - x_min) / x_range * (canvas_w - tile))
    py = int((cy - y_min) / y_range * (canvas_h - tile))
    # Flip y so that row 0 is at top
    py = canvas_h - tile - py
    img = W_pixels[m].reshape(28, 28)
    # Paste with bounds check
    py = max(0, min(py, canvas_h - tile))
    px = max(0, min(px, canvas_w - tile))
    canvas[py:py+tile, px:px+tile] = img

fig_proto, ax = plt.subplots(figsize=(12, 10))
ax.imshow(canvas, cmap='gray_r', vmin=0, vmax=255)
ax.axis('off')
ax.set_title('GTSOM — MNIST Prototype Images\n(each neuron\'s learned weight vector, back-projected to pixel space)',
             fontsize=11)
plt.tight_layout()
savefig(fig_proto, 'mnist_prototypes.png')

# ------------------------------------------------------------------
# 8. Show all figures
# ------------------------------------------------------------------
section("Done")
print(f"  Total train_time : {som.train_time:.2f}s")
print(f"  Final MQE        : {som.learn_history_[-1]['mqe']:.4f}")
print(f"  Final delBMU     : {som.learn_history_[-1]['delBMU']:.4f}")
print(f"  Empty RFs        : {(som.recaller.RFSize == 0).sum()} / {som.M}")
plt.show()
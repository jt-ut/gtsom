"""
test_blobs.py — GTSOM showcase: general (data-driven) topology learning

Unlike test_MNIST.py, which uses a fixed hexagonal grid (GTSOM.from_grid),
this script demonstrates GTSOM's namesake capability: learning on an
arbitrary, data-driven output topology via GTSOM.from_data. Instead of
choosing a lattice shape up front, the output space is built directly from
the data — prototype coordinates are computed via PCA, and the neuron
adjacency graph is the Gabriel graph of those coordinates.

Workflow:
  1. Generate 5 well-separated Gaussian blobs in 10-D (N=5,000)
  2. PCA project to 2-D (used only for visualization / score reference)
  3. Initialize GTSOM with M=100 prototypes via from_data(coord_topo='gabriel')
  4. Fit (epochs/schedule TBD — placeholder values below)
  5. Plot learning curves and the SOM lattice (all three color_by modes)

Run from the project root:
    python test_blobs.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from gtsom import GTSOM

OUTDIR   = os.path.dirname(os.path.abspath(__file__))
SEED     = 42
N        = 5_000
D        = 10
N_BLOBS  = 5
M        = 100

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
# 1. Generate blobs
# ------------------------------------------------------------------
section("1. Generating blobs")

t0 = time.perf_counter()
X_raw, y = make_blobs(
    n_samples=N,
    n_features=D,
    centers=N_BLOBS,
    cluster_std=1.0,
    random_state=SEED,
)
print(f"  Generated: {X_raw.shape}, labels: {np.unique(y)}")
print(f"  Label counts: { {c: (y==c).sum() for c in range(N_BLOBS)} }")
print(f"  Done in {time.perf_counter() - t0:.2f}s")

# ------------------------------------------------------------------
# 2. Preprocess: standardize, then PCA to 2 components (for reference /
#    visualization only — GTSOM itself trains on the standardized 10-D data)
# ------------------------------------------------------------------
section("2. Preprocessing")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw).astype(np.float32)
print(f"  Standardized: mean~0, std~1 per feature")

pca = PCA(n_components=2, random_state=SEED)
Z = pca.fit_transform(X)
print(f"  Reference 2-D PCA projection: variance explained = "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

fig_pca, ax = plt.subplots(figsize=(6, 5))
for c in range(N_BLOBS):
    mask = y == c
    ax.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.6, label=f'Blob {c}')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Reference PCA projection of the 10-D blobs')
ax.legend(fontsize=8, markerscale=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig_pca, 'blobs_pca_reference.png')

# ------------------------------------------------------------------
# 3. Initialize GTSOM — data-driven topology via from_data
# ------------------------------------------------------------------
section(f"3. Initializing GTSOM (data-driven topology, M={M})")

t0 = time.perf_counter()
som = GTSOM(
    # --- PLACEHOLDER: replace with values appropriate for this dataset ---
    rho_0=10.0,
    rho_f=0.5,
    halflife_epochs=100,
    n_jobs=-1,
    nbr_topo_alpha_0=0.5,
    nbr_topo_alpha_f=1.0,
    random_state=SEED,
    compute_dr_metrics=True,
    proto_topo='STK_CADJ',
)
som.from_data(
    X,
    M=M,
    coord_dim=2,
    coord_init='pca',
    W_init='kmeans',
    coord_topo='gabriel',
    labels=y,
)
print(f"  Init done in {time.perf_counter() - t0:.2f}s")
print(f"  {som}")
print(f"  Empty RFs at init: {(som.recaller.RFSize == 0).sum()} / {som.M}")

# Plot initial state
fig_init = som.plot(
    color_by='labels',
    title='GTSOM — Blobs (Gabriel topology)',
    subtitle='Initial state (age=0)',
)
savegg(fig_init, 'blobs_som_init.png')

# ------------------------------------------------------------------
# 4. Fit
# ------------------------------------------------------------------
section("4. Fit")

print(f"  Schedule: {som.rho_schedule}")

t0 = time.perf_counter()
# --- PLACEHOLDER: replace n_epochs/plot_every with values appropriate
#     for this dataset's size and schedule ---
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
savefig(fig_curves, 'blobs_learning_curves.png')

# ------------------------------------------------------------------
# 6. SOM lattice plots — all three color_by modes
# ------------------------------------------------------------------
section("6. SOM lattice plots")

for color_by, fname in [
    ('labels',  'blobs_som_labels.png'),
    ('mqe',     'blobs_som_mqe.png'),
    ('rfsize',  'blobs_som_rfsize.png'),
]:
    print(f"  color_by='{color_by}'")
    g = som.plot(
        color_by=color_by,
        title='GTSOM — Blobs (Gabriel topology)',
        subtitle=f'age={som.age}, color_by={color_by!r}',
    )
    savegg(g, fname)

# ------------------------------------------------------------------
# 7. Done
# ------------------------------------------------------------------
section("Done")
print(f"  Total train_time : {som.train_time:.2f}s")
print(f"  Final MQE        : {som.learn_history_[-1]['mqe']:.4f}")
print(f"  Final delBMU     : {som.learn_history_[-1]['delBMU']:.4f}")
print(f"  Empty RFs        : {(som.recaller.RFSize == 0).sum()} / {som.M}")
plt.show()
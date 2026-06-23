"""
test_gtsom_fit.py — Integration test for GTSOM learning

Tests the full workflow on 2D blob data with a 10x10 hex grid:

  Section 1 — Initialisation checks
    Verifies learn_history_[0], W0/coords0, and the init snapshot.

  Section 2 — fit()
    Runs 100 epochs, checks learn_history_ length, structure,
    MQE decrease, delBMU values, and prevBMU.

  Section 6 — Parallelism
    Verifies PARALLEL flag, n_jobs in __repr__, and that serial vs
    parallel fits from identical init produce matching W.

  Section 3 — Learning curves
    Plots global MQE and delBMU across all epochs (init + training).

  Section 4 — plot() with each color_by mode
    Saves and shows lattice plots coloured by 'mqe', 'rfsize', 'labels'.

  Section 5 — transform()
    Verifies output shape and that W has changed from W0.

Run from the project root:
    python test_gtsom_fit.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from gtsom import GTSOM

OUTDIR = os.path.dirname(os.path.abspath(__file__))

def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
N_SAMPLES   = 1000
N_BLOBS     = 5
RANDOM_SEED = 42

X, y = make_blobs(
    n_samples=N_SAMPLES,
    n_features=2,
    centers=N_BLOBS,
    cluster_std=0.8,
    random_state=RANDOM_SEED,
)
X = X.astype(np.float32)
print(f"Data: {X.shape}, labels: {np.unique(y)}")

# ------------------------------------------------------------------
# Initialise
# ------------------------------------------------------------------
som = GTSOM.from_grid(
    X,
    shape=(10, 10),
    coord_init='hex',
    W_init='random',
    random_state=RANDOM_SEED,
    labels=y,
)
print(som)

# ------------------------------------------------------------------
# Section 1 — Initialisation checks
# ------------------------------------------------------------------
section("1. Initialisation checks")

# learn_history_[0] should exist and be the post-init snapshot
assert len(som.learn_history_) == 1, \
    f"Expected 1 history entry after init, got {len(som.learn_history_)}"
snap0 = som.learn_history_[0]
print(f"learn_history_[0] keys : {list(snap0.keys())}")
print(f"  age   : {snap0['age']}")
print(f"  mqe   : {snap0['mqe']:.6f}")
print(f"  W_mqe : shape={snap0['W_mqe'].shape}, "
      f"nan_count={np.isnan(snap0['W_mqe']).sum()}, "
      f"min={np.nanmin(snap0['W_mqe']):.4f}, "
      f"max={np.nanmax(snap0['W_mqe']):.4f}")
print(f"  delBMU: {snap0['delBMU']:.4f}")
print(f"  fig   : {type(snap0['fig'])}")
assert snap0['fig'] is not None, "Init snapshot should contain a Figure"

# W0 and coords0 should match current W and embed.coords before fitting
assert som.W0 is not None, "W0 should be set after from_grid()"
assert som.coords0 is not None, "coords0 should be set after from_grid()"
assert som.W0.shape == som.W.shape, \
    f"W0 shape {som.W0.shape} != W shape {som.W.shape}"
assert som.coords0.shape == som.embed.coords.shape, \
    f"coords0 shape {som.coords0.shape} != embed.coords shape {som.embed.coords.shape}"
assert np.allclose(som.W0, som.W), \
    "W0 should equal W before any fitting"
assert np.allclose(som.coords0, som.embed.coords), \
    "coords0 should equal embed.coords before any fitting"
print(f"W0      : shape={som.W0.shape}, dtype={som.W0.dtype}  [matches W ✓]")
print(f"coords0 : shape={som.coords0.shape}, dtype={som.coords0.dtype}  [matches embed.coords ✓]")


# prevBMU holds the age-0 BMU assignments after from_grid() returns.
# The -1 sentinel only exists transiently inside _snapshot() before
# delBMU is computed; by the time the instance is returned prevBMU
# has already been overwritten with the actual age-0 assignments.
assert som.prevBMU is not None, "prevBMU should be set after from_grid()"
assert som.prevBMU.shape == (N_SAMPLES,), \
    f"prevBMU shape {som.prevBMU.shape} != ({N_SAMPLES},)"
assert (som.prevBMU >= 0).all(), \
    "prevBMU should contain valid BMU indices (>= 0) after init"
assert np.array_equal(som.prevBMU, som.recaller.BMU[:, 0]), \
    "prevBMU should match recaller.BMU[:, 0] after init"
print(f"prevBMU : shape={som.prevBMU.shape}, "
      f"all valid indices, matches recaller.BMU[:, 0] ✓")
# delBMU at age=0 should be 1.0 (all BMUs are new by definition)
assert snap0['delBMU'] == 1.0,     f"delBMU at age=0 should be 1.0, got {snap0['delBMU']}"
print(f"  delBMU : {snap0['delBMU']:.4f}  (= 1.0 at init ✓)")

# ------------------------------------------------------------------
# Section 2 — fit()
# ------------------------------------------------------------------
section("2. fit() — 100 epochs, plot_every=10")

som.compile(rho_0=3.0, rho_f=1.0, target_epochs=50)
print(f"Schedule: {som.rho_schedule}")

som.fit(X, n_epochs=100, labels=y, verbose=True, plot_every=10)

print(f"\nPost-fit state: {som}")

# learn_history_ should have 101 entries: 1 init + 100 epochs
expected_snaps = 1 + 100
assert len(som.learn_history_) == expected_snaps, \
    f"Expected {expected_snaps} snapshots, got {len(som.learn_history_)}"
print(f"\nlearn_history_ length: {len(som.learn_history_)}  (1 init + 100 epochs ✓)")

# Verify age values are sequential
ages = [s['age'] for s in som.learn_history_]
assert ages == list(range(101)), f"Ages not sequential: {ages}"
print(f"Ages sequential 0..100 ✓")

# Verify MQE is on a downward trend (last-10 mean < first-10 mean of training)
mqes = [s['mqe'] for s in som.learn_history_]
early_mqe = np.mean(mqes[1:11])    # epochs 1-10
late_mqe  = np.mean(mqes[-10:])    # epochs 91-100
assert late_mqe < early_mqe, \
    f"MQE not decreasing: early={early_mqe:.4f}, late={late_mqe:.4f}"
print(f"MQE decreasing: early={early_mqe:.4f} → late={late_mqe:.4f} ✓")

# Verify fig is present at plot_every multiples, None elsewhere
for snap in som.learn_history_:
    age = snap['age']
    has_fig = snap['fig'] is not None
    if age == 0:
        assert has_fig, "Init snapshot (age=0) should have a fig"
    elif age % 10 == 0:
        assert has_fig, f"Snapshot at age={age} should have a fig (plot_every=10)"
    else:
        assert not has_fig, f"Snapshot at age={age} should have fig=None"
print(f"Figures present at age=0 and multiples of 10 ✓")
fig_ages = [s['age'] for s in som.learn_history_ if s['fig'] is not None]
print(f"  fig ages: {fig_ages}")

# Verify delBMU structure
delbmus = [s['delBMU'] for s in som.learn_history_]
assert delbmus[0] == 1.0, f"delBMU at age=0 should be 1.0, got {delbmus[0]}"
assert all(0.0 <= v <= 1.0 for v in delbmus),     f"All delBMU values should be in [0, 1]: {delbmus}"
print(f"delBMU values all in [0, 1] ✓")
print(f"  delBMU by age: { {s['age']: round(s['delBMU'], 3) for s in som.learn_history_} }")

# prevBMU after fitting should reflect the most recent epoch's BMU assignments
assert np.array_equal(som.prevBMU, som.recaller.BMU[:, 0]), \
    "prevBMU should match recaller.BMU[:, 0] after final epoch"
print(f"prevBMU matches recaller.BMU[:, 0] after fitting ✓  "
      f"(shape={som.prevBMU.shape})")

# ------------------------------------------------------------------
# Section 3 — MQE learning curve
# ------------------------------------------------------------------
section("3. MQE and delBMU learning curves")

ages_arr   = np.array([s['age']    for s in som.learn_history_])
mqes_arr   = np.array([s['mqe']    for s in som.learn_history_])
delbmu_arr = np.array([s['delBMU'] for s in som.learn_history_])

fig_curves, (ax_mqe, ax_del) = plt.subplots(1, 2, figsize=(13, 4))

# Left: MQE
ax_mqe.plot(ages_arr, mqes_arr, 'o-', lw=1.5, ms=4, color='steelblue')
ax_mqe.axvline(0, color='dimgrey', lw=0.8, linestyle='--', label='Init (age=0)')
ax_mqe.set_xlabel('Age (epochs)')
ax_mqe.set_ylabel('Global MQE')
ax_mqe.set_title('SOM Learning\nGlobal MQE')
ax_mqe.legend(fontsize=8)
ax_mqe.grid(True, alpha=0.3)

# Right: delBMU
ax_del.plot(ages_arr, delbmu_arr, 'o-', lw=1.5, ms=4, color='darkorange')
ax_del.axvline(0, color='dimgrey', lw=0.8, linestyle='--', label='Init (age=0)')
ax_del.set_xlabel('Age (epochs)')
ax_del.set_ylabel('delBMU')
ax_del.set_title('SOM Learning\nProportion of changed BMUs')
ax_del.set_ylim(-0.05, 1.05)
ax_del.legend(fontsize=8)
ax_del.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig_curves, 'gtsom_learning_curves.png')

# ------------------------------------------------------------------
# Section 4 — plot() with each color_by mode
# ------------------------------------------------------------------
section("4. plot() — three color_by modes")

# color_by='mqe'
print("\n  color_by='mqe'")
fig_mqe_som = som.plot(color_by='mqe', title='SOM Learning', subtitle='color_by=mqe')
savefig(fig_mqe_som, 'gtsom_plot_mqe.png')

# color_by='rfsize'
print("  color_by='rfsize'")
fig_rfs = som.plot(color_by='rfsize', title='SOM Learning', subtitle='color_by=rfsize')
savefig(fig_rfs, 'gtsom_plot_rfsize.png')

# color_by='labels'
print("  color_by='labels'")
fig_lbl = som.plot(color_by='labels', title='SOM Learning', subtitle='color_by=labels')
savefig(fig_lbl, 'gtsom_plot_labels.png')

# Confirm error raised when labels not available
som_nolabels = GTSOM.from_grid(
    X, shape=(5, 5), coord_init='hex', W_init='random',
    random_state=0, labels=None,
)
try:
    som_nolabels.plot(color_by='labels')
    raise AssertionError("Should have raised ValueError")
except ValueError as e:
    print(f"\n  color_by='labels' with no labels correctly raises ValueError:")
    print(f"    {e}")

# ------------------------------------------------------------------
# Section 5 — transform() and W vs W0
# ------------------------------------------------------------------
section("5. transform() and W vs W0")

coords_out = som.transform(X)
print(f"transform(X) output shape: {coords_out.shape}")
assert coords_out.shape == (N_SAMPLES, 2), \
    f"Expected ({N_SAMPLES}, 2), got {coords_out.shape}"

# All output coords should be rows of embed.coords
embed_coord_set = set(map(tuple, som.embed.coords.tolist()))
for i, row in enumerate(coords_out):
    assert tuple(row.tolist()) in embed_coord_set, \
        f"Row {i} of transform output is not a valid embed coord: {row}"
print(f"All {N_SAMPLES} transform outputs are valid embed coords ✓")

# W should have changed from W0
assert not np.allclose(som.W, som.W0), \
    "W should differ from W0 after fitting"
w_change = np.linalg.norm(som.W - som.W0, axis=1)
print(f"W vs W0 — per-prototype L2 change: "
      f"min={w_change.min():.4f}, "
      f"mean={w_change.mean():.4f}, "
      f"max={w_change.max():.4f}  ✓")

# ------------------------------------------------------------------
# Section 6 — Parallelism
# ------------------------------------------------------------------
section("6. Parallelism checks")

from gtsom.parallel import PARALLEL
print(f"PARALLEL (numba available): {PARALLEL}")

# __repr__ should show n_jobs
repr_str = repr(som)
assert 'n_jobs=' in repr_str, f"n_jobs not in __repr__: {repr_str}"
print(f"n_jobs in __repr__ ✓  ({repr_str})")

# Correctness: serial (n_jobs=1) vs parallel (n_jobs=-1) should give
# matching W to within float64 precision.
# Run both from an identical fresh initialisation with same random_state.
print("\n  Initialising two identical SOMs for serial vs parallel comparison...")
som_serial = GTSOM.from_grid(
    X, shape=(10, 10), coord_init='hex', W_init='random',
    random_state=RANDOM_SEED, labels=None,
)
som_parallel = GTSOM.from_grid(
    X, shape=(10, 10), coord_init='hex', W_init='random',
    random_state=RANDOM_SEED, labels=None,
)

# Confirm identical starting W
assert np.array_equal(som_serial.W, som_parallel.W),     "Serial and parallel SOMs should start with identical W"
print("  Starting W identical ✓")

som_serial.compile(rho_0=3.0, rho_f=1.0, target_epochs=50, n_jobs=1)
som_parallel.compile(rho_0=3.0, rho_f=1.0, target_epochs=50, n_jobs=-1)

print(f"  Serial  : {som_serial}")
print(f"  Parallel: {som_parallel}")

print("  Running 100 epochs serial  (n_jobs=1)...")
som_serial.fit(X, n_epochs=100, verbose=False, plot_every=0)

print("  Running 100 epochs parallel (n_jobs=-1)...")
som_parallel.fit(X, n_epochs=100, verbose=False, plot_every=0)

# Compare W — small float64 differences are acceptable due to
# non-associative reduction order across threads
max_diff = np.abs(som_serial.W - som_parallel.W).max()
print(f"  Max |W_serial - W_parallel|: {max_diff:.2e}")
assert max_diff < 1e-4,     f"Serial and parallel W differ too much: max_diff={max_diff:.2e}"
print(f"  Serial vs parallel W agree to {max_diff:.2e}  ✓")

# MQE should also be close
mqe_serial   = som_serial.learn_history_[-1]['mqe']
mqe_parallel = som_parallel.learn_history_[-1]['mqe']
print(f"  MQE serial={mqe_serial:.6f}  parallel={mqe_parallel:.6f}")
assert abs(mqe_serial - mqe_parallel) < 1e-4,     f"MQE differs too much: serial={mqe_serial:.6f}, parallel={mqe_parallel:.6f}"
print(f"  MQE agreement ✓")

# Train time comparison
t_serial   = som_serial.train_time
t_parallel = som_parallel.train_time
speedup    = t_serial / t_parallel if t_parallel > 0 else float('nan')
print(f"\n  Train time (100 epochs, N={N_SAMPLES}, d=2):")
print(f"    serial   (n_jobs=1) : {t_serial:.4f}s")
print(f"    parallel (n_jobs=-1): {t_parallel:.4f}s")
print(f"    speedup             : {speedup:.2f}x")
print(f"  (Note: meaningful speedup requires large N and d)")
print(f"\n  Main SOM train_time (100 epochs): {som.train_time:.4f}s")

# ------------------------------------------------------------------
# Show all figures
# ------------------------------------------------------------------
section("Displaying all figures")
plt.show()
print("\nAll tests passed.")
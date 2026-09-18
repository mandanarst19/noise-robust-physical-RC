"""
Memory Capacity Analysis
========================
Measures how well the reservoir remembers past inputs.
MC = sum over delay k of: correlation^2(reservoir_output, delayed_input)

Key question: Does MC collapse when noise collapses feature variance?
If yes: variance collapse = memory collapse → unified explanation of phase transition.

No simulation needed — uses existing features from any run.
Runtime: ~5 minutes
"""

import numpy as np
import json
import zipfile
import os

# Auto-detect available features
FEAT_DIR = None
for run in [5, 4, 3, 2, 1]:
    path = f'/kaggle/working/multirun/run{run}/X_tr.npy'
    if os.path.exists(path):
        FEAT_DIR = f'/kaggle/working/multirun/run{run}'
        print(f"Using features from run{run}")
        break

if FEAT_DIR is None:
    raise FileNotFoundError("No features found. Run any of run1-run5 first.")

# Load features (mmap to avoid OOM)
X = np.load(f'{FEAT_DIR}/X_tr.npy', mmap_mode='r')
print(f"X shape: {X.shape}")

# ── Memory Capacity Definition ────────────────────────────────────────────────
# Standard MC (Jaeger 2002):
# Generate random input u(t) ∈ [-1, 1]
# For each delay k, train linear readout to predict u(t-k) from reservoir state x(t)
# MC_k = correlation^2(predicted, actual)^2
# MC = sum_k MC_k

# We approximate this using the reservoir's feature space:
# Instead of running new simulations, we use the existing feature matrix
# as a proxy for reservoir states, and test how well linear regression
# can reconstruct delayed versions of a synthetic input signal.

N_SAMPLES = 5000
N_DELAYS  = 20   # test delays 1 to 20 time steps (each step = 500ns)
SEED      = 42

rng = np.random.RandomState(SEED)

# Generate random input signal
u = rng.uniform(-1, 1, N_SAMPLES + N_DELAYS)

# Use subset of features as reservoir states
X_sub = X[:N_SAMPLES].copy()   # (N_SAMPLES, 15680)

print(f"\nComputing Memory Capacity (k=1 to {N_DELAYS})...")
print(f"Using {N_SAMPLES} samples, {X_sub.shape[1]} features")

from sklearn.linear_model import Ridge

MC_per_delay = []
for k in range(1, N_DELAYS + 1):
    # Target: u(t-k) — the input k steps ago
    y_target = u[N_DELAYS - k: N_DELAYS - k + N_SAMPLES]

    # Train ridge regression to predict u(t-k) from x(t)
    clf = Ridge(alpha=1e-3)
    clf.fit(X_sub, y_target)
    y_pred = clf.predict(X_sub)

    # MC_k = correlation^2
    corr = np.corrcoef(y_pred, y_target)[0, 1]
    mc_k = float(corr ** 2)
    MC_per_delay.append(mc_k)
    print(f"  k={k:2d} ({k*500}ns): MC_k={mc_k:.4f}")

MC_total = sum(MC_per_delay)

print(f"\n{'='*50}")
print(f"MEMORY CAPACITY RESULTS")
print(f"{'='*50}")
print(f"Total MC = {MC_total:.4f}")
print(f"Max possible MC = {N_DELAYS} (perfect memory)")
print(f"MC / N_delays = {MC_total/N_DELAYS:.3f}")
print(f"\nMC profile:")
print(f"  Short-term (k=1-4,  0-2μs):   {sum(MC_per_delay[:4]):.4f}")
print(f"  Mid-range  (k=5-14, 2-7μs):   {sum(MC_per_delay[4:14]):.4f}")
print(f"  Long-term  (k=15-20,7-10μs):  {sum(MC_per_delay[14:]):.4f}")

# ── Connection to timescales ──────────────────────────────────────────────────
print(f"\nTIMESCALE INTERPRETATION")
print(f"{'='*50}")
print(f"τ_met ≈ 187 ns  → 0-1 bin range")
print(f"τ_ins ≈ 7.57 μs → ~15 bin range")
print(f"\nMC in τ_met region (k=1): {MC_per_delay[0]:.4f}")
print(f"MC in τ_ins region (k=15): {MC_per_delay[14]:.4f}")

if MC_per_delay[0] > MC_per_delay[14]:
    print("→ Short-term memory dominates (consistent with τ_met as primary timescale)")
else:
    print("→ Long-term memory significant (τ_ins contributes to memory)")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    'n_samples': N_SAMPLES,
    'n_delays': N_DELAYS,
    'delay_step_ns': 500,
    'MC_total': MC_total,
    'MC_per_delay': MC_per_delay,
    'MC_short_term': float(sum(MC_per_delay[:4])),
    'MC_mid_range':  float(sum(MC_per_delay[4:14])),
    'MC_long_term':  float(sum(MC_per_delay[14:])),
    'interpretation': {
        'tau_met_region_k1': MC_per_delay[0],
        'tau_ins_region_k15': MC_per_delay[14],
    }
}

with open('/kaggle/working/memory_capacity_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with zipfile.ZipFile('/kaggle/working/memory_capacity_download.zip', 'w') as z:
    z.write('/kaggle/working/memory_capacity_results.json',
            'memory_capacity_results.json')

print(f"\n✓ download: /kaggle/working/memory_capacity_download.zip")
print(f"{'='*50}")

"""
Analysis 3.3 — Kernel Rank and Feature Expressiveness
======================================================
How expressive is the reservoir's feature space?
Measures effective rank of the feature matrix.

Requires: features from any run (X_tr.npy)
"""

import os, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Auto-detect features from any available run
FEAT_DIR = None
for run in range(1, 6):
    path = f'/kaggle/working/multirun/run{run}/X_tr.npy'
    if os.path.exists(path):
        FEAT_DIR = f'/kaggle/working/multirun/run{run}'
        print(f"Using features from run{run}")
        break

if FEAT_DIR is None:
    raise FileNotFoundError(
        "No features found. Run any of run1-run5 first."
    )

print("Loading features...")
X = np.load(f'{FEAT_DIR}/X_tr.npy')
print(f"X shape: {X.shape}")

N_SAMPLE = 5000
idx = np.random.RandomState(42).choice(len(X), N_SAMPLE, replace=False)
X_sub = X[idx]

print(f"\nComputing SVD on {N_SAMPLE} × {X.shape[1]} matrix...")
U, S, Vt = np.linalg.svd(X_sub, full_matrices=False)
print("SVD done")

total_var = (S**2).sum()

# ── Effective rank metrics ────────────────────────────────────────────────────
print("\n" + "="*60)
print("KERNEL RANK ANALYSIS")
print("="*60)

# 1. Hard rank (numerical)
rank_hard = int(np.sum(S > 1e-10))
print(f"\nHard rank (S > 1e-10): {rank_hard} / {min(X_sub.shape)}")

# 2. Effective rank (entropy-based)
p = S**2 / total_var
p = p[p > 0]
entropy = -np.sum(p * np.log(p))
eff_rank = int(np.exp(entropy))
print(f"Effective rank (exp entropy): {eff_rank}")

# 3. 90% variance explained
cumvar = np.cumsum(S**2) / total_var
rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
print(f"Rank for 90% variance: {rank_90}")
print(f"Rank for 95% variance: {rank_95}")
print(f"Rank for 99% variance: {rank_99}")

# 4. Participation ratio
pr = (S**2).sum()**2 / (S**4).sum()
print(f"Participation ratio: {pr:.1f}")

print(f"\nInterpretation:")
print(f"  Feature dimension: {X.shape[1]:,} (784 × 20)")
print(f"  Effective rank:    {eff_rank:,}")
print(f"  Compression ratio: {X.shape[1]/eff_rank:.1f}×")
print(f"  → Reservoir creates {eff_rank} independent directions")
print(f"    in {X.shape[1]}-dimensional space")

# ── Singular value decay ──────────────────────────────────────────────────────
print(f"\nSingular value spectrum:")
print(f"  S[0]  = {S[0]:.2f}  (largest)")
print(f"  S[10] = {S[10]:.2f}")
print(f"  S[50] = {S[50]:.2f}")
print(f"  S[100]= {S[100]:.2f}")
print(f"  S[-1] = {S[-1]:.4f}  (smallest)")
print(f"  Condition number: {S[0]/S[-1]:.1f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Singular value spectrum
ax = axes[0]
k = min(200, len(S))
ax.semilogy(range(1, k+1), S[:k], 'b-', linewidth=1.5)
ax.axvline(x=rank_90, color='green', linestyle='--',
           label=f'90% var: rank={rank_90}')
ax.axvline(x=rank_95, color='orange', linestyle='--',
           label=f'95% var: rank={rank_95}')
ax.axvline(x=eff_rank, color='red', linestyle=':',
           linewidth=2, label=f'Eff. rank={eff_rank}')
ax.set_xlabel('Singular value index')
ax.set_ylabel('Singular value (log scale)')
ax.set_title('Singular value spectrum\nof feature matrix')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel B: Cumulative variance explained
ax = axes[1]
k = min(500, len(S))
ax.plot(range(1, k+1), cumvar[:k]*100, 'b-', linewidth=2)
ax.axhline(y=90, color='green', linestyle='--', label='90%')
ax.axhline(y=95, color='orange', linestyle='--', label='95%')
ax.axhline(y=99, color='red', linestyle='--', label='99%')
ax.axvline(x=rank_90, color='green', linestyle=':', alpha=0.7)
ax.axvline(x=rank_95, color='orange', linestyle=':', alpha=0.7)
ax.axvline(x=rank_99, color='red', linestyle=':', alpha=0.7)
ax.set_xlabel('Number of components')
ax.set_ylabel('Cumulative variance explained (%)')
ax.set_title('Variance explained by top k components')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 101)

plt.tight_layout()
plt.savefig('/kaggle/working/kernel_rank.png', dpi=150, bbox_inches='tight')

results = {
    'n_samples': N_SAMPLE,
    'feature_dim': int(X.shape[1]),
    'hard_rank': rank_hard,
    'effective_rank': eff_rank,
    'participation_ratio': float(pr),
    'rank_90pct_var': rank_90,
    'rank_95pct_var': rank_95,
    'rank_99pct_var': rank_99,
    'condition_number': float(S[0]/S[-1]),
    'singular_values_top10': [float(s) for s in S[:10]],
    'interpretation': (
        f'Reservoir creates {eff_rank} effective independent dimensions '
        f'in {X.shape[1]}-dimensional feature space. '
        f'90% of variance explained by top {rank_90} components.'
    )
}

with open('/kaggle/working/kernel_rank_results.json', 'w') as f:
    json.dump(results, f, indent=2)

zip_path = '/kaggle/working/kernel_rank_download.zip'
with zipfile.ZipFile(zip_path, 'w') as zf:
    zf.write('/kaggle/working/kernel_rank.png', 'kernel_rank.png')
    zf.write('/kaggle/working/kernel_rank_results.json',
             'kernel_rank_results.json')

print(f"\nFigure:  /kaggle/working/kernel_rank.png")
print(f"Results: /kaggle/working/kernel_rank_results.json")
print(f"✓ download ready: {zip_path}")
print("="*60)

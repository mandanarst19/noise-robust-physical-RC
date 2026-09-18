"""
Analysis 3.2 — Mutual Information per Temporal Bin
====================================================
How much information does each temporal bin carry for each task?
Independent confirmation of timescale-task specialization.

Requires: /kaggle/working/multirun/run2/ features
"""

import os, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import RidgeClassifier

FEAT_DIR = '/kaggle/working/multirun/run2'

print("Loading features...")
X_tr = np.load(f'{FEAT_DIR}/X_tr.npy')
X_te = np.load(f'{FEAT_DIR}/X_te.npy')

# Reconstruct labels
from torchvision.datasets import MNIST
from torchvision import transforms
import torch

COLOR_PALETTE = torch.tensor([
    [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,1.0,0.0],[1.0,0.0,1.0],
    [0.0,1.0,1.0],[1.0,0.5,0.0],[0.5,0.0,1.0],[0.0,0.5,0.0],[0.5,0.5,0.5],
], dtype=torch.float32)

def get_labels(split, seed):
    rng = np.random.RandomState(seed)
    ds  = MNIST(root='/kaggle/working/data', train=(split=='train'),
                download=True, transform=transforms.ToTensor())
    yd, yc, yp = [], [], []
    for _, digit in ds:
        cid = rng.randint(0, 10)
        yd.append(digit); yc.append(cid); yp.append(digit % 2)
    return np.array(yd), np.array(yc), np.array(yp)

print("Reconstructing labels...")
y_d_tr, y_c_tr, y_p_tr = get_labels('train', 123)
y_d_tr = y_d_tr[:60000]
y_c_tr = y_c_tr[:60000]
y_p_tr = y_p_tr[:60000]

N, N_BINS = 784, 20
BIN_NS = 500

# ── Mutual information per bin ────────────────────────────────────────────────
print("\nComputing mutual information per temporal bin...")
print("(using 5000 samples for speed)")

N_SAMPLE = 5000
idx = np.random.RandomState(42).choice(60000, N_SAMPLE, replace=False)
X_sub = X_tr[idx]
y_d_sub = y_d_tr[idx]
y_c_sub = y_c_tr[idx]
y_p_sub = y_p_tr[idx]

mi_digit  = np.zeros(N_BINS)
mi_color  = np.zeros(N_BINS)
mi_parity = np.zeros(N_BINS)

for b in range(N_BINS):
    X_bin = X_sub.reshape(N_SAMPLE, N, N_BINS)[:, :, b]
    mi_digit[b]  = mutual_info_classif(X_bin, y_d_sub,
                                        discrete_features=False,
                                        random_state=42).mean()
    mi_color[b]  = mutual_info_classif(X_bin, y_c_sub,
                                        discrete_features=False,
                                        random_state=42).mean()
    mi_parity[b] = mutual_info_classif(X_bin, y_p_sub,
                                        discrete_features=False,
                                        random_state=42).mean()
    t = (b + 0.5) * BIN_NS / 1000
    print(f"  bin {b:2d} ({t:.2f}μs): "
          f"digit={mi_digit[b]:.4f}  "
          f"color={mi_color[b]:.4f}  "
          f"parity={mi_parity[b]:.4f}")

# Normalize per task
mi_d_norm = mi_digit  / mi_digit.max()
mi_c_norm = mi_color  / mi_color.max()
mi_p_norm = mi_parity / mi_parity.max()

peak_digit  = int(np.argmax(mi_digit))
peak_color  = int(np.argmax(mi_color))
peak_parity = int(np.argmax(mi_parity))

print(f"\nPeak bins:")
print(f"  Digit:  bin {peak_digit:2d}  "
      f"({(peak_digit+0.5)*BIN_NS/1000:.2f} μs)")
print(f"  Color:  bin {peak_color:2d}  "
      f"({(peak_color+0.5)*BIN_NS/1000:.2f} μs)")
print(f"  Parity: bin {peak_parity:2d}  "
      f"({(peak_parity+0.5)*BIN_NS/1000:.2f} μs)")

# Early vs late MI
early = list(range(0, 4))
late  = list(range(16, 20))

print(f"\nEarly [0-3] vs Late [16-19] MI:")
for name, mi in [('Digit', mi_digit), ('Color', mi_color), ('Parity', mi_parity)]:
    e = mi[early].mean(); l = mi[late].mean()
    print(f"  {name:6s}: early={e:.4f}  late={l:.4f}  "
          f"ratio late/early={l/e:.2f}×  "
          f"→ {'late-dominant' if l > e else 'early-dominant'}")

# ── Figure ────────────────────────────────────────────────────────────────────
bin_times = [(b + 0.5) * BIN_NS / 1000 for b in range(N_BINS)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: Raw MI per bin
ax = axes[0]
ax.plot(bin_times, mi_digit,  's-', color='royalblue',
        label='Digit',  linewidth=2, markersize=6)
ax.plot(bin_times, mi_color,  'o-', color='tomato',
        label='Color',  linewidth=2, markersize=6)
ax.plot(bin_times, mi_parity, '^-', color='seagreen',
        label='Parity', linewidth=2, markersize=6)
ax.axvline(x=0.187, color='royalblue', linestyle=':',
           alpha=0.5, label='τ_met=187ns')
ax.axvline(x=7.57,  color='tomato', linestyle=':',
           alpha=0.5, label='τ_ins=7.57μs')
ax.set_xlabel('Temporal bin center (μs)')
ax.set_ylabel('Mean mutual information (bits)')
ax.set_title('Mutual information per temporal bin')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel B: Normalized MI (heatmap style)
ax = axes[1]
data = np.vstack([mi_d_norm, mi_c_norm, mi_p_norm])
im = ax.imshow(data, aspect='auto', cmap='YlOrRd',
               extent=[0, 10, -0.5, 2.5])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Parity', 'Color', 'Digit'])
ax.set_xlabel('Time (μs)')
ax.set_title('Normalized MI per bin per task\n(brighter = more task-relevant)')
plt.colorbar(im, ax=ax, label='Normalized MI')
ax.axvline(x=0.187, color='white', linestyle='--',
           linewidth=1.5, alpha=0.8, label='τ_met')
ax.axvline(x=7.57,  color='cyan', linestyle='--',
           linewidth=1.5, alpha=0.8, label='τ_ins')
ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('/kaggle/working/mutual_info.png', dpi=150, bbox_inches='tight')

results = {
    'n_samples': N_SAMPLE,
    'peak_bins': {
        'digit':  {'bin': peak_digit,
                   'time_us': float((peak_digit+0.5)*BIN_NS/1000)},
        'color':  {'bin': peak_color,
                   'time_us': float((peak_color+0.5)*BIN_NS/1000)},
        'parity': {'bin': peak_parity,
                   'time_us': float((peak_parity+0.5)*BIN_NS/1000)},
    },
    'per_bin': {
        'times_us': [float(t) for t in bin_times],
        'digit':    [float(v) for v in mi_digit],
        'color':    [float(v) for v in mi_color],
        'parity':   [float(v) for v in mi_parity],
    },
    'early_late_ratio': {
        task: float(mi[late].mean() / mi[early].mean())
        for task, mi in [('digit', mi_digit),
                         ('color', mi_color),
                         ('parity', mi_parity)]
    }
}

with open('/kaggle/working/mutual_info_results.json', 'w') as f:
    json.dump(results, f, indent=2)

zip_path = '/kaggle/working/mutual_info_download.zip'
with zipfile.ZipFile(zip_path, 'w') as zf:
    zf.write('/kaggle/working/mutual_info.png', 'mutual_info.png')
    zf.write('/kaggle/working/mutual_info_results.json',
             'mutual_info_results.json')

print(f"\nFigure:  /kaggle/working/mutual_info.png")
print(f"Results: /kaggle/working/mutual_info_results.json")
print(f"✓ download ready: {zip_path}")
print("="*60)

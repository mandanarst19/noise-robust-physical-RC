"""
Analysis 3.1 — t-SNE Visualization of Reservoir Features
=========================================================
Projects 15,680-dimensional features to 2D.
Shows whether reservoir creates separable clusters per task.

Requires: /kaggle/working/multirun/run2/X_tr.npy
          /kaggle/working/multirun/run2/  (labels)
"""

import os, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

FEAT_DIR = '/kaggle/working/multirun/run2'

print("Loading features...")
X   = np.load(f'{FEAT_DIR}/X_tr.npy')
y_d = np.load(f'{FEAT_DIR}/y_d_tr.npy') if os.path.exists(f'{FEAT_DIR}/y_d_tr.npy') \
      else None

# Labels might be in exp01 folder
if y_d is None:
    alt = '/kaggle/working/exp01/features'
    if os.path.exists(f'{alt}/y_d_tr.npy'):
        y_d = np.load(f'{alt}/y_d_tr.npy')
        y_c = np.load(f'{alt}/y_c_tr.npy')
        y_p = np.load(f'{alt}/y_p_tr.npy')
        print(f"Labels loaded from {alt}")
    else:
        # Reconstruct labels from run2
        from torchvision.datasets import MNIST
        from torchvision import transforms
        import torch
        SEED = 123
        COLOR_PALETTE = torch.tensor([
            [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,1.0,0.0],[1.0,0.0,1.0],
            [0.0,1.0,1.0],[1.0,0.5,0.0],[0.5,0.0,1.0],[0.0,0.5,0.0],[0.5,0.5,0.5],
        ], dtype=torch.float32)
        rng = np.random.RandomState(SEED)
        ds  = MNIST(root='/kaggle/working/data', train=True, download=True,
                    transform=transforms.ToTensor())
        yd_list, yc_list, yp_list = [], [], []
        for _, digit in ds:
            cid = rng.randint(0, 10)
            yd_list.append(digit); yc_list.append(cid); yp_list.append(digit % 2)
        y_d = np.array(yd_list); y_c = np.array(yc_list); y_p = np.array(yp_list)
        print("Labels reconstructed from dataset")
else:
    y_c = np.load(f'{FEAT_DIR}/y_c_tr.npy')
    y_p = np.load(f'{FEAT_DIR}/y_p_tr.npy')

print(f"X shape: {X.shape}")

# ── Subsample for speed (t-SNE is O(n²)) ─────────────────────────────────────
N_SAMPLE = 3000
idx = np.random.RandomState(42).choice(len(X), N_SAMPLE, replace=False)
X_sub = X[idx]; y_d_sub = y_d[idx]; y_c_sub = y_c[idx]; y_p_sub = y_p[idx]

# ── Step 1: PCA to 50 dims (speeds up t-SNE) ─────────────────────────────────
print(f"Running PCA (15680 → 50 dims) on {N_SAMPLE} samples...")
pca  = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_sub)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ── Step 2: t-SNE ─────────────────────────────────────────────────────────────
print("Running t-SNE (50 → 2 dims)...")
tsne   = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_2d   = tsne.fit_transform(X_pca)
print("t-SNE done")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

color_maps = {
    'digit':  plt.cm.tab10,
    'color':  plt.cm.tab10,
    'parity': plt.cm.RdYlGn,
}

for ax, (label_name, labels, cmap) in zip(axes, [
    ('Digit (0–9)',      y_d_sub, 'tab10'),
    ('Color (10 cls)',   y_c_sub, 'tab10'),
    ('Parity (E/O)',     y_p_sub, 'RdYlGn'),
]):
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=labels, cmap=cmap,
                         s=4, alpha=0.6)
    ax.set_title(f't-SNE colored by {label_name}')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    plt.colorbar(scatter, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('VO₂ reservoir features — t-SNE projection\n'
             f'(n={N_SAMPLE} samples, PCA→50→t-SNE→2)',
             fontsize=12)
plt.tight_layout()
plt.savefig('/kaggle/working/tsne_visualization.png', dpi=150, bbox_inches='tight')

# ── Quantify cluster separation ───────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

print("\nCluster separation (k-NN accuracy in 2D t-SNE space):")
results = {}
for name, labels in [('digit', y_d_sub), ('color', y_c_sub), ('parity', y_p_sub)]:
    knn  = KNeighborsClassifier(n_neighbors=10)
    accs = cross_val_score(knn, X_2d, labels, cv=5)
    mean_acc = float(accs.mean())
    results[name] = mean_acc
    print(f"  {name:6s}: {mean_acc*100:.1f}%  "
          f"({'clusters visible' if mean_acc > 0.4 else 'overlapping'})")

output = {
    'n_samples': N_SAMPLE,
    'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
    'knn_accuracy_2d': results,
    'interpretation': (
        'k-NN accuracy in 2D t-SNE space measures cluster separability. '
        'Higher = more structured clusters for that task.'
    )
}
with open('/kaggle/working/tsne_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# ── Auto download ─────────────────────────────────────────────────────────────
zip_path = '/kaggle/working/tsne_download.zip'
with zipfile.ZipFile(zip_path, 'w') as zf:
    zf.write('/kaggle/working/tsne_visualization.png', 'tsne_visualization.png')
    zf.write('/kaggle/working/tsne_results.json', 'tsne_results.json')

print(f"\nFigure:  /kaggle/working/tsne_visualization.png")
print(f"Results: /kaggle/working/tsne_results.json")
print(f"✓ download ready: {zip_path}")
print("="*60)

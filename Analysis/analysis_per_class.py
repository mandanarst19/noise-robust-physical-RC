"""
Analysis 3.4 — Per-Class Accuracy
===================================
Which digits are hardest? Which colors? Any pattern?
Requires: /kaggle/working/multirun/run2/ features
"""

import os, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier

FEAT_DIR = '/kaggle/working/multirun/run2'

print("Loading features...")
X_tr = np.load(f'{FEAT_DIR}/X_tr.npy')
X_te = np.load(f'{FEAT_DIR}/X_te.npy')

# Load or reconstruct labels
def get_labels(split, seed):
    from torchvision.datasets import MNIST
    from torchvision import transforms
    import torch
    COLOR_PALETTE = torch.tensor([
        [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,1.0,0.0],[1.0,0.0,1.0],
        [0.0,1.0,1.0],[1.0,0.5,0.0],[0.5,0.0,1.0],[0.0,0.5,0.0],[0.5,0.5,0.5],
    ], dtype=torch.float32)
    rng = np.random.RandomState(seed)
    ds  = MNIST(root='/kaggle/working/data', train=(split=='train'),
                download=True, transform=transforms.ToTensor())
    yd, yc, yp = [], [], []
    for _, digit in ds:
        cid = rng.randint(0, 10)
        yd.append(digit); yc.append(cid); yp.append(digit % 2)
    return np.array(yd), np.array(yc), np.array(yp)

print("Reconstructing labels (seed=123)...")
y_d_tr, y_c_tr, y_p_tr = get_labels('train', 123)
y_d_te, y_c_te, y_p_te = get_labels('test',  1123)

y_d_tr = y_d_tr[:60000]; y_d_te = y_d_te[:10000]
y_c_tr = y_c_tr[:60000]; y_c_te = y_c_te[:10000]
y_p_tr = y_p_tr[:60000]; y_p_te = y_p_te[:10000]

ALPHA = 1e-3

print("Training readouts...")
clf_d = RidgeClassifier(alpha=ALPHA).fit(X_tr, y_d_tr)
clf_c = RidgeClassifier(alpha=ALPHA).fit(X_tr, y_c_tr)
clf_p = RidgeClassifier(alpha=ALPHA).fit(X_tr, y_p_tr)

pred_d = clf_d.predict(X_te)
pred_c = clf_c.predict(X_te)
pred_p = clf_p.predict(X_te)

# ── Per-class accuracy ────────────────────────────────────────────────────────
COLOR_NAMES = ['Red','Green','Blue','Yellow','Magenta',
               'Cyan','Orange','Purple','DkGreen','Gray']

print("\n" + "="*50)
print("PER-CLASS ACCURACY")
print("="*50)

digit_acc = []
print("\nDigit (0-9):")
for cls in range(10):
    mask = y_d_te == cls
    acc  = (pred_d[mask] == y_d_te[mask]).mean()
    digit_acc.append(float(acc))
    bar = '█' * int(acc * 20)
    print(f"  {cls}: {acc*100:5.1f}%  {bar}")

color_acc = []
print("\nColor:")
for cls in range(10):
    mask = y_c_te == cls
    acc  = (pred_c[mask] == y_c_te[mask]).mean()
    color_acc.append(float(acc))
    bar = '█' * int(acc * 20)
    print(f"  {COLOR_NAMES[cls]:8s}: {acc*100:5.1f}%  {bar}")

parity_acc = []
print("\nParity:")
for cls, name in [(0,'Even'), (1,'Odd')]:
    mask = y_p_te == cls
    acc  = (pred_p[mask] == y_p_te[mask]).mean()
    parity_acc.append(float(acc))
    print(f"  {name}: {acc*100:.2f}%")

print(f"\nDigit:  best={max(digit_acc)*100:.1f}%  "
      f"worst={min(digit_acc)*100:.1f}%  "
      f"range={( max(digit_acc)-min(digit_acc))*100:.1f}pp")
print(f"Color:  best={max(color_acc)*100:.1f}%  "
      f"worst={min(color_acc)*100:.1f}%  "
      f"range={(max(color_acc)-min(color_acc))*100:.1f}pp")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Digit
ax = axes[0]
bars = ax.bar(range(10), [a*100 for a in digit_acc],
              color=plt.cm.RdYlGn([a for a in digit_acc]),
              edgecolor='black', alpha=0.85)
ax.axhline(y=np.mean(digit_acc)*100, color='gray',
           linestyle='--', linewidth=1.5, label=f'Mean={np.mean(digit_acc)*100:.1f}%')
ax.set_xticks(range(10))
ax.set_xlabel('Digit class')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Per-class accuracy — Digit')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 100)

# Color
ax = axes[1]
colors_rgb = ['red','green','blue','yellow','magenta',
              'cyan','orange','purple','darkgreen','gray']
bars = ax.bar(range(10), [a*100 for a in color_acc],
              color=colors_rgb, edgecolor='black', alpha=0.85)
ax.axhline(y=np.mean(color_acc)*100, color='gray',
           linestyle='--', linewidth=1.5, label=f'Mean={np.mean(color_acc)*100:.1f}%')
ax.set_xticks(range(10))
ax.set_xticklabels([n[:3] for n in COLOR_NAMES], rotation=45, fontsize=8)
ax.set_xlabel('Color class')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Per-class accuracy — Color')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 100)

# Parity
ax = axes[2]
ax.bar(['Even', 'Odd'], [a*100 for a in parity_acc],
       color=['royalblue','tomato'], edgecolor='black', alpha=0.85)
ax.axhline(y=np.mean(parity_acc)*100, color='gray',
           linestyle='--', linewidth=1.5, label=f'Mean={np.mean(parity_acc)*100:.1f}%')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Per-class accuracy — Parity')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 100)

plt.suptitle('Per-class accuracy across all three tasks', fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/per_class_accuracy.png', dpi=150, bbox_inches='tight')

results = {
    'digit':  {'per_class': digit_acc,
               'mean': float(np.mean(digit_acc)),
               'best': int(np.argmax(digit_acc)),
               'worst': int(np.argmin(digit_acc)),
               'range_pp': float((max(digit_acc)-min(digit_acc))*100)},
    'color':  {'per_class': color_acc,
               'class_names': COLOR_NAMES,
               'mean': float(np.mean(color_acc)),
               'best': COLOR_NAMES[int(np.argmax(color_acc))],
               'worst': COLOR_NAMES[int(np.argmin(color_acc))],
               'range_pp': float((max(color_acc)-min(color_acc))*100)},
    'parity': {'per_class': parity_acc,
               'class_names': ['Even','Odd'],
               'mean': float(np.mean(parity_acc))},
}

with open('/kaggle/working/per_class_results.json', 'w') as f:
    json.dump(results, f, indent=2)

zip_path = '/kaggle/working/per_class_download.zip'
with zipfile.ZipFile(zip_path, 'w') as zf:
    zf.write('/kaggle/working/per_class_accuracy.png', 'per_class_accuracy.png')
    zf.write('/kaggle/working/per_class_results.json', 'per_class_results.json')

print(f"\nFigure:  /kaggle/working/per_class_accuracy.png")
print(f"Results: /kaggle/working/per_class_results.json")
print(f"✓ download ready: {zip_path}")
print("="*60)

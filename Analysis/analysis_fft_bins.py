"""
Analysis — FFT of Temporal Bins (Gallicchio 2017 analog)
=========================================================
Gallicchio (2017) showed that higher layers in deepESN
encode lower frequencies. We test whether temporal bins
in VO2 show the same pattern:

  Early bins (τ_met region) → high frequency content
  Late bins  (τ_ins region) → low frequency content

If yes: VO2 implements hierarchical temporal representation
physically — without stacked layers.

No simulation needed — uses existing features.
Runtime: ~2 minutes
"""

import numpy as np
import json
import zipfile
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Auto-detect features
FEAT_DIR = None
for run in [5, 4, 3, 2, 1]:
    path = f'/kaggle/working/multirun/run{run}/X_tr.npy'
    if os.path.exists(path):
        FEAT_DIR = f'/kaggle/working/multirun/run{run}'
        print(f"Using features from run{run}")
        break

if FEAT_DIR is None:
    raise FileNotFoundError("No features found.")

X = np.load(f'{FEAT_DIR}/X_tr.npy', mmap_mode='r')
N_SAMPLES = 5000
N_NEURONS = 784
N_BINS    = 20

X_sub = X[:N_SAMPLES].reshape(N_SAMPLES, N_NEURONS, N_BINS)
print(f"Shape: {X_sub.shape}  (samples, neurons, bins)")

# ── FFT analysis ──────────────────────────────────────────────────────────────
# For each bin b: take the time series across N_SAMPLES
# FFT of this gives frequency content of that bin's activity
# "frequency" here = how fast the activity varies across different input images

print("\nComputing FFT for each temporal bin...")

fft_magnitudes = []
for b in range(N_BINS):
    # Activity of all neurons at bin b, across samples
    activity = X_sub[:, :, b]   # (N_SAMPLES, N_NEURONS)
    
    # Mean across neurons → scalar time series
    mean_activity = activity.mean(axis=1)   # (N_SAMPLES,)
    
    # FFT
    fft_vals = np.abs(np.fft.rfft(mean_activity))
    fft_vals = fft_vals / fft_vals.sum()   # normalize
    fft_magnitudes.append(fft_vals)

fft_magnitudes = np.array(fft_magnitudes)   # (N_BINS, N_freq)
freqs = np.fft.rfftfreq(N_SAMPLES)

# ── Summary statistics ─────────────────────────────────────────────────────────
# For each bin: compute mean frequency (centroid)
centroids = []
for b in range(N_BINS):
    centroid = float(np.sum(freqs * fft_magnitudes[b]) / np.sum(fft_magnitudes[b]))
    centroids.append(centroid)
    t_bin = (b + 0.5) * 500
    print(f"  bin {b:2d} ({t_bin:5.0f}ns): freq centroid = {centroid:.4f}")

# ── Key comparison ─────────────────────────────────────────────────────────────
early_bins = list(range(0, 4))
late_bins  = list(range(15, 20))

early_centroid = np.mean([centroids[b] for b in early_bins])
late_centroid  = np.mean([centroids[b] for b in late_bins])

print(f"\n{'='*55}")
print("HIERARCHICAL TEMPORAL REPRESENTATION")
print("="*55)
print(f"Early bins [0-3]  (τ_met, 0-2μs):   mean freq = {early_centroid:.4f}")
print(f"Late  bins [15-19](τ_ins, 7-10μs):   mean freq = {late_centroid:.4f}")

if early_centroid > late_centroid:
    print(f"\n✓ RESULT: Early bins have HIGHER frequency content")
    print(f"  Ratio: {early_centroid/late_centroid:.2f}×")
    print(f"  → VO₂ bins implement hierarchical temporal representation")
    print(f"  → Analogous to Gallicchio (2017) deepESN layers")
else:
    print(f"\n⚠ Unexpected: Late bins have higher frequency")
    print(f"  May indicate different encoding mechanism")

# ── Figure (Gallicchio-style) ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('FFT of VO₂ temporal bins\n(Gallicchio 2017 analog: early=fast, late=slow)',
             fontsize=13)

bin_labels = [(0, 'bin 0 (τ_met, 0-500ns)'),
              (4, 'bin 4 (1-2μs)'),
              (14, 'bin 14 (τ_ins, 7-7.5μs)'),
              (19, 'bin 19 (9.5-10μs)')]

for ax, (b, label) in zip(axes.flat, bin_labels):
    # Show only low frequencies (first 50)
    k = min(50, len(freqs))
    ax.bar(range(k), fft_magnitudes[b, :k], color='royalblue', alpha=0.7)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Frequency component')
    ax.set_ylabel('Normalized magnitude')
    ax.set_xlim(0, k)

plt.tight_layout()
plt.savefig('/kaggle/working/fft_bins.png', dpi=150, bbox_inches='tight')

# Panel B: frequency centroid vs bin
fig2, ax2 = plt.subplots(figsize=(9, 5))
bin_times = [(b + 0.5) * 500 / 1000 for b in range(N_BINS)]
ax2.plot(bin_times, centroids, 'o-', color='royalblue', linewidth=2, markersize=7)
ax2.axvspan(0, 2, alpha=0.1, color='blue',  label='τ_met region (0-2μs)')
ax2.axvspan(7, 10, alpha=0.1, color='red',   label='τ_ins region (7-10μs)')
ax2.set_xlabel('Temporal bin center (μs)')
ax2.set_ylabel('Frequency centroid')
ax2.set_title('Frequency content per temporal bin\n(higher = faster dynamics)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/freq_centroid_vs_bin.png', dpi=150, bbox_inches='tight')

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    'centroids_per_bin': centroids,
    'bin_times_us': bin_times,
    'early_bins_mean_freq': float(early_centroid),
    'late_bins_mean_freq':  float(late_centroid),
    'ratio_early_over_late': float(early_centroid / late_centroid),
    'interpretation': (
        'Early bins (τ_met region) show higher frequency content than '
        'late bins (τ_ins region), consistent with hierarchical temporal '
        'representation analogous to Gallicchio (2017) deepESN layers.'
        if early_centroid > late_centroid else
        'Unexpected result — further investigation needed.'
    )
}

with open('/kaggle/working/fft_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with zipfile.ZipFile('/kaggle/working/fft_download.zip', 'w') as z:
    z.write('/kaggle/working/fft_results.json',       'fft_results.json')
    z.write('/kaggle/working/fft_bins.png',            'fft_bins.png')
    z.write('/kaggle/working/freq_centroid_vs_bin.png','freq_centroid_vs_bin.png')

print(f"\n✓ download: /kaggle/working/fft_download.zip")
print("="*55)

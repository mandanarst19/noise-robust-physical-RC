"""
Objective 1 — Universal Scaling Curve
======================================
Tests σ*(τ) ∝ τ^{-β} with normalized units
so VO₂ and ESN data points fall on the same curve.

Key: normalize both axes:
  x: τ/τ_ref  (τ_ref = smallest τ in dataset)
  y: σ*/σ_c   (σ_c = measured σ* at τ_ref)

If universal: all points collapse to a single power law

Runtime: ~45 min in Kaggle
"""
import numpy as np
import json, zipfile, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import stats

os.makedirs('/tmp/obj1_universal', exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
N_RESERVOIR  = 500
N_STEPS      = 5000
N_WARMUP     = 300
N_TRAIN      = 3000
N_TEST       = 1000
SEED         = 42
ALPHA_RIDGE  = 1e-3
N_RUNS       = 10

# More k values for better fit
NARMA_K      = [1, 2, 4, 8, 16]
LEAK_RATES   = [0.50, 0.90, 0.95, 0.99]
SIGMA_VALUES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
THRESHOLD    = 0.8

# ── VO₂ data points (from thesis) ────────────────────────────────────────────
# τ in steps (dt=10ns)
vo2_points = [
    {'label': 'VO₂ color (τ_ins)', 'tau_steps': 7570/10,  'sigma_star': 2.5e-4},
    {'label': 'VO₂ parity (τ_met)','tau_steps': 187/10,   'sigma_star': 1.53e-3},
]

# ── NARMA ─────────────────────────────────────────────────────────────────────
def generate_narma(k, n, seed=0):
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 0.5, n+k)
    y = np.zeros(n+k)
    for t in range(k, n+k):
        y[t] = np.clip(
            0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-k:t])
            + 1.5*u[t-k]*u[t] + 0.1, -10, 10)
    return u[k:], y[k:]

def nrmse(y_true, y_pred):
    v = np.var(y_true)
    return np.sqrt(np.mean((y_true-y_pred)**2)/v) if v > 0 else np.inf

# ── Main loop ─────────────────────────────────────────────────────────────────
results = {}

print(f"{'='*65}")
print(f"OBJECTIVE 1 UNIVERSAL — k={NARMA_K}")
print(f"{'='*65}")

for alpha in LEAK_RATES:
    tau = -1/np.log(1-alpha)
    results[alpha] = {}
    print(f"\nα={alpha:.2f}  τ={tau:.2f}")

    for k in NARMA_K:
        results[alpha][k] = {}
        u_seq, y_tgt = generate_narma(k, N_STEPS+N_WARMUP, seed=SEED)

        for sigma in SIGMA_VALUES:
            vals = []
            for run in range(N_RUNS):
                rng_r = np.random.RandomState(SEED+run*100)
                W = rng_r.randn(N_RESERVOIR, N_RESERVOIR)
                sr = np.max(np.abs(np.linalg.eigvals(W)))
                W = W / sr * 0.9
                W_in = rng_r.randn(N_RESERVOIR, 1) * 0.1

                rng_n = np.random.RandomState(run*50)
                x = np.zeros(N_RESERVOIR)
                X = np.zeros((N_STEPS+N_WARMUP, N_RESERVOIR))
                for t in range(N_STEPS+N_WARMUP):
                    noise = sigma * rng_n.randn(N_RESERVOIR)
                    x = ((1-alpha)*x
                         + alpha*np.tanh(W@x + W_in[:,0]*u_seq[t] + noise))
                    X[t] = x

                X_use = X[N_WARMUP:]
                y_use = y_tgt[N_WARMUP:]
                if np.any(~np.isfinite(y_use)):
                    vals.append(np.inf); continue
                clf = Ridge(alpha=ALPHA_RIDGE).fit(
                    X_use[:N_TRAIN], y_use[:N_TRAIN])
                vals.append(nrmse(
                    y_use[N_TRAIN:N_TRAIN+N_TEST],
                    clf.predict(X_use[N_TRAIN:N_TRAIN+N_TEST])))

            results[alpha][k][sigma] = {
                'mean': float(np.nanmean(vals)),
                'std':  float(np.nanstd(vals))}

        means = [results[alpha][k][s]['mean'] for s in sorted(SIGMA_VALUES)]
        print(f"  k={k:2d}: "+" ".join([f"{m:.3f}" for m in means[:5]]))

# ── σ* per (α, k) ────────────────────────────────────────────────────────────
sigma_stars = {}
for alpha in LEAK_RATES:
    sigma_stars[alpha] = {}
    for k in NARMA_K:
        sigmas_s = sorted(SIGMA_VALUES)
        ss = None
        for i in range(len(sigmas_s)-1):
            s1,s2 = sigmas_s[i], sigmas_s[i+1]
            n1 = results[alpha][k][s1]['mean']
            n2 = results[alpha][k][s2]['mean']
            if n1 <= THRESHOLD <= n2:
                t = (THRESHOLD-n1)/(n2-n1) if n2!=n1 else 0
                ss = s1 + t*(s2-s1)
                break
        sigma_stars[alpha][k] = float(ss) if ss else float(sigmas_s[-1])

# ── Universal normalization ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"UNIVERSAL SCALING — Normalized units")
print(f"x: τ/τ_ref   y: σ*/σ_ref")
print(f"{'='*65}")

# For each α: τ_ref = τ(k=1), σ_ref = σ*(k=1)
all_tau_norm  = []
all_sigma_norm = []
all_labels    = []

betas_all = []

for alpha in LEAK_RATES:
    tau = -1/np.log(1-alpha)
    sigma_ref = sigma_stars[alpha][1]  # k=1 as reference
    tau_ref_k = 1.0  # k=1

    if sigma_ref <= 0: continue

    ks_fit=[]; ss_fit=[]
    for k in NARMA_K:
        sv = sigma_stars[alpha][k]
        tau_norm   = k / tau_ref_k         # τ/τ_ref
        sigma_norm = sv / sigma_ref        # σ*/σ_ref

        all_tau_norm.append(tau_norm)
        all_sigma_norm.append(sigma_norm)
        all_labels.append(f'α={alpha}')

        if 0 < sv < SIGMA_VALUES[-1]:
            ks_fit.append(np.log(k))
            ss_fit.append(np.log(sv))

    if len(ks_fit) >= 3:
        beta, intercept, r, p, se = stats.linregress(ks_fit, ss_fit)
        betas_all.append({
            'alpha': alpha, 'tau': tau,
            'beta': beta, 'se': se, 'p': p, 'r2': r**2})
        print(f"  α={alpha:.2f} τ={tau:.2f}: "
              f"β={beta:.3f}±{se:.3f}  R²={r**2:.3f}  p={p:.3f}")

# Global fit across all α
valid_mask = [0 < s < 1.0 for s in all_sigma_norm]
tau_fit    = np.log(np.array(all_tau_norm)[valid_mask])
sigma_fit  = np.log(np.array(all_sigma_norm)[valid_mask])
if len(tau_fit) >= 4:
    beta_global, ic_global, r_g, p_g, se_g = stats.linregress(
        tau_fit, sigma_fit)
    print(f"\n  GLOBAL fit (all α): "
          f"β={beta_global:.3f}±{se_g:.3f}  R²={r_g**2:.3f}  p={p_g:.4f}")

# ── VO₂ normalized ───────────────────────────────────────────────────────────
# Normalize VO₂ to its own reference (τ_met, σ*(τ_met))
vo2_tau_ref   = vo2_points[1]['tau_steps']   # τ_met = 18.7 steps
vo2_sigma_ref = vo2_points[1]['sigma_star']  # σ*(τ_met) = 1.53e-3

vo2_tau_norm   = [p['tau_steps']/vo2_tau_ref  for p in vo2_points]
vo2_sigma_norm = [p['sigma_star']/vo2_sigma_ref for p in vo2_points]

print(f"\n  VO₂ normalized points:")
for i, p in enumerate(vo2_points):
    print(f"  {p['label']}: τ/τ_ref={vo2_tau_norm[i]:.1f}  "
          f"σ*/σ_ref={vo2_sigma_norm[i]:.3f}")

# Check VO₂ β
if len(vo2_points) >= 2:
    x_vo2 = np.log([p['tau_steps'] for p in vo2_points])
    y_vo2 = np.log([p['sigma_star'] for p in vo2_points])
    beta_vo2 = (y_vo2[0]-y_vo2[1])/(x_vo2[0]-x_vo2[1])
    print(f"\n  VO₂ β = {beta_vo2:.3f}  (Langevin prediction: -0.5)")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Objective 1 — Universal σ*(τ) Scaling\n'
             'Leaky ESN + VO₂ on normalized axes',
             fontsize=11)

colors = plt.cm.viridis(np.linspace(0, 1, len(LEAK_RATES)))

# Panel 1: Universal collapse
ax = axes[0]
markers_esn = ['o','s','^','D']
for i, alpha in enumerate(LEAK_RATES):
    tau = -1/np.log(1-alpha)
    sigma_ref = sigma_stars[alpha].get(1, None)
    if sigma_ref is None or sigma_ref <= 0: continue
    ks  = NARMA_K
    ss  = [sigma_stars[alpha][k] for k in ks]
    tau_n   = [k/1.0 for k in ks]
    sigma_n = [s/sigma_ref for s in ss]
    valid = [(t,s) for t,s in zip(tau_n,sigma_n) if 0<s<5]
    if valid:
        tv, sv = zip(*valid)
        ax.loglog(tv, sv, f'{markers_esn[i]}-',
                  color=colors[i], lw=2, ms=8,
                  label=f'ESN α={alpha} (τ={tau:.2f})',
                  alpha=0.8)

# VO₂ points
ax.loglog(vo2_tau_norm, vo2_sigma_norm,
          'r*', ms=18, zorder=10,
          label='VO₂ (color/parity)')

# Theory lines
tau_range = np.logspace(-0.1, 1.3, 100)
ax.loglog(tau_range, tau_range**(-0.5), 'k--', lw=2,
          label='β=-0.5 (Langevin theory)')
if 'beta_global' in dir():
    ax.loglog(tau_range, tau_range**beta_global, 'k:',  lw=1.5,
              label=f'β={beta_global:.2f} (empirical)')

ax.set_xlabel('τ / τ_ref (normalized timescale)', fontsize=12)
ax.set_ylabel('σ*(τ) / σ_ref (normalized threshold)', fontsize=12)
ax.set_title('Universal collapse\n(ESN + VO₂ normalized)', fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')
ax.set_xlim(0.8, 25)

# Panel 2: β values with CI
ax = axes[1]
if betas_all:
    alphas_b = [d['alpha'] for d in betas_all]
    taus_b   = [d['tau']   for d in betas_all]
    betas_b  = [d['beta']  for d in betas_all]
    ses_b    = [d['se']    for d in betas_all]

    ax.errorbar(taus_b, betas_b,
                yerr=[1.96*se for se in ses_b],
                fmt='o-', color='steelblue', lw=2, ms=10,
                capsize=5, label='ESN β ± 95% CI')
    ax.axhline(-0.5, color='red', ls='--', lw=2,
               label='β=-0.5 (Langevin)')
    ax.fill_between([0.1, 2.0], -0.75, -0.25,
                    alpha=0.1, color='red',
                    label='±0.25 band')

    if 'beta_global' in dir():
        ax.axhline(beta_global, color='navy', ls=':', lw=1.5,
                   label=f'Global β={beta_global:.2f}')

    # VO₂ point
    if 'beta_vo2' in dir():
        ax.scatter([0.05], [beta_vo2], s=200, color='tomato',
                   marker='*', zorder=10,
                   label=f'VO₂ β={beta_vo2:.2f}')

ax.set_xscale('log')
ax.set_xlabel('ESN timescale τ (steps, log)', fontsize=12)
ax.set_ylabel('Scaling exponent β', fontsize=12)
ax.set_title('β across substrates\n(top 1%: CI should not cross -0.5)', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_ylim(-2.5, 0.5)

plt.tight_layout()
fig_path = '/tmp/obj1_universal/obj1_universal.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'n_runs': N_RUNS, 'n_reservoir': N_RESERVOIR,
    'narma_k': NARMA_K,
    'betas': betas_all,
    'beta_global': float(beta_global) if 'beta_global' in dir() else None,
    'beta_vo2': float(beta_vo2) if 'beta_vo2' in dir() else None,
    'sigma_stars': {str(a):{str(k):v for k,v in kd.items()}
                    for a,kd in sigma_stars.items()},
    'vo2': vo2_points
}
with open('/tmp/obj1_universal/results.json','w') as f:
    json.dump(out, f, indent=2)
with zipfile.ZipFile('/tmp/obj1_universal_download.zip','w') as z:
    z.write('/tmp/obj1_universal/results.json','obj1_universal_results.json')
    z.write(fig_path,'obj1_universal.png')

print(f"\n✓ download: /tmp/obj1_universal_download.zip")
print(f"{'='*65}")

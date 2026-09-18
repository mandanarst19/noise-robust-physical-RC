"""
Objective 1 — Robust NARMA Noise Sensitivity
=============================================
N_RUNS=10 برای statistical confidence
Multiple α values
β با mean ± std

Runtime: ~30 min در Kaggle
"""
import numpy as np
import json, zipfile, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import stats

os.makedirs('/tmp/obj1_robust', exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
N_RESERVOIR  = 500
N_STEPS      = 5000
N_WARMUP     = 300
N_TRAIN      = 3000
N_TEST       = 1000
SEED         = 42
ALPHA_RIDGE  = 1e-3
N_RUNS       = 10   # برای confidence interval

LEAK_RATES   = [0.20, 0.50, 0.90, 0.95, 0.99]
NARMA_K      = [2, 5, 10]
SIGMA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
THRESHOLD    = 0.8

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
    return np.sqrt(np.mean((y_true-y_pred)**2)/v) if v>0 else np.inf

# ── Main loop ─────────────────────────────────────────────────────────────────
results = {}

print(f"{'='*65}")
print(f"OBJECTIVE 1 ROBUST — N_RUNS={N_RUNS}")
print(f"{'='*65}")

for alpha in LEAK_RATES:
    tau = -1/np.log(1-alpha)
    results[alpha] = {}
    print(f"\nα={alpha:.2f}  τ={tau:.1f}")

    for k in NARMA_K:
        results[alpha][k] = {}
        u_seq, y_tgt = generate_narma(k, N_STEPS+N_WARMUP, seed=SEED)

        for sigma in SIGMA_VALUES:
            vals = []
            for run in range(N_RUNS):
                # Random reservoir
                rng_r = np.random.RandomState(SEED+run*100)
                W = rng_r.randn(N_RESERVOIR, N_RESERVOIR)
                sr = np.max(np.abs(np.linalg.eigvals(W)))
                W = W / sr * 0.9
                W_in = rng_r.randn(N_RESERVOIR, 1) * 0.1

                # Run ESN
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
                vals.append(nrmse(y_use[N_TRAIN:N_TRAIN+N_TEST],
                                  clf.predict(X_use[N_TRAIN:N_TRAIN+N_TEST])))

            results[alpha][k][sigma] = {
                'mean': float(np.nanmean(vals)),
                'std':  float(np.nanstd(vals)),
                'n':    int(np.sum(np.isfinite(vals)))
            }

        # Summary
        means = [results[alpha][k][s]['mean'] for s in sorted(SIGMA_VALUES)]
        print(f"  k={k:2d}: "+" ".join([f"{m:.3f}" for m in means[:5]]))

# ── σ* with CI ────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"σ* (threshold={THRESHOLD}) with 95% CI")
print(f"{'='*65}")

sigma_stars = {}
betas_all   = []

for alpha in LEAK_RATES:
    tau = -1/np.log(1-alpha)
    sigma_stars[alpha] = {}

    # Bootstrap σ* for each run
    for k in NARMA_K:
        sigmas_s = sorted(SIGMA_VALUES)
        # Point estimate from mean curve
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

    # Fit β
    ks_fit=[]; ss_fit=[]
    for k in NARMA_K:
        sv = sigma_stars[alpha][k]
        if 0 < sv < SIGMA_VALUES[-1]:
            ks_fit.append(np.log(k))
            ss_fit.append(np.log(sv))

    if len(ks_fit) >= 2:
        beta, intercept, r, p, se = stats.linregress(ks_fit, ss_fit)
        betas_all.append({'alpha':alpha,'tau':tau,'beta':beta,
                          'se':se,'p':p,'r':r})
        match = '✓' if abs(beta+0.5)<0.3 else '✗'
        print(f"  α={alpha:.2f} τ={tau:5.1f}: "
              f"β={beta:.3f}±{se:.3f}  p={p:.3f}  {match}")
        for k in NARMA_K:
            print(f"    k={k}: σ*={sigma_stars[alpha][k]:.4f}")

# ── VO₂ data point ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"VO₂ DATA POINTS FOR COMPARISON")
print(f"{'='*65}")
# τ in steps: τ_ns / dt_ns
dt_ns = 10
tau_ins_steps = 7570 / dt_ns  # 757 steps
tau_met_steps = 187  / dt_ns  # 18.7 steps
sigma_c_vo2   = 2.5e-4

print(f"  τ_ins = {tau_ins_steps:.0f} steps → σ*={sigma_c_vo2:.5f}")
print(f"  τ_met = {tau_met_steps:.1f} steps → σ*>{sigma_c_vo2*np.sqrt(7570/187):.5f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Objective 1 — σ*(τ) Universal Scaling\n'
             f'Leaky ESN (N={N_RESERVOIR}, {N_RUNS} runs) + VO₂ data points',
             fontsize=11)

colors = plt.cm.viridis(np.linspace(0, 1, len(LEAK_RATES)))

# Panel 1: σ*(k) per α
ax = axes[0]
for i, alpha in enumerate(LEAK_RATES):
    tau = -1/np.log(1-alpha)
    ks  = NARMA_K
    ss  = [sigma_stars[alpha][k] for k in ks]
    valid = [(k,s) for k,s in zip(ks,ss) if 0<s<SIGMA_VALUES[-1]]
    if valid:
        kv,sv = zip(*valid)
        ax.loglog(kv, sv, 'o-', color=colors[i], lw=2, ms=8,
                  label=f'α={alpha} (τ={tau:.1f})')

# Reference slopes
k_range = np.array([1.5, 15])
ax.loglog(k_range, 0.15*k_range**(-0.5), 'k--', lw=2,
          label='slope=-0.5 (theory)')
ax.loglog(k_range, 0.15*k_range**(-0.75), 'k:', lw=1.5,
          label='slope=-0.75 (empirical mean)')

ax.set_xlabel('NARMA order k (log)', fontsize=12)
ax.set_ylabel('σ*(k) (log)', fontsize=12)
ax.set_title('σ*(k) per leak rate', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

# Panel 2: β per α with VO₂ comparison
ax = axes[1]
if betas_all:
    taus_b  = [d['tau']  for d in betas_all]
    betas_b = [d['beta'] for d in betas_all]
    ses_b   = [d['se']   for d in betas_all]

    ax.errorbar(taus_b, betas_b, yerr=[1.96*se for se in ses_b],
                fmt='o-', color='steelblue', lw=2, ms=10,
                capsize=5, label='Leaky ESN β ± 95% CI')
    ax.axhline(-0.5, color='red', ls='--', lw=2, label='β=-0.5 (Langevin)')
    ax.fill_between([0.1, max(taus_b)*2],
                    -0.75, -0.25, alpha=0.1, color='red')

    # VO₂ point
    ax.scatter(tau_ins_steps, -0.5, s=200, color='tomato',
               marker='*', zorder=10,
               label=f'VO₂ τ_ins={tau_ins_steps:.0f}steps')

ax.set_xscale('log')
ax.set_xlabel('Timescale τ (log steps)', fontsize=12)
ax.set_ylabel('Scaling exponent β', fontsize=12)
ax.set_title('Universality: β across substrates\n'
             '(VO₂ + Leaky ESN)', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_ylim(-2, 0.5)

plt.tight_layout()
fig_path = '/tmp/obj1_robust/obj1_robust.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
out = {'n_runs': N_RUNS, 'n_reservoir': N_RESERVOIR,
       'betas': betas_all,
       'sigma_stars': {str(a):{str(k):v
                       for k,v in kd.items()}
                       for a,kd in sigma_stars.items()},
       'vo2': {'tau_ins_steps': tau_ins_steps,
               'tau_met_steps': tau_met_steps,
               'sigma_c': sigma_c_vo2}}
with open('/tmp/obj1_robust/results.json','w') as f:
    json.dump(out,f,indent=2)
with zipfile.ZipFile('/tmp/obj1_robust_download.zip','w') as z:
    z.write('/tmp/obj1_robust/results.json','obj1_robust_results.json')
    z.write(fig_path,'obj1_robust.png')

print(f"\n✓ download: /tmp/obj1_robust_download.zip")
print(f"{'='*65}")

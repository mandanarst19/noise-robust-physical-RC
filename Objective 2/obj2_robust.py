"""
Objective 2 — Architecture Comparison Robust
=============================================
Metrics:
  1. Memory Capacity (MC) — Jaeger 2002
  2. σ*(k) and β — from NARMA tasks
  3. Correlation: does higher MC → β closer to -0.5?

Architectures:
  1. Standard ESN (random W)
  2. Ring topology
  3. Orthogonal (random orthogonal)
  4. Antisymmetric/EuSN

Runtime: ~45 min in Kaggle
"""
import numpy as np
import json, zipfile, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import stats
from scipy.linalg import expm, ortho_group

os.makedirs('/tmp/obj2_robust', exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
N_RESERVOIR  = 500
N_STEPS_MC   = 3000   # for MC measurement
N_STEPS_NARMA= 5000   # for NARMA
N_WARMUP     = 300
N_TRAIN      = 2000
N_TEST       = 800
SEED         = 42
ALPHA_RIDGE  = 1e-3
N_RUNS       = 5      # per architecture
ALPHA_LEAK   = 0.90   # fixed leak rate
NARMA_K      = [1, 2, 4, 8, 16]
SIGMA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
THRESHOLD    = 0.8
MAX_DELAY_MC = 50     # MC sum up to k=50

# ── Reservoir architectures ───────────────────────────────────────────────────
def make_standard(n, seed):
    rng = np.random.RandomState(seed)
    W   = rng.randn(n, n)
    return W / np.max(np.abs(np.linalg.eigvals(W))) * 0.9

def make_ring(n, seed):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i-1)%n] = 0.9
    rng = np.random.RandomState(seed)
    W  += rng.randn(n, n) * 0.01
    return W / np.max(np.abs(np.linalg.eigvals(W))) * 0.9

def make_orthogonal(n, seed):
    rng = np.random.RandomState(seed)
    Q   = ortho_group.rvs(n, random_state=rng)
    return Q * 0.9

def make_antisymmetric(n, seed):
    rng = np.random.RandomState(seed)
    A   = rng.randn(n, n) * 0.3
    S   = A - A.T
    return expm(S) * 0.9

ARCHITECTURES = {
    'Standard ESN':    make_standard,
    'Ring':            make_ring,
    'Orthogonal':      make_orthogonal,
    'Antisymmetric':   make_antisymmetric,
}

# ── Memory Capacity (Jaeger 2002) ─────────────────────────────────────────────
def measure_MC(W, W_in, alpha, max_delay=MAX_DELAY_MC,
               n_steps=N_STEPS_MC, n_warmup=N_WARMUP, seed=0):
    """
    MC = Σ_k R²(k)  for k=1,...,max_delay
    Input: white noise u ~ U(-0.5, 0.5)
    Target: u[t-k]
    """
    rng   = np.random.RandomState(seed)
    u     = rng.uniform(-0.5, 0.5, n_steps + max_delay + n_warmup)
    n     = W.shape[0]
    x     = np.zeros(n)
    X     = np.zeros((n_steps + max_delay + n_warmup, n))

    for t in range(n_steps + max_delay + n_warmup):
        x = (1-alpha)*x + alpha*np.tanh(W@x + W_in[:,0]*u[t])
        X[t] = x

    X_use = X[n_warmup:]
    u_use = u[n_warmup:]

    MC_total = 0.0
    MC_per_k = []
    for k in range(1, max_delay+1):
        X_tr = X_use[k:N_TRAIN+k]
        y_tr = u_use[:N_TRAIN]
        X_te = X_use[N_TRAIN+k:N_TRAIN+k+N_TEST]
        y_te = u_use[N_TRAIN:N_TRAIN+N_TEST]

        if len(X_tr) < 100: break
        clf = Ridge(alpha=1e-6).fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        ss_res = np.sum((y_te-y_pred)**2)
        ss_tot = np.sum((y_te-y_te.mean())**2)
        r2 = max(0.0, 1 - ss_res/ss_tot) if ss_tot>0 else 0.0
        MC_per_k.append(r2)
        MC_total += r2

    return MC_total, MC_per_k

# ── NARMA ────────────────────────────────────────────────────────────────────
def generate_narma(k, n, seed=0):
    rng = np.random.RandomState(seed)
    u   = rng.uniform(0, 0.5, n+k)
    y   = np.zeros(n+k)
    for t in range(k, n+k):
        y[t] = np.clip(
            0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-k:t])
            + 1.5*u[t-k]*u[t] + 0.1, -10, 10)
    return u[k:], y[k:]

def nrmse(y_true, y_pred):
    v = np.var(y_true)
    return np.sqrt(np.mean((y_true-y_pred)**2)/v) if v>0 else np.inf

# ── Main experiment ───────────────────────────────────────────────────────────
results_mc   = {}
results_narma= {}

print(f"{'='*65}")
print(f"OBJECTIVE 2 ROBUST — Architecture Comparison")
print(f"N={N_RESERVOIR}  N_RUNS={N_RUNS}  α={ALPHA_LEAK}")
print(f"{'='*65}")

for arch_name, make_W in ARCHITECTURES.items():
    print(f"\n── {arch_name} ──")
    results_mc[arch_name]    = []
    results_narma[arch_name] = {k:{} for k in NARMA_K}

    for run in range(N_RUNS):
        rng_r = np.random.RandomState(SEED+run*100)
        W     = make_W(N_RESERVOIR, SEED+run*100)
        W_in  = rng_r.randn(N_RESERVOIR,1)*0.1

        # MC measurement
        mc, mc_per_k = measure_MC(W, W_in, ALPHA_LEAK,
                                   seed=SEED+run*200)
        results_mc[arch_name].append(mc)
        if run==0:
            print(f"  Run {run}: MC={mc:.2f}")

    # NARMA noise sweep (use mean W across runs)
    for k in NARMA_K:
        u_seq, y_tgt = generate_narma(
            k, N_STEPS_NARMA+N_WARMUP, seed=SEED)
        for sigma in SIGMA_VALUES:
            vals = []
            for run in range(N_RUNS):
                W    = make_W(N_RESERVOIR, SEED+run*100)
                rng_r= np.random.RandomState(SEED+run*100)
                W_in = rng_r.randn(N_RESERVOIR,1)*0.1
                rng_n= np.random.RandomState(run*50)
                x    = np.zeros(N_RESERVOIR)
                X    = np.zeros((N_STEPS_NARMA+N_WARMUP,
                                 N_RESERVOIR))
                for t in range(N_STEPS_NARMA+N_WARMUP):
                    noise = sigma*rng_n.randn(N_RESERVOIR)
                    x = ((1-ALPHA_LEAK)*x
                         + ALPHA_LEAK*np.tanh(
                             W@x + W_in[:,0]*u_seq[t] + noise))
                    X[t] = x
                X_use = X[N_WARMUP:]
                y_use = y_tgt[N_WARMUP:]
                if np.any(~np.isfinite(y_use)):
                    vals.append(np.inf); continue
                clf = Ridge(alpha=ALPHA_RIDGE).fit(
                    X_use[:N_TRAIN], y_use[:N_TRAIN])
                vals.append(nrmse(
                    y_use[N_TRAIN:N_TRAIN+N_TEST],
                    clf.predict(
                        X_use[N_TRAIN:N_TRAIN+N_TEST])))
            results_narma[arch_name][k][sigma] = {
                'mean': float(np.nanmean(vals)),
                'std':  float(np.nanstd(vals))}
        means=[results_narma[arch_name][k][s]['mean']
               for s in sorted(SIGMA_VALUES)]
        print(f"  k={k:2d}: "
              +" ".join([f"{m:.3f}" for m in means[:4]]))

# ── σ* and β per architecture ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"σ* and β per Architecture")
print(f"{'='*65}")

sigma_stars_arch = {}
betas_arch       = {}
mc_means         = {}

for arch_name in ARCHITECTURES:
    mc_means[arch_name] = float(np.mean(results_mc[arch_name]))
    sigma_stars_arch[arch_name] = {}

    for k in NARMA_K:
        sigmas_s = sorted(SIGMA_VALUES)
        ss = None
        for i in range(len(sigmas_s)-1):
            s1,s2 = sigmas_s[i], sigmas_s[i+1]
            n1 = results_narma[arch_name][k][s1]['mean']
            n2 = results_narma[arch_name][k][s2]['mean']
            if n1 <= THRESHOLD <= n2:
                t = (THRESHOLD-n1)/(n2-n1) if n2!=n1 else 0
                ss = s1 + t*(s2-s1)
                break
        sigma_stars_arch[arch_name][k] = (
            float(ss) if ss else float(sigmas_s[-1]))

    # Fit β
    ks_fit=[]; ss_fit=[]
    for k in NARMA_K:
        sv = sigma_stars_arch[arch_name][k]
        if 0 < sv < SIGMA_VALUES[-1]:
            ks_fit.append(np.log(k))
            ss_fit.append(np.log(sv))
    if len(ks_fit) >= 3:
        b,_,r,p,se = stats.linregress(ks_fit, ss_fit)
        betas_arch[arch_name] = {
            'beta':b,'se':se,'r2':r**2,'p':p}

print(f"\n  {'Architecture':<22} {'MC':>8} "
      f"{'β':>8} {'SE':>8} {'R²':>8}")
print(f"  {'-'*58}")
for arch_name in ARCHITECTURES:
    mc  = mc_means[arch_name]
    b   = betas_arch.get(arch_name,{})
    print(f"  {arch_name:<22} {mc:>8.2f} "
          f"{b.get('beta',0):>8.3f} "
          f"{b.get('se',0):>8.3f} "
          f"{b.get('r2',0):>8.3f}")

# σ* improvement over Standard ESN
print(f"\n  σ* improvement over Standard ESN:")
baseline = sigma_stars_arch['Standard ESN']
print(f"  {'Architecture':<22} "
      + "".join([f"k={k:2d}   " for k in NARMA_K]))
for arch_name in ARCHITECTURES:
    if arch_name=='Standard ESN': continue
    ratios = [sigma_stars_arch[arch_name][k]/baseline[k]
              if baseline[k]>0 else 1.0 for k in NARMA_K]
    print(f"  {arch_name:<22} "
          + "".join([f"{r:>6.2f}× " for r in ratios]))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15,5))
fig.suptitle('Objective 2 — Architecture Noise Robustness\n'
             f'N={N_RESERVOIR}, {N_RUNS} runs, α={ALPHA_LEAK}',
             fontsize=11)

colors_arch = ['steelblue','tomato','seagreen','purple']
markers     = ['o','s','^','D']

# Panel 1: MC per architecture
ax = axes[0]
arch_names = list(ARCHITECTURES.keys())
mc_vals    = [mc_means[a] for a in arch_names]
mc_stds    = [np.std(results_mc[a]) for a in arch_names]
bars = ax.bar(range(len(arch_names)), mc_vals,
              yerr=mc_stds, color=colors_arch,
              alpha=0.85, edgecolor='black',
              capsize=5)
ax.set_xticks(range(len(arch_names)))
ax.set_xticklabels([a.replace(' ','\n') for a in arch_names],
                   fontsize=9)
ax.set_ylabel('Memory Capacity (MC)', fontsize=12)
ax.set_title('Memory Capacity\n(higher = more memory)', fontsize=10)
ax.grid(axis='y', alpha=0.3)
for bar,val in zip(bars,mc_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3,
            f'{val:.1f}', ha='center', fontsize=9)

# Panel 2: σ*(k) per architecture
ax = axes[1]
for i,arch_name in enumerate(ARCHITECTURES):
    ks = NARMA_K
    ss = [sigma_stars_arch[arch_name][k] for k in ks]
    valid = [(k,s) for k,s in zip(ks,ss)
             if 0<s<SIGMA_VALUES[-1]]
    if valid:
        kv,sv = zip(*valid)
        ax.loglog(kv, sv, f'{markers[i]}-',
                  color=colors_arch[i], lw=2, ms=8,
                  label=arch_name)

k_range = np.array([0.8, 20])
ax.loglog(k_range, 0.15*k_range**(-0.5),
          'k--', lw=1.5, label='slope=-0.5')
ax.set_xlabel('NARMA k (log)', fontsize=12)
ax.set_ylabel('σ*(k) (log)', fontsize=12)
ax.set_title('σ*(k) per architecture', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

# Panel 3: β vs MC — the key relationship
ax = axes[2]
for i,arch_name in enumerate(ARCHITECTURES):
    mc  = mc_means[arch_name]
    b   = betas_arch.get(arch_name,{})
    beta= b.get('beta', None)
    se  = b.get('se', 0)
    if beta:
        ax.errorbar(mc, beta, yerr=1.96*se,
                    fmt=markers[i], color=colors_arch[i],
                    ms=12, capsize=5,
                    label=arch_name)
        ax.annotate(arch_name,
                    xy=(mc, beta),
                    xytext=(mc+0.5, beta+0.02),
                    fontsize=8)

ax.axhline(-0.5, color='red', ls='--', lw=2,
           label='β=-0.5 (Langevin)')
ax.axhline(-0.77, color='navy', ls=':', lw=1.5,
           label='β=-0.77 (Standard ESN)')
ax.set_xlabel('Memory Capacity (MC)', fontsize=12)
ax.set_ylabel('Scaling exponent β', fontsize=12)
ax.set_title('β vs MC — key relationship\n'
             'Higher MC → β closer to -0.5?', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = '/tmp/obj2_robust/obj2_robust.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'n_runs': N_RUNS, 'n_reservoir': N_RESERVOIR,
    'alpha_leak': ALPHA_LEAK,
    'mc_means': mc_means,
    'betas': {a:betas_arch.get(a,{}) for a in ARCHITECTURES},
    'sigma_stars': {a:{str(k):v for k,v in kd.items()}
                    for a,kd in sigma_stars_arch.items()},
}
with open('/tmp/obj2_robust/results.json','w') as f:
    json.dump(out, f, indent=2)
with zipfile.ZipFile('/tmp/obj2_robust_download.zip','w') as z:
    z.write('/tmp/obj2_robust/results.json','obj2_robust_results.json')
    z.write(fig_path,'obj2_robust.png')

print(f"\n✓ download: /tmp/obj2_robust_download.zip")
print(f"{'='*65}")

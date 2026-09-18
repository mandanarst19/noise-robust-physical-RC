"""
Objective 2 Final — Architecture Comparison
=============================================
Metrics:
  1. Memory Capacity (MC)
  2. Kernel rank (effective rank from SVD)
  3. σ*(k) and β from NARMA (σ up to 5.0)

Key question: Higher MC → β closer to -0.5?

Runtime: ~60 min in Kaggle
"""
import numpy as np
import json, zipfile, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import stats
from scipy.linalg import expm
from scipy.stats import ortho_group

os.makedirs('/tmp/obj2_final', exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
N_RESERVOIR  = 300
N_STEPS_MC   = 3000
N_STEPS_NARMA= 3000
N_WARMUP     = 200
N_TRAIN      = 2000
N_TEST       = 600
SEED         = 42
ALPHA_RIDGE  = 1e-3
N_RUNS       = 5
ALPHA_LEAK   = 0.90
NARMA_K      = [1, 2, 4, 8]
SIGMA_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
THRESHOLD    = 0.8
MAX_DELAY_MC = 50

# ── Architectures ─────────────────────────────────────────────────────────────
def make_standard(n, seed):
    rng = np.random.RandomState(seed)
    W   = rng.randn(n, n)
    return W / np.max(np.abs(np.linalg.eigvals(W))) * 0.9

def make_ring(n, seed):
    W = np.zeros((n, n))
    for i in range(n): W[i,(i-1)%n] = 0.9
    rng = np.random.RandomState(seed)
    W  += rng.randn(n,n)*0.01
    return W / np.max(np.abs(np.linalg.eigvals(W))) * 0.9

def make_orthogonal(n, seed):
    return ortho_group.rvs(n, random_state=seed) * 0.9

def make_antisymmetric(n, seed):
    rng = np.random.RandomState(seed)
    A   = rng.randn(n,n)*0.3
    return expm(A-A.T)*0.9

ARCHITECTURES = {
    'Standard ESN':  make_standard,
    'Ring':          make_ring,
    'Orthogonal':    make_orthogonal,
    'Antisymmetric': make_antisymmetric,
}

# ── Memory Capacity ───────────────────────────────────────────────────────────
def measure_MC(W, W_in, alpha, seed=0):
    rng = np.random.RandomState(seed)
    n   = W.shape[0]
    u   = rng.uniform(-0.5, 0.5,
                      N_STEPS_MC+MAX_DELAY_MC+N_WARMUP)
    x   = np.zeros(n)
    X   = np.zeros((N_STEPS_MC+MAX_DELAY_MC+N_WARMUP, n))
    for t in range(N_STEPS_MC+MAX_DELAY_MC+N_WARMUP):
        x = (1-alpha)*x + alpha*np.tanh(W@x+W_in[:,0]*u[t])
        X[t] = x
    X_use = X[N_WARMUP:]; u_use = u[N_WARMUP:]
    MC = 0.0
    for k in range(1, MAX_DELAY_MC+1):
        X_tr = X_use[k:N_TRAIN+k]; y_tr = u_use[:N_TRAIN]
        X_te = X_use[N_TRAIN+k:N_TRAIN+k+N_TEST]
        y_te = u_use[N_TRAIN:N_TRAIN+N_TEST]
        if len(X_tr)<100: break
        clf = Ridge(alpha=1e-6).fit(X_tr, y_tr)
        ss_res = np.sum((y_te-clf.predict(X_te))**2)
        ss_tot = np.sum((y_te-y_te.mean())**2)
        MC += max(0.0, 1-ss_res/ss_tot) if ss_tot>0 else 0.0
    return MC

# ── Kernel Rank ───────────────────────────────────────────────────────────────
def measure_kernel_rank(W, W_in, alpha, n_samples=500, seed=0):
    """Effective rank of reservoir state matrix"""
    rng = np.random.RandomState(seed)
    u   = rng.uniform(-0.5, 0.5, n_samples+N_WARMUP)
    x   = np.zeros(W.shape[0])
    X   = np.zeros((n_samples, W.shape[0]))
    for t in range(n_samples+N_WARMUP):
        x = (1-alpha)*x + alpha*np.tanh(W@x+W_in[:,0]*u[t])
        if t >= N_WARMUP:
            X[t-N_WARMUP] = x
    # Effective rank = exp(H) where H = entropy of normalized singular values
    sv  = np.linalg.svd(X, compute_uv=False)
    sv  = sv[sv > 1e-10]
    p   = sv**2 / np.sum(sv**2)
    H   = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H))

# ── NARMA ─────────────────────────────────────────────────────────────────────
def generate_narma(k, n, seed=0):
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 0.5, n+k)
    y = np.zeros(n+k)
    for t in range(k, n+k):
        y[t] = np.clip(0.3*y[t-1]+0.05*y[t-1]*np.sum(y[t-k:t])
                       +1.5*u[t-k]*u[t]+0.1, -10, 10)
    return u[k:], y[k:]

def nrmse(y_true, y_pred):
    v = np.var(y_true)
    return np.sqrt(np.mean((y_true-y_pred)**2)/v) if v>0 else np.inf

# ── Main ──────────────────────────────────────────────────────────────────────
results_mc   = {}
results_kr   = {}
results_narma= {}

print(f"{'='*65}")
print(f"OBJECTIVE 2 FINAL")
print(f"{'='*65}")

for arch_name, make_W in ARCHITECTURES.items():
    print(f"\n── {arch_name} ──")
    results_mc[arch_name] = []
    results_kr[arch_name] = []
    results_narma[arch_name] = {k:{} for k in NARMA_K}

    for run in range(N_RUNS):
        rng_r = np.random.RandomState(SEED+run*100)
        W     = make_W(N_RESERVOIR, SEED+run*100)
        W_in  = rng_r.randn(N_RESERVOIR,1)*0.1
        results_mc[arch_name].append(
            measure_MC(W, W_in, ALPHA_LEAK, seed=SEED+run*200))
        results_kr[arch_name].append(
            measure_kernel_rank(W, W_in, ALPHA_LEAK, seed=SEED+run*300))

    print(f"  MC={np.mean(results_mc[arch_name]):.2f}"
          f"±{np.std(results_mc[arch_name]):.2f}")
    print(f"  KR={np.mean(results_kr[arch_name]):.2f}"
          f"±{np.std(results_kr[arch_name]):.2f}")

    for k in NARMA_K:
        u_seq,y_tgt = generate_narma(k,N_STEPS_NARMA+N_WARMUP,seed=SEED)
        for sigma in SIGMA_VALUES:
            vals=[]
            for run in range(N_RUNS):
                W    = make_W(N_RESERVOIR, SEED+run*100)
                rng_r= np.random.RandomState(SEED+run*100)
                W_in = rng_r.randn(N_RESERVOIR,1)*0.1
                rng_n= np.random.RandomState(run*50)
                x    = np.zeros(N_RESERVOIR)
                X    = np.zeros((N_STEPS_NARMA+N_WARMUP,N_RESERVOIR))
                for t in range(N_STEPS_NARMA+N_WARMUP):
                    noise=sigma*rng_n.randn(N_RESERVOIR)
                    x=((1-ALPHA_LEAK)*x
                       +ALPHA_LEAK*np.tanh(W@x+W_in[:,0]*u_seq[t]+noise))
                    X[t]=x
                X_use=X[N_WARMUP:]; y_use=y_tgt[N_WARMUP:]
                if np.any(~np.isfinite(y_use)):
                    vals.append(np.inf); continue
                clf=Ridge(alpha=ALPHA_RIDGE).fit(
                    X_use[:N_TRAIN],y_use[:N_TRAIN])
                vals.append(nrmse(
                    y_use[N_TRAIN:N_TRAIN+N_TEST],
                    clf.predict(X_use[N_TRAIN:N_TRAIN+N_TEST])))
            results_narma[arch_name][k][sigma]={
                'mean':float(np.nanmean(vals)),
                'std': float(np.nanstd(vals))}
        means=[results_narma[arch_name][k][s]['mean']
               for s in SIGMA_VALUES]
        print(f"  k={k}: "+" ".join([f"{m:.3f}" for m in means[:5]]))

# ── σ* and β ──────────────────────────────────────────────────────────────────
sigma_stars={}; betas={}
mc_means={}; kr_means={}

for arch_name in ARCHITECTURES:
    mc_means[arch_name]=float(np.mean(results_mc[arch_name]))
    kr_means[arch_name]=float(np.mean(results_kr[arch_name]))
    sigma_stars[arch_name]={}
    for k in NARMA_K:
        sigmas_s=SIGMA_VALUES; ss=None
        for i in range(len(sigmas_s)-1):
            s1,s2=sigmas_s[i],sigmas_s[i+1]
            n1=results_narma[arch_name][k][s1]['mean']
            n2=results_narma[arch_name][k][s2]['mean']
            if n1<=THRESHOLD<=n2:
                t=(THRESHOLD-n1)/(n2-n1) if n2!=n1 else 0
                ss=s1+t*(s2-s1); break
        sigma_stars[arch_name][k]=float(ss) if ss else float(sigmas_s[-1])
    ks_f=[]; ss_f=[]
    for k in NARMA_K:
        sv=sigma_stars[arch_name][k]
        if 0<sv<5.0:
            ks_f.append(np.log(k)); ss_f.append(np.log(sv))
    if len(ks_f)>=3:
        b,_,r,p,se=stats.linregress(ks_f,ss_f)
        betas[arch_name]={'beta':b,'se':se,'r2':r**2}

print(f"\n{'='*65}")
print(f"{'Architecture':<22} {'MC':>8} {'KR':>8} {'β':>8} {'SE':>8}")
print(f"{'-'*55}")
for arch_name in ARCHITECTURES:
    mc=mc_means[arch_name]; kr=kr_means[arch_name]
    b=betas.get(arch_name,{})
    print(f"{arch_name:<22} {mc:>8.2f} {kr:>8.2f} "
          f"{b.get('beta',0):>8.3f} {b.get('se',0):>8.3f}")

# Correlation MC vs β
mc_v  = [mc_means[a] for a in ARCHITECTURES]
bet_v = [betas.get(a,{}).get('beta',0) for a in ARCHITECTURES]
r_mb,p_mb = stats.pearsonr(mc_v, bet_v)
print(f"\nCorrelation MC vs β: r={r_mb:.3f}  p={p_mb:.3f}")
if r_mb > 0.7:
    print("✓ Higher MC → β closer to -0.5")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(15,5))
fig.suptitle('Objective 2 — Architecture Noise Robustness\n'
             'Key: Higher MC → β closer to Langevin (-0.5)?',
             fontsize=11, fontweight='bold')

colors_arch=['steelblue','tomato','seagreen','purple']
markers=['o','s','^','D']
arch_list=list(ARCHITECTURES.keys())

# Panel 1: MC
ax=axes[0]
mc_vals=[mc_means[a] for a in arch_list]
mc_stds=[np.std(results_mc[a]) for a in arch_list]
bars=ax.bar(range(len(arch_list)),mc_vals,
            yerr=mc_stds,color=colors_arch,
            alpha=0.85,edgecolor='black',capsize=5)
ax.set_xticks(range(len(arch_list)))
ax.set_xticklabels([a.replace(' ','\n') for a in arch_list],fontsize=9)
ax.set_ylabel('Memory Capacity (MC)',fontsize=12)
ax.set_title('Memory Capacity\n(Jaeger 2002)',fontsize=10)
ax.grid(axis='y',alpha=0.3)
for bar,val in zip(bars,mc_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3,f'{val:.1f}',
            ha='center',fontsize=9)

# Panel 2: σ*(k)
ax=axes[1]
for i,arch_name in enumerate(arch_list):
    ks=NARMA_K
    ss=[sigma_stars[arch_name][k] for k in ks]
    valid=[(k,s) for k,s in zip(ks,ss) if 0<s<5.0]
    if valid:
        kv,sv=zip(*valid)
        ax.loglog(kv,sv,f'{markers[i]}-',
                  color=colors_arch[i],lw=2,ms=8,
                  label=arch_name)
k_range=np.array([0.8,10])
ax.loglog(k_range,0.15*k_range**(-0.5),'k--',lw=2,label='β=-0.5')
ax.set_xlabel('NARMA k (log)',fontsize=12)
ax.set_ylabel('σ*(k) (log)',fontsize=12)
ax.set_title('σ*(k) per architecture',fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3,which='both')

# Panel 3: β vs MC
ax=axes[2]
for i,arch_name in enumerate(arch_list):
    mc=mc_means[arch_name]
    b=betas.get(arch_name,{})
    beta=b.get('beta',None)
    se=b.get('se',0)
    if beta:
        ax.errorbar(mc,beta,yerr=1.96*se,
                    fmt=markers[i],color=colors_arch[i],
                    ms=12,capsize=5,
                    label=arch_name,zorder=5)
        ax.annotate(arch_name.replace(' ','\n'),
                    xy=(mc,beta),
                    xytext=(mc+0.3,beta+0.01),
                    fontsize=7)

ax.axhline(-0.5,color='red',ls='--',lw=2,
           label='Theory β=-0.5')

# Trend line
if len(mc_v)>=3:
    mc_arr=np.array(mc_v)
    b_arr=np.array(bet_v)
    z=np.polyfit(mc_arr,b_arr,1)
    mc_range=np.linspace(min(mc_v)-1,max(mc_v)+1,100)
    ax.plot(mc_range,np.polyval(z,mc_range),
            'gray',ls=':',lw=1.5,
            label=f'Trend (r={r_mb:.2f})')

ax.set_xlabel('Memory Capacity (MC)',fontsize=12)
ax.set_ylabel('β (closer to -0.5 = better)',fontsize=12)
ax.set_title('MC vs β — Design Principle\n'
             'Higher MC → Langevin-like scaling?',fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig_path='/tmp/obj2_final/obj2_final.png'
plt.savefig(fig_path,dpi=150,bbox_inches='tight')
plt.close()

out={'mc_means':mc_means,'kr_means':kr_means,
     'betas':{a:betas.get(a,{}) for a in ARCHITECTURES},
     'sigma_stars':{a:{str(k):v for k,v in kd.items()}
                    for a,kd in sigma_stars.items()},
     'correlation_mc_beta':{'r':r_mb,'p':p_mb}}
with open('/tmp/obj2_final/results.json','w') as f:
    json.dump(out,f,indent=2)
with zipfile.ZipFile('/tmp/obj2_final_download.zip','w') as z:
    z.write('/tmp/obj2_final/results.json','obj2_final_results.json')
    z.write(fig_path,'obj2_final.png')
print(f"\n✓ download: /tmp/obj2_final_download.zip")
print(f"{'='*65}")

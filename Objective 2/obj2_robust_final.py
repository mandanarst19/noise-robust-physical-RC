"""
Objective 2 — Robust Multi-Alpha Architecture Comparison
=========================================================
Tests: Standard ESN, Ring, Orthogonal, Antisymmetric
Across: α=0.50, 0.90, 0.95, 0.99
Metrics: MC, KR, β from NARMA (σ up to 5.0)

Key question: Is the MC-β trade-off robust across leak rates?

Runtime: ~90 min in Kaggle
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

os.makedirs('/tmp/obj2_robust_final', exist_ok=True)

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
NARMA_K      = [1, 2, 4, 8]
SIGMA_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
THRESHOLD    = 0.8
MAX_DELAY_MC = 50

# Multiple leak rates — key for robustness
LEAK_RATES = [0.50, 0.90, 0.95, 0.99]

# ── Architectures ─────────────────────────────────────────────────────────────
def make_standard(n, seed):
    rng = np.random.RandomState(seed)
    W = rng.randn(n, n)
    return W / np.max(np.abs(np.linalg.eigvals(W))) * 0.9

def make_ring(n, seed):
    W = np.zeros((n, n))
    for i in range(n): W[i,(i-1)%n] = 0.9
    rng = np.random.RandomState(seed)
    W += rng.randn(n,n)*0.01
    return W / np.max(np.abs(np.linalg.eigvals(W))) * 0.9

def make_orthogonal(n, seed):
    return ortho_group.rvs(n, random_state=seed) * 0.9

def make_antisymmetric(n, seed):
    rng = np.random.RandomState(seed)
    A = rng.randn(n,n)*0.3
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
    n = W.shape[0]
    u = rng.uniform(-0.5, 0.5, N_STEPS_MC+MAX_DELAY_MC+N_WARMUP)
    x = np.zeros(n)
    X = np.zeros((N_STEPS_MC+MAX_DELAY_MC+N_WARMUP, n))
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

# ── Crash-safe checkpoint ─────────────────────────────────────────────────────
CKPT = '/tmp/obj2_robust_final/checkpoint.json'
if os.path.exists(CKPT):
    with open(CKPT) as f: results = json.load(f)
    print(f"Resuming from checkpoint: {len(results)} done")
else:
    results = {}

def save_ckpt():
    with open(CKPT,'w') as f: json.dump(results,f)

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"{'='*70}")
print(f"OBJECTIVE 2 ROBUST — {len(ARCHITECTURES)} archs × {len(LEAK_RATES)} α")
print(f"{'='*70}")

for arch_name, make_W in ARCHITECTURES.items():
    if arch_name not in results:
        results[arch_name] = {}

    for alpha in LEAK_RATES:
        alpha_key = str(alpha)
        if alpha_key in results[arch_name]:
            print(f"  {arch_name} α={alpha}: skip (done)")
            continue

        print(f"\n── {arch_name}  α={alpha} ──")
        results[arch_name][alpha_key] = {
            'mc': [], 'narma': {str(k):{} for k in NARMA_K}}

        # MC
        for run in range(N_RUNS):
            rng_r = np.random.RandomState(SEED+run*100)
            W     = make_W(N_RESERVOIR, SEED+run*100)
            W_in  = rng_r.randn(N_RESERVOIR,1)*0.1
            mc    = measure_MC(W, W_in, alpha, seed=SEED+run*200)
            results[arch_name][alpha_key]['mc'].append(mc)
        print(f"  MC={np.mean(results[arch_name][alpha_key]['mc']):.2f}"
              f"±{np.std(results[arch_name][alpha_key]['mc']):.2f}")

        # NARMA
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
                        x=((1-alpha)*x
                           +alpha*np.tanh(W@x+W_in[:,0]*u_seq[t]+noise))
                        X[t]=x
                    X_use=X[N_WARMUP:]; y_use=y_tgt[N_WARMUP:]
                    if np.any(~np.isfinite(y_use)):
                        vals.append(np.inf); continue
                    clf=Ridge(alpha=ALPHA_RIDGE).fit(
                        X_use[:N_TRAIN],y_use[:N_TRAIN])
                    vals.append(nrmse(
                        y_use[N_TRAIN:N_TRAIN+N_TEST],
                        clf.predict(X_use[N_TRAIN:N_TRAIN+N_TEST])))
                results[arch_name][alpha_key]['narma'][str(k)][str(sigma)]={
                    'mean':float(np.nanmean(vals)),
                    'std': float(np.nanstd(vals))}

            means=[results[arch_name][alpha_key]['narma'][str(k)][str(s)]['mean']
                   for s in SIGMA_VALUES]
            print(f"  k={k}: "+" ".join([f"{m:.3f}" for m in means[:5]]))
        save_ckpt()

# ── Analysis ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"RESULTS — β per (architecture, α)")
print(f"{'='*70}")

all_mc=[]; all_beta=[]
summary = {}

for arch_name in ARCHITECTURES:
    summary[arch_name] = {}
    for alpha in LEAK_RATES:
        alpha_key = str(alpha)
        if alpha_key not in results.get(arch_name,{}):
            continue

        mc = float(np.mean(results[arch_name][alpha_key]['mc']))

        # σ* and β
        sigma_stars={}
        for k in NARMA_K:
            narma_k = results[arch_name][alpha_key]['narma'][str(k)]
            ss=None
            for i in range(len(SIGMA_VALUES)-1):
                s1,s2=SIGMA_VALUES[i],SIGMA_VALUES[i+1]
                n1=narma_k[str(s1)]['mean']
                n2=narma_k[str(s2)]['mean']
                if n1<=THRESHOLD<=n2:
                    t=(THRESHOLD-n1)/(n2-n1) if n2!=n1 else 0
                    ss=s1+t*(s2-s1); break
            sigma_stars[k]=float(ss) if ss else 5.0

        ks_f=[]; ss_f=[]
        for k in NARMA_K:
            sv=sigma_stars[k]
            if 0<sv<5.0:
                ks_f.append(np.log(k)); ss_f.append(np.log(sv))
        if len(ks_f)>=3:
            b,_,r,p,se=stats.linregress(ks_f,ss_f)
            summary[arch_name][alpha]={
                'mc':mc,'beta':b,'se':se,'r2':r**2}
            all_mc.append(mc); all_beta.append(b)
            print(f"  {arch_name:<20} α={alpha:.2f}  "
                  f"MC={mc:.2f}  β={b:.3f}±{se:.3f}  R²={r**2:.3f}")

# Global correlation MC vs β
if len(all_mc)>=4:
    r_mb,p_mb=stats.pearsonr(all_mc,all_beta)
    print(f"\nGlobal r(MC,β)={r_mb:.3f}  p={p_mb:.4f}")
    print(f"Standard ESN β closest to -0.5: "
          f"{min(all_beta, key=lambda x: abs(x+0.5)):.3f}")
    print(f"Orthogonal β (flattest): "
          f"{max(all_beta):.3f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(15,5))
fig.suptitle('Objective 2 — Architecture Noise Scaling\n'
             'Trade-off: Memory Capacity vs Timescale Selectivity',
             fontsize=11, fontweight='bold')

colors_arch=['steelblue','tomato','seagreen','purple']
markers_arch=['o','s','^','D']
arch_list=list(ARCHITECTURES.keys())
colors_alpha=plt.cm.cool(np.linspace(0,1,len(LEAK_RATES)))

# Panel 1: MC per architecture (α=0.90)
ax=axes[0]
for i,arch_name in enumerate(arch_list):
    mc_vals=[summary[arch_name].get(a,{}).get('mc',0)
             for a in LEAK_RATES]
    mc_vals=[v for v in mc_vals if v>0]
    if mc_vals:
        ax.bar(i, np.mean(mc_vals),
               yerr=np.std(mc_vals),
               color=colors_arch[i], alpha=0.85,
               edgecolor='black', capsize=5)
        ax.text(i, np.mean(mc_vals)+0.3,
                f'{np.mean(mc_vals):.1f}',
                ha='center', fontsize=9)
ax.set_xticks(range(len(arch_list)))
ax.set_xticklabels([a.replace(' ','\n') for a in arch_list],fontsize=9)
ax.set_ylabel('Memory Capacity (MC)',fontsize=12)
ax.set_title('Memory Capacity\n(mean across α)',fontsize=10)
ax.grid(axis='y',alpha=0.3)

# Panel 2: β per architecture across α
ax=axes[1]
for i,arch_name in enumerate(arch_list):
    betas_a=[summary[arch_name].get(a,{}).get('beta',None)
             for a in LEAK_RATES]
    ses_a  =[summary[arch_name].get(a,{}).get('se',0)
             for a in LEAK_RATES]
    alphas_done=[a for a,b in zip(LEAK_RATES,betas_a) if b is not None]
    betas_done =[b for b in betas_a if b is not None]
    ses_done   =[s for s,b in zip(ses_a,betas_a) if b is not None]
    if betas_done:
        ax.errorbar(alphas_done, betas_done,
                    yerr=[1.96*s for s in ses_done],
                    fmt=f'{markers_arch[i]}-',
                    color=colors_arch[i],lw=2,ms=8,
                    capsize=4,label=arch_name)

ax.axhline(-0.5,color='red',ls='--',lw=2,
           label='Langevin β=-0.5')
ax.set_xlabel('Leak rate α',fontsize=12)
ax.set_ylabel('β (noise scaling exponent)',fontsize=12)
ax.set_title('β per architecture across α\n'
             'Robustness check',fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3: MC vs β trade-off
ax=axes[2]
for i,arch_name in enumerate(arch_list):
    mc_v  =[summary[arch_name].get(a,{}).get('mc',None)
            for a in LEAK_RATES]
    beta_v=[summary[arch_name].get(a,{}).get('beta',None)
            for a in LEAK_RATES]
    se_v  =[summary[arch_name].get(a,{}).get('se',0)
            for a in LEAK_RATES]
    for j,(mc,beta,se) in enumerate(zip(mc_v,beta_v,se_v)):
        if mc and beta:
            ax.errorbar(mc,beta,yerr=1.96*se,
                        fmt=markers_arch[i],
                        color=colors_arch[i],
                        ms=10 if j==0 else 6,
                        alpha=0.5+0.5*(j==0),
                        capsize=3)
    # label only once
    mc_m=np.mean([v for v in mc_v if v])
    b_m =np.mean([v for v in beta_v if v])
    ax.annotate(arch_name.split()[0],
                xy=(mc_m,b_m),
                xytext=(mc_m+0.3,b_m+0.01),
                fontsize=8,color=colors_arch[i])

ax.axhline(-0.5,color='red',ls='--',lw=2,
           label='Langevin β=-0.5')
if len(all_mc)>=4:
    mc_range=np.linspace(min(all_mc)-1,max(all_mc)+1,100)
    z=np.polyfit(all_mc,all_beta,1)
    ax.plot(mc_range,np.polyval(z,mc_range),
            'gray',ls=':',lw=1.5,
            label=f'Trend r={r_mb:.2f}')

ax.set_xlabel('Memory Capacity (MC)',fontsize=12)
ax.set_ylabel('β (noise scaling)',fontsize=12)
ax.set_title('MC-β Trade-off\nHigh MC → uniform sensitivity',fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig_path='/tmp/obj2_robust_final/obj2_robust_final.png'
plt.savefig(fig_path,dpi=150,bbox_inches='tight')
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
out={'summary':summary,'correlation_mc_beta':{'r':r_mb,'p':p_mb}
     if len(all_mc)>=4 else {}}
with open('/tmp/obj2_robust_final/results.json','w') as f:
    json.dump(out,f,indent=2)
with zipfile.ZipFile('/tmp/obj2_robust_final_download.zip','w') as z:
    z.write('/tmp/obj2_robust_final/results.json','obj2_robust_final.json')
    z.write(fig_path,'obj2_robust_final.png')
print(f"\n✓ download: /tmp/obj2_robust_final_download.zip")
print(f"{'='*70}")

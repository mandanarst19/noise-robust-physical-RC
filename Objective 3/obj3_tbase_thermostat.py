"""
Objective 3 — T_base Thermostat Homeostasis
=============================================
No threshold — continuous proportional control
like a thermostat.

Rule:
  error = log(target_featvar) - log(current_featvar)
  T_base_new = T_base - η_T × error

This is purely unsupervised:
  - No labeled data needed
  - No threshold to tune
  - Proportional to drift magnitude

Runtime: ~4-5 hours (Kaggle T4 GPU)
Crash-safe: checkpoint after each step
"""
import os, sys, json, time, zipfile
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.linear_model import RidgeClassifier

SRC = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'model.py' in files: SRC = root; break
sys.path.insert(0, SRC)
from model import Circuit2D

# ── Config ────────────────────────────────────────────────────────────────────
NOISE        = 0.0002
CTH          = 0.15
COUPLE       = 0.02
V_MIN        = 10.5
V_MAX        = 12.2
R_LOAD       = 12.0
DT           = 10
T_MAX        = 10000
N_BINS       = 20
LEN_Y        = 50
N            = 784
BATCH        = 32
N_TRAIN      = 60000
N_TEST       = 10000
SEED         = 42
ALPHA        = 1e-3

T_BASE_NOMINAL = 325.0
T_BASE_DRIFT   = 331.0
T_BASE_MIN     = 320.0
T_BASE_MAX     = 335.0

# ── Thermostat parameters ─────────────────────────────────────────────────────
TARGET_FEATVAR = 3.162   # baseline FeatVar at T=325K
ETA_T          = 1.0     # learning rate for T_base (K per unit error)
MAX_STEPS      = 8       # maximum homeostasis steps
CONVERGE_TOL   = 0.10    # stop if |log(current/target)| < 0.10 (10% error)

OUT_DIR = '/kaggle/working/obj3_thermostat'
os.makedirs(OUT_DIR, exist_ok=True)
CKPT = f'{OUT_DIR}/checkpoint.json'

# ── Color palette ─────────────────────────────────────────────────────────────
COLOR_PALETTE = torch.tensor([
    [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],
    [1.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,1.0],
    [1.0,0.5,0.0],[0.5,0.0,1.0],[0.0,0.5,0.0],
    [0.5,0.5,0.5],
], dtype=torch.float32)

# ── Dataset ───────────────────────────────────────────────────────────────────
def make_dataset(split):
    root='/kaggle/working/data'; os.makedirs(root,exist_ok=True)
    rng=np.random.RandomState(SEED if split=='train' else SEED+1000)
    ds=MNIST(root=root,train=(split=='train'),download=True,
             transform=transforms.ToTensor())
    imgs,yd,yc,yp=[],[],[],[]
    for img,digit in ds:
        gray=img.squeeze(0); cid=rng.randint(0,10)
        col=torch.zeros(3,28,28)
        for ch in range(3): col[ch]=gray*COLOR_PALETTE[cid,ch]
        imgs.append(col); yd.append(digit)
        yc.append(cid); yp.append(digit%2)
    return (torch.stack(imgs),torch.tensor(yd),
            torch.tensor(yc),torch.tensor(yp))

# ── Feature extraction ────────────────────────────────────────────────────────
def extract(images, n, t_base, tag):
    path = f'{OUT_DIR}/{tag}.npy'
    if os.path.exists(path):
        X=np.load(path)
        print(f"  loaded {tag}  var={np.var(X):.4f}")
        return X
    images=images[:n]
    R=images[:,0]; G=images[:,1]; B=images[:,2]
    V_all=(V_MIN+(V_MAX-V_MIN)*(0.299*R+0.587*G+0.114*B)).view(n,N)
    feats=[]; t0=time.time()
    for s in range(0,n,BATCH):
        e=min(s+BATCH,n); bsz=e-s
        c=Circuit2D(batch=bsz,Nx=28,Ny=28,V=V_MIN,R=R_LOAD,
                    noise_strength=NOISE,Cth_factor=CTH,
                    couple_factor=COUPLE,width_factor=1.0,
                    T_base=t_base)
        c.set_input(V=V_all[s:e])
        y0=torch.stack([torch.zeros(bsz,N),
                        torch.ones(bsz,N)*t_base],dim=1)
        with torch.no_grad(): _,I=c.solve(y0,T_MAX,DT)
        feats.append(I.view(bsz,N,N_BINS,LEN_Y).max(-1)
                     .values.view(bsz,-1).cpu().numpy())
        elapsed=time.time()-t0; rate=e/elapsed if elapsed>0 else 1
        print(f"  [{e}/{n}]  {elapsed:.0f}s  "
              f"ETA {(n-e)/rate:.0f}s",end='\r')
    print()
    X=np.concatenate(feats); np.save(path,X)
    print(f"  saved  var={np.var(X):.4f}")
    return X

def train_eval(X_tr, X_te, y_d_tr, y_d_te,
               y_c_tr, y_c_te, y_p_tr, y_p_te):
    results={}
    for name,y_tr,y_te in [('digit',y_d_tr,y_d_te),
                             ('color',y_c_tr,y_c_te),
                             ('parity',y_p_tr,y_p_te)]:
        clf=RidgeClassifier(alpha=ALPHA).fit(X_tr,y_tr)
        results[name]=float((clf.predict(X_te)==y_te).mean())
    return results

# ── Thermostat rule ───────────────────────────────────────────────────────────
def thermostat_update(T_current, current_fv, target_fv, eta):
    """
    Proportional control in log-space:
    error = log(target) - log(current)
    T_new = T_current - eta × error

    If current_fv < target_fv: error > 0 → T decreases → τ_ins increases
    If current_fv > target_fv: error < 0 → T increases → τ_ins decreases
    """
    error  = np.log(target_fv) - np.log(current_fv + 1e-10)
    T_new  = T_current - eta * error
    T_new  = float(np.clip(T_new, T_BASE_MIN, T_BASE_MAX))
    return T_new, error

# ── Load checkpoint ───────────────────────────────────────────────────────────
if os.path.exists(CKPT):
    with open(CKPT) as f: ckpt=json.load(f)
    print(f"Resuming from checkpoint")
else:
    ckpt={}

def save_ckpt():
    with open(CKPT,'w') as f: json.dump(ckpt,f,indent=2)

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
tr_imgs,tr_d,tr_c,tr_p=make_dataset('train')
te_imgs,te_d,te_c,te_p=make_dataset('test')
y_d_tr=tr_d[:N_TRAIN].numpy(); y_d_te=te_d[:N_TEST].numpy()
y_c_tr=tr_c[:N_TRAIN].numpy(); y_c_te=te_c[:N_TEST].numpy()
y_p_tr=tr_p[:N_TRAIN].numpy(); y_p_te=te_p[:N_TEST].numpy()

# ── Step A: Baseline ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"[A] BASELINE  T_base={T_BASE_NOMINAL}K")
print(f"{'='*65}")

if 'baseline' not in ckpt:
    X_tr=extract(tr_imgs,N_TRAIN,T_BASE_NOMINAL,'Xtr_325')
    X_te=extract(te_imgs,N_TEST, T_BASE_NOMINAL,'Xte_325')
    res=train_eval(X_tr,X_te,y_d_tr,y_d_te,
                   y_c_tr,y_c_te,y_p_tr,y_p_te)
    fv=float(np.var(X_tr))
    ckpt['baseline']={
        'T_base':T_BASE_NOMINAL,'results':res,'feat_var':fv}
    save_ckpt()

b=ckpt['baseline']
print(f"  digit={b['results']['digit']*100:.2f}%  "
      f"color={b['results']['color']*100:.2f}%  "
      f"parity={b['results']['parity']*100:.2f}%")
print(f"  FeatVar={b['feat_var']:.4f}  (= TARGET)")

# ── Step B: Drift ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"[B] DRIFT  T_base={T_BASE_DRIFT}K")
print(f"{'='*65}")

if 'drift' not in ckpt:
    X_tr=extract(tr_imgs,N_TRAIN,T_BASE_DRIFT,'Xtr_331')
    X_te=extract(te_imgs,N_TEST, T_BASE_DRIFT,'Xte_331')
    res=train_eval(X_tr,X_te,y_d_tr,y_d_te,
                   y_c_tr,y_c_te,y_p_tr,y_p_te)
    fv=float(np.var(X_tr))
    ckpt['drift']={
        'T_base':T_BASE_DRIFT,'results':res,'feat_var':fv}
    save_ckpt()

d=ckpt['drift']
print(f"  digit={d['results']['digit']*100:.2f}%  "
      f"color={d['results']['color']*100:.2f}%  "
      f"parity={d['results']['parity']*100:.2f}%")
print(f"  FeatVar={d['feat_var']:.4f}")

# First thermostat update from drift
T_next, err = thermostat_update(
    T_BASE_DRIFT, d['feat_var'], TARGET_FEATVAR, ETA_T)
print(f"\n  Thermostat prediction:")
print(f"  error = log({TARGET_FEATVAR:.3f}) - log({d['feat_var']:.4f}) = {err:.3f}")
print(f"  T_new = {T_BASE_DRIFT} - {ETA_T}×{err:.3f} = {T_next:.2f}K")

# ── Step C: Thermostat homeostasis ────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"[C] THERMOSTAT HOMEOSTASIS")
print(f"  Target FeatVar: {TARGET_FEATVAR:.3f}  (baseline)")
print(f"  η_T = {ETA_T}  (K per unit log-error)")
print(f"  Convergence: |error| < {CONVERGE_TOL}")
print(f"  No threshold — continuous proportional control")
print(f"{'='*65}")

if 'homeostasis' not in ckpt:
    ckpt['homeostasis']={
        'steps':[],'current_T':T_BASE_DRIFT,
        'current_fv':d['feat_var']}
    save_ckpt()

T_current =ckpt['homeostasis']['current_T']
current_fv=ckpt['homeostasis']['current_fv']
steps_done=ckpt['homeostasis']['steps']
step_num  =len(steps_done)

for iteration in range(step_num, MAX_STEPS):
    # Compute thermostat update
    T_new, error = thermostat_update(
        T_current, current_fv, TARGET_FEATVAR, ETA_T)

    print(f"\n  Step {iteration+1}:")
    print(f"    Current: T={T_current:.2f}K  FeatVar={current_fv:.4f}")
    print(f"    Error:   log({TARGET_FEATVAR:.3f}/({current_fv:.4f})) = {error:.3f}")
    print(f"    Update:  T_new = {T_current:.2f} - {ETA_T}×{error:.3f} "
          f"= {T_new:.2f}K")

    # Check convergence
    if abs(error) < CONVERGE_TOL:
        print(f"  ✓ Converged! |error|={abs(error):.3f} < {CONVERGE_TOL}")
        break

    # Round T_new to nearest 0.5K for simulation
    T_sim = round(T_new * 2) / 2
    print(f"    Simulating at T={T_sim:.1f}K...")

    tag_tr=f'Xtr_{T_sim:.1f}'.replace('.','p')
    tag_te=f'Xte_{T_sim:.1f}'.replace('.','p')

    X_tr=extract(tr_imgs,N_TRAIN,T_sim,tag_tr)
    X_te=extract(te_imgs,N_TEST, T_sim,tag_te)
    res=train_eval(X_tr,X_te,y_d_tr,y_d_te,
                   y_c_tr,y_c_te,y_p_tr,y_p_te)
    fv=float(np.var(X_tr))

    print(f"    Result: digit={res['digit']*100:.2f}%  "
          f"color={res['color']*100:.2f}%  "
          f"parity={res['parity']*100:.2f}%")
    print(f"    FeatVar={fv:.4f}  "
          f"(target={TARGET_FEATVAR:.3f}  "
          f"error={abs(np.log(TARGET_FEATVAR)-np.log(fv+1e-10)):.3f})")

    steps_done.append({
        'step':iteration+1,
        'T_sim':T_sim,'T_predicted':T_new,
        'error_before':error,
        'results':res,
        'digit':res['digit'],'color':res['color'],
        'parity':res['parity'],'feat_var':fv})

    ckpt['homeostasis']['steps']=steps_done
    ckpt['homeostasis']['current_T']=T_sim
    ckpt['homeostasis']['current_fv']=fv
    save_ckpt()

    T_current =T_sim
    current_fv=fv

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"THERMOSTAT HOMEOSTASIS SUMMARY")
print(f"Target FeatVar = {TARGET_FEATVAR:.3f} (baseline at T=325K)")
print(f"{'='*65}")

b_color=ckpt['baseline']['results']['color']
b_fv   =ckpt['baseline']['feat_var']
d_color=ckpt['drift']['results']['color']
d_fv   =ckpt['drift']['feat_var']

print(f"\n{'Step':<8} {'T(K)':>7} {'Color':>8} "
      f"{'FeatVar':>10} {'FV error':>10} {'Color rec':>11}")
print(f"{'-'*60}")
print(f"  {'Base':<6} {T_BASE_NOMINAL:>7.1f} "
      f"{b_color*100:>7.2f}%  {b_fv:>10.4f}  {'0.000':>10}  {'baseline':>10}")
print(f"  {'Drift':<6} {T_BASE_DRIFT:>7.1f} "
      f"{d_color*100:>7.2f}%  {d_fv:>10.4f}  "
      f"{abs(np.log(TARGET_FEATVAR)-np.log(d_fv+1e-10)):>10.3f}  "
      f"{'0.0%':>10}")

for s in steps_done:
    fv_err=abs(np.log(TARGET_FEATVAR)-np.log(s['feat_var']+1e-10))
    dist_drift=abs(d_color-b_color)
    dist_curr =abs(s['color']-b_color)
    rec=(1-dist_curr/dist_drift)*100 if dist_drift>0 else 0
    print(f"  {s['step']:<6} {s['T_sim']:>7.1f} "
          f"{s['color']*100:>7.2f}%  {s['feat_var']:>10.4f}  "
          f"{fv_err:>10.3f}  {rec:>10.1f}%")

if steps_done:
    final=steps_done[-1]
    h_color=final['color']; h_fv=final['feat_var']
    dist_drift=abs(d_color-b_color)
    dist_final=abs(h_color-b_color)
    rec_c=(1-dist_final/dist_drift)*100 if dist_drift>0 else 0

    dist_drift_fv=abs(d_fv-b_fv)
    dist_final_fv=abs(h_fv-b_fv)
    rec_fv=(1-dist_final_fv/dist_drift_fv)*100 if dist_drift_fv>0 else 0

    print(f"\n  Color degradation:   {abs(d_color-b_color)*100:.1f}pp")
    print(f"  Color recovery:      {rec_c:.1f}%")
    print(f"  FeatVar degradation: {abs(d_fv-b_fv):.3f}")
    print(f"  FeatVar recovery:    {rec_fv:.1f}%")
    print(f"  Steps required:      {len(steps_done)}")
    print(f"  T_base trajectory:   {T_BASE_DRIFT}K → "
          f"{' → '.join([str(s['T_sim'])+'K' for s in steps_done])}")
    print(f"\n  Key advantage of thermostat approach:")
    print(f"  ✓ No threshold to tune")
    print(f"  ✓ Unsupervised (FeatVar only — no labels)")
    print(f"  ✓ Proportional: large drift → large correction")
    print(f"  ✓ Self-limiting: small error → small correction")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(CKPT,'w') as f: json.dump(ckpt,f,indent=2)
with zipfile.ZipFile(
    '/kaggle/working/obj3_thermostat_download.zip','w') as z:
    z.write(CKPT,'obj3_thermostat_results.json')
print(f"\n✓ download: /kaggle/working/obj3_thermostat_download.zip")

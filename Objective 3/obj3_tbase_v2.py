"""
Objective 3 — T_base Homeostasis v2
=====================================
Monitors BOTH:
  1. Color accuracy  (task performance proxy)
  2. Feature variance (unsupervised proxy)

Rule:
  if color < COLOR_THRESHOLD OR featvar < FEATVAR_THRESHOLD:
      T_base -= T_STEP

This shows both signals agree and either alone is sufficient.

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

T_BASE_NOMINAL   = 325.0
T_BASE_DRIFT     = 331.0
T_BASE_MIN       = 320.0

# ── Homeostasis thresholds ────────────────────────────────────────────────────
COLOR_THRESHOLD  = 0.65    # if color < this → adjust
FEATVAR_THRESHOLD= 0.50    # if featvar < this → adjust
                           # baseline featvar ≈ 3.16
                           # collapsed featvar ≈ 0.41
                           # threshold = midpoint ≈ 0.50
T_STEP           = 1.0    # K per adjustment

OUT_DIR = '/kaggle/working/obj3_tbase_v2'
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

# ── Homeostasis trigger ───────────────────────────────────────────────────────
def should_adjust(color_acc, feat_var):
    """
    Returns (bool, reason):
    True if either signal indicates drift
    """
    color_alert  = color_acc  < COLOR_THRESHOLD
    featvar_alert= feat_var   < FEATVAR_THRESHOLD

    if color_alert and featvar_alert:
        return True, "BOTH signals below threshold"
    elif color_alert:
        return True, f"color={color_acc*100:.1f}% < {COLOR_THRESHOLD*100:.0f}%"
    elif featvar_alert:
        return True, f"featvar={feat_var:.3f} < {FEATVAR_THRESHOLD:.2f}"
    else:
        return False, "both signals OK"

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
    ckpt['baseline']={
        'T_base':T_BASE_NOMINAL,'results':res,
        'feat_var':float(np.var(X_tr))}
    save_ckpt()

b=ckpt['baseline']
print(f"  digit={b['results']['digit']*100:.2f}%  "
      f"color={b['results']['color']*100:.2f}%  "
      f"parity={b['results']['parity']*100:.2f}%")
print(f"  FeatVar={b['feat_var']:.4f}")
trigger, reason = should_adjust(
    b['results']['color'], b['feat_var'])
print(f"  Homeostasis trigger: {trigger} ({reason})")

# ── Step B: Drift ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"[B] DRIFT  T_base={T_BASE_DRIFT}K")
print(f"{'='*65}")

if 'drift' not in ckpt:
    X_tr=extract(tr_imgs,N_TRAIN,T_BASE_DRIFT,'Xtr_331')
    X_te=extract(te_imgs,N_TEST, T_BASE_DRIFT,'Xte_331')
    res=train_eval(X_tr,X_te,y_d_tr,y_d_te,
                   y_c_tr,y_c_te,y_p_tr,y_p_te)
    ckpt['drift']={
        'T_base':T_BASE_DRIFT,'results':res,
        'feat_var':float(np.var(X_tr))}
    save_ckpt()

d=ckpt['drift']
print(f"  digit={d['results']['digit']*100:.2f}%  "
      f"color={d['results']['color']*100:.2f}%  "
      f"parity={d['results']['parity']*100:.2f}%")
print(f"  FeatVar={d['feat_var']:.4f}")
trigger, reason = should_adjust(
    d['results']['color'], d['feat_var'])
print(f"  Homeostasis trigger: {trigger} ({reason})")

# ── Step C: Homeostasis ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"[C] HOMEOSTASIS (dual monitoring)")
print(f"  Color threshold:  {COLOR_THRESHOLD*100:.0f}%")
print(f"  FeatVar threshold: {FEATVAR_THRESHOLD:.2f}")
print(f"  Rule: if EITHER signal below threshold → T_base -= {T_STEP}K")
print(f"{'='*65}")

if 'homeostasis' not in ckpt:
    ckpt['homeostasis']={'steps':[],'current_T':T_BASE_DRIFT}
    save_ckpt()

T_current=ckpt['homeostasis']['current_T']
steps_done=ckpt['homeostasis']['steps']

# Get last signal values
if steps_done:
    last_color  =steps_done[-1]['color']
    last_featvar=steps_done[-1]['feat_var']
else:
    last_color  =d['results']['color']
    last_featvar=d['feat_var']

step_num=len(steps_done)

while T_current > T_BASE_MIN:
    trigger, reason = should_adjust(last_color, last_featvar)
    if not trigger:
        print(f"\n  ✓ Both signals OK at T={T_current}K")
        print(f"    color={last_color*100:.1f}%  "
              f"featvar={last_featvar:.3f}")
        break

    step_num+=1
    T_new = T_current - T_STEP
    T_new = max(T_new, T_BASE_MIN)

    print(f"\n  Step {step_num}: T={T_current:.1f}K → {T_new:.1f}K")
    print(f"    Trigger: {reason}")

    tag_tr=f'Xtr_{T_new:.0f}'
    tag_te=f'Xte_{T_new:.0f}'

    X_tr=extract(tr_imgs,N_TRAIN,T_new,tag_tr)
    X_te=extract(te_imgs,N_TEST, T_new,tag_te)
    res=train_eval(X_tr,X_te,y_d_tr,y_d_te,
                   y_c_tr,y_c_te,y_p_tr,y_p_te)
    fv=float(np.var(X_tr))

    print(f"  T={T_new}K: digit={res['digit']*100:.2f}%  "
          f"color={res['color']*100:.2f}%  "
          f"parity={res['parity']*100:.2f}%")
    print(f"  FeatVar={fv:.4f}")

    # Check which signals triggered
    c_ok  = res['color'] >= COLOR_THRESHOLD
    fv_ok = fv >= FEATVAR_THRESHOLD
    print(f"  Color signal:  {'✓ OK' if c_ok  else '✗ still below threshold'}")
    print(f"  FeatVar signal:{'✓ OK' if fv_ok else '✗ still below threshold'}")

    steps_done.append({
        'T_base':T_new,'results':res,
        'digit':res['digit'],'color':res['color'],
        'parity':res['parity'],'feat_var':fv,
        'trigger_reason':reason})
    ckpt['homeostasis']['steps']=steps_done
    ckpt['homeostasis']['current_T']=T_new
    save_ckpt()

    last_color  =res['color']
    last_featvar=fv
    T_current   =T_new

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"DUAL-SIGNAL HOMEOSTASIS SUMMARY")
print(f"{'='*65}")

b_color =ckpt['baseline']['results']['color']
b_fv    =ckpt['baseline']['feat_var']
d_color =ckpt['drift']['results']['color']
d_fv    =ckpt['drift']['feat_var']

print(f"\n{'State':<20} {'T(K)':>6} {'Color':>8} {'FeatVar':>10} "
      f"{'C-trigger':>10} {'FV-trigger':>11}")
print(f"{'-'*68}")
print(f"  {'Baseline':<18} {T_BASE_NOMINAL:>6.1f} "
      f"{b_color*100:>7.2f}%  {b_fv:>10.4f}  "
      f"{'No':>10}  {'No':>10}")
print(f"  {'Drift':<18} {T_BASE_DRIFT:>6.1f} "
      f"{d_color*100:>7.2f}%  {d_fv:>10.4f}  "
      f"{'Yes' if d_color<COLOR_THRESHOLD else 'No':>10}  "
      f"{'Yes' if d_fv<FEATVAR_THRESHOLD else 'No':>10}")

for s in steps_done:
    c_trig ='Yes' if s['color']<COLOR_THRESHOLD else 'No'
    fv_trig='Yes' if s['feat_var']<FEATVAR_THRESHOLD else 'No'
    print(f"  {'Step→'+str(s['T_base'])+'K':<18} "
          f"{s['T_base']:>6.1f} "
          f"{s['color']*100:>7.2f}%  {s['feat_var']:>10.4f}  "
          f"{c_trig:>10}  {fv_trig:>10}")

if steps_done:
    final=steps_done[-1]
    h_color=final['color']; h_fv=final['feat_var']

    dist_drift_c =abs(d_color-b_color)
    dist_final_c =abs(h_color-b_color)
    recovery_c   =(1-dist_final_c/dist_drift_c)*100 if dist_drift_c>0 else 0

    dist_drift_fv=abs(d_fv-b_fv)
    dist_final_fv=abs(h_fv-b_fv)
    recovery_fv  =(1-dist_final_fv/dist_drift_fv)*100 if dist_drift_fv>0 else 0

    print(f"\n  Color degradation:   {abs(d_color-b_color)*100:.1f}pp")
    print(f"  Color recovery:      {recovery_c:.1f}%")
    print(f"  FeatVar degradation: {abs(d_fv-b_fv):.3f}")
    print(f"  FeatVar recovery:    {recovery_fv:.1f}%")
    print(f"  T_base adjusted:     {T_BASE_DRIFT}K → {final['T_base']}K "
          f"({T_BASE_DRIFT-final['T_base']:.0f}K reduction)")

    # Signal agreement
    print(f"\n  Signal agreement analysis:")
    for s in steps_done:
        c_trig =s['color']  < COLOR_THRESHOLD
        fv_trig=s['feat_var']< FEATVAR_THRESHOLD
        if c_trig and fv_trig:
            agree="BOTH triggered ← signals agree ✓"
        elif c_trig and not fv_trig:
            agree="color only ← color more sensitive"
        elif not c_trig and fv_trig:
            agree="featvar only ← featvar more sensitive"
        else:
            agree="neither ← both recovered ✓"
        print(f"    T={s['T_base']}K: {agree}")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(CKPT,'w') as f: json.dump(ckpt,f,indent=2)
with zipfile.ZipFile(
    '/kaggle/working/obj3_tbase_v2_download.zip','w') as z:
    z.write(CKPT,'obj3_tbase_v2_results.json')
print(f"\n✓ download: /kaggle/working/obj3_tbase_v2_download.zip")

"""
Objective 3 — T_base Homeostasis in VO₂
==========================================
Rule: monitor color accuracy online
      if color < threshold → decrease T_base → increase τ_ins
      → σ*(τ_ins) increases → color recovers

This directly connects to:
- Objective 1: σ*(τ) ∝ τ^{-0.5}
- Thesis causal result: T_base → τ_ins → color accuracy

Experiment:
1. Baseline: T_base=325K, σ=0.0002 → color≈78%
2. Drift: T_base drifts to 331K → color≈37%  
3. Homeostasis: T_base adjusted back → color recovers

Runtime: ~4-5 hours (3 full simulations × 60000 samples)
Use Kaggle T4 GPU for speed.
"""
import os, sys, json, time, zipfile
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.linear_model import RidgeClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'model.py' in files: SRC = root; break
sys.path.insert(0, SRC)
from model import Circuit2D

# ── Parameters ────────────────────────────────────────────────────────────────
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

# T_base values
T_BASE_NOMINAL  = 325.0   # baseline
T_BASE_DRIFT    = 331.0   # after drift
T_BASE_MIN      = 323.0   # homeostasis lower bound
T_BASE_MAX      = 333.0   # homeostasis upper bound

# Homeostasis parameters
COLOR_THRESHOLD = 0.65    # if color < this → adjust T_base
T_STEP          = 1.0     # K per adjustment step

OUT_DIR = '/kaggle/working/obj3_tbase'
os.makedirs(OUT_DIR, exist_ok=True)
CKPT = f'{OUT_DIR}/checkpoint.json'

# ── Dataset ───────────────────────────────────────────────────────────────────
COLOR_PALETTE = torch.tensor([
    [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,1.0,0.0],[1.0,0.0,1.0],
    [0.0,1.0,1.0],[1.0,0.5,0.0],[0.5,0.0,1.0],[0.0,0.5,0.0],[0.5,0.5,0.5],
], dtype=torch.float32)

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

def extract_features(images, n, t_base, tag):
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
        print(f"  [{e}/{n}]  {elapsed:.0f}s  ETA {(n-e)/rate:.0f}s",end='\r')
    print()
    X=np.concatenate(feats); np.save(path,X)
    print(f"  saved  var={np.var(X):.4f}")
    return X

def train_and_eval(X_tr, X_te, y_d_tr, y_d_te,
                   y_c_tr, y_c_te, y_p_tr, y_p_te):
    results={}
    for name,y_tr,y_te in [('digit',y_d_tr,y_d_te),
                             ('color',y_c_tr,y_c_te),
                             ('parity',y_p_tr,y_p_te)]:
        clf=RidgeClassifier(alpha=ALPHA).fit(X_tr,y_tr)
        acc=float((clf.predict(X_te)==y_te).mean())
        results[name]=acc
    return results

# ── Load checkpoint ───────────────────────────────────────────────────────────
if os.path.exists(CKPT):
    with open(CKPT) as f: ckpt=json.load(f)
    print(f"Resuming from checkpoint")
else:
    ckpt={}

def save_ckpt():
    with open(CKPT,'w') as f: json.dump(ckpt,f,indent=2)

# ── Load dataset once ─────────────────────────────────────────────────────────
print("Loading dataset...")
tr_imgs,tr_d,tr_c,tr_p=make_dataset('train')
te_imgs,te_d,te_c,te_p=make_dataset('test')
y_d_tr=tr_d[:N_TRAIN].numpy(); y_d_te=te_d[:N_TEST].numpy()
y_c_tr=tr_c[:N_TRAIN].numpy(); y_c_te=te_c[:N_TEST].numpy()
y_p_tr=tr_p[:N_TRAIN].numpy(); y_p_te=te_p[:N_TEST].numpy()

# ── Step A: Baseline T=325K ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[A] BASELINE  T_base={T_BASE_NOMINAL}K")
print(f"{'='*60}")

if 'baseline' not in ckpt:
    X_tr=extract_features(tr_imgs,N_TRAIN,T_BASE_NOMINAL,'Xtr_325')
    X_te=extract_features(te_imgs,N_TEST, T_BASE_NOMINAL,'Xte_325')
    res=train_and_eval(X_tr,X_te,y_d_tr,y_d_te,
                       y_c_tr,y_c_te,y_p_tr,y_p_te)
    ckpt['baseline']={'T_base':T_BASE_NOMINAL,'results':res}
    save_ckpt()

res=ckpt['baseline']['results']
print(f"  digit={res['digit']*100:.2f}%  "
      f"color={res['color']*100:.2f}%  "
      f"parity={res['parity']*100:.2f}%")

# ── Step B: Drift T=331K ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[B] DRIFT  T_base={T_BASE_DRIFT}K")
print(f"{'='*60}")

if 'drift' not in ckpt:
    X_tr=extract_features(tr_imgs,N_TRAIN,T_BASE_DRIFT,'Xtr_331')
    X_te=extract_features(te_imgs,N_TEST, T_BASE_DRIFT,'Xte_331')
    res=train_and_eval(X_tr,X_te,y_d_tr,y_d_te,
                       y_c_tr,y_c_te,y_p_tr,y_p_te)
    ckpt['drift']={'T_base':T_BASE_DRIFT,'results':res}
    save_ckpt()

res=ckpt['drift']['results']
print(f"  digit={res['digit']*100:.2f}%  "
      f"color={res['color']*100:.2f}%  "
      f"parity={res['parity']*100:.2f}%")

color_drift=res['color']

# ── Step C: Homeostasis ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[C] HOMEOSTASIS")
print(f"  Rule: if color < {COLOR_THRESHOLD*100:.0f}% → T_base -= {T_STEP}K")
print(f"{'='*60}")

if 'homeostasis' not in ckpt:
    ckpt['homeostasis']={'steps':[],'current_T':T_BASE_DRIFT}
    save_ckpt()

T_current=ckpt['homeostasis']['current_T']
steps_done=ckpt['homeostasis']['steps']

# Check if color already recovered
if steps_done:
    last_color=steps_done[-1]['color']
else:
    last_color=color_drift

step_num=len(steps_done)

while last_color < COLOR_THRESHOLD and T_current > T_BASE_MIN:
    step_num+=1
    print(f"\n  Step {step_num}: T_base={T_current:.1f}K "
          f"(color was {last_color*100:.1f}%)")

    # Adjust T_base
    T_new=T_current-T_STEP
    T_new=max(T_new, T_BASE_MIN)

    tag_tr=f'Xtr_{T_new:.0f}'.replace('.','p')
    tag_te=f'Xte_{T_new:.0f}'.replace('.','p')

    X_tr=extract_features(tr_imgs,N_TRAIN,T_new,tag_tr)
    X_te=extract_features(te_imgs,N_TEST, T_new,tag_te)
    res=train_and_eval(X_tr,X_te,y_d_tr,y_d_te,
                       y_c_tr,y_c_te,y_p_tr,y_p_te)

    print(f"  T={T_new}K: digit={res['digit']*100:.2f}%  "
          f"color={res['color']*100:.2f}%  "
          f"parity={res['parity']*100:.2f}%")

    steps_done.append({'T_base':T_new,'results':res,
                       'digit':res['digit'],'color':res['color'],
                       'parity':res['parity']})
    ckpt['homeostasis']['steps']=steps_done
    ckpt['homeostasis']['current_T']=T_new
    save_ckpt()

    last_color=res['color']
    T_current=T_new

    if last_color >= COLOR_THRESHOLD:
        print(f"\n  ✓ Color recovered to {last_color*100:.1f}% "
              f"at T_base={T_current:.1f}K")
        break

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"HOMEOSTASIS SUMMARY")
print(f"{'='*60}")

b_color=ckpt['baseline']['results']['color']
d_color=ckpt['drift']['results']['color']

print(f"  Baseline  T={T_BASE_NOMINAL}K: color={b_color*100:.2f}%")
print(f"  Drift     T={T_BASE_DRIFT}K: color={d_color*100:.2f}%")

if steps_done:
    final=steps_done[-1]
    h_color=final['color']
    h_T=final['T_base']
    print(f"  Homeostasis T={h_T}K: color={h_color*100:.2f}%")

    dist_drift=abs(d_color-b_color)
    dist_final=abs(h_color-b_color)
    recovery=(1-dist_final/dist_drift)*100 if dist_drift>0 else 0
    print(f"\n  Color degradation: {abs(d_color-b_color)*100:.1f}pp")
    print(f"  Color recovery:    {recovery:.1f}%")
    print(f"  T_base adjusted:   {T_BASE_DRIFT}K → {h_T}K "
          f"({T_BASE_DRIFT-h_T:.0f}K reduction)")

# ── Figure ────────────────────────────────────────────────────────────────────
if steps_done:
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    fig.suptitle('Objective 3B — T_base Homeostasis in VO₂\n'
                 'Thermal operating point adjustment restores color accuracy',
                 fontsize=11,fontweight='bold')

    # Panel 1: color accuracy vs T_base
    ax=axes[0]
    T_vals=[T_BASE_NOMINAL, T_BASE_DRIFT]+[s['T_base'] for s in steps_done]
    c_vals=[b_color, d_color]+[s['color'] for s in steps_done]
    labels=['Baseline','Drift']+[f'Step {i+1}' for i in range(len(steps_done))]

    colors_pts=['steelblue','tomato']+['seagreen']*len(steps_done)
    for i,(T,c,col) in enumerate(zip(T_vals,c_vals,colors_pts)):
        ax.scatter(T,c*100,s=150,color=col,zorder=5)
        ax.annotate(labels[i],xy=(T,c*100),
                    xytext=(T+0.1,c*100+1),fontsize=8)

    ax.axhline(COLOR_THRESHOLD*100,color='red',ls='--',lw=1.5,
               label=f'Threshold={COLOR_THRESHOLD*100:.0f}%')
    ax.axhline(b_color*100,color='gray',ls=':',lw=1.5,
               label=f'Baseline={b_color*100:.1f}%')

    ax.set_xlabel('T_base (K)',fontsize=12)
    ax.set_ylabel('Color accuracy (%)',fontsize=12)
    ax.set_title('Color accuracy vs T_base\nHomeostasis reduces T_base',
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: step-by-step recovery
    ax=axes[1]
    step_colors=[steps_done[i]['color']*100 for i in range(len(steps_done))]
    step_T     =[steps_done[i]['T_base'] for i in range(len(steps_done))]

    ax.plot([0],[d_color*100],'s',color='tomato',ms=12,
            label=f'Drift: {d_color*100:.1f}%')
    ax.plot(range(1,len(steps_done)+1),step_colors,
            '^-',color='seagreen',lw=2,ms=10,
            label='Homeostasis steps')
    ax.axhline(COLOR_THRESHOLD*100,color='red',ls='--',lw=1.5,
               label='Recovery threshold')
    ax.axhline(b_color*100,color='steelblue',ls=':',lw=1.5,
               label=f'Baseline: {b_color*100:.1f}%')

    ax.set_xlabel('Homeostasis step',fontsize=12)
    ax.set_ylabel('Color accuracy (%)',fontsize=12)
    ax.set_title('Step-by-step recovery\n'
                 f'T_base: {T_BASE_DRIFT}K → {step_T[-1]}K',fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path=f'{OUT_DIR}/obj3_tbase.png'
    plt.savefig(fig_path,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

# ── Save zip ──────────────────────────────────────────────────────────────────
with zipfile.ZipFile('/kaggle/working/obj3_tbase_download.zip','w') as z:
    z.write(CKPT,'obj3_tbase_results.json')
    if os.path.exists(f'{OUT_DIR}/obj3_tbase.png'):
        z.write(f'{OUT_DIR}/obj3_tbase.png','obj3_tbase.png')
print(f"✓ download: /kaggle/working/obj3_tbase_download.zip")

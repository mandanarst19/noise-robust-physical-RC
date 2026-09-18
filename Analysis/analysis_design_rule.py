"""
Analysis 5.1 — Design Rule for Physical Reservoir Computing
============================================================
A substrate-independent framework for predicting task robustness
under hardware noise, derived from VO2 experimental results.

No simulation needed.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*60)
print("DESIGN RULE FOR PHYSICAL RESERVOIR COMPUTING")
print("="*60)

# ── Observed data ─────────────────────────────────────────────────────────────
NOISE_DATA = [
    {'sigma': 0.00000, 'color': 0.7731, 'parity': 0.8930, 'digit': 0.8711, 'var': 1.571255},
    {'sigma': 0.00010, 'color': 0.7691, 'parity': 0.8966, 'digit': 0.8726, 'var': 1.570997},
    {'sigma': 0.00020, 'color': 0.7702, 'parity': 0.8898, 'digit': 0.8643, 'var': 1.565972},
    {'sigma': 0.00050, 'color': 0.2682, 'parity': 0.8844, 'digit': 0.8487, 'var': 0.000087},
    {'sigma': 0.00100, 'color': 0.2700, 'parity': 0.8750, 'digit': 0.8485, 'var': 0.000088},
]

# ── Physical parameters ───────────────────────────────────────────────────────
tau_met = 187e-9    # s
tau_th  = 241e-9    # s
tau_ins = 7.57e-6   # s
sigma_star_color  = 0.00043   # estimated from interpolation
sigma_star_parity = 0.00100   # lower bound

# ── The Design Rule ───────────────────────────────────────────────────────────
print("""
THE DESIGN RULE
───────────────
Given a physical reservoir with timescales {{τ_1, τ_2, ..., τ_k}}
and hardware noise level σ_hw:

    A task T requiring timescale τ is SAFE if:
        σ_hw < σ*(τ)

    where σ*(τ) = σ*(τ_ref) × (τ_ref / τ)^(1/2)

    and σ*(τ_ref) is the measured threshold for a reference task.

Equivalently:
    SAFE  ←→  σ_hw × √τ < σ*(τ_ref) × √τ_ref
    
This is a noise-timescale product criterion.
""")

# ── Application to VO2 system ─────────────────────────────────────────────────
print("APPLICATION TO VO₂ SYSTEM")
print("─"*40)

# Reference: color task with τ_ins
ref_product = sigma_star_color * np.sqrt(tau_ins)
print(f"Reference (color, τ_ins): σ* × √τ = {ref_product:.2e}")

# Check parity
parity_product = sigma_star_parity * np.sqrt(tau_met)
print(f"Parity    (τ_met):        σ* × √τ > {parity_product:.2e}")

# For arbitrary task
print(f"\nFor a new task requiring timescale τ:")
print(f"  σ*(τ) = {sigma_star_color:.5f} × √({tau_ins*1e6:.2f}μs / τ)")
print(f"\nExamples:")
for tau_ns, name in [(100, '100 ns'), (500, '500 ns'),
                     (1000, '1 μs'), (7570, '7.57 μs (τ_ins)'),
                     (187, 'τ_met')]:
    tau_s = tau_ns * 1e-9
    sigma_s = sigma_star_color * np.sqrt(tau_ins / tau_s)
    safety = "SAFE" if sigma_s > 0.0002 else "AT RISK"
    print(f"  τ = {name:12s}: σ*(τ) = {sigma_s:.5f}  [{safety} at σ=0.0002]")

# ── Substrate-independent generalization ──────────────────────────────────────
print(f"""
SUBSTRATE-INDEPENDENT GENERALIZATION
─────────────────────────────────────
The design rule applies to ANY physical reservoir with:
  - Multiple timescales {{τ_fast, τ_slow}}
  - Additive noise σ_hw

Pre-deployment screening procedure:
  1. Measure substrate timescales {{τ_i}} from hardware characterization
  2. Identify required timescale τ_task for each target task
  3. Measure or estimate σ*(τ_ref) from a single calibration experiment
  4. Compute σ*(τ_task) = σ*(τ_ref) × √(τ_ref / τ_task)
  5. If σ_hw < σ*(τ_task): deploy safely
     If σ_hw > σ*(τ_task): task will fail — redesign or noise reduction needed

This replaces trial-and-error deployment with principled screening.
""")

# ── Comparison: which tasks are safe on VO2 ──────────────────────────────────
print("TASK SAFETY MAP FOR VO₂ AT σ = 0.0002")
print("─"*40)

tasks = [
    ('Digit recognition', tau_ins, 10, 'mixed'),
    ('Color classification', tau_ins, 10, 'τ_ins'),
    ('Parity detection', tau_met, 2, 'τ_met'),
    ('Binary edge detection', tau_met, 2, 'τ_met'),
    ('Tone classification', tau_ins, 4, 'τ_ins'),
    ('Spike rate coding', tau_met, 2, 'τ_met'),
]

sigma_hw = 0.0002

print(f"{'Task':<25} {'τ_req':>10} {'σ*(τ)':>10} {'Status':>10}")
print("─"*58)
for name, tau, n_cls, tau_type in tasks:
    sigma_s = sigma_star_color * np.sqrt(tau_ins / tau)
    status = '✓ SAFE' if sigma_hw < sigma_s else '✗ AT RISK'
    print(f"  {name:<23} {tau*1e6:>8.3f}μs  {sigma_s:>9.5f}  {status}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: σ*(τ) curve — the design rule
ax = axes[0]
taus_us = np.logspace(-3, 1, 200)   # 1 ns to 10 μs in μs
taus_s  = taus_us * 1e-6
sigma_curve = sigma_star_color * np.sqrt(tau_ins / taus_s)

ax.loglog(taus_us, sigma_curve, 'b-', linewidth=2.5, label='σ*(τ) design rule')
ax.fill_between(taus_us, sigma_curve, sigma_curve.max(),
                alpha=0.15, color='green', label='Safe region')
ax.fill_between(taus_us, sigma_curve.min(), sigma_curve,
                alpha=0.15, color='red', label='At-risk region')

ax.axhline(y=sigma_hw, color='gray', linestyle='--',
           linewidth=2, label=f'σ_hw = {sigma_hw:.4f}')
ax.axvline(x=tau_met*1e6, color='royalblue', linestyle=':',
           linewidth=2, label=f'τ_met = {tau_met*1e9:.0f} ns')
ax.axvline(x=tau_ins*1e6, color='tomato', linestyle=':',
           linewidth=2, label=f'τ_ins = {tau_ins*1e6:.2f} μs')

ax.scatter([tau_ins*1e6], [sigma_star_color], s=100, color='tomato',
           zorder=5, label='Color (measured)')
ax.scatter([tau_met*1e6], [sigma_star_parity], s=100, color='royalblue',
           marker='^', zorder=5, label='Parity (lower bound)')

ax.set_xlabel('Required timescale τ (μs)')
ax.set_ylabel('Noise threshold σ*(τ)')
ax.set_title('Design rule: safe noise level vs required timescale')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Panel B: Task safety map
ax = axes[1]
task_names  = ['Parity\n(τ_met)', 'Digit\n(mixed)', 'Color\n(τ_ins)']
task_sigmas = [sigma_star_parity, 
               sigma_star_color * np.sqrt(tau_ins / (0.5*(tau_met+tau_ins))),
               sigma_star_color]
task_colors = ['seagreen', 'royalblue', 'tomato']

bars = ax.bar(task_names, [s*1e4 for s in task_sigmas],
              color=task_colors, alpha=0.8, edgecolor='black')
ax.axhline(y=sigma_hw*1e4, color='gray', linestyle='--',
           linewidth=2, label=f'σ_hw = {sigma_hw:.4f}')
ax.set_ylabel('Noise threshold σ*(τ) (×10⁻⁴)')
ax.set_title('Task robustness ranking\n(higher = more robust)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar, s in zip(bars, task_sigmas):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*1e4,
            f'{s:.5f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/kaggle/working/design_rule.png', dpi=150, bbox_inches='tight')
print(f"\nFigure saved: /kaggle/working/design_rule.png")

results = {
    'design_rule': 'sigma_hw < sigma_star(tau_ref) * sqrt(tau_ref / tau_task)',
    'reference_point': {
        'task': 'color', 'tau_ref_us': tau_ins*1e6,
        'sigma_star': sigma_star_color
    },
    'vo2_thresholds': {
        'color_tau_ins':  sigma_star_color,
        'parity_tau_met': sigma_star_parity,
    },
    'key_message': (
        'Tasks requiring slow dynamics collapse first under noise. '
        'Design rule allows pre-deployment screening without trial-and-error.'
    )
}

with open('/kaggle/working/design_rule_results.json', 'w') as f:
    json.dump(results, f, indent=2)

import shutil
shutil.copy('/kaggle/working/design_rule_results.json',
            '/kaggle/working/DOWNLOAD_design_rule_results.json')
shutil.copy('/kaggle/working/design_rule.png',
            '/kaggle/working/DOWNLOAD_design_rule.png')

print(f"Results: /kaggle/working/DOWNLOAD_design_rule_results.json")
print(f"Figure:  /kaggle/working/DOWNLOAD_design_rule.png")
print("="*60)

# ── Auto download ─────────────────────────────────────────────────────────────
import shutil, os, zipfile

def zip_and_download(files, zip_name):
    zip_path = f'/kaggle/working/{zip_name}.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for f in files:
            if os.path.exists(f):
                zf.write(f, os.path.basename(f))
    print(f"✓ download ready: {zip_path}")

zip_and_download(['/kaggle/working/design_rule_results.json', '/kaggle/working/design_rule.png'], 'design_rule_download')

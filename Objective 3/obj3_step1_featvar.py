"""
Objective 3 — Step 1: FeatVar Monitoring
==========================================
Shows that:
1. Drift causes FeatVar collapse
2. Homeostasis (gain adjustment) restores FeatVar

Rule: var(x_i) → target_var via gain adjustment
      gain[i] *= exp(η × (log(target_var) - log(var(x_i))))

This is analogous to σ_c monitoring from Objective 1.

Runtime: ~5 min (no NARMA, no Ridge)
"""
import numpy as np
import json, zipfile, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('/tmp/obj3_step1', exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
N_RESERVOIR  = 300
N_STEPS      = 2000   # total steps
N_DRIFT_ON   = 500    # drift starts at step 500
N_HOMEO_ON   = 1000   # homeostasis starts at step 1000
SEED         = 42
ALPHA_LEAK   = 0.90

# Noise levels
SIGMA_BASE   = 0.05   # baseline noise
SIGMA_DRIFT  = 0.30   # drift noise (6× baseline)

# Homeostatic parameters
ETA_GAIN     = 0.01   # learning rate
TARGET_VAR   = 0.08   # target variance per neuron
WINDOW       = 100    # steps for running variance

# ── Fixed reservoir ───────────────────────────────────────────────────────────
rng  = np.random.RandomState(SEED)
W    = rng.randn(N_RESERVOIR, N_RESERVOIR)
sr   = np.max(np.abs(np.linalg.eigvals(W)))
W    = W / sr * 0.9
W_in = rng.randn(N_RESERVOIR, 1) * 0.1

# ── Input ─────────────────────────────────────────────────────────────────────
rng_u = np.random.RandomState(SEED+1)
u_seq = rng_u.uniform(-0.5, 0.5, N_STEPS)

# ── Run with monitoring ───────────────────────────────────────────────────────
rng_n = np.random.RandomState(SEED+2)

x        = np.zeros(N_RESERVOIR)
gain     = np.ones(N_RESERVOIR)
x_buffer = np.zeros((WINDOW, N_RESERVOIR))

# Track over time
featvar_history  = []
gain_mean_history= []
sigma_history    = []

print(f"{'='*60}")
print(f"OBJECTIVE 3 STEP 1 — FeatVar Monitoring")
print(f"{'='*60}")
print(f"Step 0-{N_DRIFT_ON}: baseline σ={SIGMA_BASE}")
print(f"Step {N_DRIFT_ON}-{N_HOMEO_ON}: drift σ={SIGMA_DRIFT}")
print(f"Step {N_HOMEO_ON}-{N_STEPS}: drift + homeostasis")

for t in range(N_STEPS):
    # Determine current noise
    if t < N_DRIFT_ON:
        sigma = SIGMA_BASE
        use_homeo = False
    elif t < N_HOMEO_ON:
        sigma = SIGMA_DRIFT
        use_homeo = False
    else:
        sigma = SIGMA_DRIFT
        use_homeo = True

    noise = sigma * rng_n.randn(N_RESERVOIR)

    # ESN step with gain
    x = ((1-ALPHA_LEAK)*x
         + ALPHA_LEAK*np.tanh(
             gain*(W@x + W_in[:,0]*u_seq[t]) + noise))

    # Update buffer
    x_buffer[t % WINDOW] = x

    # Homeostatic update
    if use_homeo and t >= N_HOMEO_ON + WINDOW:
        var_x = x_buffer.var(axis=0) + 1e-10
        # Log-space gain adjustment
        gain *= np.exp(ETA_GAIN * (np.log(TARGET_VAR) - np.log(var_x)))
        gain  = np.clip(gain, 0.05, 20.0)

    # Track FeatVar = mean variance across neurons
    if t >= WINDOW:
        featvar = x_buffer.var(axis=0).mean()
        featvar_history.append(featvar)
        gain_mean_history.append(gain.mean())
        sigma_history.append(sigma)

# Print key moments
t_vals = list(range(WINDOW, N_STEPS))
print(f"\nKey FeatVar values:")
print(f"  Baseline end    (t={N_DRIFT_ON-1}):   "
      f"{featvar_history[N_DRIFT_ON-WINDOW-1]:.4f}")
print(f"  Drift end       (t={N_HOMEO_ON-1}):  "
      f"{featvar_history[N_HOMEO_ON-WINDOW-1]:.4f}")
print(f"  After homeo     (t={N_STEPS-1}):  "
      f"{featvar_history[-1]:.4f}")
print(f"  Target:                        {TARGET_VAR:.4f}")

baseline_var = featvar_history[N_DRIFT_ON-WINDOW-1]
drift_var    = featvar_history[N_HOMEO_ON-WINDOW-1]
final_var    = featvar_history[-1]

degradation = abs(drift_var - baseline_var)/baseline_var*100
dist_drift  = abs(drift_var - TARGET_VAR)
dist_final  = abs(final_var - TARGET_VAR)
recovery    = (1 - dist_final/dist_drift)*100 if dist_drift > 0 else 0

print(f"\nFeatVar degradation under drift: {degradation:.1f}%")
print(f"FeatVar recovery toward target:  {recovery:.1f}%")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Objective 3 Step 1 — FeatVar Monitoring\n'
             'Homeostasis restores feature variance after drift',
             fontsize=11, fontweight='bold')

t_plot = np.array(range(len(featvar_history)))

# Panel 1: FeatVar over time
ax = axes[0]
ax.plot(t_plot, featvar_history, 'steelblue', lw=1.5,
        label='Feature variance V(t)')
ax.axhline(TARGET_VAR, color='green', ls='--', lw=2,
           label=f'Target variance = {TARGET_VAR}')
ax.axhline(baseline_var, color='gray', ls=':', lw=1.5,
           label=f'Baseline = {baseline_var:.3f}')

# Shade regions
ax.axvspan(0, N_DRIFT_ON-WINDOW, alpha=0.08, color='green',
           label='Baseline')
ax.axvspan(N_DRIFT_ON-WINDOW, N_HOMEO_ON-WINDOW,
           alpha=0.08, color='red', label='Drift (no homeo)')
ax.axvspan(N_HOMEO_ON-WINDOW, len(t_plot),
           alpha=0.08, color='blue', label='Drift + Homeostasis')

ax.text((N_DRIFT_ON-WINDOW)//2, max(featvar_history)*0.9,
        'Baseline', ha='center', fontsize=9, color='green')
ax.text((N_DRIFT_ON+N_HOMEO_ON)//2-WINDOW, max(featvar_history)*0.9,
        'Drift\n(no homeo)', ha='center', fontsize=9, color='red')
ax.text((N_HOMEO_ON+N_STEPS)//2-WINDOW, max(featvar_history)*0.9,
        'Drift +\nHomeostasis', ha='center', fontsize=9, color='blue')

ax.set_ylabel('Feature variance V(t)', fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# Panel 2: Gain over time
ax = axes[1]
ax.plot(t_plot, gain_mean_history, 'tomato', lw=1.5,
        label='Mean gain g(t)')
ax.axhline(1.0, color='gray', ls='--', lw=1.5,
           label='Initial gain = 1.0')
ax.axvspan(N_HOMEO_ON-WINDOW, len(t_plot),
           alpha=0.08, color='blue', label='Homeostasis active')

ax.set_xlabel('Time step', fontsize=12)
ax.set_ylabel('Mean gain g(t)', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = '/tmp/obj3_step1/obj3_step1.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'baseline_featvar': float(baseline_var),
    'drift_featvar':    float(drift_var),
    'final_featvar':    float(final_var),
    'target_var':       TARGET_VAR,
    'degradation_pct':  float(degradation),
    'recovery_pct':     float(recovery),
    'sigma_base':       SIGMA_BASE,
    'sigma_drift':      SIGMA_DRIFT,
}
with open('/tmp/obj3_step1/results.json','w') as f:
    json.dump(out, f, indent=2)
with zipfile.ZipFile('/tmp/obj3_step1_download.zip','w') as z:
    z.write('/tmp/obj3_step1/results.json', 'obj3_step1_results.json')
    z.write(fig_path, 'obj3_step1.png')
print(f"\n✓ download: /tmp/obj3_step1_download.zip")
print(f"{'='*60}")

# Noise-Robust Physical Reservoir Computing

**Universal Scaling Laws, Architecture Design, and Homeostatic Adaptation**

PhD Research Proposal Code — Mandana Roosta  
MSc Physics, Shahid Beheshti University  
Applying: Politecnico di Milano — Computer Science and Engineering

---

## Overview

This repository contains the experimental code for a PhD research proposal on noise robustness in physical reservoir computing (RC).

The proposal addresses a fundamental open problem: when RC runs on real hardware, additive noise causes **timescale-selective memory collapse** — slow-timescale tasks fail while fast-timescale tasks survive. Three objectives characterise this mechanism, engineer architecture to control it, and compensate for residual drift through unsupervised feedback.

### Related repository
Thesis code (multi-task VO₂ RC): [Multi-Task-Learning-on-Phase-Change-Material](https://github.com/mandanarst19/Multi-Task-Learning-on-Phase-Change-Material)

---

## Key Results

| Objective | Finding | Result |
|-----------|---------|--------|
| **1** | Universal noise scaling law | β = −0.511 ± 0.027 (ESN), β = −0.490 (VO₂), R² = 0.963 |
| **2** | Architecture controls β | r(MC, β) = 0.781, p = 0.0004 across 16 conditions |
| **3** | Homeostatic recovery | 97.8% color accuracy recovery in 2 steps |

---

## Repository Structure

```
noise-robust-physical-RC/
│
├── README.md
│
├── objective1/                    # Universal noise scaling law
│   ├── obj1_universal.py          # PRIMARY — β=-0.511±0.027, R²=0.963
│   └── obj1_robust.py             # Supporting — alternative run
│
├── objective2/                    # Architecture comparison
│   ├── obj2_robust_final.py       # PRIMARY — r(MC,β)=0.781, p=0.0004
│   ├── obj2_final.py              # α=0.9 only — r=0.973, p=0.027
│   └── obj2_robust.py             # Older version
│
├── objective3/                    # Homeostatic recovery
│   ├── obj3_tbase_thermostat.py   # PRIMARY — unsupervised thermostat
│   ├── obj3_tbase_homeostasis.py  # Threshold-based rule (same result)
│   ├── obj3_tbase_v2.py           # Dual-signal extension
│   └── obj3_step1_featvar.py      # ESN gain homeostasis (proof of concept)
│
└── analysis/                      # Supporting analyses
    ├── analysis_design_rule.py    # σ*(τ) = σ_c × √(τ_ins/τ) validation
    ├── analysis_fft_bins.py       # FFT analysis of temporal bins
    ├── analysis_per_class.py      # Per-class accuracy (luminance bottleneck)
    ├── analysis_tsne.py           # t-SNE feature space visualization
    ├── analysis_kernel_rank.py    # Kernel rank per architecture
    ├── analysis_mutual_info.py    # Task independence (MI ≈ 0.03 nat)
    ├── analysis_memory_capacity.py# MC measurement per architecture
    └── generate_all_figures.py    # Paper-ready figures from verified numbers
```

---

## Objective 1 — Universal Noise Scaling Law

**Question:** Given a physical substrate with known noise floor σ_hw, which computational tasks will survive — and can this be predicted before fabrication?

**Theory:** Noise accumulates as σ_T(τ) = (σ/C_th)·√τ over τ/dt integration steps.  
Setting this equal to signal amplitude yields: **σ*(τ) ∝ τ^{−1/2}**

**Validation across two substrates:**

| Substrate | β | R² | p |
|-----------|---|----|---|
| VO₂ thermal neuristors | −0.490 | — | — |
| Leaky ESN (16 conditions) | −0.511 ± 0.027 | 0.963 | < 0.0001 |
| Langevin theory | −0.500 | — | — |

**Design rule:** σ*(τ_task) = σ_c × √(τ_ref / τ_task)

**Primary script:** `objective1/obj1_universal.py`
```
N=500, 10 runs, NARMA k∈{1,2,4,8}, α∈{0.50,0.90,0.95,0.99}, σ up to 5.0
Runtime: ~45 min on Kaggle T4 GPU
```

---

## Objective 2 — Architecture-Controlled Noise Scaling

**Question:** Can β be engineered through reservoir topology?

**Mechanism:** Eigenvalue structure determines β.
- Standard random matrices: eigenvalues scattered → uneven memory → β ≈ −0.5 (Langevin-like)
- Orthogonal matrices: eigenvalues on unit circle → uniform memory → β ≈ −0.36 (flat)

**Results across 16 conditions (4 architectures × 4 leak rates):**

| Architecture | MC | KR | β |
|---|---|---|---|
| Standard ESN | 30.42 | 6.83 | −0.534 |
| Ring | 35.14 | 6.85 | −0.402 |
| Antisymmetric (EuSN) | 36.64 | 7.19 | −0.409 |
| Orthogonal | 37.52 | 7.29 | −0.357 |

**Correlation:** r(MC, β) = 0.781, p = 0.0004

**Design trade-off:** High-MC architectures sacrifice timescale selectivity for uniform noise robustness.

**Primary script:** `objective2/obj2_robust_final.py`
```
N=300, 5 runs, 4 architectures × 4 leak rates
Runtime: ~20 min on Kaggle T4 GPU
```

---

## Objective 3 — Unsupervised Homeostatic Recovery

**Question:** Can the system restore its own operating point after hardware drift — without labeled data?

**Failure mode:** +6K thermal drift (T_base: 325K→331K) collapses color accuracy by 40.7pp.

**Thermostat rule:**
```
T_new = T_current − η · log(V_target / V_current)
```
where V_target = 3.162 (baseline feature variance), η = 1.0 K.

**Properties:**
- Proportional to drift magnitude
- Symmetric in log-space
- Self-limiting as V_current → V_target
- No threshold, no labeled data, no device physics knowledge required

**Results:**

| State | T (K) | Color | FeatVar |
|-------|-------|-------|---------|
| Baseline | 325 | 78.06% | 3.162 |
| Drift | 331 | 37.37% | 0.410 |
| Step 1 | 330 | 58.68% | 0.745 |
| Step 2 | 329 | 78.82% ✓ | 1.433 |

**Recovery: 97.8% in 2 steps**

**Unexpected finding:** T=329K outperforms original T=325K across all three tasks (digit +3.9pp, parity +3.5pp).

**Primary script:** `objective3/obj3_tbase_thermostat.py`
```
Full 60,000-image training at each operating point
Runtime: ~5 hours on Kaggle T4 GPU (crash-safe checkpointing)
```

---

## Requirements

```bash
# All scripts run on Kaggle with T4 GPU
# Required packages (standard Kaggle environment):
numpy
scipy
scikit-learn
matplotlib
torch
torchvision

# For VO2 objectives, also requires Circuit2D simulator:
# Zhang et al., Nature Communications 15, 6986 (2024)
```

---

## Running on Kaggle

### Objective 1 (no VO2 simulator needed):
```python
exec(open('/kaggle/working/obj1_universal.py').read())
# Runtime: ~45 min
```

### Objective 2 (no VO2 simulator needed):
```python
exec(open('/kaggle/working/obj2_robust_final.py').read())
# Runtime: ~20 min
```

### Objective 3 (requires Circuit2D dataset):
```python
# Attach Circuit2D dataset to Kaggle notebook first
exec(open('/kaggle/working/obj3_tbase_thermostat.py').read())
# Runtime: ~5 hours (crash-safe — resumes from checkpoint)
```

---

## Verified Results

All results verified across multiple seeds and conditions:

```
Objective 1: β=-0.511±0.027  R²=0.963  p<0.0001  ✓
Objective 2: r(MC,β)=0.781   p=0.0004             ✓
Objective 3: 97.8% recovery  in 2 steps            ✓
```

---

## References

[1] Milano, G., Pedretti, G., Ielmini, D. et al. *Nature Materials* **21**, 195–202 (2022).  
[2] Ceni, A. et al. *Neurocomputing* **675**, 132952 (2026).  
[3] Zhang, Y.-H. et al. *Nature Communications* **15**, 6986 (2024).  
[4] Pedretti, G. & Ielmini, D. *Electronics* **10**, 1063 (2021).  
[5] Gallicchio, C. *Neurocomputing* **579**, 127411 (2024).  

---

## Citation

```bibtex
@misc{roosta2025noise,
  author = {Roosta, Mandana},
  title  = {Noise-Robust Physical Reservoir Computing},
  year   = {2025},
  url    = {https://github.com/mandanarst19/noise-robust-physical-RC}
}
```

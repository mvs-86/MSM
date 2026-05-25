# Asymmetric MSM: Leverage Effect Design

**Date:** 2026-05-25
**Scope:** Univariate (Msm) and bivariate (Bmsm) leverage extensions

---

## Problem

Standard MSM is symmetric by construction: `r_t = σ·√(M_t)·ε_t` with ε ~ N(0,1). The emission density treats positive and negative return shocks identically. The model cannot capture the leverage effect — the empirical regularity that negative return shocks increase future volatility more than positive shocks of equal magnitude.

---

## Three Approaches

### Approach 1 — σ-Scaling (`Amsm_scale`, `Bamsm_scale`)

**Mechanism:** Scale the base σ multiplicatively by an asymmetry factor conditioned on the sign of the lagged return.

```
σ_eff_t = σ · (1 + λ · 𝟏[r_{t-1} < 0])
r_t | M_t = g_m  ~  N(0, σ_eff_t² · g_m)
```

**Parameter:** `lev` = λ
- λ = 0: reduces to standard MSM
- λ > 0: negative shocks amplify next-period conditional σ
- Bounds: lb = −0.99, ub = 10

**Implementation:** `sigma_gm_t = sigma_eff_t * g.m` is computed inside the filter loop (time-varying). New C++ kernels required.

**Bivariate:** `lev1`, `lev2` — independent leverage for each series. Each series σ scales by its own lagged return sign.

---

### Approach 2 — GJR Additive (`Amsm_gjr`, `Bamsm_gjr`)

**Mechanism:** Augment the state-conditional variance with a GJR-type additive term.

```
Var(r_t | M_t = g_m) = σ²·g_m + γ · 𝟏[r_{t-1} < 0] · r_{t-1}²
σ_eff_t(m) = sqrt(σ²·g_m + γ · max(r_{t-1}, 0)²⁻)
```

where `max(r_{t-1}, 0)⁻ = (min(r_{t-1}, 0))²`.

**Parameter:** `lev` = γ
- γ = 0: reduces to standard MSM
- γ > 0: negative lagged returns add directly to state-conditional variance
- Bounds: lb = 0, ub = 10

**Implementation:** Effective σ is both time- and state-varying. New C++ kernels required.

**Bivariate:** `lev1`, `lev2` — independent GJR term per series. State-conditional variance for each series augmented separately.

---

### Approach 3 — Asymmetric Transition Probabilities (`Amsm_atp`, `Bamsm_atp`)

**Mechanism:** Use two distinct transition matrices depending on the sign of the lagged return. Negative returns trigger a scaled-up switching probability, inducing more volatile state changes.

```
A_pos = Msm_A(b, γ_k,        kbar)   # used when r_{t-1} ≥ 0
A_neg = Msm_A(b, γ_k·(1+λ),  kbar)   # used when r_{t-1} < 0
```

**Parameter:** `lev` = λ
- λ = 0: reduces to standard MSM
- λ > 0: switching accelerates after negative returns → higher expected future volatility
- Bounds: lb = −0.99, ub = 5 (γ_k·(1+λ) must remain in (0,1))

**Implementation:**
- kbar < 8: precompute `A_pos` and `A_neg` before the loop; select per step based on `dat[t-1]`
- kbar ≥ 8: store `gamma_k_pos` and `gamma_k_neg` vectors; apply correct kron at each step

**Bivariate:** Single shared `lev` — the common volatility component transition accelerates when *either* series has a negative lagged return. Rationale: bivariate MSM has a single shared M_t, so a single transition asymmetry is coherent.

---

## S3 Classes

- `amsmmodel` — used by all three univariate variants; `$type` field = `"scale"`, `"gjr"`, or `"atp"`
- `bamsmmodel` — used by all three bivariate variants; `$type` field = `"scale"`, `"gjr"`, or `"atp"`

Shared S3 methods for each class: `print`, `summary`, `predict`, `plot`, `coef`. All dispatch on `$type` only where output differs (e.g., parameter labels in summary).

---

## Parameter Vectors

### Univariate (`amsmmodel`)
All approaches: `[m0, b, γ_k, σ, lev]` — 5 parameters (kbar=1); `[m0, b, γ_k, σ, lev]` regardless of approach (same position, different interpretation).

### Bivariate (`bamsmmodel`)
- Scale / GJR: `[m01, m02, σ1, σ2, γ_k, b, ρ_e, λ, ρ_m, lev1, lev2]` — 11 parameters
- ATP: `[m01, m02, σ1, σ2, γ_k, b, ρ_e, λ, ρ_m, lev]` — 10 parameters

Note: `λ` above is Bmsm's existing bivariate switching correlation parameter — distinct from `lev`.

---

## New Files

### R
```
R/Amsm_scale.R          # Amsm_scale() + likelihood wrapper
R/Amsm_gjr.R            # Amsm_gjr() + likelihood wrapper
R/Amsm_atp.R            # Amsm_atp() + likelihood wrapper
R/Amsm_methods.R        # S3: print/summary/predict/plot/coef for amsmmodel
R/Bamsm_scale.R         # Bamsm_scale() + likelihood wrapper
R/Bamsm_gjr.R           # Bamsm_gjr() + likelihood wrapper
R/Bamsm_atp.R           # Bamsm_atp() + likelihood wrapper
R/Bamsm_methods.R       # S3: print/summary/predict/plot/coef for bamsmmodel
```

### C++ (src/)
```
src/Amsm_scale_cpp.cpp  # Amsm_scale_fast_cpp, Amsm_scale_kron_cpp
src/Amsm_gjr_cpp.cpp    # Amsm_gjr_fast_cpp, Amsm_gjr_kron_cpp
src/Amsm_atp_cpp.cpp    # Amsm_atp_fast_cpp, Amsm_atp_kron_cpp
src/Bamsm_scale_cpp.cpp # Bamsm_scale_fast_cpp, Bamsm_scale_kron_cpp
src/Bamsm_gjr_cpp.cpp   # Bamsm_gjr_fast_cpp, Bamsm_gjr_kron_cpp
src/Bamsm_atp_cpp.cpp   # Bamsm_atp_fast_cpp, Bamsm_atp_kron_cpp
```

---

## Optimization

Each function follows the existing pattern:
1. Parameter check (bounds, starting values)
2. `nlminb` on the negative log-likelihood
3. Likelihood call for filtered probabilities
4. Standard error via numerical Hessian (Newey-West for univariate)

Bivariate variants use the existing two-stage estimation:
- Stage 1: `[m01, m02, σ1, σ2, γ_k, b]` — same as Bmsm stage 1; leverage params excluded
- Stage 2: `[ρ_e, λ, ρ_m, lev1, lev2]` (scale/gjr) or `[ρ_e, λ, ρ_m, lev]` (atp) — correlation + leverage jointly

Rationale: leverage parameters interact with the bivariate emission density (stage 2 objective), not the marginal volatility structure (stage 1).

Starting value for `lev`: 0 (symmetric, standard MSM). Leverage expected to be positive for equity returns.

---

## Testing

- `Amsm_scale(ret, lev=0)` LL must match `Msm(ret)` LL to floating-point precision
- `Amsm_gjr(ret, lev=0)` LL must match `Msm(ret)` LL
- `Amsm_atp(ret, lev=0)` LL must match `Msm(ret)` LL
- Same nesting tests for bivariate
- Sign of estimated `lev` expected positive on equity return data (calvet2004data)
- `predict`, `plot` work without error for all 6 model types

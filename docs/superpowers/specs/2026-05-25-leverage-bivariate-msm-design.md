# Design: Asymmetry / Leverage Effect for Bivariate MSM (Scale variant)

**Date:** 2026-05-25
**Status:** Approved

---

## Problem

Standard `Bmsm` is symmetric by construction — the bivariate normal emission density and the shared transition structure treat positive and negative return shocks identically. It cannot capture the well-known leverage effect (negative returns → higher future volatility).

Three univariate leverage mechanisms already exist (`Amsm_gjr`, `Amsm_atp`, `Amsm_scale`). This spec adds the **sigma-scaling** variant to the bivariate model.

---

## Model Specification

### Leverage mechanism

At each time `t`, each series' state-conditional sigma is scaled up if that series' own lagged return was negative:

```
sigma1_eff(m, t) = sigma1 * g1_m * (1 + lev1 * I(r1_{t-1} < 0))
sigma2_eff(m, t) = sigma2 * g2_m * (1 + lev2 * I(r2_{t-1} < 0))
```

- Own-series trigger only: `sigma1` reacts to `r1_{t-1}`, `sigma2` to `r2_{t-1}`
- Separate leverage coefficients `lev1`, `lev2` — each series has independent asymmetry magnitude
- `lev = 0` recovers symmetric `Bmsm` exactly (nesting property)

### Parameter vector (11 params)

```
[m01, m02, sigma1, sigma2, gammak, b, lev1, lev2, rhoe, lambda, rhom]
```

| Param | Meaning | Bounds |
|-------|---------|--------|
| m01, m02 | Multiplier component mean, series 1 & 2 | (1, 2) |
| sigma1, sigma2 | Base annualised vol | (0, ∞) |
| gammak | Highest-freq switching probability | (0, 1) |
| b | Frequency spacing | (1, 50) |
| lev1, lev2 | Scale leverage, series 1 & 2 | (-0.99, 10) |
| rhoe | Innovation correlation | (-1, 1) |
| lambda | State-correlation parameter | (0, 1) |
| rhom | Multiplier correlation | (-1, 1) |

---

## Estimation

Two-stage MLE, mirroring `Bmsm`:

### Stage 1 — 8 params: `[m01, m02, sigma1, sigma2, gammak, b, lev1, lev2]`

Objective: sum of two independent univariate scale LLs.

```r
ll1 <- Amsm_scale_ll(c(m01, b, gammak, sigma1, lev1), kbar, ret[,1], n.vol)
ll2 <- Amsm_scale_ll(c(m02, b, gammak, sigma2, lev2), kbar, ret[,2], n.vol)
ll  <- ll1 + ll2
```

`b` and `gammak` shared across series (same as existing `Bmsm_stage1_likelihood`).

### Stage 2 — 3 params: `[rhoe, lambda, rhom]`

Stage 1 params (including `lev1`, `lev2`) held fixed. Bivariate joint LL computed via new C++ filtered function that applies sigma scaling per period using lagged returns.

---

## Implementation

### New files (zero changes to existing files)

```
R/Bmsm_scale.R                    — top-level function + S3 methods
R/Bmsm_scale_stage1_likelihood.R  — stage 1 LL (calls Amsm_scale_ll twice)
R/Bmsm_scale_stage2_likelihood.R  — stage 2 LL (wraps new C++ filtered fn)
R/Bmsm_scale_std_err.R            — numerical SE for 11-param vector
src/Bmsm_scale_filtered_cpp.cpp   — bivariate filter with per-period sigma scaling
tests/testthat/test-Bmsm_scale.R  — unit tests incl. nesting test
```

### C++ change (in `Bmsm_scale_filtered_cpp.cpp`)

Copy of `Bmsm_filtered_cpp.cpp`. Single change in inner loop — scale `sg1`/`sg2` by leverage factor before computing bivariate normal weight `w`:

```cpp
double s1 = (t > 0 && dat(t-1, 0) < 0.0) ? (1.0 + lev1) : 1.0;
double s2 = (t > 0 && dat(t-1, 1) < 0.0) ? (1.0 + lev2) : 1.0;
arma::rowvec sg1_eff  = sg1 * s1;
arma::rowvec sg2_eff  = sg2 * s2;
arma::rowvec sg12_eff = sg1_eff % sg2_eff;
// use sg1_eff, sg2_eff, sg12_eff in bivariate normal w computation
```

Two exported functions: `Bmsm_scale_filtered_cpp` (kbar < 4) and `Bmsm_scale_filtered_kron` (kbar ≥ 4).

### Top-level function

```r
Bmsm_scale(ret, kbar = 1, n = 252, para0 = NULL, s.err = TRUE)
```

- `para0` order: `c(m01, m02, sigma1, sigma2, gammak, b, lev1, lev2, rhoe, lambda, rhom)`
- Default start: stage 1 start from `Bmsm_parameter_check` + `lev1 = lev2 = 0`
- Returns object of class `"bmsmmodel"` — reuses all existing S3 methods (`summary`, `print`, `plot`, `predict`)
- `rownames(coef)`: `c("m01","m02","sigma1","sigma2","gammak","b","lev1","lev2","rhoe","lambda","rhom")`
- `summary` degrees of freedom: `nrow(ret) - 11`

---

## Testing

Minimum test coverage in `test-Bmsm_scale.R`:

1. **Nesting** — `lev1 = lev2 = 0` stage 2 LL matches `Bmsm_stage2_likelihood2` exactly (tolerance 1e-6)
2. **Estimation** — `Bmsm_scale(ret, kbar=1)` returns `"bmsmmodel"`, 11 coefficients, finite LL
3. **Leverage sign** — fitted `lev1`, `lev2` positive on equity return data (known leverage effect)
4. **SE** — `s.err = TRUE` produces finite SEs; `s.err = FALSE` returns `NA` matrix

---

## Out of scope

- GJR and ATP leverage variants for `Bmsm` (can follow this same pattern later)
- Cross-leverage (`sigma1` reacting to `r2_{t-1}`)
- Joint single-stage MLE

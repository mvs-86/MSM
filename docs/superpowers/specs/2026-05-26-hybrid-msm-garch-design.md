# Hybrid MSM-GJR-GARCH Design

**Date:** 2026-05-26
**Scope:** Univariate — `Msm_garch()`

---

## Motivation

MSM captures long-run volatility persistence through multi-frequency Markov switching but ignores the fast ARCH effect visible at horizons of 1–5 days. GJR-GARCH captures short-run clustering and the leverage effect but lacks the long-memory structure that drives MSM's superior long-horizon forecasts (Calvet & Fisher 2004, 2008). The component-GARCH literature (Engle & Lee 1999) shows that a multiplicative permanent × transitory decomposition of variance yields better fit than either pure model. This design implements that decomposition: MSM as the slow permanent component, GJR-GARCH as the fast transitory component.

---

## Model

```
r_t = σ · √(M_t · h_t) · ε_t,    ε_t ~ iid N(0,1)
```

**MSM component** `M_t`: standard cascade of `kbar` Markov-switching multipliers, identical to `Msm()`. Parameters `{m0, b, gammak, sigma}`.

**GJR-GARCH component** `h_t`: short-run transitory factor with unit unconditional mean.

```
Mhat_{t-1} = Σ_m π_{t-1}[m] · g_m          # E[M_{t-1} | r_{1..t-1}]
ẽ²_{t-1}   = r²_{t-1} / (σ² · Mhat_{t-1})  # MSM-normalised innovation

h_t = (1 - α - β - γ/2)
      + (α + γ · 𝟏[r_{t-1} < 0]) · ẽ²_{t-1}
      + β · h_{t-1}
h_1 = 1
```

Stationarity: `α + β + γ/2 < 1`.
Nesting: `α = β = γ = 0` ⟹ `h_t ≡ 1` ⟹ reduces exactly to `Msm()`.

**Emission density** (state `m`):

```
r_t | M_t = g_m  ~  N(0, σ² · g_m · h_t)
```

---

## Parameter Vector

7 parameters total:

| # | Name | Meaning | Bounds |
|---|------|---------|--------|
| 1 | `m0` | MSM multiplier base | [1, 1.99] |
| 2 | `b` | Cascade frequency ratio | [1, 50] — `NA` at `kbar=1` |
| 3 | `gammak` | Highest-freq switching prob | (0, 1) |
| 4 | `sigma` | Baseline volatility (annualised × √n.vol internally) | (0, ∞) |
| 5 | `alpha` | ARCH coefficient | [0, 0.99) |
| 6 | `beta` | GARCH persistence | [0, 0.99) |
| 7 | `gamma_gjr` | GJR leverage | [0, 2) |

`sigma` stored internally as annualised × √n.vol; divided back in `$coefficients` and `$se` on output — same convention as `Msm()`.

---

## Forward Filter (per-step)

1. **Transition:** `π_t ← kron_apply(π_{t-1}, γ_k)` (kbar ≥ 4) or `π_{t-1} · A` (kbar < 4)
2. **Expected M:** `Mhat_{t-1} = dot(π_t_pred, g_m)` — uses pre-transition distribution
3. **GJR update:** `h_t` from recursion above using `r_{t-1}`, `Mhat_{t-1}`, `h_{t-1}`
4. **Emission:** `ω_t[m] = (1/√2π) · exp(−r_t² / (2σ²·g_m·h_t)) / (σ·√(g_m·h_t))`
5. **Normalise:** `π_t = (ω_t ⊙ π_t) / Σ(ω_t ⊙ π_t)`

`h_t` is a scalar carried alongside `π_t`; no state expansion.

---

## New Files

### R

```
R/Msm_garch.R                  # Msm_garch() top-level function
R/Msm_garch_likelihood.R       # full output: filtered, LLs, h vector
R/Msm_garch_ll.R               # scalar LL wrapper for nlminb
R/Msm_garch_std_err.R          # sandwich SE (Newey-West OPG + 2-sided Hessian)
R/Msm_garch_parameter_check.R  # input validation, bounds, starting values
```

### C++ (`src/`)

```
src/Msm_garch_cpp.cpp
```

Four exported functions following existing dual-dispatch pattern:

```cpp
double Msm_garch_ll_fast_cpp(dat, sigma_gm, A, alpha, beta, gamma_gjr)
double Msm_garch_ll_kron_cpp(dat, sigma_gm, b, gammak, kbar, alpha, beta, gamma_gjr)
List   Msm_garch_fast_cpp(dat, sigma_gm, A, alpha, beta, gamma_gjr)
List   Msm_garch_kron_cpp(dat, sigma_gm, b, gammak, kbar, alpha, beta, gamma_gjr)
```

`_fast_cpp` used when `kbar < 4`; `_kron_cpp` when `kbar >= 4`.

`List` variants return `list(filtered, LL, LLs, h)` — `h` is the `N`-length vector of `h_t` values.

---

## `Msm_garch()` R Function

```r
Msm_garch(ret, kbar = 1, n.vol = 252, para0 = NULL, nw.lag = 0)
```

**Starting values:** `{m0, b, gammak, sigma}` from `Msm_parameter_check()`; `alpha=0.05`, `beta=0.85`, `gamma_gjr=0.05`.

**Optimisation:** `nlminb` with `eval.max=1000`, `iter.max=500`. Box bounds enforce stationarity without interior penalty.

**Returns `msmgarchmodel` list:**

| Field | Content |
|-------|---------|
| `LL` | scalar log-likelihood at optimum |
| `LLs` | N-vector of per-observation log-likelihoods |
| `filtered` | N × 2^kbar matrix of filtered state probabilities |
| `h` | N-vector of `h_t` values |
| `A` | transition matrix |
| `g.m` | state volatility values |
| `coefficients` | 7×1 matrix (sigma back-scaled, b=NA at kbar=1) |
| `se` | 7×1 matrix (same conventions) |
| `optim.msg`, `optim.convergence`, `optim.iter` | from nlminb |

---

## Standard Errors

Identical approach to `Msm_std_err()`:
- Outer product of gradients (score), Newey-West with `nw.lag` lags
- Two-sided numerical Hessian
- Sandwich: `H^{-1} · B · H^{-1}`
- Singular Hessian → `MASS::ginv` fallback with warning

---

## S3 Methods

New class `msmgarchmodel`. Implement:

- `print.msmgarchmodel` — summary table with params, SE, LL
- `summary.msmgarchmodel` — same as print (matching `msmmodel` pattern)
- `plot.msmgarchmodel` — conditional volatility `σ·√(M_t·h_t)` and filtered probs
- `predict.msmgarchmodel` — forward propagate both MSM state distribution and `h_{T+h}` forecast

`predict` uses `h_{T+h} = (1-α-β-γ/2) · (1 + E[ẽ²·𝟏_neg]) + (α+γ/2)·ẽ²_T·E[...] + β·h_{T+h-1}`. For h > 1, `E[ẽ²_{T+h-1}] = 1` under stationarity, so `h_{T+h} → 1` as h → ∞. Long-run forecast reverts to MSM-only component.

---

## Tests

All in `tests/testthat/test-Msm_garch.R`:

| Test | Criterion |
|------|-----------|
| Nesting: `Msm_garch(ret, kbar=k)` LL with `alpha=beta=gamma_gjr=0` matches `Msm(ret, kbar=k)` LL | ≤ 1e-6 |
| Convergence on `calvet2004data`, kbar=1 and kbar=2 | `optim.convergence == 0` |
| `length(coef)` = 7, `is.finite(coef[-2])`, `is.na(coef[2])` at kbar=1 | pass |
| `length(se)` = 7, `is.finite(se[-2])`, `is.na(se[2])` at kbar=1 | pass |
| `length($h)` = `nrow(ret)`, all `$h > 0` | pass |
| `rowSums($filtered)` all ≈ 1 | pass |
| `alpha + beta + gamma_gjr/2 < 1` at optimum | pass |
| `gamma_gjr > 0` on equity returns (informational, not hard) | informational |

---

## Scientific References

- Calvet & Fisher (2004). "How to Forecast Long-Run Volatility." *J. Financial Econometrics*.
- Calvet & Fisher (2008). *Multifractal Volatility.* Academic Press. Ch. 9.
- Engle & Lee (1999). "A Long-Run and Short-Run Component Model of Stock Return Volatility." In *Cointegration, Causality, and Forecasting*.
- Lux & Morales-Arias (2010). "Forecasting volatility under fractality, regime-switching, long memory and student-t innovations." *Computational Statistics & Data Analysis*.
- Glosten, Jagannathan & Runkle (1993). "On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *J. Finance*. (GJR-GARCH source)

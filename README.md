R Package for Markov Switching Multifractal Models
================

<!-- README.md is generated from README.Rmd. Please edit that file -->

## The Markov Switching Multifractal Model

The Markov Switching Multifractal (MSM) model is a stochastic volatility model where conditional volatility is a product of finitely many latent state variables (volatility components) with varying degrees of persistence. Let *r*<sub>*t*</sub> ≡ ln(*P*<sub>*t*</sub>/*P*<sub>*t*−1</sub>). Then:

*r*<sub>*t*</sub> = *σ*(*M*<sub>1,*t*</sub> · *M*<sub>2,*t*</sub> · ... · *M*<sub>*k*,*t*</sub>)<sup>1/2</sup> *ε*<sub>*t*</sub>

where *ε*<sub>*t*</sub> ~ *i.i.d.* 𝒩(0,1) and *σ* > 0 is the unconditional standard deviation. The univariate MSM(*k*) model has four parameters (*m*<sub>0</sub>, *b*, *γ*<sub>*k*</sub>, *σ*):

- *m*<sub>0</sub> ∈ (1, 2]: size of each volatility component
- *b* ∈ (1, ∞): controls switching probability spacing
- *γ*<sub>*k*</sub> ∈ (0, 1): highest-frequency switching probability
- *σ* ∈ (0, ∞): unconditional standard deviation

For background, see Calvet, Fisher & Mandelbrot (1997); Calvet & Fisher (2001, 2002, 2004); Calvet, Fisher & Thompson (2006).

---

## Models

| Function | Model | Params | Class |
|----------|-------|--------|-------|
| `Msm` | Univariate MSM | 4 | `msmmodel` |
| `Msm_rolling` | Rolling/expanding univariate MSM | — | list |
| `Amsm_scale` | Univariate MSM — sigma-scaling leverage | 5 | `amsmmodel` |
| `Amsm_gjr` | Univariate MSM — GJR additive leverage | 5 | `amsmmodel` |
| `Amsm_atp` | Univariate MSM — asymmetric transition probabilities | 5 | `amsmmodel` |
| `Bmsm` | Bivariate MSM | 9 | `bmsmmodel` |
| `Bmsm_rolling` | Rolling/expanding bivariate MSM | — | list |
| `Bmsm_scale` | Bivariate MSM — sigma-scaling leverage | 11 | `bmsmmodel` |

---

## Installation

```r
# install.packages("devtools")
devtools::install_github("mvsantoss/MSMr")
```

---

## Univariate MSM

Basic estimation:

```r
library(MSM)
data("calvet2006returns")
ret <- as.matrix(calvet2006returns[1:2000, 2]) * 100

fit <- Msm(ret, kbar = 5, n.vol = 252)
summary(fit)
#> *----------------------------------------------------------------------------*
#>   Markov Switching Multifractal Model With 5 Volatility Component(s)
#> *----------------------------------------------------------------------------*
#>
#>        Estimate Std. Error t.value p.value
#> m0       1.5083     0.0266  56.780  <2e-16 ***
#> b        4.9436     1.3273   3.725  0.0002 ***
#> gammak   0.1246     0.0399   3.127  0.0018 **
#> sigma    0.4185     0.0354  11.812  <2e-16 ***
#>
#> LogLikelihood: -514.4351
```

Note: for *k* = 1, *b* is not identified and reported as `NA`.

Fitted and forecast volatility:

```r
yhat <- predict(fit)          # fitted vol
yhat <- predict(fit, h = 10)  # 10-step ahead forecast
```

Decompose into volatility components:

```r
em <- Msm_decompose(fit)
plot(em)
```

---

## Bivariate MSM

The bivariate MSM extends the univariate model to two return series with a shared transition structure:

*r*<sub>*t*</sub><sup>*α*</sup> = *σ*<sub>*α*</sub>(*M*<sub>1,*t*</sub><sup>*α*</sup> · ... · *M*<sub>*k*,*t*</sub><sup>*α*</sup>)<sup>1/2</sup> *ε*<sub>*α*,*t*</sub>

*r*<sub>*t*</sub><sup>*β*</sup> = *σ*<sub>*β*</sub>(*M*<sub>1,*t*</sub><sup>*β*</sup> · ... · *M*<sub>*k*,*t*</sub><sup>*β*</sup>)<sup>1/2</sup> *ε*<sub>*β*,*t*</sub>

where (*ε*<sub>*α*,*t*</sub>, *ε*<sub>*β*,*t*</sub>) ~ 𝒩(0, Σ). The 9-parameter vector is:

Θ = (*m*<sub>0</sub><sup>*α*</sup>, *m*<sub>0</sub><sup>*β*</sup>, *σ*<sub>*α*</sub>, *σ*<sub>*β*</sub>, *b*, *γ*<sub>*k*</sub>, *ρ*<sub>*ε*</sub>, *λ*, *ρ*<sub>*m*</sub>)

```r
rets <- 100 * calvet2006returns[1:2000, 3:4]

fit2 <- Bmsm(as.matrix(rets), kbar = 2, n = 252)
summary(fit2)
#> *-------------------------------------------------------------------------------------*
#>   Bivariate Markov Switching Multifractal Model With 2 Volatility Component(s)
#> *-------------------------------------------------------------------------------------*
#>
#>        Estimate Std. Error t.value p.value
#> m01      1.8678     0.0110 169.232  <2e-16 ***
#> m02      1.8298     0.0148 123.994  <2e-16 ***
#> sigma1   0.5304     0.0450  11.796  <2e-16 ***
#> sigma2   0.4571     0.0410  11.151  <2e-16 ***
#> gammak   0.3226     0.0424   7.606  <2e-16 ***
#> b       13.4536     4.4497   3.023  0.0025 **
#> rhoe    -0.2310     0.0247  -9.356  <2e-16 ***
#> lambda   0.3942     0.2294   1.718  0.0859 .
#> rhom     0.7134     0.3611   1.976  0.0484 *
#>
#> LogLikelihood: 1599.973
```

Forecast bivariate volatility, covariance, and correlation:

```r
yhat2 <- predict(fit2)        # fitted vol1, vol2, cov, corr
yhat2 <- predict(fit2, h = 5) # 5-step ahead
```

---

## Asymmetric Univariate MSM

Three leverage mechanisms for the univariate model. All add a `lev` parameter (5 params total) and return an `amsmmodel` object with the same `summary`, `print`, `predict` S3 methods as `msmmodel`.

### Amsm_scale — sigma-scaling

Scales state-conditional sigma when last return was negative:

```
sigma_eff(t) = sigma * g_m * (1 + lev * I(r_{t-1} < 0))
```

`lev = 0` recovers symmetric `Msm` exactly.

```r
fit_s <- Amsm_scale(ret, kbar = 2, n.vol = 252)
summary(fit_s)
#>        Estimate Std. Error
#> m0       1.5241     0.0281
#> b        5.1033     1.4102
#> gammak   0.1318     0.0412
#> sigma    0.4071     0.0348
#> lev      0.2847     0.0631
```

### Amsm_gjr — GJR additive variance

Adds to state-conditional variance (not sigma) when last return was negative:

```
Var(r_t | M_t = g_m) = sigma^2 * g_m + lev * I(r_{t-1} < 0) * r_{t-1}^2
```

`lev >= 0` only (variance must stay positive).

```r
fit_g <- Amsm_gjr(ret, kbar = 2, n.vol = 252)
```

### Amsm_atp — asymmetric transition probabilities

Uses two transition matrices: `A_pos` for positive lagged returns, `A_neg` built from `gamma_k * (1 + lev)` for negative ones. Positive `lev` accelerates volatility switching after downturns.

```r
fit_a <- Amsm_atp(ret, kbar = 2, n.vol = 252)
```

All three share the same interface:

```r
# Common pattern
fit <- Amsm_scale(ret, kbar = 2)   # or Amsm_gjr / Amsm_atp
summary(fit)
predict(fit)          # fitted vol
predict(fit, h = 5)   # h-step ahead
```

---

## Rolling and Expanding Window Estimation

Both `Msm_rolling` and `Bmsm_rolling` re-fit the model on successive windows and return per-window parameters and forecasts.

### Msm_rolling

```r
result <- Msm_rolling(
  ret,
  window = 250,   # fixed window length
  h      = 1,     # forecast horizon
  type   = "rolling",   # or "expanding"
  kbar   = 2,
  n.vol  = 252
)

head(result$parameters)
#   window_end    m0        b   gammak    sigma converged
# 1        250 1.512   4.8831   0.1201   0.4193      TRUE
# 2        251 1.509   4.9102   0.1224   0.4181      TRUE

head(result$forecasts)
#   window_end h      vol   vol.sq
# 1        250 1   0.1521  0.02313
# 2        251 1   0.1488  0.02214
```

`type = "expanding"` grows the window from observation 1 rather than sliding it.

### Bmsm_rolling

```r
result2 <- Bmsm_rolling(
  as.matrix(rets),
  window = 250,
  h      = 1,
  type   = "rolling",
  kbar   = 1,
  n      = 252,
  s.err  = FALSE   # skip SE for speed in rolling context
)

head(result2$parameters)
#   window_end  m01   m02  sigma1 sigma2 gammak    b  rhoe lambda  rhom converged
# 1        250 1.87  1.83   0.531  0.458  0.318 NA   -0.231 0.394 0.714      TRUE

head(result2$forecasts)
#   window_end h   vol1   vol2    covt   rho.t
# 1        250 1  0.532  0.459  0.0562  0.231
```

Output columns for `Bmsm_rolling` forecasts: `vol1`, `vol2`, `covt` (conditional covariance), `rho.t` (conditional correlation).

---

## Bivariate MSM with Leverage (Bmsm_scale)

Standard `Bmsm` is symmetric — it cannot distinguish the direction of return shocks. `Bmsm_scale` adds the well-known **leverage effect**: negative lagged returns scale up next-period conditional volatility for that series.

### Model

Each series gets its own leverage coefficient applied to the state-conditional sigma:

```
sigma1_eff(m, t) = sigma1 * g1_m * (1 + lev1 * I(r1_{t-1} < 0))
sigma2_eff(m, t) = sigma2 * g2_m * (1 + lev2 * I(r2_{t-1} < 0))
```

- Own-series trigger only: `sigma1` reacts to lagged `r1`, `sigma2` to lagged `r2`
- Separate leverage coefficients per series
- `lev1 = lev2 = 0` recovers symmetric `Bmsm` exactly

### Parameters (11)

| Parameter | Meaning | Bounds |
|-----------|---------|--------|
| m01, m02 | Multiplier component mean | (1, 2) |
| sigma1, sigma2 | Base annualised vol | (0, ∞) |
| gammak | Highest-freq switching probability | (0, 1) |
| b | Frequency spacing | (1, 50) |
| **lev1, lev2** | **Scale leverage per series** | **(-0.99, 10)** |
| rhoe | Innovation correlation | (-1, 1) |
| lambda | State-correlation parameter | (0, 1) |
| rhom | Multiplier correlation | (-1, 1) |

### Usage

```r
rets <- 100 * calvet2006returns[1:2000, 3:4]

fit_lev <- Bmsm_scale(as.matrix(rets), kbar = 2, n = 252)
summary(fit_lev)
```

The function uses two-stage MLE:
- **Stage 1** (8 params): marginal dynamics — estimates *m*<sub>0</sub>, *σ*, *γ*<sub>*k*</sub>, *b*, *lev* independently per series
- **Stage 2** (3 params): joint dynamics — estimates *ρ*<sub>*ε*</sub>, *λ*, *ρ*<sub>*m*</sub> with stage 1 fixed

Returns a `bmsmmodel` object — all existing S3 methods (`summary`, `print`, `plot`, `predict`) work unchanged.

```r
yhat_lev <- predict(fit_lev)        # fitted vol1, vol2
yhat_lev <- predict(fit_lev, h = 5) # h-step ahead
```

Disable standard errors for speed:

```r
fit_lev <- Bmsm_scale(as.matrix(rets), kbar = 2, s.err = FALSE)
```

---

## Performance Note

The transition matrix grows as 2<sup>*k*</sup> × 2<sup>*k*</sup>. At *k* = 10 that is 1,048,576 elements. The inner filter loop runs over every observation. The package uses C++/RcppArmadillo for all likelihood computations and switches to a Kronecker-factored algorithm for *k* ≥ 4, but estimation time still grows rapidly with *k*. Recommended: *k* ≤ 10.

---

## References

Calvet, L. E., and A. J. Fisher. 2004. "How to Forecast Long-Run Volatility: Regime Switching and the Estimation of Multifractal Processes." *Journal of Financial Econometrics* 2(1): 49–83.

Calvet, L. E., A. J. Fisher, and S. B. Thompson. 2006. "Volatility Comovement: A Multifrequency Approach." *Journal of Econometrics* 131(1–2): 179–215.

Calvet, L., and A. Fisher. 2001. "Forecasting Multifractal Volatility." *Journal of Econometrics* 105(1): 27–58.

———. 2002. "Multifractality in Asset Returns: Theory and Evidence." *Review of Economics and Statistics* 84(3): 381–406.

Calvet, L., A. Fisher, and B. Mandelbrot. 1997. "Large Deviations and the Distribution of Price Changes." Cowles Foundation Discussion Paper 1165.

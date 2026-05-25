# Bmsm_scale Leverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sigma-scaling leverage effect to Bivariate MSM so negative return shocks raise next-period volatility for each series independently.

**Architecture:** Approach A — new files only, zero changes to existing code. Four new C++ functions in one new `.cpp` file handle leverage-scaled bivariate filtering. Five new R files implement stage 1/2 likelihoods, SE helpers, std err, and the top-level `Bmsm_scale()` function. Returns `"bmsmmodel"` class so all existing S3 methods work.

**Tech Stack:** R, Rcpp/RcppArmadillo, `nlminb`, `devtools`, `testthat`

---

## File Map

| File | Purpose |
|------|---------|
| `src/Bmsm_scale_filtered_cpp.cpp` | C++: 4 exported functions — `Bmsm_scale_ll_cpp`, `Bmsm_scale_ll_kron` (stage 2 opt), `Bmsm_scale_filtered_cpp`, `Bmsm_scale_filtered_kron` (full output) |
| `R/Bmsm_scale_stage1_likelihood.R` | Stage 1 LL: sum of two `Amsm_scale_ll` calls |
| `R/Bmsm_scale_stage2_likelihood.R` | Stage 2 LL: wraps `Bmsm_scale_ll_cpp` / `Bmsm_scale_ll_kron` |
| `R/Bmsm_scale_filtered2.R` | Post-fit full output: wraps `Bmsm_scale_filtered_cpp` / `Bmsm_scale_filtered_kron` |
| `R/Bmsm_scale_stage1_LLs.R` | Per-obs LLs for stage 1 SE gradient |
| `R/Bmsm_scale_stage1hess_LL.R` | Scalar LL for stage 1 SE hessian |
| `R/Bmsm_scale_stage2_LLs.R` | Per-obs LLs for stage 2 SE gradient |
| `R/Bmsm_scale_stage2hess_LL.R` | Scalar LL for stage 2 SE hessian |
| `R/Bmsm_scale_std_err.R` | Sandwich SE for 11-param vector |
| `R/Bmsm_scale.R` | Top-level `Bmsm_scale()` function |
| `tests/testthat/test-Bmsm_scale.R` | Tests: nesting, estimation, SE |

---

## Task 1: C++ Filtered Functions with Leverage

**Files:**
- Create: `src/Bmsm_scale_filtered_cpp.cpp`

- [ ] **Step 1.1: Write the C++ file**

```cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace Rcpp;

static inline void bkron_apply_sc(arma::rowvec& pi, const arma::vec& gamma_k,
                                   double lamda, double rho_m) {
    int kbar = gamma_k.n_elem, n = pi.n_elem;
    for (int k = 0; k < kbar; k++) {
        double g  = gamma_k(k);
        double bv = g * ((1.0 - lamda) * g + lamda);
        double g2 = 0.5 * g;
        double p  = 1.0 - g + bv * (1.0 + rho_m) * 0.25;
        double q  = 1.0 - g + bv * (1.0 - rho_m) * 0.25;
        double t00 = p,          t01 = 1.0 - g2 - p, t03 = g - 1.0 + p;
        double t11 = q,          t10 = 1.0 - g2 - q, t12 = g - 1.0 + q;
        int stride = 1;
        for (int s = 0; s < k; s++) stride *= 4;
        int block = stride * 4;
        for (int base2 = 0; base2 < n; base2 += block) {
            for (int i = 0; i < stride; i++) {
                double v0 = pi(base2 + i),          v1 = pi(base2 + i + stride);
                double v2 = pi(base2 + i + 2*stride), v3 = pi(base2 + i + 3*stride);
                double s03 = v0 + v3, s12 = v1 + v2;
                pi(base2 + i)            = t00*v0 + t10*s12 + t03*v3;
                pi(base2 + i + stride)   = t01*s03 + t11*v1 + t12*v2;
                pi(base2 + i + 2*stride) = t01*s03 + t12*v1 + t11*v2;
                pi(base2 + i + 3*stride) = t03*v0 + t10*s12 + t00*v3;
            }
        }
    }
}

static inline arma::vec bmsm_scale_make_gamma_k(double b, double gamma_kbar, int kbar) {
    arma::vec gamma_k(kbar);
    gamma_k(0) = 1.0 - std::pow(1.0 - gamma_kbar, 1.0 / std::pow(b, kbar - 1));
    for (int i = 1; i < kbar; i++)
        gamma_k(i) = 1.0 - std::pow(1.0 - gamma_k(0), std::pow(b, i));
    return gamma_k;
}

// Stage 2 scalar LL for optimization, kbar < 4.
// sigma1_eff(t,m) = sigma1*g1_m*(1+lev1*I(r1_{t-1}<0)), same for series 2.
// [[Rcpp::export]]
NumericVector Bmsm_scale_ll_cpp(const arma::mat& dat, const arma::mat& A, const arma::mat& gm,
    const double& rhoe, const double& sigma1, const double& sigma2,
    double lev1, double lev2) {
    int T = dat.n_rows, k = A.n_cols;
    double pa       = 1.0 / (2.0 * M_PI * std::sqrt(1.0 - rhoe * rhoe));
    double inv_2var = 1.0 / (2.0 * (1.0 - rhoe * rhoe));
    const arma::rowvec sg1_base = sigma1 * gm.row(0);
    const arma::rowvec sg2_base = sigma2 * gm.row(1);
    arma::rowvec piA(k), C(k), r1(k), r2(k), w(k);
    arma::rowvec pia(k, arma::fill::zeros);
    arma::colvec LLs(T, arma::fill::zeros);
    piA.fill(1.0 / k);
    piA    = piA * A;
    pia(0) = 1.0;
    for (int t = 0; t < T; t++) {
        double s1 = (t > 0 && dat(t-1, 0) < 0.0) ? (1.0 + lev1) : 1.0;
        double s2 = (t > 0 && dat(t-1, 1) < 0.0) ? (1.0 + lev2) : 1.0;
        arma::rowvec sg1  = sg1_base * s1;
        arma::rowvec sg2  = sg2_base * s2;
        arma::rowvec sg12 = sg1 % sg2;
        r1 = dat(t, 0) / sg1;
        r2 = dat(t, 1) / sg2;
        w  = pa * arma::exp(-(r1%r1 + r2%r2 - 2.0*rhoe*(r1%r2)) * inv_2var) / sg12 + 1e-16;
        C  = w % piA;
        double ft = arma::accu(C);
        LLs(t)  = std::log(ft);
        if (ft == 0.0) { piA = pia * A; } else { piA = (C / ft) * A; }
    }
    return NumericVector::create(Named("LL") = -arma::accu(LLs));
}

// Stage 2 scalar LL for optimization, kbar >= 4.
// [[Rcpp::export]]
NumericVector Bmsm_scale_ll_kron(const arma::mat& dat, const arma::mat& gm,
    const double& rhoe, const double& sigma1, const double& sigma2,
    double b, double gamma_kbar, double lamda, double rho_m, int kbar,
    double lev1, double lev2) {
    int T = dat.n_rows, k = 1 << (2 * kbar);
    double pa       = 1.0 / (2.0 * M_PI * std::sqrt(1.0 - rhoe * rhoe));
    double inv_2var = 1.0 / (2.0 * (1.0 - rhoe * rhoe));
    const arma::vec  gamma_k    = bmsm_scale_make_gamma_k(b, gamma_kbar, kbar);
    const arma::rowvec sg1_base = sigma1 * gm.row(0);
    const arma::rowvec sg2_base = sigma2 * gm.row(1);
    arma::rowvec pi_t(k), C(k), r1(k), r2(k), w(k);
    arma::colvec LLs(T, arma::fill::zeros);
    pi_t.fill(1.0 / k);
    bkron_apply_sc(pi_t, gamma_k, lamda, rho_m);
    for (int t = 0; t < T; t++) {
        double s1 = (t > 0 && dat(t-1, 0) < 0.0) ? (1.0 + lev1) : 1.0;
        double s2 = (t > 0 && dat(t-1, 1) < 0.0) ? (1.0 + lev2) : 1.0;
        arma::rowvec sg1  = sg1_base * s1;
        arma::rowvec sg2  = sg2_base * s2;
        arma::rowvec sg12 = sg1 % sg2;
        r1 = dat(t, 0) / sg1;
        r2 = dat(t, 1) / sg2;
        w  = pa * arma::exp(-(r1%r1 + r2%r2 - 2.0*rhoe*(r1%r2)) * inv_2var) / sg12 + 1e-16;
        C  = w % pi_t;
        double ft = arma::accu(C);
        LLs(t)  = std::log(ft);
        if (ft == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else           { pi_t = C / ft; }
        bkron_apply_sc(pi_t, gamma_k, lamda, rho_m);
    }
    return NumericVector::create(Named("LL") = -arma::accu(LLs));
}

// Full filtered output (filtered.P + LL + LLs), kbar < 4.
// [[Rcpp::export]]
List Bmsm_scale_filtered_cpp(const arma::mat& dat, const arma::mat& A, const arma::mat& gm,
    const double& rhoe, const double& sigma1, const double& sigma2,
    double lev1, double lev2) {
    int T = dat.n_rows, k = A.n_cols;
    double pa       = 1.0 / (2.0 * M_PI * std::sqrt(1.0 - rhoe * rhoe));
    double inv_2var = 1.0 / (2.0 * (1.0 - rhoe * rhoe));
    const arma::rowvec sg1_base = sigma1 * gm.row(0);
    const arma::rowvec sg2_base = sigma2 * gm.row(1);
    arma::mat P(T, k, arma::fill::zeros);
    arma::rowvec piA(k), C(k), r1(k), r2(k), w(k);
    arma::rowvec pia(k, arma::fill::zeros);
    arma::colvec LLs(T, arma::fill::zeros);
    piA.fill(1.0 / k);
    piA    = piA * A;
    pia(0) = 1.0;
    for (int t = 0; t < T; t++) {
        double s1 = (t > 0 && dat(t-1, 0) < 0.0) ? (1.0 + lev1) : 1.0;
        double s2 = (t > 0 && dat(t-1, 1) < 0.0) ? (1.0 + lev2) : 1.0;
        arma::rowvec sg1  = sg1_base * s1;
        arma::rowvec sg2  = sg2_base * s2;
        arma::rowvec sg12 = sg1 % sg2;
        r1 = dat(t, 0) / sg1;
        r2 = dat(t, 1) / sg2;
        w  = pa * arma::exp(-(r1%r1 + r2%r2 - 2.0*rhoe*(r1%r2)) * inv_2var) / sg12 + 1e-16;
        C  = w % piA;
        double ft = arma::accu(C);
        LLs(t)  = std::log(ft);
        if (ft == 0.0) { piA = pia * A; } else { piA = (C / ft) * A; }
        P.row(t) = piA;
    }
    return List::create(
        Named("filtered.P") = P,
        Named("LL")         = -arma::accu(LLs),
        Named("LLs")        = LLs);
}

// Full filtered output (filtered.P + LL + LLs), kbar >= 4.
// [[Rcpp::export]]
List Bmsm_scale_filtered_kron(const arma::mat& dat, const arma::mat& gm,
    const double& rhoe, const double& sigma1, const double& sigma2,
    double b, double gamma_kbar, double lamda, double rho_m, int kbar,
    double lev1, double lev2) {
    int T = dat.n_rows, k = 1 << (2 * kbar);
    double pa       = 1.0 / (2.0 * M_PI * std::sqrt(1.0 - rhoe * rhoe));
    double inv_2var = 1.0 / (2.0 * (1.0 - rhoe * rhoe));
    const arma::vec  gamma_k    = bmsm_scale_make_gamma_k(b, gamma_kbar, kbar);
    const arma::rowvec sg1_base = sigma1 * gm.row(0);
    const arma::rowvec sg2_base = sigma2 * gm.row(1);
    arma::mat P(T, k, arma::fill::zeros);
    arma::rowvec pi_t(k), C(k), r1(k), r2(k), w(k);
    arma::colvec LLs(T, arma::fill::zeros);
    pi_t.fill(1.0 / k);
    bkron_apply_sc(pi_t, gamma_k, lamda, rho_m);
    for (int t = 0; t < T; t++) {
        double s1 = (t > 0 && dat(t-1, 0) < 0.0) ? (1.0 + lev1) : 1.0;
        double s2 = (t > 0 && dat(t-1, 1) < 0.0) ? (1.0 + lev2) : 1.0;
        arma::rowvec sg1  = sg1_base * s1;
        arma::rowvec sg2  = sg2_base * s2;
        arma::rowvec sg12 = sg1 % sg2;
        r1 = dat(t, 0) / sg1;
        r2 = dat(t, 1) / sg2;
        w  = pa * arma::exp(-(r1%r1 + r2%r2 - 2.0*rhoe*(r1%r2)) * inv_2var) / sg12 + 1e-16;
        C  = w % pi_t;
        double ft = arma::accu(C);
        LLs(t)  = std::log(ft);
        if (ft == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else           { pi_t = C / ft; }
        bkron_apply_sc(pi_t, gamma_k, lamda, rho_m);
        P.row(t) = pi_t;
    }
    return List::create(
        Named("filtered.P") = P,
        Named("LL")         = -arma::accu(LLs),
        Named("LLs")        = LLs);
}
```

- [ ] **Step 1.2: Regenerate Rcpp exports and compile**

Run from R (in `/home/jovyan/workspace/MSMr`):
```r
Rcpp::compileAttributes()
devtools::load_all(".")
```
Expected: compiles without errors. `Bmsm_scale_ll_cpp`, `Bmsm_scale_ll_kron`, `Bmsm_scale_filtered_cpp`, `Bmsm_scale_filtered_kron` appear in the loaded namespace.

- [ ] **Step 1.3: Verify C++ functions are callable**

```r
library(MSM)  # or after load_all
data("calvet2006returns")
ret <- as.matrix(calvet2006returns[,2:3]) * 100
ret_small <- ret[1:100, ]
gm <- Bmsm_states(1.5, 1.5, 1)
A  <- Bmsm_A(1, 2.5, 0.9, 0.9, 0.9)
result <- Bmsm_scale_ll_cpp(ret_small, A, gm, 0.1, 0.02, 0.03, 0.0, 0.0)
stopifnot(is.finite(result[["LL"]]))
```
Expected: returns a finite scalar.

- [ ] **Step 1.4: Commit**

```bash
git add src/Bmsm_scale_filtered_cpp.cpp src/RcppExports.cpp R/RcppExports.R
git commit -m "feat: add C++ bivariate filtered functions with sigma-scaling leverage"
```

---

## Task 2: Stage 1 Likelihood

**Files:**
- Create: `R/Bmsm_scale_stage1_likelihood.R`
- Test: `tests/testthat/test-Bmsm_scale.R`

- [ ] **Step 2.1: Write failing test**

Create `tests/testthat/test-Bmsm_scale.R`:
```r
library(MSM)

data("calvet2006returns")
ret     <- as.matrix(calvet2006returns[, 2:3]) * 100
ret_small <- ret[1:100, ]

# --- Stage 1: lev=0 must reproduce Bmsm_stage1_likelihood ---

test_that("stage1 LL with lev=0 matches Bmsm_stage1_likelihood", {
  para_bmsm  <- c(1.4330, 1.5768, 5.0433, 7.6284, 0.7299, 8.2716)
  para_scale <- c(para_bmsm[1:6], 0, 0)  # append lev1=0, lev2=0

  ll_bmsm  <- Bmsm_stage1_likelihood(para_bmsm,  ret_small, kbar = 1, n.vol = 252)
  ll_scale <- Bmsm_scale_stage1_likelihood(para_scale, ret_small, kbar = 1, n.vol = 252)

  expect_equal(as.numeric(ll_scale), as.numeric(ll_bmsm), tolerance = 1e-6)
})
```

- [ ] **Step 2.2: Run test to verify it fails**

```r
devtools::test(filter = "Bmsm_scale")
```
Expected: FAIL — `"could not find function "Bmsm_scale_stage1_likelihood""`

- [ ] **Step 2.3: Implement stage 1 likelihood**

Create `R/Bmsm_scale_stage1_likelihood.R`:
```r
#' Stage 1 log-likelihood for Bmsm_scale model.
#'
#' Sum of two independent univariate scale-leverage MSM log-likelihoods.
#' Estimates [m01, m02, sigma1, sigma2, gammak, b, lev1, lev2].
#'
#' @param para length-8 vector [m01, m02, sigma1, sigma2, gammak, b, lev1, lev2].
#' @param dat N x 2 return matrix.
#' @param kbar number of frequency components.
#' @param n.vol trading days per year.
#' @return scalar log-likelihood (negative, for minimisation).
#' @export
Bmsm_scale_stage1_likelihood <- function(para, dat, kbar, n.vol) {
  m01     <- para[1]
  m02     <- para[2]
  sigma1  <- para[3]
  sigma2  <- para[4]
  gamma.k <- para[5]
  b       <- para[6]
  lev1    <- para[7]
  lev2    <- para[8]

  ret1 <- matrix(dat[, 1], ncol = 1)
  ret2 <- matrix(dat[, 2], ncol = 1)
  par1 <- c(m01, b, gamma.k, sigma1, lev1)
  par2 <- c(m02, b, gamma.k, sigma2, lev2)

  ll <- Amsm_scale_ll(par1, kbar, ret1, n.vol) +
        Amsm_scale_ll(par2, kbar, ret2, n.vol)
  return(ll)
}
```

- [ ] **Step 2.4: Run test to verify it passes**

```r
devtools::load_all(".")
devtools::test(filter = "Bmsm_scale")
```
Expected: stage1 nesting test PASS.

- [ ] **Step 2.5: Commit**

```bash
git add R/Bmsm_scale_stage1_likelihood.R tests/testthat/test-Bmsm_scale.R
git commit -m "feat: add Bmsm_scale stage 1 likelihood + nesting test"
```

---

## Task 3: Stage 2 Likelihood and Filtered Output

**Files:**
- Create: `R/Bmsm_scale_stage2_likelihood.R`
- Create: `R/Bmsm_scale_filtered2.R`
- Modify: `tests/testthat/test-Bmsm_scale.R`

- [ ] **Step 3.1: Add failing test for stage 2 nesting**

Append to `tests/testthat/test-Bmsm_scale.R`:
```r
# --- Stage 2: lev=0 must reproduce Bmsm_stage2_likelihood2 ---

test_that("stage2 LL with lev=0 matches Bmsm_stage2_likelihood2", {
  para1_bmsm  <- c(1.4330, 1.5768, 5.0433, 7.6284, 0.7299, 8.2716)
  para1_scale <- c(para1_bmsm, 0, 0)
  para2 <- c(0.1090, 0.4421, 0.8823)

  ll_bmsm  <- Bmsm_stage2_likelihood2(para2, 1, ret_small, para1_bmsm, 252)
  ll_scale <- Bmsm_scale_stage2_likelihood(para2, 1, ret_small, para1_scale, 252)

  expect_equal(as.numeric(ll_scale), as.numeric(ll_bmsm), tolerance = 1e-6)
})

# --- filtered2: lev=0 must reproduce Bmsm_filtered2 LL ---

test_that("Bmsm_scale_filtered2 with lev=0 matches Bmsm_filtered2 LL", {
  para_bmsm  <- c(1.4330, 1.5768, 5.0433, 7.6284, 0.7299, 8.2716, 0.1090, 0.4421, 0.8823)
  para_scale <- c(para_bmsm[1:6], 0, 0, para_bmsm[7:9])

  ref <- Bmsm_filtered2(para_bmsm,  1, ret_small, 252)
  res <- Bmsm_scale_filtered2(para_scale, 1, ret_small, 252)

  expect_equal(res$LL, ref$LL, tolerance = 1e-6)
  expect_equal(dim(res$filtered.P), dim(ref$filtered.P))
})
```

- [ ] **Step 3.2: Run tests to verify they fail**

```r
devtools::test(filter = "Bmsm_scale")
```
Expected: 2 new FAILs — `"could not find function"` for `Bmsm_scale_stage2_likelihood` and `Bmsm_scale_filtered2`.

- [ ] **Step 3.3: Implement stage 2 likelihood**

Create `R/Bmsm_scale_stage2_likelihood.R`:
```r
#' Stage 2 log-likelihood for Bmsm_scale model.
#'
#' Joint bivariate LL with sigma-scaling leverage. Stage 1 params held fixed.
#'
#' @param para length-3 vector [rhoe, lambda, rhom].
#' @param kbar number of frequency components.
#' @param dat N x 2 return matrix.
#' @param para1 length-8 stage 1 params [m01,m02,sigma1,sigma2,gammak,b,lev1,lev2].
#' @param n.vol trading days per year.
#' @return scalar log-likelihood (negative, for minimisation).
#' @export
Bmsm_scale_stage2_likelihood <- function(para, kbar, dat, para1, n.vol) {
  rhoe  <- para[1]
  lamda <- para[2]
  rho.m <- para[3]

  m01     <- para1[1]
  m02     <- para1[2]
  sigma1  <- para1[3] / sqrt(n.vol)
  sigma2  <- para1[4] / sqrt(n.vol)
  gamma.k <- para1[5]
  b       <- para1[6]
  lev1    <- para1[7]
  lev2    <- para1[8]

  gm <- Bmsm_states(m01, m02, kbar)

  if (kbar >= 4) {
    LL <- Bmsm_scale_ll_kron(dat, gm, rhoe, sigma1, sigma2,
                              b, gamma.k, lamda, rho.m, kbar, lev1, lev2)
  } else {
    A  <- Bmsm_A(kbar, b, gamma.k, lamda, rho.m)
    LL <- Bmsm_scale_ll_cpp(dat, A, gm, rhoe, sigma1, sigma2, lev1, lev2)
  }

  if (!is.finite(LL))
    warning("Log-likelihood is inf. Probably due to all zeros in conditional probability.")
  return(LL)
}
```

- [ ] **Step 3.4: Implement filtered2**

Create `R/Bmsm_scale_filtered2.R`:
```r
#' Filtered probabilities and log-likelihood for Bmsm_scale model.
#'
#' Called after optimisation to produce full output.
#'
#' @param para length-11 param vector [m01,m02,sigma1,sigma2,gammak,b,lev1,lev2,rhoe,lambda,rhom].
#' @param kbar number of frequency components.
#' @param dat N x 2 return matrix.
#' @param n.vol trading days per year.
#' @return list with filtered.P, LL, LLs, A, g.m, para, kbar, n.
#' @export
Bmsm_scale_filtered2 <- function(para, kbar, dat, n.vol) {
  m01     <- para[1]
  m02     <- para[2]
  sigma1  <- para[3] / sqrt(n.vol)
  sigma2  <- para[4] / sqrt(n.vol)
  gamma.k <- para[5]
  b       <- para[6]
  lev1    <- para[7]
  lev2    <- para[8]
  rhoe    <- para[9]
  lamda   <- para[10]
  rho.m   <- para[11]

  gm <- Bmsm_states(m01, m02, kbar)
  A  <- Bmsm_A(kbar, b, gamma.k, lamda, rho.m)

  if (kbar >= 4) {
    likelihood <- Bmsm_scale_filtered_kron(dat, gm, rhoe, sigma1, sigma2,
                                            b, gamma.k, lamda, rho.m, kbar, lev1, lev2)
  } else {
    likelihood <- Bmsm_scale_filtered_cpp(dat, A, gm, rhoe, sigma1, sigma2, lev1, lev2)
  }

  likelihood$A    <- A
  likelihood$g.m  <- gm
  likelihood$para <- para
  likelihood$kbar <- kbar
  likelihood$n    <- n.vol
  return(likelihood)
}
```

- [ ] **Step 3.5: Run tests to verify they pass**

```r
devtools::load_all(".")
devtools::test(filter = "Bmsm_scale")
```
Expected: all 3 tests PASS.

- [ ] **Step 3.6: Commit**

```bash
git add R/Bmsm_scale_stage2_likelihood.R R/Bmsm_scale_filtered2.R tests/testthat/test-Bmsm_scale.R
git commit -m "feat: add Bmsm_scale stage 2 likelihood and filtered2 with nesting tests"
```

---

## Task 4: SE Helper Functions

**Files:**
- Create: `R/Bmsm_scale_stage1_LLs.R`
- Create: `R/Bmsm_scale_stage1hess_LL.R`
- Create: `R/Bmsm_scale_stage2_LLs.R`
- Create: `R/Bmsm_scale_stage2hess_LL.R`

No tests for these — they are internal helpers tested indirectly through `Bmsm_scale_std_err`. Implement all four and commit together.

- [ ] **Step 4.1: Create `R/Bmsm_scale_stage1_LLs.R`**

```r
Bmsm_scale_stage1_LLs <- function(para, kbar, dat, n.vol) {
  if (para[1] >= 2) para[1] <- 1.9999
  if (para[2] >= 2) para[2] <- 1.9999
  if (para[5] >= 1) para[5] <- 0.9999

  ret1 <- matrix(dat[, 1], ncol = 1)
  ret2 <- matrix(dat[, 2], ncol = 1)
  par1 <- c(para[1], para[6], para[5], para[3], para[7])
  par2 <- c(para[2], para[6], para[5], para[4], para[8])

  LLs <- Amsm_scale_likelihood(par1, kbar, ret1, n.vol)$LLs +
         Amsm_scale_likelihood(par2, kbar, ret2, n.vol)$LLs
  return(LLs)
}
```

- [ ] **Step 4.2: Create `R/Bmsm_scale_stage1hess_LL.R`**

```r
Bmsm_scale_stage1hess_LL <- function(para, kbar, dat, n.vol) {
  if (para[1] >= 2) para[1] <- 1.9999
  if (para[2] >= 2) para[2] <- 1.9999
  if (para[5] >= 1) para[5] <- 0.9999

  ret1 <- matrix(dat[, 1], ncol = 1)
  ret2 <- matrix(dat[, 2], ncol = 1)
  par1 <- c(para[1], para[6], para[5], para[3], para[7])
  par2 <- c(para[2], para[6], para[5], para[4], para[8])

  LL <- Amsm_scale_likelihood(par1, kbar, ret1, n.vol)$LL +
        Amsm_scale_likelihood(par2, kbar, ret2, n.vol)$LL
  return(LL)
}
```

- [ ] **Step 4.3: Create `R/Bmsm_scale_stage2_LLs.R`**

```r
Bmsm_scale_stage2_LLs <- function(para, kbar, dat, n.vol, para1) {
  if (para[1] >=  1) para[1] <-  0.9999
  if (para[1] <= -1) para[1] <- -0.9999
  if (para[2] >=  1) para[2] <-  0.9999
  if (para[3] >=  1) para[3] <-  0.9999
  if (para[3] <= -1) para[3] <- -0.9999

  rhoe  <- para[1]; lamda <- para[2]; rho.m <- para[3]
  m01     <- para1[1]; m02     <- para1[2]
  sigma1  <- para1[3] / sqrt(n.vol); sigma2  <- para1[4] / sqrt(n.vol)
  gamma.k <- para1[5]; b <- para1[6]; lev1 <- para1[7]; lev2 <- para1[8]

  gm <- Bmsm_states(m01, m02, kbar)

  if (kbar >= 4) {
    LLs <- Bmsm_scale_filtered_kron(dat, gm, rhoe, sigma1, sigma2,
                                     b, gamma.k, lamda, rho.m, kbar, lev1, lev2)$LLs
  } else {
    A   <- Bmsm_A(kbar, b, gamma.k, lamda, rho.m)
    LLs <- Bmsm_scale_filtered_cpp(dat, A, gm, rhoe, sigma1, sigma2, lev1, lev2)$LLs
  }
  return(LLs)
}
```

- [ ] **Step 4.4: Create `R/Bmsm_scale_stage2hess_LL.R`**

```r
Bmsm_scale_stage2hess_LL <- function(para, kbar, dat, n.vol, para1) {
  if (para[1] >=  1) para[1] <-  0.9999
  if (para[1] <= -1) para[1] <- -0.9999
  if (para[2] >=  1) para[2] <-  0.9999
  if (para[3] >=  1) para[3] <-  0.9999
  if (para[3] <= -1) para[3] <- -0.9999

  rhoe  <- para[1]; lamda <- para[2]; rho.m <- para[3]
  m01     <- para1[1]; m02     <- para1[2]
  sigma1  <- para1[3] / sqrt(n.vol); sigma2  <- para1[4] / sqrt(n.vol)
  gamma.k <- para1[5]; b <- para1[6]; lev1 <- para1[7]; lev2 <- para1[8]

  gm <- Bmsm_states(m01, m02, kbar)

  if (kbar >= 4) {
    LL <- Bmsm_scale_filtered_kron(dat, gm, rhoe, sigma1, sigma2,
                                    b, gamma.k, lamda, rho.m, kbar, lev1, lev2)$LL
  } else {
    A  <- Bmsm_A(kbar, b, gamma.k, lamda, rho.m)
    LL <- Bmsm_scale_filtered_cpp(dat, A, gm, rhoe, sigma1, sigma2, lev1, lev2)$LL
  }
  return(LL)
}
```

- [ ] **Step 4.5: Load and verify helpers are callable**

```r
devtools::load_all(".")
data("calvet2006returns")
ret_small <- as.matrix(calvet2006returns[1:100, 2:3]) * 100
para8 <- c(1.4330, 1.5768, 5.0433, 7.6284, 0.7299, 8.2716, 0, 0)
lls <- Bmsm_scale_stage1_LLs(para8, 1, ret_small, 252)
stopifnot(length(lls) == nrow(ret_small), all(is.finite(lls)))
```
Expected: returns length-100 numeric vector of finite values.

- [ ] **Step 4.6: Commit**

```bash
git add R/Bmsm_scale_stage1_LLs.R R/Bmsm_scale_stage1hess_LL.R \
        R/Bmsm_scale_stage2_LLs.R R/Bmsm_scale_stage2hess_LL.R
git commit -m "feat: add Bmsm_scale SE helper functions (LLs, hess)"
```

---

## Task 5: Standard Errors

**Files:**
- Create: `R/Bmsm_scale_std_err.R`
- Modify: `tests/testthat/test-Bmsm_scale.R`

- [ ] **Step 5.1: Add failing test**

Append to `tests/testthat/test-Bmsm_scale.R`:
```r
# --- SE: lev=0 SE matches Bmsm_std_err (approximately) ---

test_that("Bmsm_scale_std_err with lev=0 is finite and length 11", {
  para_scale <- c(1.4330, 1.5768, 5.0433, 7.6284, 0.7299, 8.2716, 0, 0, 0.1090, 0.4421, 0.8823)
  se <- Bmsm_scale_std_err(para_scale, 1, ret_small, 252)
  expect_equal(nrow(se), 11)
  # b (index 6) is NA for kbar=1; all others finite
  expect_true(is.na(se[6, 1]))
  expect_true(all(is.finite(se[-6, 1])))
})
```

- [ ] **Step 5.2: Run test to verify it fails**

```r
devtools::test(filter = "Bmsm_scale")
```
Expected: FAIL — `"could not find function "Bmsm_scale_std_err""`

- [ ] **Step 5.3: Implement std err**

Create `R/Bmsm_scale_std_err.R`:
```r
#' Sandwich standard errors for Bmsm_scale model.
#'
#' @param para length-11 param vector [m01,m02,sigma1,sigma2,gammak,b,lev1,lev2,rhoe,lambda,rhom].
#' @param kbar number of frequency components.
#' @param ret N x 2 return matrix.
#' @param n.vol trading days per year.
#' @return 11 x 1 matrix of standard errors (NA for b when kbar=1).
#' @export
Bmsm_scale_std_err <- function(para, kbar, ret, n.vol) {
  grad1 <- Bmsm_grad("Bmsm_scale_stage1_LLs",
                      arg.list = list(para = para[1:8], kbar = kbar,
                                      dat = ret, n.vol = n.vol))
  grad2 <- Bmsm_grad("Bmsm_scale_stage2_LLs",
                      arg.list = list(para = para[9:11], kbar = kbar,
                                      dat = ret, n.vol = n.vol, para1 = para[1:8]))

  H1 <- Bmsm_hessian_2_sided("Bmsm_scale_stage1hess_LL",
                               arg.list = list(para = para[1:8], kbar = kbar,
                                               dat = ret, n.vol = n.vol))
  H2 <- Bmsm_hessian_2_sided("Bmsm_scale_stage2hess_LL",
                               arg.list = list(para = para[9:11], kbar = kbar,
                                               dat = ret, n.vol = n.vol, para1 = para[1:8]))

  if (kbar == 1) {
    grad1 <- grad1[, -6]   # drop b column
    H1    <- H1[-6, -6]    # drop b row/col
  }

  N  <- nrow(ret)
  J1 <- t(grad1) %*% grad1
  J2 <- t(grad2) %*% grad2

  se1 <- sqrt(diag(solve(H1/N) %*% (J1/N^2) %*% solve(H1/N)))
  se2 <- sqrt(diag(solve(H2/N) %*% (J2/N^2) %*% solve(H2/N)))

  if (kbar == 1)
    se1 <- c(se1[1:5], NA, se1[6:7])  # re-insert NA for b at position 6

  se <- matrix(c(se1, se2), ncol = 1)
  return(se)
}
```

- [ ] **Step 5.4: Run test to verify it passes**

```r
devtools::load_all(".")
devtools::test(filter = "Bmsm_scale")
```
Expected: all tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add R/Bmsm_scale_std_err.R tests/testthat/test-Bmsm_scale.R
git commit -m "feat: add Bmsm_scale_std_err with sandwich SE"
```

---

## Task 6: Top-Level Function and Integration Tests

**Files:**
- Create: `R/Bmsm_scale.R`
- Modify: `tests/testthat/test-Bmsm_scale.R`

- [ ] **Step 6.1: Add integration tests**

Append to `tests/testthat/test-Bmsm_scale.R`:
```r
# --- Integration: Bmsm_scale estimation ---

test_that("Bmsm_scale returns bmsmmodel with 11 coefficients", {
  fit <- Bmsm_scale(ret_small, kbar = 1, s.err = FALSE)
  expect_s3_class(fit, "bmsmmodel")
  expect_equal(nrow(fit$coefficients), 11)
  expect_true(is.finite(fit$LL))
  expect_equal(
    rownames(fit$coefficients),
    c("m01","m02","sigma1","sigma2","gammak","b","lev1","lev2","rhoe","lambda","rhom")
  )
})

test_that("Bmsm_scale lev=0 LL close to Bmsm LL on same data", {
  fit_bmsm  <- Bmsm(ret_small,       kbar = 1, s.err = FALSE)
  fit_scale <- Bmsm_scale(ret_small, kbar = 1, s.err = FALSE)
  # Bmsm_scale with free lev should fit at least as well
  expect_lte(fit_scale$LL, fit_bmsm$LL + 1e-3)
})

test_that("Bmsm_scale s.err=TRUE returns finite SE except b (kbar=1)", {
  fit <- Bmsm_scale(ret_small, kbar = 1, s.err = TRUE)
  expect_true(is.na(fit$se[6, 1]))
  expect_true(all(is.finite(fit$se[-6, 1])))
})

test_that("Bmsm_scale summary and predict work via S3 methods", {
  fit <- Bmsm_scale(ret_small, kbar = 1, s.err = FALSE)
  s   <- summary(fit)
  expect_s3_class(s, "summary.bmsmmodel")
  pred <- predict(fit)
  expect_true(!is.null(pred$vol1))
  expect_true(!is.null(pred$vol2))
})
```

- [ ] **Step 6.2: Run tests to verify they fail**

```r
devtools::test(filter = "Bmsm_scale")
```
Expected: 4 new FAILs — `"could not find function "Bmsm_scale""`

- [ ] **Step 6.3: Implement top-level function**

Create `R/Bmsm_scale.R`:
```r
#' Bivariate MSM with Sigma-Scaling Leverage Effect.
#'
#' Estimates BMSM(k) with per-series sigma-scaling leverage:
#' sigma_eff_i(t) = sigma_i * g_m * (1 + lev_i * I(r_{i,t-1} < 0)).
#' Negative own-series lagged returns scale next-period conditional volatility up.
#'
#' @param ret N x 2 return matrix.
#' @param kbar number of frequency components. Default 1.
#' @param n trading days per year. Default 252.
#' @param para0 optional starting values length 11:
#'   c(m01,m02,sigma1,sigma2,gammak,b,lev1,lev2,rhoe,lambda,rhom). Default NULL.
#' @param s.err logical; estimate standard errors? Default TRUE.
#'
#' @return a \code{bmsmmodel} object. Coefficients:
#'   m01, m02, sigma1, sigma2, gammak, b, lev1, lev2, rhoe, lambda, rhom.
#'
#' @export
Bmsm_scale <- function(ret, kbar = 1, n = 252, para0 = NULL, s.err = TRUE) {
  bmsm.check <- Bmsm_parameter_check(ret, kbar, NULL, n)
  ret  <- bmsm.check$dat
  kbar <- bmsm.check$kbar

  if (!is.null(para0)) {
    if (length(para0) != 11)
      stop("para0 must be length 11: c(m01,m02,sigma1,sigma2,gammak,b,lev1,lev2,rhoe,lambda,rhom)")
    x0    <- para0
    x0[3] <- x0[3] * sqrt(n)
    x0[4] <- x0[4] * sqrt(n)
  } else {
    base  <- bmsm.check$start.value  # [m01,m02,sigma1,sigma2,gammak,b,rhoe,lambda,rhom]
    x0    <- c(base[1:6], 0, 0, base[7:9])
    x0[3] <- x0[3] * sqrt(n)
    x0[4] <- x0[4] * sqrt(n)
  }

  lb1 <- c(1,      1,      0.00001, 0.00001, 0.001,   1, -0.99, -0.99)
  ub1 <- c(1.9999, 1.9999, 100,     100,     0.9999, 100,  10,   10)
  lb2 <- c(-0.9999, 0.0001, -0.9999)
  ub2 <- c( 0.9999, 0.9999,  0.9999)

  para1 <- x0[1:8]
  para2 <- x0[9:11]

  stage1.fit <- nlminb(para1, Bmsm_scale_stage1_likelihood,
                        lower = lb1, upper = ub1,
                        dat = ret, kbar = kbar, n.vol = n,
                        control = list(eval.max = 500, iter.max = 300))

  para1 <- stage1.fit$par

  stage2.fit <- nlminb(para2, Bmsm_scale_stage2_likelihood,
                        lower = lb2, upper = ub2,
                        kbar = kbar, dat = ret, para1 = para1, n.vol = n,
                        control = list(eval.max = 500, iter.max = 300))

  para          <- matrix(c(para1, stage2.fit$par), ncol = 1)
  bmsm.estimate <- Bmsm_scale_filtered2(c(para1, stage2.fit$par), kbar, ret, n)

  se <- matrix(NA, 11, 1)
  if (isTRUE(s.err)) se <- Bmsm_scale_std_err(c(para1, stage2.fit$par), kbar, ret, n)

  if (kbar == 1) para[6] <- NA
  coef      <- para
  coef[3:4] <- coef[3:4] / sqrt(n)
  se[3:4]   <- se[3:4] / sqrt(n)

  rownames(coef) <- c("m01","m02","sigma1","sigma2","gammak","b","lev1","lev2",
                       "rhoe","lambda","rhom")
  colnames(coef) <- "Estimate"
  colnames(se)   <- "Std. Error"

  bmsm.estimate$optim.msg <- c(stage1.message = stage1.fit$message,
                                stage2.message = stage2.fit$message)
  bmsm.estimate$optim.convergence <- c(stage1.convergence = stage1.fit$convergence,
                                        stage2.convergence = stage2.fit$convergence)
  bmsm.estimate$optim.iter <- c(stage1.iteration = stage1.fit$iterations,
                                  stage2.iteration = stage2.fit$iterations)
  bmsm.estimate$coefficients <- coef
  bmsm.estimate$call         <- match.call()
  bmsm.estimate$ret          <- ret
  bmsm.estimate$LL1          <- stage1.fit$objective
  bmsm.estimate$se           <- se

  class(bmsm.estimate) <- "bmsmmodel"
  bmsm.estimate
}
```

- [ ] **Step 6.4: Regenerate NAMESPACE and load**

```r
devtools::document()   # processes @export tags, updates NAMESPACE
devtools::load_all(".")
```
Expected: `Bmsm_scale`, `Bmsm_scale_filtered2`, `Bmsm_scale_stage1_likelihood`, `Bmsm_scale_stage2_likelihood`, `Bmsm_scale_std_err` appear in NAMESPACE exports.

- [ ] **Step 6.5: Run all tests**

```r
devtools::test(filter = "Bmsm_scale")
```
Expected: all tests PASS. If SE test is slow, it is expected (hessian is O(N·p²) evaluations).

- [ ] **Step 6.6: Run full test suite to confirm no regressions**

```r
devtools::test()
```
Expected: all pre-existing tests still PASS.

- [ ] **Step 6.7: Commit**

```bash
git add R/Bmsm_scale.R NAMESPACE tests/testthat/test-Bmsm_scale.R
git commit -m "feat: add Bmsm_scale top-level function with integration tests"
```

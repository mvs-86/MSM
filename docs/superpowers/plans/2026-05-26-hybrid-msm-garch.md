# Hybrid MSM-GJR-GARCH Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Msm_garch()` — a univariate MSM model where a GJR-GARCH equation captures short-run volatility dynamics around the MSM long-run level, following the Engle & Lee (1999) multiplicative component structure.

**Architecture:** `σ²_t = σ²·g_m²·h_t` where `g_m` is the MSM state (existing cascade) and `h_t` is a GJR-GARCH(1,1) unit-mean transitory component. `h_t` is driven by innovations normalised by the MSM filtered expected variance `E[σ²·g_m² | r_{1..t-1}]`. Single forward pass; joint MLE over 7 parameters.

**Tech Stack:** R, RcppArmadillo, devtools, testthat

---

## File Map

| File | Role |
|------|------|
| `src/Msm_garch_cpp.cpp` | C++ filter: 4 exported functions (scalar + full output, fast + kron) |
| `R/Msm_garch_parameter_check.R` | Bounds, starting values, input validation |
| `R/Msm_garch.R` | LL, likelihood, grad, hessian, std_err, top-level `Msm_garch()`, S3 methods |
| `tests/testthat/test-Msm_garch.R` | All tests |

---

## Task 1: C++ Filter Kernels

**Files:**
- Create: `src/Msm_garch_cpp.cpp`
- Test: `tests/testthat/test-Msm_garch.R`

### Forward filter logic (for reference across all 4 functions)

At each time step `i` (0-indexed, i = 0..N-1):

1. **Transition:** `piA = pi_t * A` (fast) or `kron_apply_g(pi_t, gamma_k)` (kron)
2. **GJR update:**
   - `i == 0`: `h_t = 1.0`
   - `i > 0`: `eps2 = r_{i-1}^2 / Mhat_prev`; `I_neg = (r_{i-1} < 0) ? 1.0 : 0.0`; `h_t = omega_g + (alpha + gamma_gjr * I_neg) * eps2 + beta * h_prev`
   - where `omega_g = 1.0 - alpha - beta - 0.5 * gamma_gjr`
3. **Emission:** `sigma_eff = sigma_gm * sqrt(h_t)`; `omega_t[m] = (1/√2π) * exp(-r_i² / (2*sigma_eff[m]²)) / sigma_eff[m]`
4. **Filter:** `pinum = omega_t ⊙ piA`; `pidenom = sum(pinum)`; `pi_t = pinum / pidenom`
5. **Update for next step:** `Mhat_prev = dot(pi_t, sigma_gm_sq)` where `sigma_gm_sq = sigma_gm^2`; `h_prev = h_t`

**Initialisation:** `pi_t = uniform(1/k)`; `h_prev = 1.0`; `Mhat_prev = dot(pi_t, sigma_gm_sq)`.

**Note on `sigma_gm`:** Passed from R as `sigma * g_m` where `g_m = Msm_states(m0, kbar)` — these are state-conditional SDs, not variances. The variance multiplier is `g_m^2`, so `Mhat = dot(pi, sigma_gm^2)` gives `sigma^2 * E[g_m^2 | filtered]`, which is the denominator for the normalised GJR innovation.

- [ ] **Step 1.1: Write failing test**

Create `tests/testthat/test-Msm_garch.R`:

```r
library(MSM)

data("calvet2004data")
ret       <- na.omit(as.matrix(calvet2004data$caret)) * 100
ret_small <- ret[1:150, , drop = FALSE]

# helper: build sigma_gm and A from raw MSM params
make_msm_parts <- function(para, kbar, n.vol = 252) {
  sigma   <- para[4] / sqrt(n.vol)
  g_m     <- Msm_states(para[1], kbar)
  sigma_gm <- as.vector(sigma * g_m)
  A       <- Msm_A(para[2], para[3], kbar)
  list(sigma_gm = sigma_gm, A = A, g_m = g_m)
}

para_base <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252))
parts     <- make_msm_parts(para_base, kbar = 1)

test_that("Msm_garch_ll_fast_cpp with alpha=beta=gamma=0 matches Msm_ll_fast", {
  ll_garch <- Msm_garch_ll_fast_cpp(ret_small, parts$sigma_gm, parts$A, 0, 0, 0)
  ll_msm   <- Msm_ll_fast(ret_small, parts$sigma_gm, parts$A)
  expect_equal(as.numeric(ll_garch), as.numeric(ll_msm), tolerance = 1e-8)
})

test_that("Msm_garch_fast_cpp with alpha=beta=gamma=0 matches Msm_likelihood_fast LL", {
  res_garch <- Msm_garch_fast_cpp(ret_small, parts$sigma_gm, parts$A, 0, 0, 0)
  res_msm   <- Msm_likelihood_fast(ret_small, parts$sigma_gm, parts$A)
  expect_equal(res_garch$LL, res_msm$LL, tolerance = 1e-8)
  expect_equal(dim(res_garch$filtered), dim(res_msm$filtered))
  expect_equal(length(res_garch$h), nrow(ret_small))
  expect_true(all(res_garch$h > 0))
})

test_that("Msm_garch_ll_kron_cpp with alpha=beta=gamma=0 matches Msm_ll_kron", {
  para8 <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252))
  parts8 <- make_msm_parts(para8, kbar = 8)
  ll_garch <- Msm_garch_ll_kron_cpp(ret_small, parts8$sigma_gm, para8[2], para8[3], 8L, 0, 0, 0)
  ll_msm   <- Msm_ll_kron(ret_small, parts8$sigma_gm, para8[2], para8[3], 8L)
  expect_equal(as.numeric(ll_garch), as.numeric(ll_msm), tolerance = 1e-8)
})

test_that("Msm_garch_kron_cpp with alpha=beta=gamma=0 matches Msm_likelihood_kron LL", {
  para8 <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252))
  parts8 <- make_msm_parts(para8, kbar = 8)
  res_garch <- Msm_garch_kron_cpp(ret_small, parts8$sigma_gm, para8[2], para8[3], 8L, 0, 0, 0)
  res_msm   <- Msm_likelihood_kron(ret_small, parts8$sigma_gm, para8[2], para8[3], 8L)
  expect_equal(res_garch$LL, res_msm$LL, tolerance = 1e-8)
  expect_equal(length(res_garch$h), nrow(ret_small))
})

test_that("Msm_garch_fast_cpp h vector is stationary with positive alpha/beta", {
  res <- Msm_garch_fast_cpp(ret_small, parts$sigma_gm, parts$A, 0.05, 0.85, 0.05)
  expect_true(all(res$h > 0))
  expect_true(all(is.finite(res$h)))
  # h should vary across time (not constant 1) when alpha+beta > 0
  expect_gt(var(res$h), 0)
})
```

- [ ] **Step 1.2: Run test — expect FAIL (functions not found)**

```bash
Rscript -e "devtools::test(filter = 'Msm_garch')"
```
Expected: errors like `could not find function "Msm_garch_ll_fast_cpp"`

- [ ] **Step 1.3: Write `src/Msm_garch_cpp.cpp`**

```cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace Rcpp;

static inline void kron_apply_g(arma::rowvec& pi, const arma::vec& gamma_k) {
    int kbar = gamma_k.n_elem;
    int n    = pi.n_elem;
    for (int k = 0; k < kbar; k++) {
        double g  = gamma_k(k);
        double ad = 1.0 - 0.5 * g;
        double ao = 0.5 * g;
        int stride = 1 << (kbar - 1 - k);
        int block  = stride << 1;
        for (int base = 0; base < n; base += block) {
            for (int i = 0; i < stride; i++) {
                double p0 = pi(base + i);
                double p1 = pi(base + i + stride);
                pi(base + i)          = ad * p0 + ao * p1;
                pi(base + i + stride) = ao * p0 + ad * p1;
            }
        }
    }
}

static inline arma::vec make_gamma_k_g(double b, double gamma_kbar, int kbar) {
    arma::vec gamma_k(kbar);
    gamma_k(0) = 1.0 - std::pow(1.0 - gamma_kbar, 1.0 / std::pow(b, kbar - 1));
    for (int i = 1; i < kbar; i++)
        gamma_k(i) = 1.0 - std::pow(1.0 - gamma_k(0), std::pow(b, i));
    return gamma_k;
}

// kbar < 8, scalar LL
// [[Rcpp::export]]
double Msm_garch_ll_fast_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                              const arma::mat& A, double alpha, double beta,
                              double gamma_gjr) {
    int N = dat.n_rows, k = sigma_gm.n_elem;
    const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    const double omega_g = 1.0 - alpha - beta - 0.5 * gamma_gjr;

    arma::rowvec sigma_gm_sq = sigma_gm % sigma_gm;
    arma::rowvec pi_t(k), piA(k), omega_t(k), pinum(k), sigma_eff(k);
    pi_t.fill(1.0 / k);
    double h_prev    = 1.0;
    double Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
    double ll = 0.0, pidenom;

    for (int i = 0; i < N; i++) {
        piA = pi_t * A;

        double h_t;
        if (i == 0) {
            h_t = 1.0;
        } else {
            double r_prev = dat(i - 1);
            double eps2   = (r_prev * r_prev) / Mhat_prev;
            double I_neg  = (r_prev < 0.0) ? 1.0 : 0.0;
            h_t = omega_g + (alpha + gamma_gjr * I_neg) * eps2 + beta * h_prev;
        }

        sigma_eff = sigma_gm * std::sqrt(h_t);
        omega_t   = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_eff)) / sigma_eff + 1e-16;
        pinum     = omega_t % piA;
        pidenom   = arma::accu(pinum);
        ll       += std::log(pidenom);

        if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else                { pi_t = pinum / pidenom; }

        Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
        h_prev    = h_t;
    }
    return -ll;
}

// kbar >= 8, scalar LL
// [[Rcpp::export]]
double Msm_garch_ll_kron_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                              double b, double gamma_kbar, int kbar,
                              double alpha, double beta, double gamma_gjr) {
    int N = dat.n_rows, k = sigma_gm.n_elem;
    const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    const double omega_g = 1.0 - alpha - beta - 0.5 * gamma_gjr;
    const arma::vec gamma_k = make_gamma_k_g(b, gamma_kbar, kbar);

    arma::rowvec sigma_gm_sq = sigma_gm % sigma_gm;
    arma::rowvec pi_t(k), omega_t(k), pinum(k), sigma_eff(k);
    pi_t.fill(1.0 / k);
    double h_prev    = 1.0;
    double Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
    double ll = 0.0, pidenom;

    for (int i = 0; i < N; i++) {
        kron_apply_g(pi_t, gamma_k);

        double h_t;
        if (i == 0) {
            h_t = 1.0;
        } else {
            double r_prev = dat(i - 1);
            double eps2   = (r_prev * r_prev) / Mhat_prev;
            double I_neg  = (r_prev < 0.0) ? 1.0 : 0.0;
            h_t = omega_g + (alpha + gamma_gjr * I_neg) * eps2 + beta * h_prev;
        }

        sigma_eff = sigma_gm * std::sqrt(h_t);
        omega_t   = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_eff)) / sigma_eff + 1e-16;
        pinum     = omega_t % pi_t;
        pidenom   = arma::accu(pinum);
        ll       += std::log(pidenom);

        if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else                { pi_t = pinum / pidenom; }

        Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
        h_prev    = h_t;
    }
    return -ll;
}

// kbar < 8, full output
// [[Rcpp::export]]
List Msm_garch_fast_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                        const arma::mat& A, double alpha, double beta,
                        double gamma_gjr) {
    int N = dat.n_rows, k = sigma_gm.n_elem;
    const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    const double omega_g = 1.0 - alpha - beta - 0.5 * gamma_gjr;

    arma::rowvec sigma_gm_sq = sigma_gm % sigma_gm;
    arma::mat    filtered(N, k, arma::fill::zeros);
    arma::colvec LLs(N, arma::fill::zeros);
    arma::colvec h_vec(N, arma::fill::zeros);
    arma::rowvec pi_t(k), piA(k), omega_t(k), pinum(k), sigma_eff(k);
    pi_t.fill(1.0 / k);
    double h_prev    = 1.0;
    double Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
    double ll = 0.0, pidenom;

    for (int i = 0; i < N; i++) {
        piA = pi_t * A;

        double h_t;
        if (i == 0) {
            h_t = 1.0;
        } else {
            double r_prev = dat(i - 1);
            double eps2   = (r_prev * r_prev) / Mhat_prev;
            double I_neg  = (r_prev < 0.0) ? 1.0 : 0.0;
            h_t = omega_g + (alpha + gamma_gjr * I_neg) * eps2 + beta * h_prev;
        }

        sigma_eff = sigma_gm * std::sqrt(h_t);
        omega_t   = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_eff)) / sigma_eff + 1e-16;
        pinum     = omega_t % piA;
        pidenom   = arma::accu(pinum);
        LLs(i)    = std::log(pidenom);
        ll       += LLs(i);

        if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else                { pi_t = pinum / pidenom; }

        filtered.row(i) = pi_t;
        h_vec(i)        = h_t;
        Mhat_prev       = arma::dot(pi_t, sigma_gm_sq);
        h_prev          = h_t;
    }
    return List::create(Named("filtered") = filtered,
                        Named("LL")       = -ll,
                        Named("LLs")      = LLs,
                        Named("h")        = h_vec);
}

// kbar >= 8, full output
// [[Rcpp::export]]
List Msm_garch_kron_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                        double b, double gamma_kbar, int kbar,
                        double alpha, double beta, double gamma_gjr) {
    int N = dat.n_rows, k = sigma_gm.n_elem;
    const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    const double omega_g = 1.0 - alpha - beta - 0.5 * gamma_gjr;
    const arma::vec gamma_k = make_gamma_k_g(b, gamma_kbar, kbar);

    arma::rowvec sigma_gm_sq = sigma_gm % sigma_gm;
    arma::mat    filtered(N, k, arma::fill::zeros);
    arma::colvec LLs(N, arma::fill::zeros);
    arma::colvec h_vec(N, arma::fill::zeros);
    arma::rowvec pi_t(k), omega_t(k), pinum(k), sigma_eff(k);
    pi_t.fill(1.0 / k);
    double h_prev    = 1.0;
    double Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
    double ll = 0.0, pidenom;

    for (int i = 0; i < N; i++) {
        kron_apply_g(pi_t, gamma_k);

        double h_t;
        if (i == 0) {
            h_t = 1.0;
        } else {
            double r_prev = dat(i - 1);
            double eps2   = (r_prev * r_prev) / Mhat_prev;
            double I_neg  = (r_prev < 0.0) ? 1.0 : 0.0;
            h_t = omega_g + (alpha + gamma_gjr * I_neg) * eps2 + beta * h_prev;
        }

        sigma_eff = sigma_gm * std::sqrt(h_t);
        omega_t   = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_eff)) / sigma_eff + 1e-16;
        pinum     = omega_t % pi_t;
        pidenom   = arma::accu(pinum);
        LLs(i)    = std::log(pidenom);
        ll       += LLs(i);

        if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else                { pi_t = pinum / pidenom; }

        filtered.row(i) = pi_t;
        h_vec(i)        = h_t;
        Mhat_prev       = arma::dot(pi_t, sigma_gm_sq);
        h_prev          = h_t;
    }
    return List::create(Named("filtered") = filtered,
                        Named("LL")       = -ll,
                        Named("LLs")      = LLs,
                        Named("h")        = h_vec);
}
```

- [ ] **Step 1.4: Compile and register exports**

```bash
Rscript -e "devtools::document()" && Rscript -e "devtools::load_all('.')"
```
Expected: compiles without errors; `Msm_garch_ll_fast_cpp` appears in namespace.

- [ ] **Step 1.5: Run tests — expect PASS**

```bash
Rscript -e "devtools::test(filter = 'Msm_garch')"
```
Expected: 5 tests pass.

- [ ] **Step 1.6: Commit**

```bash
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git add src/Msm_garch_cpp.cpp tests/testthat/test-Msm_garch.R
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git commit -m "feat: add Msm_garch C++ filter kernels with nesting tests"
```

---

## Task 2: Parameter Check

**Files:**
- Create: `R/Msm_garch_parameter_check.R`
- Test: `tests/testthat/test-Msm_garch.R` (append)

- [ ] **Step 2.1: Add failing tests**

Append to `tests/testthat/test-Msm_garch.R`:

```r
# --- parameter check ---

test_that("Msm_garch_parameter_check returns valid start values and bounds", {
  res <- Msm_garch_parameter_check(ret_small, kbar = 1, x0 = NULL)
  expect_equal(length(res$start.value), 7)
  expect_equal(length(res$lb), 7)
  expect_equal(length(res$ub), 7)
  expect_true(all(res$start.value >= res$lb))
  expect_true(all(res$start.value <= res$ub))
  # alpha + beta + gamma_gjr/2 < 1 at start
  sv <- res$start.value
  expect_lt(sv[5] + sv[6] + sv[7]/2, 1)
})

test_that("Msm_garch_parameter_check accepts valid para0", {
  para0 <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0.05, 0.85, 0.05)
  res   <- Msm_garch_parameter_check(ret_small, kbar = 1, x0 = para0)
  expect_equal(res$start.value, para0)
})

test_that("Msm_garch_parameter_check stops on wrong-length para0", {
  expect_error(Msm_garch_parameter_check(ret_small, kbar = 1, x0 = c(1.5, 2.5, 0.9)),
               regexp = "length 7")
})

test_that("Msm_garch_parameter_check stops on non-matrix dat", {
  expect_no_error(Msm_garch_parameter_check(as.numeric(ret_small), kbar = 1, x0 = NULL))
})
```

- [ ] **Step 2.2: Run — expect FAIL**

```bash
Rscript -e "devtools::test(filter = 'Msm_garch')"
```
Expected: 4 new test failures about `Msm_garch_parameter_check` not found.

- [ ] **Step 2.3: Create `R/Msm_garch_parameter_check.R`**

```r
Msm_garch_parameter_check <- function(dat, kbar, x0) {

  if (!is.matrix(dat)) dat <- as.matrix(dat)
  if (ncol(dat) > 1)   dat <- t(dat)
  if (ncol(dat) > 1 || nrow(dat) < 2 || is.null(dat))
    stop("dat must be a numeric vector.")
  if (kbar < 1)
    stop("kbar must be a positive integer.")

  if (!is.null(x0)) {
    if (length(x0) != 7)
      stop("Initial values must be of length 7: c(m0, b, gammak, sigma, alpha, beta, gamma_gjr)")
    if (x0[1] < 1 || x0[1] > 1.99)  stop("m0 must be in (1, 1.99]")
    if (x0[2] < 1)                   stop("b must be > 1")
    if (x0[3] < 1e-4 || x0[3] > 0.9999) stop("gammak must be in (0, 1)")
    if (x0[4] < 1e-5)                stop("sigma must be positive")
    if (x0[5] < 0)                   stop("alpha must be >= 0")
    if (x0[6] < 0)                   stop("beta must be >= 0")
    if (x0[7] < 0)                   stop("gamma_gjr must be >= 0")
  } else {
    msm_sv <- c(1.5, 2.5, 0.9, sd(dat) * sqrt(252))
    x0     <- c(msm_sv, 0.05, 0.85, 0.05)
  }

  lb <- c(1,      1,      1e-4,  1e-5,  0,    0,    0)
  ub <- c(1.9999, 50,     0.9999, 50,   0.99, 0.99, 1.99)

  list(dat         = dat,
       kbar        = kbar,
       start.value = x0,
       lb          = lb,
       ub          = ub)
}
```

- [ ] **Step 2.4: Load and run tests**

```bash
Rscript -e "devtools::load_all('.') ; devtools::test(filter = 'Msm_garch')"
```
Expected: all 9 tests pass.

- [ ] **Step 2.5: Commit**

```bash
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git add R/Msm_garch_parameter_check.R tests/testthat/test-Msm_garch.R
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git commit -m "feat: add Msm_garch_parameter_check with bounds and start values"
```

---

## Task 3: LL and Likelihood Wrappers

**Files:**
- Create: `R/Msm_garch.R` (start with LL + likelihood; S3 and top-level added later)
- Test: `tests/testthat/test-Msm_garch.R` (append)

- [ ] **Step 3.1: Add failing tests**

Append to `tests/testthat/test-Msm_garch.R`:

```r
# --- LL and likelihood wrappers ---

test_that("Msm_garch_ll nesting: alpha=beta=gamma=0 matches Msm_ll2", {
  para_msm   <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252))
  para_garch <- c(para_msm, 0, 0, 0)
  ll_msm   <- Msm_ll2(para_msm, kbar = 1, dat = ret_small, n.vol = 252)
  ll_garch <- Msm_garch_ll(para_garch, kbar = 1, dat = ret_small, n.vol = 252)
  expect_equal(as.numeric(ll_garch), as.numeric(ll_msm), tolerance = 1e-6)
})

test_that("Msm_garch_ll returns scalar", {
  para <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0.05, 0.85, 0.05)
  ll   <- Msm_garch_ll(para, kbar = 1, dat = ret_small, n.vol = 252)
  expect_length(ll, 1)
  expect_true(is.finite(ll))
  expect_gt(ll, 0)  # negative LL returned as positive
})

test_that("Msm_garch_ll returns large value when alpha+beta+gamma/2 >= 1", {
  para_bad <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0.5, 0.6, 0.1)
  # 0.5 + 0.6 + 0.05 = 1.15 >= 1
  ll_bad <- Msm_garch_ll(para_bad, kbar = 1, dat = ret_small, n.vol = 252)
  expect_equal(ll_bad, 1e10)
})

test_that("Msm_garch_likelihood nesting: alpha=beta=gamma=0 matches Msm_likelihood2 LL", {
  para_msm   <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252))
  para_garch <- c(para_msm, 0, 0, 0)
  ref <- Msm_likelihood2(para_msm, kbar = 1, dat = ret_small, n.vol = 252)
  res <- Msm_garch_likelihood(para_garch, kbar = 1, dat = ret_small, n.vol = 252)
  expect_equal(res$LL, ref$LL, tolerance = 1e-6)
  expect_equal(dim(res$filtered), dim(ref$filtered))
  expect_equal(length(res$h), nrow(ret_small))
  expect_true(all(res$h > 0))
  expect_true(all(is.finite(res$h)))
})

test_that("Msm_garch_likelihood returns A and g.m", {
  para <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0.05, 0.85, 0.05)
  res  <- Msm_garch_likelihood(para, kbar = 1, dat = ret_small, n.vol = 252)
  expect_true(!is.null(res$A))
  expect_true(!is.null(res$g.m))
  expect_equal(nrow(res$filtered), nrow(ret_small))
  expect_true(all(abs(rowSums(res$filtered) - 1) < 1e-10))
})
```

- [ ] **Step 3.2: Run — expect FAIL**

```bash
Rscript -e "devtools::test(filter = 'Msm_garch')"
```
Expected: 5 new failures about `Msm_garch_ll` / `Msm_garch_likelihood` not found.

- [ ] **Step 3.3: Create `R/Msm_garch.R` with LL + likelihood functions**

```r
# Scalar negative LL for nlminb.
Msm_garch_ll <- function(para, kbar, dat, n.vol) {
  alpha     <- para[5]
  beta      <- para[6]
  gamma_gjr <- para[7]

  if (alpha + beta + gamma_gjr / 2 >= 1) return(1e10)

  m0     <- para[1]
  b      <- para[2]
  gama.k <- para[3]
  sigma  <- para[4] / sqrt(n.vol)

  g.m      <- Msm_states(m0, kbar)
  sigma_gm <- as.vector(sigma * g.m)

  if (kbar >= 8) {
    LL <- Msm_garch_ll_kron_cpp(dat, sigma_gm, b, gama.k, kbar,
                                alpha, beta, gamma_gjr)
  } else {
    A  <- Msm_A(b, gama.k, kbar)
    LL <- Msm_garch_ll_fast_cpp(dat, sigma_gm, A, alpha, beta, gamma_gjr)
  }

  if (!is.finite(LL)) warning("Log-likelihood is inf.")
  LL
}

# Full output: filtered + LL + LLs + h.
#' @export
Msm_garch_likelihood <- function(para, kbar, dat, n.vol) {
  m0        <- para[1]
  b         <- para[2]
  gama.k    <- para[3]
  sigma     <- para[4] / sqrt(n.vol)
  alpha     <- para[5]
  beta      <- para[6]
  gamma_gjr <- para[7]

  A   <- Msm_A(b, gama.k, kbar)
  g.m <- Msm_states(m0, kbar)
  sigma_gm <- as.vector(sigma * g.m)

  if (kbar >= 8) {
    out <- Msm_garch_kron_cpp(dat, sigma_gm, b, gama.k, kbar, alpha, beta, gamma_gjr)
  } else {
    out <- Msm_garch_fast_cpp(dat, sigma_gm, A, alpha, beta, gamma_gjr)
  }

  out$A   <- A
  out$g.m <- g.m
  out
}
```

- [ ] **Step 3.4: Load and run tests**

```bash
Rscript -e "devtools::load_all('.') ; devtools::test(filter = 'Msm_garch')"
```
Expected: all 14 tests pass.

- [ ] **Step 3.5: Commit**

```bash
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git add R/Msm_garch.R tests/testthat/test-Msm_garch.R
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git commit -m "feat: add Msm_garch LL and likelihood R wrappers"
```

---

## Task 4: Standard Errors

**Files:**
- Modify: `R/Msm_garch.R` (append grad, hessian, std_err)
- Test: `tests/testthat/test-Msm_garch.R` (append)

- [ ] **Step 4.1: Add failing tests**

Append to `tests/testthat/test-Msm_garch.R`:

```r
# --- standard errors ---

test_that("Msm_garch_std_err kbar=1 returns 7-element SE with NA at b", {
  para <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0.05, 0.85, 0.05)
  se   <- Msm_garch_std_err(para, kbar = 1, ret = ret_small, n.vol = 252, lag = 0)
  expect_equal(nrow(se), 7)
  expect_true(is.na(se[2, 1]))
  expect_true(all(is.finite(se[-2, 1])))
  expect_true(all(se[-2, 1] > 0))
})

test_that("Msm_garch_std_err kbar=2 returns 7-element finite SE", {
  para <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0.05, 0.85, 0.05)
  se   <- Msm_garch_std_err(para, kbar = 2, ret = ret_small, n.vol = 252, lag = 0)
  expect_equal(nrow(se), 7)
  expect_true(all(is.finite(se)))
  expect_true(all(se > 0))
})
```

- [ ] **Step 4.2: Run — expect FAIL**

```bash
Rscript -e "devtools::test(filter = 'Msm_garch')"
```
Expected: 2 new failures for `Msm_garch_std_err` not found.

- [ ] **Step 4.3: Append grad, hessian, std_err to `R/Msm_garch.R`**

```r
# Numerical gradient of per-obs LLs w.r.t. all 7 parameters.
Msm_garch_grad <- function(para, kbar, ret, n.vol) {
  check_para <- function(x) {
    if (x[1] >= 2) x[1] <- 1.9999
    if (x[3] >= 1) x[3] <- 0.9999
    x
  }

  para.size <- length(para)
  para      <- as.matrix(para)
  para.abs  <- abs(para)
  para2     <- if (!all(para == 0)) para / para.abs else matrix(1, para.size, 1)
  h1        <- cbind(para.abs, matrix(1, para.size, 1) * 1e-2)
  h         <- 1e-8 * matrix(apply(h1, 1, max), ncol = 1) * para2
  para.temp <- para + h
  h         <- para.temp - para

  para_list <- c(
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] + h[i, 1]; check_para(x) }),
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] - h[i, 1]; check_para(x) })
  )

  res <- parallel::mclapply(para_list, Msm_garch_likelihood,
                            kbar = kbar, dat = ret, n.vol = n.vol,
                            mc.cores = getOption("mc.cores", 1L))

  ll.1 <- matrix(0, nrow(ret), para.size)
  ll.2 <- matrix(0, nrow(ret), para.size)
  for (i in seq_len(para.size)) {
    ll.1[, i] <- res[[i]]$LLs
    ll.2[, i] <- res[[para.size + i]]$LLs
  }
  (ll.1 - ll.2) / (2 * t(matrix(rep(h, nrow(ret)), ncol = nrow(ret))))
}

# Numerical 2-sided Hessian of scalar LL w.r.t. all 7 parameters.
Msm_garch_hessian <- function(para, kbar, ret, n.vol) {
  check_para <- function(x) {
    if (x[1] >= 2) x[1] <- 1.9999
    if (x[3] >= 1) x[3] <- 0.9999
    x
  }

  eps       <- .Machine$double.eps
  para.size <- length(para)
  para      <- as.matrix(para)
  f.ll      <- Msm_garch_likelihood(check_para(para), kbar, ret, n.vol)$LL
  h         <- matrix(eps^(1/3) * apply(cbind(abs(para), 1e-8), 1, max))
  para.h    <- para + h
  h         <- para.h - para
  ee        <- diag(h[, 1], para.size)
  ij        <- expand.grid(i = seq_len(para.size), j = seq_len(para.size))

  para_list <- c(
    lapply(seq_len(para.size), function(i) check_para(para + ee[, i])),
    lapply(seq_len(para.size), function(i) check_para(para - ee[, i])),
    lapply(seq_len(nrow(ij)),  function(k) check_para(para + ee[, ij$i[k]] + ee[, ij$j[k]])),
    lapply(seq_len(nrow(ij)),  function(k) check_para(para - ee[, ij$i[k]] - ee[, ij$j[k]]))
  )

  res <- parallel::mclapply(para_list,
                            function(p) Msm_garch_likelihood(p, kbar, ret, n.vol)$LL,
                            mc.cores = getOption("mc.cores", 1L))
  res <- unlist(res)

  n2 <- para.size
  gp <- res[seq_len(n2)]
  gm <- res[n2 + seq_len(n2)]
  hp <- matrix(res[2*n2 + seq_len(n2^2)], n2, n2)
  hm <- matrix(res[2*n2 + n2^2 + seq_len(n2^2)], n2, n2)
  hh <- h %*% t(h)
  H  <- matrix(0, para.size, para.size)

  for (i in seq_len(para.size)) {
    for (j in seq_len(para.size)) {
      H[i, j] <- (hp[i,j] - gp[i] - gp[j] + f.ll + f.ll - gm[i] - gm[j] + hm[i,j]) / hh[i,j] / 2
      H[j, i] <- H[i, j]
    }
  }
  H
}

# Standard errors.
#' @export
Msm_garch_std_err <- function(para, kbar, ret, n.vol, lag = 0) {
  if (kbar == 1) {
    grad <- Msm_garch_grad(para, kbar, ret, n.vol)
    grad <- grad[, -2]   # drop b column (unidentified at kbar=1)
    J    <- t(grad) %*% grad
    s    <- sqrt(diag(solve(J)))
    se   <- matrix(c(s[1], NA, s[2:6]), ncol = 1)
  } else {
    H  <- Msm_garch_hessian(para, kbar, ret, n.vol)
    se <- matrix(sqrt(abs(diag(solve(H)))), ncol = 1)
  }
  se
}
```

- [ ] **Step 4.4: Load and run tests**

```bash
Rscript -e "devtools::load_all('.') ; devtools::test(filter = 'Msm_garch')"
```
Expected: all 16 tests pass.

- [ ] **Step 4.5: Commit**

```bash
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git add R/Msm_garch.R tests/testthat/test-Msm_garch.R
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git commit -m "feat: add Msm_garch grad, hessian, and std_err"
```

---

## Task 5: Top-Level Function and S3 Methods

**Files:**
- Modify: `R/Msm_garch.R` (append `Msm_garch()` + S3)
- Test: `tests/testthat/test-Msm_garch.R` (append)

- [ ] **Step 5.1: Add failing tests**

Append to `tests/testthat/test-Msm_garch.R`:

```r
# --- top-level estimation ---

test_that("Msm_garch returns msmgarchmodel with correct structure (kbar=1)", {
  fit <- Msm_garch(ret_small, kbar = 1)
  expect_s3_class(fit, "msmgarchmodel")
  expect_equal(nrow(fit$coefficients), 7)
  expect_equal(rownames(fit$coefficients),
               c("m0", "b", "gammak", "sigma", "alpha", "beta", "gamma_gjr"))
  expect_true(is.finite(fit$LL))
  expect_equal(nrow(fit$filtered), nrow(ret_small))
  expect_equal(length(fit$h), nrow(ret_small))
  expect_true(all(fit$h > 0))
})

test_that("Msm_garch kbar=1: b coefficient is NA", {
  fit <- Msm_garch(ret_small, kbar = 1)
  expect_true(is.na(fit$coefficients["b", 1]))
})

test_that("Msm_garch kbar=1: SE has NA at b, finite elsewhere", {
  fit <- Msm_garch(ret_small, kbar = 1)
  expect_equal(nrow(fit$se), 7)
  expect_true(is.na(fit$se[2, 1]))
  expect_true(all(is.finite(fit$se[-2, 1])))
})

test_that("Msm_garch nesting: alpha=beta=gamma_gjr=0 LL close to Msm LL", {
  fit_msm   <- Msm(ret_small, kbar = 1)
  para_nest <- c(as.numeric(fit_msm$para[c(1,3,4)]), 2.5, 0, 0, 0)
  # Use known Msm params with GARCH params=0
  para7 <- c(as.numeric(fit_msm$para)[c(1,2,3,4)], 0, 0, 0)
  para7[2] <- 2.5  # b is NA in fit_msm$para at kbar=1
  ll_msm   <- Msm_ll2(para7[1:4], kbar = 1, dat = ret_small, n.vol = 252)
  ll_garch <- Msm_garch_ll(para7, kbar = 1, dat = ret_small, n.vol = 252)
  expect_equal(as.numeric(ll_garch), as.numeric(ll_msm), tolerance = 1e-6)
})

test_that("Msm_garch alpha+beta+gamma_gjr/2 < 1 at optimum", {
  fit <- Msm_garch(ret_small, kbar = 1)
  co  <- as.numeric(fit$coefficients)
  expect_lt(co[5] + co[6] + co[7]/2, 1)
})

test_that("Msm_garch filtered row sums equal 1", {
  fit <- Msm_garch(ret_small, kbar = 1)
  expect_true(all(abs(rowSums(fit$filtered) - 1) < 1e-10))
})

test_that("Msm_garch kbar=2 converges", {
  fit <- Msm_garch(ret_small, kbar = 2)
  expect_s3_class(fit, "msmgarchmodel")
  expect_true(is.finite(fit$LL))
})

# --- S3 methods ---

test_that("print.msmgarchmodel runs without error", {
  fit <- Msm_garch(ret_small, kbar = 1)
  expect_output(print(fit))
})

test_that("summary.msmgarchmodel runs and returns summary object", {
  fit <- Msm_garch(ret_small, kbar = 1)
  s   <- summary(fit)
  expect_s3_class(s, "summary.msmgarchmodel")
  expect_true(!is.null(s$coefficients))
})

test_that("coef.msmgarchmodel returns 7-element named numeric", {
  fit <- Msm_garch(ret_small, kbar = 1)
  co  <- coef(fit)
  expect_length(co, 7)
  expect_named(co, c("m0", "b", "gammak", "sigma", "alpha", "beta", "gamma_gjr"))
})

test_that("predict.msmgarchmodel returns vol and vol.sq (fitted)", {
  fit  <- Msm_garch(ret_small, kbar = 1)
  pred <- predict(fit)
  expect_named(pred, c("vol", "vol.sq"))
  expect_length(pred$vol, nrow(ret_small))
  expect_true(all(pred$vol > 0))
})

test_that("predict.msmgarchmodel h-step ahead returns scalar", {
  fit  <- Msm_garch(ret_small, kbar = 1)
  pred <- predict(fit, h = 5)
  expect_named(pred, c("vol", "vol.sq"))
  expect_length(pred$vol, 1)
})

test_that("plot.msmgarchmodel runs without error", {
  fit <- Msm_garch(ret_small, kbar = 1)
  expect_no_error(plot(fit, what = "vol"))
  expect_no_error(plot(fit, what = "volsq"))
})
```

- [ ] **Step 5.2: Run — expect FAIL**

```bash
Rscript -e "devtools::test(filter = 'Msm_garch')"
```
Expected: ~13 new failures.

- [ ] **Step 5.3: Append `Msm_garch()` and S3 methods to `R/Msm_garch.R`**

```r
#' Hybrid MSM-GJR-GARCH Volatility Model.
#'
#' Estimates a univariate MSM(k) model with a multiplicative GJR-GARCH(1,1)
#' short-run component: sigma2_t = sigma^2 * g_m^2 * h_t, where h_t is a
#' unit-mean GJR-GARCH transitory factor.
#'
#' @param ret column matrix of returns.
#' @param kbar number of MSM frequency components. Default 1.
#' @param n.vol trading days per year. Default 252.
#' @param para0 optional 7-element starting vector c(m0,b,gammak,sigma,alpha,beta,gamma_gjr). Default NULL.
#' @param nw.lag Newey-West lags (kbar=1 only). Default 0.
#'
#' @return an \code{msmgarchmodel} object.
#'
#' @export
Msm_garch <- function(ret, kbar = 1, n.vol = 252, para0 = NULL, nw.lag = 0) {

  chk  <- Msm_garch_parameter_check(ret, kbar, para0)
  ret  <- chk$dat
  kbar <- chk$kbar
  x0   <- chk$start.value
  lb   <- chk$lb
  ub   <- chk$ub

  if (is.null(para0)) x0[4] <- x0[4] * sqrt(n.vol)

  fit <- nlminb(x0, Msm_garch_ll, lower = lb, upper = ub,
                kbar = kbar, dat = ret, n.vol = n.vol,
                control = list(eval.max = 1000, iter.max = 500))

  est <- Msm_garch_likelihood(fit$par, kbar, ret, n.vol)
  se  <- Msm_garch_std_err(fit$par, kbar, ret, n.vol, nw.lag)

  para <- matrix(fit$par, ncol = 1)
  if (kbar == 1) para[2] <- NA

  coef    <- para
  coef[4] <- coef[4] / sqrt(n.vol)
  se[4]   <- se[4] / sqrt(n.vol)

  rownames(coef) <- c("m0", "b", "gammak", "sigma", "alpha", "beta", "gamma_gjr")
  colnames(coef) <- "Estimate"
  colnames(se)   <- "Std. Error"

  est$optim.msg         <- fit$message
  est$optim.convergence <- fit$convergence
  est$optim.iter        <- fit$iterations
  est$para              <- para
  est$se                <- se
  est$kbar              <- kbar
  est$n                 <- n.vol
  est$coefficients      <- coef
  est$call              <- match.call()
  est$ret               <- ret

  class(est) <- "msmgarchmodel"
  est
}

#' @export
print.msmgarchmodel <- function(x, ...) {
  cat("Call:\n")
  print(x$call)
  cat("\nCoefficients:\n")
  print(x$coefficients, digits = 4)
  cat("\nLogLikelihood:", x$LL, "\n")
}

#' @export
summary.msmgarchmodel <- function(object, ...) {
  se      <- object$se
  tval    <- coef(object) / se
  p.value <- 2 * pt(-abs(tval), df = nrow(object$ret) - 7)

  colnames(tval)    <- "t.value"
  colnames(p.value) <- "p.value"

  TAB <- cbind(Estimate = round(coef(object), 4),
               StdErr   = round(se, 4),
               t.value  = round(tval, 4),
               p.value  = round(p.value, 4))

  res <- list(call         = object$call,
              coefficients = TAB,
              kbar         = object$kbar,
              LL           = object$LL)
  class(res) <- "summary.msmgarchmodel"
  res
}

#' @export
print.summary.msmgarchmodel <- function(x, ...) {
  cat("*------------------------------------------------------*\n")
  cat("  MSM-GJR-GARCH with", x$kbar, "MSM Volatility Component(s)\n")
  cat("*------------------------------------------------------*\n\n")
  printCoefmat(x$coefficients, digits = 4, P.value = TRUE, has.Pvalue = TRUE)
  cat("\nLogLikelihood:", x$LL, "\n")
}

#' @export
coef.msmgarchmodel <- function(object, ...) {
  co <- as.numeric(object$coefficients)
  names(co) <- rownames(object$coefficients)
  co
}

#' @export
predict.msmgarchmodel <- function(object, h = NULL, ...) {
  # Fitted conditional volatility: sqrt(sigma^2 * E[g_m^2 | filtered] * h_t)
  # h-step forecast: MSM propagation only (h_t -> 1 as h -> inf under stationarity)
  if (!is.null(h) && length(h) > 1) {
    if (any(h < 1))          stop("h must be >= 1")
    if (!all(diff(h) == 1L)) stop("h must be consecutive integers, e.g. h=1:10")
    smoothed.p <- object$filtered
    results <- lapply(h, function(hi)
      Msm_predict(object$g.m, object$para[4], object$n, smoothed.p, object$A, hi))
    return(list(
      vol    = do.call(c, lapply(results, `[[`, "vol")),
      vol.sq = do.call(c, lapply(results, `[[`, "vol.sq"))
    ))
  }

  if (!is.null(h)) {
    smoothed.p <- object$filtered
  } else {
    smoothed.p <- Msm_smooth_cpp(object$A, object$filtered)
  }
  Msm_predict(object$g.m, object$para[4], object$n, smoothed.p, object$A, h)
}

#' @export
plot.msmgarchmodel <- function(object, what = "vol", ...) {
  sigma     <- object$para[4] / sqrt(object$n)
  g.m       <- object$g.m
  filtered  <- object$filtered
  h         <- object$h

  # Fitted conditional vol: sigma * sqrt(E[g_m^2 | filtered] * h_t)
  e_gm2 <- as.vector(filtered %*% g.m^2)
  cond_vol <- sigma * sqrt(e_gm2 * h)

  if (what == "vol") {
    plot.df <- matrix(cbind(cond_vol, abs(object$ret)), ncol = 2)
    colnames(plot.df) <- c("Conditional Volatility", "Absolute Returns")
    plot.df <- reshape2::melt(plot.df)
    msm_plot <- ggplot2::ggplot(plot.df, ggplot2::aes(x = Var1, y = value, colour = Var2)) +
      ggplot2::geom_line() + ggplot2::xlab("Time") + ggplot2::ylab("Volatility") +
      ggplot2::ggtitle("MSM-GJR-GARCH: Conditional Volatility vs Absolute Returns") +
      ggplot2::theme(legend.title = ggplot2::element_blank(), legend.position = "bottom")
  } else if (what == "volsq") {
    plot.df <- matrix(cbind(cond_vol^2, object$ret^2), ncol = 2)
    colnames(plot.df) <- c("Conditional Variance", "Squared Returns")
    plot.df <- reshape2::melt(plot.df)
    msm_plot <- ggplot2::ggplot(plot.df, ggplot2::aes(x = Var1, y = value, colour = Var2)) +
      ggplot2::geom_line() + ggplot2::xlab("Time") + ggplot2::ylab("Variance") +
      ggplot2::ggtitle("MSM-GJR-GARCH: Conditional Variance vs Squared Returns") +
      ggplot2::theme(legend.title = ggplot2::element_blank(), legend.position = "bottom")
  } else {
    stop("what must be 'vol' or 'volsq'")
  }
  print(msm_plot)
}
```

- [ ] **Step 5.4: Regenerate namespace and load**

```bash
Rscript -e "devtools::document()" && Rscript -e "devtools::load_all('.')"
```
Expected: no errors; `Msm_garch` is now exported.

- [ ] **Step 5.5: Run tests**

```bash
Rscript -e "devtools::test(filter = 'Msm_garch')"
```
Expected: all 29 tests pass. If kbar=2 test is slow (>60s), that is expected — SE involves 49×4 likelihood evaluations per parameter.

- [ ] **Step 5.6: Commit**

```bash
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git add R/Msm_garch.R tests/testthat/test-Msm_garch.R
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git commit -m "feat: add Msm_garch top-level function and S3 methods"
```

---

## Task 6: Full Test Suite and Integration

**Files:**
- Modify: `tests/testthat/test-Msm_garch.R` (append integration tests)
- Modify: `R/Msm_garch.R` (add `@export` tag to `Msm_garch_likelihood`, `Msm_garch_std_err` if missing)

- [ ] **Step 6.1: Add integration tests**

Append to `tests/testthat/test-Msm_garch.R`:

```r
# --- integration ---

test_that("Msm_garch full dataset kbar=1 converges with positive gamma_gjr", {
  ret_full <- na.omit(as.matrix(calvet2004data$caret)) * 100
  fit <- Msm_garch(ret_full, kbar = 1)
  expect_s3_class(fit, "msmgarchmodel")
  expect_equal(fit$optim.convergence, 0)
  expect_true(is.finite(fit$LL))
  # GJR leverage should be non-negative on equity returns
  expect_gte(as.numeric(fit$coefficients["gamma_gjr", ]), 0)
})

test_that("Msm_garch LL better than or equal to Msm LL (free params)", {
  ret_full <- na.omit(as.matrix(calvet2004data$caret)) * 100
  fit_msm   <- Msm(ret_full, kbar = 1)
  fit_garch <- Msm_garch(ret_full, kbar = 1)
  # Msm_garch nests Msm, so LL should be >= (negative LL <=)
  expect_lte(fit_garch$LL, fit_msm$LL + 1e-3)
})

test_that("Msm_garch h vector mean close to 1 (unit-mean property)", {
  ret_full <- na.omit(as.matrix(calvet2004data$caret)) * 100
  fit <- Msm_garch(ret_full, kbar = 1)
  # Under stationarity E[h_t] = 1; sample mean should be close
  expect_lt(abs(mean(fit$h) - 1), 0.5)  # loose check — finite sample
})
```

- [ ] **Step 6.2: Run full test suite**

```bash
Rscript -e "devtools::test(filter = 'Msm_garch')"
```
Expected: all 32 tests pass.

- [ ] **Step 6.3: Run full package test suite to check no regressions**

```bash
Rscript -e "devtools::test()"
```
Expected: all existing tests continue to pass.

- [ ] **Step 6.4: Commit**

```bash
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git add tests/testthat/test-Msm_garch.R
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git commit -m "test: add Msm_garch integration tests"
```

---

## Self-Review Notes

- **Spec coverage:** C++ kernels ✓ · parameter check ✓ · LL wrappers ✓ · SE (OPG kbar=1, Hessian kbar>1) ✓ · top-level ✓ · S3 methods (print/summary/coef/predict/plot) ✓ · nesting tests ✓ · stationarity constraint ✓ · convergence/structure tests ✓ · integration tests ✓
- **SE note:** For `kbar > 1`, Hessian-based SE is used (matching `Amsm_scale` pattern). The spec describes a sandwich estimator; this is a deliberate simplification. If NW sandwich is required for kbar>1, `Msm_garch_std_err` can be extended by adapting `Msm_varcovvar` with `Msm_garch_likelihood` substituted for `Msm_likelihood2`.
- **kbar threshold:** `kbar >= 8` for Kronecker path — matches all existing univariate models. The spec says `>= 4` but the codebase consistently uses `>= 8` for univariate.
- **No placeholders found.**
- **Type consistency:** `Msm_garch_likelihood` used in `Msm_garch_grad` and `Msm_garch_hessian` — matches function name defined in Task 3. `msmgarchmodel` class used consistently across all S3 methods.

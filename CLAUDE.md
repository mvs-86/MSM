# MSM R Package — Claude Context

## Project

R package implementing Markov Switching Multifractal (MSM) volatility models. Univariate, bivariate, symmetric, and asymmetric (leverage) variants. C++/RcppArmadillo for all inner-loop likelihood computations.

## Build & Test

```bash
# Recompile C++ and regenerate NAMESPACE (after any src/ or @export change)
Rscript -e "devtools::document()"
Rscript -e "devtools::load_all('.')"

# Run full test suite
Rscript -e "devtools::test()"

# Run single test file
Rscript -e "devtools::test(filter = 'Bmsm_scale')"

# Build package
Rscript -e "devtools::build()"
```

## Git

Repository has an ownership mismatch. All git commands need:

```bash
GIT_DIR=/home/jovyan/workspace/MSMr/.git GIT_WORK_TREE=/home/jovyan/workspace/MSMr git <command>
```

`git config --global` fails ("Device or resource busy") — do not attempt it.

## File Structure

```
R/          # R source — one function per file (roughly)
src/        # C++/RcppArmadillo — compiled via Rcpp::compileAttributes()
tests/testthat/
docs/superpowers/
  specs/    # design specs
  plans/    # implementation plans
```

## Model Inventory

| Function | Model | Params | Return class |
|----------|-------|--------|--------------|
| `Msm` | Univariate MSM | 4 | `msmmodel` |
| `Msm_rolling` | Rolling/expanding univariate MSM | — | list |
| `Amsm_scale` | Univariate — sigma-scaling leverage | 5 | `amsmmodel` |
| `Amsm_gjr` | Univariate — GJR additive variance leverage | 5 | `amsmmodel` |
| `Amsm_atp` | Univariate — asymmetric transition probabilities | 5 | `amsmmodel` |
| `Bmsm` | Bivariate MSM | 9 | `bmsmmodel` |
| `Bmsm_rolling` | Rolling/expanding bivariate MSM | — | list |
| `Bmsm_scale` | Bivariate — sigma-scaling leverage | 11 | `bmsmmodel` |

## Architecture Conventions

**Two-stage MLE** (bivariate models): Stage 1 = marginal params via `nlminb`; Stage 2 = joint/correlation params with Stage 1 fixed.

**C++ dispatch pattern**: `kbar >= 4` (or `>= 8` for some univariate) uses Kronecker-factored algorithm (`_kron` suffix); smaller `kbar` uses dense matrix (`_cpp` suffix).

**Parameter order** for `Bmsm_scale` (11 params):
`[m01, m02, sigma1, sigma2, gammak, b, lev1, lev2, rhoe, lambda, rhom]`

**sigma convention**: internally stored as annualised × √n.vol; divided back in `$coefficients` on output.

**kbar = 1**: `b` is unidentified — set to `NA` in coefficients and SE.

**Standard errors**: sandwich (Newey-West outer product of gradients / two-sided Hessian). Singular Hessian → `MASS::ginv` fallback with warning.

**S3 classes**: `msmmodel`, `amsmmodel`, `bmsmmodel`. New leverage models reuse existing class so `summary`, `print`, `plot`, `predict` work without changes.

## Adding a New Model

1. Add C++ file in `src/` with `// [[Rcpp::export]]` tags
2. Run `devtools::document()` to regenerate `R/RcppExports.R` and `NAMESPACE`
3. Add R wrappers following existing pattern (stage1_likelihood, stage2_likelihood, filtered2, std_err, top-level)
4. Write tests with nesting check (`lev=0` must reproduce symmetric model to 1e-6)
5. Declare any new package dependencies in `DESCRIPTION` under `Imports:`

## Testing Conventions

- TDD: write failing test first, then implement
- Nesting tests mandatory for all leverage models
- SE tests: check `length`, `is.finite`, and `NA` at `b` position for `kbar=1`
- `ret_small <- as.matrix(calvet2006returns[1:100, 2:3]) * 100` — standard small dataset used across bivariate tests

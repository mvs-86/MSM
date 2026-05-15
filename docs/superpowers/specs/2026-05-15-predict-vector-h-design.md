# Design: Multi-step forecast for `predict.msmmodel` and `predict.bmsmmodel`

**Date:** 2026-05-15

## Problem

`predict.msmmodel` and `predict.bmsmmodel` accept `h` as a scalar integer, returning a single-point forecast. When users want forecasts at multiple horizons (e.g., `h=1:10`), they must call `predict` ten times and assemble results manually.

## Goal

Support vector `h` (consecutive integers only, e.g. `h=1:10`). Return list with same element names but each element is a vector of length `length(h)`, one value per horizon. Scalar `h` behavior unchanged.

## Scope

Changes confined to `predict.msmmodel` (in `R/Msm.R`) and `predict.bmsmmodel` (in `R/Bmsm.R`). Inner functions `Msm_predict` and `Bmsm_predict` are not modified.

## Design

### Return structure

When `h` is scalar: existing behavior, list elements are scalars (backward compatible).

When `h` is a vector of consecutive integers:

- `predict.msmmodel` returns `list(vol = numeric(length(h)), vol.sq = numeric(length(h)))`
- `predict.bmsmmodel` returns `list(vol1, vol2, covt, rho.t, vol1.sq, vol2.sq)` each `numeric(length(h))`

Row `i` corresponds to `h[i]`-step-ahead forecast.

### Implementation

Both wrappers detect `length(h) > 1`, validate, then `lapply` over `h` calling the existing scalar inner function, and bind results:

```r
predict.msmmodel <- function(object, h = NULL, ...) {
  if (!is.null(h) && length(h) > 1) {
    if (!all(diff(h) == 1L)) stop("h must be consecutive integers, e.g. h=1:10")
    if (any(h < 1))          stop("h must be >= 1")
    smoothed.p <- object$filtered
    results    <- lapply(h, function(hi) Msm_predict(object$g.m, object$para[4], object$n, smoothed.p, object$A, hi))
    return(list(
      vol    = do.call(c, lapply(results, `[[`, "vol")),
      vol.sq = do.call(c, lapply(results, `[[`, "vol.sq"))
    ))
  }
  # scalar path (unchanged)
  smoothed.p <- if (!is.null(h)) object$filtered else Msm_smooth_cpp(object$A, object$filtered)
  Msm_predict(object$g.m, object$para[4], object$n, smoothed.p, object$A, h)
}
```

Same pattern for `predict.bmsmmodel`, binding all six output fields.

### Validation

- Non-consecutive `h` (e.g. `c(1,5,10)`): error with clear message
- `h` containing values < 1: error
- Scalar `h`: delegates unchanged to inner function (which does its own validation)

## Verification checks

1. `predict(fit, h=5)$vol` — length 1, same value as before
2. `predict(fit, h=1:10)$vol` — length 10
3. `predict(fit, h=3)$vol == predict(fit, h=1:3)$vol[3]` — TRUE
4. Same three checks for `Bmsm` fit on all six output fields

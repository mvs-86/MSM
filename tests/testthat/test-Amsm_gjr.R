library(MSM)

data("calvet2004data")
ret <- na.omit(as.matrix(calvet2004data$caret)) * 100
ret_small <- ret[1:150, , drop = FALSE]

# --- nesting: lev=0 must reproduce Msm LL exactly ---

test_that("Amsm_gjr_ll with lev=0 matches Msm_ll2", {
  para_lev0 <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0)
  ll_msm  <- Msm_ll2(para_lev0[1:4], kbar = 1, dat = ret_small, n.vol = 252)
  ll_gjr  <- Amsm_gjr_ll(para_lev0,  kbar = 1, dat = ret_small, n.vol = 252)
  expect_equal(as.numeric(ll_gjr), as.numeric(ll_msm), tolerance = 1e-6)
})

test_that("Amsm_gjr_likelihood with lev=0 matches Msm_likelihood2 LL", {
  para_lev0 <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0)
  ref <- Msm_likelihood2(para_lev0[1:4], kbar = 1, dat = ret_small, n.vol = 252)
  res <- Amsm_gjr_likelihood(para_lev0,  kbar = 1, dat = ret_small, n.vol = 252)
  expect_equal(res$LL, ref$LL, tolerance = 1e-6)
  expect_equal(dim(res$filtered), dim(ref$filtered))
})

# --- estimation ---

test_that("Amsm_gjr returns amsmmodel with type gjr", {
  fit <- Amsm_gjr(ret_small, kbar = 1)
  expect_s3_class(fit, "amsmmodel")
  expect_equal(fit$type, "gjr")
  expect_equal(nrow(fit$coefficients), 5)
  expect_equal(rownames(fit$coefficients), c("m0", "b", "gammak", "sigma", "lev"))
  expect_true(is.finite(fit$LL))
})

test_that("Amsm_gjr lev is finite and non-negative", {
  fit <- Amsm_gjr(ret_small, kbar = 1)
  lev <- as.numeric(fit$coefficients["lev", ])
  expect_true(is.finite(lev))
  expect_gte(lev, 0)
})

test_that("Amsm_gjr kbar=2 runs and returns finite LL", {
  fit <- Amsm_gjr(ret_small, kbar = 2)
  expect_s3_class(fit, "amsmmodel")
  expect_true(is.finite(fit$LL))
})

# --- S3 methods reuse amsmmodel class ---

test_that("print, summary, predict, plot, coef all work for Amsm_gjr", {
  fit <- Amsm_gjr(ret_small, kbar = 1)
  expect_output(print(fit))
  s <- summary(fit)
  expect_s3_class(s, "summary.amsmmodel")
  pred <- predict(fit)
  expect_named(pred, c("vol", "vol.sq"))
  expect_true(all(pred$vol > 0))
  expect_no_error(plot(fit, what = "vol"))
  co <- coef(fit)
  expect_length(co, 5)
  expect_named(co, c("m0", "b", "gammak", "sigma", "lev"))
})

test_that("predict.amsmmodel h-step ahead works for gjr fit", {
  fit  <- Amsm_gjr(ret_small, kbar = 1)
  pred <- predict(fit, h = 3)
  expect_named(pred, c("vol", "vol.sq"))
  expect_length(pred$vol, 1)
})

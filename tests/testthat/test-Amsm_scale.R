library(MSM)

data("calvet2004data")
ret <- na.omit(as.matrix(calvet2004data$caret)) * 100
ret_small <- ret[1:150, , drop = FALSE]

# --- nesting: lev=0 must reproduce Msm LL exactly ---

test_that("Amsm_scale_ll with lev=0 matches Msm_ll2", {
  msm_fit  <- Msm(ret_small, kbar = 1)
  para_base <- as.numeric(msm_fit$para)
  para_base[2] <- 2.5  # b (NA when kbar=1 in $para, set to something valid)
  para_lev0 <- c(para_base[1], 2.5, para_base[3], para_base[4], 0)

  ll_msm   <- Msm_ll2(para_lev0[1:4], kbar = 1, dat = ret_small, n.vol = 252)
  ll_amsm  <- Amsm_scale_ll(para_lev0, kbar = 1, dat = ret_small, n.vol = 252)

  expect_equal(as.numeric(ll_amsm), as.numeric(ll_msm), tolerance = 1e-6)
})

test_that("Amsm_scale_likelihood with lev=0 matches Msm_likelihood2 LL", {
  para_lev0 <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0)

  ref  <- Msm_likelihood2(para_lev0[1:4], kbar = 1, dat = ret_small, n.vol = 252)
  res  <- Amsm_scale_likelihood(para_lev0, kbar = 1, dat = ret_small, n.vol = 252)

  expect_equal(res$LL, ref$LL, tolerance = 1e-6)
  expect_equal(dim(res$filtered), dim(ref$filtered))
})

# --- estimation ---

test_that("Amsm_scale returns amsmmodel object with correct structure", {
  fit <- Amsm_scale(ret_small, kbar = 1)
  expect_s3_class(fit, "amsmmodel")
  expect_equal(fit$type, "scale")
  expect_equal(nrow(fit$coefficients), 5)
  expect_equal(rownames(fit$coefficients), c("m0", "b", "gammak", "sigma", "lev"))
  expect_true(is.numeric(fit$LL))
  expect_true(is.finite(fit$LL))
})

test_that("Amsm_scale lev is finite and within bounds", {
  fit <- Amsm_scale(ret_small, kbar = 1)
  lev <- as.numeric(fit$coefficients["lev", ])
  expect_true(is.finite(lev))
  expect_gt(lev, -1)   # lower bound
})

test_that("Amsm_scale kbar=2 runs and returns finite LL", {
  fit <- Amsm_scale(ret_small, kbar = 2)
  expect_s3_class(fit, "amsmmodel")
  expect_true(is.finite(fit$LL))
})

# --- S3 methods ---

test_that("print.amsmmodel runs without error", {
  fit <- Amsm_scale(ret_small, kbar = 1)
  expect_output(print(fit))
})

test_that("summary.amsmmodel runs and has coefficients", {
  fit <- Amsm_scale(ret_small, kbar = 1)
  s   <- summary(fit)
  expect_s3_class(s, "summary.amsmmodel")
  expect_true(!is.null(s$coefficients))
})

test_that("predict.amsmmodel returns vol and vol.sq", {
  fit  <- Amsm_scale(ret_small, kbar = 1)
  pred <- predict(fit)
  expect_named(pred, c("vol", "vol.sq"))
  expect_true(all(pred$vol > 0))
})

test_that("predict.amsmmodel h-step ahead works", {
  fit  <- Amsm_scale(ret_small, kbar = 1)
  pred <- predict(fit, h = 5)
  expect_named(pred, c("vol", "vol.sq"))
  expect_length(pred$vol, 1)
})

test_that("plot.amsmmodel runs without error for vol and volsq", {
  fit <- Amsm_scale(ret_small, kbar = 1)
  expect_no_error(plot(fit, what = "vol"))
  expect_no_error(plot(fit, what = "volsq"))
})

test_that("coef.amsmmodel returns 5-element named vector", {
  fit <- Amsm_scale(ret_small, kbar = 1)
  co  <- coef(fit)
  expect_length(co, 5)
  expect_named(co, c("m0", "b", "gammak", "sigma", "lev"))
})

library(MSM)

data("calvet2004data")
ret <- na.omit(as.matrix(calvet2004data$caret)) * 100
ret_small <- ret[1:150, , drop = FALSE]

# --- nesting: lev=0 must reproduce Msm LL exactly ---

test_that("Amsm_atp_ll with lev=0 matches Msm_ll2", {
  para_lev0 <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0)
  ll_msm  <- Msm_ll2(para_lev0[1:4], kbar = 1, dat = ret_small, n.vol = 252)
  ll_atp  <- Amsm_atp_ll(para_lev0,  kbar = 1, dat = ret_small, n.vol = 252)
  expect_equal(as.numeric(ll_atp), as.numeric(ll_msm), tolerance = 1e-6)
})

test_that("Amsm_atp_likelihood with lev=0 matches Msm_likelihood2 LL", {
  para_lev0 <- c(1.5, 2.5, 0.9, 0.3 * sqrt(252), 0)
  ref <- Msm_likelihood2(para_lev0[1:4], kbar = 1, dat = ret_small, n.vol = 252)
  res <- Amsm_atp_likelihood(para_lev0,  kbar = 1, dat = ret_small, n.vol = 252)
  expect_equal(res$LL, ref$LL, tolerance = 1e-6)
  expect_equal(dim(res$filtered), dim(ref$filtered))
})

# negative lev: A_neg = Msm_A(b, gammak*(1+lev), kbar); lev=-1 -> gammak*0 -> A_neg=identity-ish
test_that("Amsm_atp_ll with lev=-1 gives different LL than lev=0", {
  para_lev0  <- c(1.5, 2.5, 0.5, 0.3 * sqrt(252), 0)
  para_levm1 <- c(1.5, 2.5, 0.5, 0.3 * sqrt(252), -0.5)
  ll0  <- Amsm_atp_ll(para_lev0,  kbar = 1, dat = ret_small, n.vol = 252)
  llm1 <- Amsm_atp_ll(para_levm1, kbar = 1, dat = ret_small, n.vol = 252)
  expect_false(isTRUE(all.equal(as.numeric(ll0), as.numeric(llm1))))
})

# --- estimation ---

test_that("Amsm_atp returns amsmmodel with type atp", {
  fit <- Amsm_atp(ret_small, kbar = 1)
  expect_s3_class(fit, "amsmmodel")
  expect_equal(fit$type, "atp")
  expect_equal(nrow(fit$coefficients), 5)
  expect_equal(rownames(fit$coefficients), c("m0", "b", "gammak", "sigma", "lev"))
  expect_true(is.finite(fit$LL))
})

test_that("Amsm_atp lev is finite and within bounds", {
  fit <- Amsm_atp(ret_small, kbar = 1)
  lev <- as.numeric(fit$coefficients["lev", ])
  expect_true(is.finite(lev))
  expect_gt(lev, -1)
})

test_that("Amsm_atp kbar=2 runs and returns finite LL", {
  fit <- Amsm_atp(ret_small, kbar = 2)
  expect_s3_class(fit, "amsmmodel")
  expect_true(is.finite(fit$LL))
})

# --- S3 methods ---

test_that("print, summary, predict, plot, coef all work for Amsm_atp", {
  fit <- Amsm_atp(ret_small, kbar = 1)
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

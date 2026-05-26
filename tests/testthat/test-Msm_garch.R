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

test_that("Msm_garch_ll_fast_cpp and Msm_garch_fast_cpp return same LL (nonzero params)", {
  ll_scalar <- Msm_garch_ll_fast_cpp(ret_small, parts$sigma_gm, parts$A, 0.05, 0.85, 0.05)
  ll_full   <- Msm_garch_fast_cpp(ret_small, parts$sigma_gm, parts$A, 0.05, 0.85, 0.05)$LL
  expect_equal(ll_scalar, ll_full, tolerance = 1e-10)
})

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
  expect_true(ll != 0)  # negative LL is non-zero
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

test_that("predict.msmgarchmodel h=NULL vol matches plot computation", {
  fit   <- Msm_garch(ret_small, kbar = 1)
  pred  <- predict(fit)
  sigma <- fit$para[4] / sqrt(fit$n)
  smoothed.p <- Msm_smooth_cpp(fit$A, fit$filtered)
  e_gm2 <- as.vector(smoothed.p %*% t(fit$g.m^2))
  expected_vol <- sigma * sqrt(e_gm2 * fit$h)
  expect_equal(pred$vol, expected_vol, tolerance = 1e-10)
})

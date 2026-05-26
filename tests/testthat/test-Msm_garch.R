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

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

# --- SE: Bmsm_scale_std_err returns length-11 SE with NA for b (kbar=1) ---

test_that("Bmsm_scale_std_err with lev=0 is finite and length 11", {
  para_scale <- c(1.4330, 1.5768, 5.0433, 7.6284, 0.7299, 8.2716, 0, 0, 0.1090, 0.4421, 0.8823)
  se <- Bmsm_scale_std_err(para_scale, 1, ret_small, 252)
  expect_equal(nrow(se), 11)
  # b (index 6) is NA for kbar=1; all others finite
  expect_true(is.na(se[6, 1]))
  expect_true(all(is.finite(se[-6, 1])))
})

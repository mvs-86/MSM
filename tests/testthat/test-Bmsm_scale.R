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

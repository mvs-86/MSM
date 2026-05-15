library(MSM)

data("calvet2004data")
ret <- na.omit(as.matrix(calvet2004data$caret)) * 100
fit_msm <- Msm(ret, kbar = 2, n.vol = 252, nw.lag = 2)

test_that("predict.msmmodel scalar h still returns length-1 elements", {
  pred <- predict(fit_msm, h = 5)
  expect_length(pred$vol,    1)
  expect_length(pred$vol.sq, 1)
})

test_that("predict.msmmodel vector h returns correct length", {
  pred <- predict(fit_msm, h = 1:10)
  expect_length(pred$vol,    10)
  expect_length(pred$vol.sq, 10)
})

test_that("predict.msmmodel vector h[i] matches scalar h=i", {
  pred_vec    <- predict(fit_msm, h = 1:3)
  pred_scalar <- predict(fit_msm, h = 3)
  expect_equal(pred_vec$vol[3],    as.numeric(pred_scalar$vol))
  expect_equal(pred_vec$vol.sq[3], as.numeric(pred_scalar$vol.sq))
})

test_that("predict.msmmodel rejects non-consecutive h", {
  expect_error(predict(fit_msm, h = c(1, 5, 10)), "consecutive")
})

test_that("predict.msmmodel rejects h < 1", {
  expect_error(predict(fit_msm, h = 0:5), ">= 1")
})

data("calvet2006returns")
ret2 <- as.matrix(calvet2006returns[, 2:3]) * 100
fit_bmsm <- Bmsm(ret2, kbar = 2, n = 252, s.err = FALSE)

test_that("predict.bmsmmodel scalar h returns length-1 elements", {
  pred <- predict(fit_bmsm, h = 5)
  expect_length(pred$vol1,    1)
  expect_length(pred$vol2,    1)
  expect_length(pred$covt,    1)
  expect_length(pred$rho.t,   1)
  expect_length(pred$vol1.sq, 1)
  expect_length(pred$vol2.sq, 1)
})

test_that("predict.bmsmmodel vector h returns correct length", {
  pred <- predict(fit_bmsm, h = 1:10)
  expect_length(pred$vol1,    10)
  expect_length(pred$vol2,    10)
  expect_length(pred$covt,    10)
  expect_length(pred$rho.t,   10)
  expect_length(pred$vol1.sq, 10)
  expect_length(pred$vol2.sq, 10)
})

test_that("predict.bmsmmodel vector h[i] matches scalar h=i", {
  pred_vec    <- predict(fit_bmsm, h = 1:3)
  pred_scalar <- predict(fit_bmsm, h = 3)
  expect_equal(pred_vec$vol1[3],    as.numeric(pred_scalar$vol1))
  expect_equal(pred_vec$vol2[3],    as.numeric(pred_scalar$vol2))
  expect_equal(pred_vec$covt[3],    as.numeric(pred_scalar$covt))
  expect_equal(pred_vec$rho.t[3],   as.numeric(pred_scalar$rho.t))
  expect_equal(pred_vec$vol1.sq[3], as.numeric(pred_scalar$vol1.sq))
  expect_equal(pred_vec$vol2.sq[3], as.numeric(pred_scalar$vol2.sq))
})

test_that("predict.bmsmmodel rejects non-consecutive h", {
  expect_error(predict(fit_bmsm, h = c(1, 5, 10)), "consecutive")
})

test_that("predict.bmsmmodel rejects h < 1", {
  expect_error(predict(fit_bmsm, h = 0:5), ">= 1")
})

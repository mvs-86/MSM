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
  expect_equal(pred_vec$vol[3],    pred_scalar$vol)
  expect_equal(pred_vec$vol.sq[3], pred_scalar$vol.sq)
})

test_that("predict.msmmodel rejects non-consecutive h", {
  expect_error(predict(fit_msm, h = c(1, 5, 10)), "consecutive")
})

test_that("predict.msmmodel rejects h < 1", {
  expect_error(predict(fit_msm, h = 0:5), ">= 1")
})

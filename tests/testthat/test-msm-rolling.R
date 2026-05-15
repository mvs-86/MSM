library(MSM)

data("calvet2004data")
ret <- na.omit(as.matrix(calvet2004data$caret)) * 100
ret_small <- ret[1:60, , drop = FALSE]   # 60 rows, fast fitting

test_that("Msm_rolling returns list with parameters and forecasts data frames", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 1)
  expect_type(result, "list")
  expect_named(result, c("parameters", "forecasts"))
  expect_s3_class(result$parameters, "data.frame")
  expect_s3_class(result$forecasts,  "data.frame")
})

test_that("Msm_rolling rolling: correct number of rows (T - window)", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 1, type = "rolling")
  expect_equal(nrow(result$parameters), nrow(ret_small) - 50)
  expect_equal(nrow(result$forecasts),  nrow(ret_small) - 50)
})

test_that("Msm_rolling expanding: correct number of rows (T - window)", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 1, type = "expanding")
  expect_equal(nrow(result$parameters), nrow(ret_small) - 50)
  expect_equal(nrow(result$forecasts),  nrow(ret_small) - 50)
})

test_that("Msm_rolling parameters has correct columns", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 1)
  expect_named(result$parameters, c("window_end", "m0", "b", "gammak", "sigma", "converged"))
})

test_that("Msm_rolling forecasts has correct columns", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 1)
  expect_named(result$forecasts, c("window_end", "h", "vol", "vol.sq"))
})

test_that("Msm_rolling rolling: window_end is last index of each window", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 1, type = "rolling")
  # i=1: end=50, i=2: end=51, ..., i=10: end=59
  expect_equal(result$parameters$window_end, seq(50, nrow(ret_small) - 1))
})

test_that("Msm_rolling expanding: window_end grows correctly", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 1, type = "expanding")
  # i=1: rows 1:50, end=50; i=2: rows 1:51, end=51; ...
  expect_equal(result$parameters$window_end, seq(50, nrow(ret_small) - 1))
})

test_that("Msm_rolling converged column is logical", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 1)
  expect_type(result$parameters$converged, "logical")
})

test_that("Msm_rolling vol and vol.sq are positive", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 1)
  expect_true(all(result$forecasts$vol    > 0))
  expect_true(all(result$forecasts$vol.sq > 0))
})

test_that("Msm_rolling h column equals supplied h", {
  result <- Msm_rolling(ret_small, window = 50, h = 3, kbar = 1)
  expect_true(all(result$forecasts$h == 3))
})

test_that("Msm_rolling rejects window < 10", {
  expect_error(Msm_rolling(ret_small, window = 5), "window must be >= 10")
})

test_that("Msm_rolling rejects h < 1", {
  expect_error(Msm_rolling(ret_small, window = 50, h = 0), "h must be >= 1")
})

test_that("Msm_rolling rejects invalid type", {
  expect_error(Msm_rolling(ret_small, window = 50, type = "banana"), "should be one of")
})

test_that("Msm_rolling rejects multi-column ret", {
  ret2 <- cbind(ret_small, ret_small)
  expect_error(Msm_rolling(ret2, window = 50), "single-column")
})

test_that("Msm_rolling rejects window >= T", {
  expect_error(Msm_rolling(ret_small, window = 60), "too few rows")
})

test_that("Msm_rolling kbar=2 completes and warm-start code path is exercised", {
  result <- Msm_rolling(ret_small, window = 50, h = 1, kbar = 2)
  expect_equal(nrow(result$parameters), nrow(ret_small) - 50)
  expect_true(all(result$forecasts$vol > 0))
})

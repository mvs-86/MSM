library(MSM)

data("calvet2006returns")
ret2 <- as.matrix(calvet2006returns[, 2:3]) * 100
ret2_small <- ret2[1:60, ]   # 60 rows for speed

test_that("Bmsm_rolling returns list with parameters and forecasts data frames", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1)
  expect_type(result, "list")
  expect_named(result, c("parameters", "forecasts"))
  expect_s3_class(result$parameters, "data.frame")
  expect_s3_class(result$forecasts,  "data.frame")
})

test_that("Bmsm_rolling rolling: correct number of rows (T - window)", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1, type = "rolling")
  expect_equal(nrow(result$parameters), nrow(ret2_small) - 50)
  expect_equal(nrow(result$forecasts),  nrow(ret2_small) - 50)
})

test_that("Bmsm_rolling expanding: correct number of rows (T - window)", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1, type = "expanding")
  expect_equal(nrow(result$parameters), nrow(ret2_small) - 50)
  expect_equal(nrow(result$forecasts),  nrow(ret2_small) - 50)
})

test_that("Bmsm_rolling parameters has correct columns", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1)
  expect_named(result$parameters,
    c("window_end", "m01", "m02", "sigma1", "sigma2",
      "gammak", "b", "rhoe", "lambda", "rhom", "converged"))
})

test_that("Bmsm_rolling forecasts has correct columns", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1)
  expect_named(result$forecasts, c("window_end", "h", "vol1", "vol2", "covt", "rho.t"))
})

test_that("Bmsm_rolling rolling: window_end is last index of each window", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1, type = "rolling")
  expect_equal(result$parameters$window_end, seq(50, nrow(ret2_small) - 1))
})

test_that("Bmsm_rolling expanding: window_end grows correctly", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1, type = "expanding")
  expect_equal(result$parameters$window_end, seq(50, nrow(ret2_small) - 1))
})

test_that("Bmsm_rolling converged column is logical", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1)
  expect_type(result$parameters$converged, "logical")
})

test_that("Bmsm_rolling vol1 and vol2 are positive", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1)
  expect_true(all(result$forecasts$vol1 > 0))
  expect_true(all(result$forecasts$vol2 > 0))
})

test_that("Bmsm_rolling rho.t is between -1 and 1", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 1, kbar = 1)
  expect_true(all(result$forecasts$rho.t >= -1 & result$forecasts$rho.t <= 1))
})

test_that("Bmsm_rolling h column equals supplied h", {
  result <- Bmsm_rolling(ret2_small, window = 50, h = 2, kbar = 1)
  expect_true(all(result$forecasts$h == 2))
})

test_that("Bmsm_rolling rejects window < 10", {
  expect_error(Bmsm_rolling(ret2_small, window = 5), "window must be >= 10")
})

test_that("Bmsm_rolling rejects h < 1", {
  expect_error(Bmsm_rolling(ret2_small, window = 50, h = 0), "h must be >= 1")
})

test_that("Bmsm_rolling rejects invalid type", {
  expect_error(Bmsm_rolling(ret2_small, window = 50, type = "banana"), "should be one of")
})

test_that("Bmsm_rolling rejects single-column ret", {
  expect_error(Bmsm_rolling(ret2_small[, 1, drop = FALSE], window = 50), "two-column")
})

test_that("Bmsm_rolling rejects window >= T", {
  expect_error(Bmsm_rolling(ret2_small, window = 60), "too few rows")
})

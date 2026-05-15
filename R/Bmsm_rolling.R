# R/Bmsm_rolling.R

#' Rolling/Expanding Window Estimation for the Bivariate MSM Model.
#'
#' Re-fits \code{\link{Bmsm}} on successive rolling or expanding windows and
#' returns estimated parameters and h-step-ahead forecasts for each window.
#'
#' @param ret Two-column matrix of returns.
#' @param window Integer >= 10. Fixed window length (rolling) or minimum window
#'   length (expanding).
#' @param h Integer >= 1. Forecast horizon applied at the end of each window.
#' @param type One of \code{"rolling"} (fixed-width) or \code{"expanding"}
#'   (growing from start). Default \code{"rolling"}.
#' @param kbar Number of MSM frequency components. Default 1.
#' @param n Integer. Trading days per year. Default 252.
#' @param s.err Logical. Compute standard errors? Default \code{FALSE}
#'   (expensive in rolling context).
#'
#' @return A named list with two data frames:
#' \describe{
#'   \item{parameters}{One row per window: \code{window_end}, \code{m01},
#'     \code{m02}, \code{sigma1}, \code{sigma2}, \code{gammak}, \code{b},
#'     \code{rhoe}, \code{lambda}, \code{rhom}, \code{converged}.}
#'   \item{forecasts}{One row per window: \code{window_end}, \code{h},
#'     \code{vol1}, \code{vol2}, \code{covt}, \code{rho.t}.}
#' }
#'
#' @examples
#' data("calvet2006returns")
#' ret <- as.matrix(calvet2006returns[, 2:3]) * 100
#' ret_sub <- ret[1:60, ]
#' result <- Bmsm_rolling(ret_sub, window = 50, h = 1, kbar = 1)
#' head(result$parameters)
#' head(result$forecasts)
#'
#' @export
Bmsm_rolling <- function(ret, window, h = 1,
                         type  = c("rolling", "expanding"),
                         kbar  = 1, n = 252, s.err = FALSE) {
  type <- match.arg(type)

  if (!is.matrix(ret)) ret <- as.matrix(ret)
  if (ncol(ret) != 2)  stop("ret must be a two-column matrix for Bmsm_rolling")

  N <- nrow(ret)
  if (window < 10)    stop("window must be >= 10")
  if (h < 1)          stop("h must be >= 1")
  n_windows <- N - window
  if (n_windows < 1)  stop("ret has too few rows for the given window size")

  params_list   <- vector("list", n_windows)
  forecast_list <- vector("list", n_windows)
  prev_para     <- NULL

  for (i in seq_len(n_windows)) {
    idx        <- if (type == "rolling") i:(i + window - 1L) else 1L:(window + i - 1L)
    window_end <- idx[length(idx)]

    fit  <- Bmsm(ret[idx, ], kbar = kbar, n = n, para0 = prev_para, s.err = s.err)
    pred <- predict(fit, h = h)

    prev_para <- if (anyNA(fit$para)) NULL else as.numeric(fit$para)

    params_list[[i]] <- data.frame(
      window_end = window_end,
      m01        = as.numeric(fit$coefficients["m01",    ]),
      m02        = as.numeric(fit$coefficients["m02",    ]),
      sigma1     = as.numeric(fit$coefficients["sigma1", ]),
      sigma2     = as.numeric(fit$coefficients["sigma2", ]),
      gammak     = as.numeric(fit$coefficients["gammak", ]),
      b          = as.numeric(fit$coefficients["b",      ]),
      rhoe       = as.numeric(fit$coefficients["rhoe",   ]),
      lambda     = as.numeric(fit$coefficients["lambda", ]),
      rhom       = as.numeric(fit$coefficients["rhom",   ]),
      converged  = all(fit$optim.convergence == 0L)
    )

    forecast_list[[i]] <- data.frame(
      window_end = window_end,
      h          = h,
      vol1       = as.numeric(pred$vol1),
      vol2       = as.numeric(pred$vol2),
      covt       = as.numeric(pred$covt),
      rho.t      = as.numeric(pred$rho.t)
    )
  }

  list(
    parameters = do.call(rbind, params_list),
    forecasts  = do.call(rbind, forecast_list)
  )
}

#' Rolling/Expanding Window Estimation for the Univariate MSM Model.
#'
#' Re-fits \code{\link{Msm}} on successive rolling or expanding windows and
#' returns estimated parameters and h-step-ahead forecasts for each window.
#'
#' @param ret Column matrix of returns.
#' @param window Integer >= 10. Fixed window length (rolling) or minimum window
#'   length (expanding).
#' @param h Integer >= 1. Forecast horizon applied at the end of each window.
#' @param type One of \code{"rolling"} (fixed-width) or \code{"expanding"}
#'   (growing from start). Default \code{"rolling"}.
#' @param kbar Number of MSM frequency components. Default 1.
#' @param n.vol Trading days per year. Default 252.
#' @param nw.lag Newey-West lags for standard errors. Default 0.
#'
#' @return A named list with two data frames:
#' \describe{
#'   \item{parameters}{One row per window: \code{window_end}, \code{m0},
#'     \code{b}, \code{gammak}, \code{sigma}, \code{converged}.}
#'   \item{forecasts}{One row per window: \code{window_end}, \code{h},
#'     \code{vol}, \code{vol.sq}.}
#' }
#'
#' @examples
#' data("calvet2004data")
#' ret <- na.omit(as.matrix(calvet2004data$caret)) * 100
#' result <- Msm_rolling(ret, window = 200, h = 1, kbar = 1)
#' head(result$parameters)
#' head(result$forecasts)
#'
#' @export
Msm_rolling <- function(ret, window, h = 1,
                        type   = c("rolling", "expanding"),
                        kbar   = 1, n.vol = 252, nw.lag = 0) {
  type <- match.arg(type)

  if (!is.matrix(ret)) ret <- as.matrix(ret)
  if (ncol(ret) != 1)  stop("ret must be a single-column matrix for Msm_rolling")

  T <- nrow(ret)
  if (window < 10)        stop("window must be >= 10")
  if (h < 1)              stop("h must be >= 1")
  n_windows <- T - window
  if (n_windows < 1)      stop("ret has too few rows for the given window size")

  params_list   <- vector("list", n_windows)
  forecast_list <- vector("list", n_windows)
  prev_para     <- NULL

  for (i in seq_len(n_windows)) {
    idx        <- if (type == "rolling") i:(i + window - 1L) else 1L:(window + i - 1L)
    window_end <- idx[length(idx)]

    fit  <- Msm(ret[idx, , drop = FALSE], kbar = kbar, n.vol = n.vol,
                para0 = prev_para, nw.lag = nw.lag)
    pred <- predict(fit, h = h)

    prev_para <- if (anyNA(fit$para)) NULL else as.numeric(fit$para)

    params_list[[i]] <- data.frame(
      window_end = window_end,
      m0         = as.numeric(fit$coefficients["m0",     ]),
      b          = as.numeric(fit$coefficients["b",      ]),
      gammak     = as.numeric(fit$coefficients["gammak", ]),
      sigma      = as.numeric(fit$coefficients["sigma",  ]),
      converged  = fit$optim.convergence == 0L
    )

    forecast_list[[i]] <- data.frame(
      window_end = window_end,
      h          = h,
      vol        = as.numeric(pred$vol),
      vol.sq     = as.numeric(pred$vol.sq)
    )
  }

  list(
    parameters = do.call(rbind, params_list),
    forecasts  = do.call(rbind, forecast_list)
  )
}

#' Boundary Checks on \code{Msm_garch} Model Parameters.
#'
#' Performs boundary checks on \code{Msm_garch} model parameter values.
#'
#' @param dat is a column matrix/dataframe of returns.
#' @param kbar is the number of frequency components in the model.
#' @param x0 is the initial parameter values passed to Msm_garch. Length 7:
#'   \code{c(m0, b, gammak, sigma, alpha, beta, gamma_gjr)}.
#'
#' @return a list consisting of:
#' \item{dat}{column matrix of returns}
#' \item{kbar}{number of frequency components}
#' \item{start.value}{initial parameter values}
#' \item{lb}{lower bounds on parameters}
#' \item{ub}{upper bounds on parameters}
#'
Msm_garch_parameter_check <- function(dat, kbar, x0, n.vol = 252) {

  if (!is.matrix(dat)) dat <- as.matrix(dat)
  if (ncol(dat) > 1)   dat <- t(dat)
  if (ncol(dat) > 1 || nrow(dat) < 2 || is.null(dat))
    stop("dat must be a numeric vector.")
  if (kbar < 1)
    stop("kbar must be a positive integer.")

  if (!is.null(x0)) {
    if (length(x0) != 7)
      stop("Initial values must be of length 7: c(m0, b, gammak, sigma, alpha, beta, gamma_gjr)")
    if (x0[1] < 1 || x0[1] > 1.99)      stop("m0 must be in (1, 1.99]")
    if (x0[2] < 1)                        stop("b must be > 1")
    if (x0[3] < 1e-4 || x0[3] > 0.9999) stop("gammak must be in (0, 1)")
    if (x0[4] < 1e-5)                     stop("sigma must be positive")
    if (x0[5] < 0)                        stop("alpha must be >= 0")
    if (x0[6] < 0)                        stop("beta must be >= 0")
    if (x0[7] < 0)                        stop("gamma_gjr must be >= 0")
  } else {
    msm_sv <- c(1.5, 2.5, 0.9, sd(dat) * sqrt(n.vol))
    x0     <- c(msm_sv, 0.05, 0.85, 0.05)
  }

  lb <- c(1,      1,      1e-4,  1e-5,  0,    0,    0)
  ub <- c(1.9999, 50,     0.9999, 50,   0.99, 0.99, 1.99)

  list(dat         = dat,
       kbar        = kbar,
       start.value = x0,
       lb          = lb,
       ub          = ub)
}

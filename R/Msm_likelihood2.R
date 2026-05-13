#' Log-Likelihood for \code{\link{Msm}}(k) model.
#'
#' Calculates log-likelihood and filtered probability values for \code{\link{Msm}}(k) model.
#'
#' @param para is a vector of parameters [m0, b, gamma_k, sigma].
#' @param kbar is the number of frequency components in the \code{\link{Msm}}(k) model.
#' @param dat is a column matrix of returns.
#' @param n.vol is the number of trading days in a year, if data is on a daily frequency. Default is 252.
#' If data is monthly, then set n.vol to 12. If data is quarterly, then n.vol should be 4.
#'
#' @return a list consisting of:
#' \item{LL}{sum of loglikelihood values at optimum}
#' \item{LLs}{a vector of loglikelihood values at optimum}
#' \item{filtered}{a matrix of filtered probabilities}
#' \item{A}{the estimated transition matrix}
#' \item{g.m}{the estimated state values}
#'
#'
#' @examples
#' data("calvet2004data")
#' ret <- na.omit(as.matrix(calvet2004data$caret))*100
#' vol <- Msm_likelihood2(c(1.556, 10.92, 0.109, 0.278*sqrt(252)), 2, ret, n.vol=252)
#'
#' @export

Msm_likelihood2 <- function(para, kbar, dat, n.vol){


  m0     <- para[1]
  b      <- para[2]
  gama.k <- para[3]
  sigma  <- para[4]/sqrt(n.vol)
  k2     <- 2^kbar

  A   <- Msm_A(b, gama.k, kbar)
  g.m <- Msm_states(m0, kbar)
  N   <- nrow(dat)

  sigma_gm <- as.vector(sigma * g.m)

  likelihood = Msm_likelihood_fast(dat, sigma_gm, A)

  likelihood$A   <- A
  likelihood$g.m <- g.m

  return(likelihood)

}

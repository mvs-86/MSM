#' Sum of Log-Likelihood for \code{\link{Msm}}(k) model.
#'
#' Calculates the sum of the log-likelihood values for \code{\link{Msm}}(k) model.
#'
#' @param para is a vector of parameters returned by \code{\link{Msm}}.
#' @param kbar is the number of frequency components in the \code{\link{Msm}}(k) model.
#' @param dat is a column matrix of returns.
#' @param n.vol is the number of trading days in a year, if data is on a daily frequency. Default is 252.
#'
#' @return sum of the log-likelihood values
#'
#'
#' @examples
#' data("calvet2004data")
#' ret <- na.omit(as.matrix(calvet2004data$caret))*100
#' Msm_ll2(c(1.556, 10.92, 0.109, 0.278*sqrt(252)), 2, ret, n.vol=252)
#'
#' @export

Msm_ll2 <- function(para, kbar, dat, n.vol){

  m0     <- para[1]
  b      <- para[2]
  gama.k <- para[3]
  sigma  <- para[4]/sqrt(n.vol)
  k2     <- 2^kbar

  A   <- Msm_A(b, gama.k, kbar)
  g.m <- Msm_states(m0, kbar)
  N  <- nrow(dat)


  sigma_gm <- as.vector(sigma * g.m)

  LL = Msm_ll_fast(dat, sigma_gm, A)
  if(!is.finite(LL)){

    warning('Log-likelihood is inf. Probably due to all zeros in conditional probability.')
  }

  return(LL)

}

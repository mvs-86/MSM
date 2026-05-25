#' Stage 1 log-likelihood for Bmsm_scale model.
#'
#' Sum of two independent univariate scale-leverage MSM log-likelihoods.
#' Estimates [m01, m02, sigma1, sigma2, gammak, b, lev1, lev2].
#'
#' @param para length-8 vector [m01, m02, sigma1, sigma2, gammak, b, lev1, lev2].
#' @param dat N x 2 return matrix.
#' @param kbar number of frequency components.
#' @param n.vol trading days per year.
#' @return scalar log-likelihood (negative, for minimisation).
#' @export
Bmsm_scale_stage1_likelihood <- function(para, dat, kbar, n.vol) {
  m01     <- para[1]
  m02     <- para[2]
  sigma1  <- para[3]
  sigma2  <- para[4]
  gamma.k <- para[5]
  b       <- para[6]
  lev1    <- para[7]
  lev2    <- para[8]

  ret1 <- matrix(dat[, 1], ncol = 1)
  ret2 <- matrix(dat[, 2], ncol = 1)
  par1 <- c(m01, b, gamma.k, sigma1, lev1)
  par2 <- c(m02, b, gamma.k, sigma2, lev2)

  ll <- Amsm_scale_ll(par1, kbar, ret1, n.vol) +
        Amsm_scale_ll(par2, kbar, ret2, n.vol)
  return(ll)
}

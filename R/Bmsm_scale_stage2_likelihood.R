#' Stage 2 log-likelihood for Bmsm_scale model.
#'
#' Joint bivariate LL with sigma-scaling leverage. Stage 1 params held fixed.
#'
#' @param para length-3 vector [rhoe, lambda, rhom].
#' @param kbar number of frequency components.
#' @param dat N x 2 return matrix.
#' @param para1 length-8 stage 1 params [m01,m02,sigma1,sigma2,gammak,b,lev1,lev2].
#' @param n.vol trading days per year.
#' @return scalar log-likelihood (negative, for minimisation).
#' @export
Bmsm_scale_stage2_likelihood <- function(para, kbar, dat, para1, n.vol) {
  rhoe  <- para[1]
  lamda <- para[2]
  rho.m <- para[3]

  m01     <- para1[1]
  m02     <- para1[2]
  sigma1  <- para1[3] / sqrt(n.vol)
  sigma2  <- para1[4] / sqrt(n.vol)
  gamma.k <- para1[5]
  b       <- para1[6]
  lev1    <- para1[7]
  lev2    <- para1[8]

  gm <- Bmsm_states(m01, m02, kbar)

  if (kbar >= 4) {
    LL <- Bmsm_scale_ll_kron(dat, gm, rhoe, sigma1, sigma2,
                              b, gamma.k, lamda, rho.m, kbar, lev1, lev2)
  } else {
    A  <- Bmsm_A(kbar, b, gamma.k, lamda, rho.m)
    LL <- Bmsm_scale_ll_cpp(dat, A, gm, rhoe, sigma1, sigma2, lev1, lev2)
  }

  if (!is.finite(LL))
    warning("Log-likelihood is inf. Probably due to all zeros in conditional probability.")
  return(LL)
}

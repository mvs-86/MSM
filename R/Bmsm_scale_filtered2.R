#' Filtered probabilities and log-likelihood for Bmsm_scale model.
#'
#' Called after optimisation to produce full output.
#'
#' @param para length-11 param vector [m01,m02,sigma1,sigma2,gammak,b,lev1,lev2,rhoe,lambda,rhom].
#' @param kbar number of frequency components.
#' @param dat N x 2 return matrix.
#' @param n.vol trading days per year.
#' @return list with filtered.P, LL, LLs, A, g.m, para, kbar, n.
#' @export
Bmsm_scale_filtered2 <- function(para, kbar, dat, n.vol) {
  m01     <- para[1]
  m02     <- para[2]
  sigma1  <- para[3] / sqrt(n.vol)
  sigma2  <- para[4] / sqrt(n.vol)
  gamma.k <- para[5]
  b       <- para[6]
  lev1    <- para[7]
  lev2    <- para[8]
  rhoe    <- para[9]
  lamda   <- para[10]
  rho.m   <- para[11]

  gm <- Bmsm_states(m01, m02, kbar)
  A  <- Bmsm_A(kbar, b, gamma.k, lamda, rho.m)

  if (kbar >= 4) {
    likelihood <- Bmsm_scale_filtered_kron(dat, gm, rhoe, sigma1, sigma2,
                                            b, gamma.k, lamda, rho.m, kbar, lev1, lev2)
  } else {
    likelihood <- Bmsm_scale_filtered_cpp(dat, A, gm, rhoe, sigma1, sigma2, lev1, lev2)
  }

  likelihood$A    <- A
  likelihood$g.m  <- gm
  likelihood$para <- para
  likelihood$kbar <- kbar
  likelihood$n    <- n.vol
  return(likelihood)
}

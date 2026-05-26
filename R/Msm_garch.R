# Scalar negative LL for nlminb.
Msm_garch_ll <- function(para, kbar, dat, n.vol) {
  alpha     <- para[5]
  beta      <- para[6]
  gamma_gjr <- para[7]

  if (alpha + beta + gamma_gjr / 2 >= 1) return(1e10)

  m0     <- para[1]
  b      <- para[2]
  gama.k <- para[3]
  sigma  <- para[4] / sqrt(n.vol)

  g.m      <- Msm_states(m0, kbar)
  sigma_gm <- as.vector(sigma * g.m)

  if (kbar >= 8) {
    LL <- Msm_garch_ll_kron_cpp(dat, sigma_gm, b, gama.k, kbar,
                                alpha, beta, gamma_gjr)
  } else {
    A  <- Msm_A(b, gama.k, kbar)
    LL <- Msm_garch_ll_fast_cpp(dat, sigma_gm, A, alpha, beta, gamma_gjr)
  }

  if (!is.finite(LL)) warning("Log-likelihood is inf.")
  LL
}

# Full output: filtered + LL + LLs + h.
#' @export
Msm_garch_likelihood <- function(para, kbar, dat, n.vol) {
  m0        <- para[1]
  b         <- para[2]
  gama.k    <- para[3]
  sigma     <- para[4] / sqrt(n.vol)
  alpha     <- para[5]
  beta      <- para[6]
  gamma_gjr <- para[7]

  A   <- Msm_A(b, gama.k, kbar)
  g.m <- Msm_states(m0, kbar)
  sigma_gm <- as.vector(sigma * g.m)

  if (kbar >= 8) {
    out <- Msm_garch_kron_cpp(dat, sigma_gm, b, gama.k, kbar, alpha, beta, gamma_gjr)
  } else {
    out <- Msm_garch_fast_cpp(dat, sigma_gm, A, alpha, beta, gamma_gjr)
  }

  out$A   <- A
  out$g.m <- g.m
  out
}

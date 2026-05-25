#' Bivariate MSM with Sigma-Scaling Leverage Effect.
#'
#' Estimates BMSM(k) with per-series sigma-scaling leverage:
#' sigma_eff_i(t) = sigma_i * g_m * (1 + lev_i * I(r_{i,t-1} < 0)).
#' Negative own-series lagged returns scale next-period conditional volatility up.
#'
#' @param ret N x 2 return matrix.
#' @param kbar number of frequency components. Default 1.
#' @param n trading days per year. Default 252.
#' @param para0 optional starting values length 11:
#'   c(m01,m02,sigma1,sigma2,gammak,b,lev1,lev2,rhoe,lambda,rhom). Default NULL.
#' @param s.err logical; estimate standard errors? Default TRUE.
#'
#' @return a \code{bmsmmodel} object. Coefficients:
#'   m01, m02, sigma1, sigma2, gammak, b, lev1, lev2, rhoe, lambda, rhom.
#'
#' @export
Bmsm_scale <- function(ret, kbar = 1, n = 252, para0 = NULL, s.err = TRUE) {
  bmsm.check <- Bmsm_parameter_check(ret, kbar, NULL, n)
  ret  <- bmsm.check$dat
  kbar <- bmsm.check$kbar

  if (!is.null(para0)) {
    if (length(para0) != 11)
      stop("para0 must be length 11: c(m01,m02,sigma1,sigma2,gammak,b,lev1,lev2,rhoe,lambda,rhom)")
    x0    <- para0
    x0[3] <- x0[3] * sqrt(n)
    x0[4] <- x0[4] * sqrt(n)
  } else {
    base  <- bmsm.check$start.value  # [m01,m02,sigma1,sigma2,gammak,b,rhoe,lambda,rhom]
    x0    <- c(base[1:6], 0, 0, base[7:9])
    x0[3] <- x0[3] * sqrt(n)
    x0[4] <- x0[4] * sqrt(n)
  }

  lb1 <- c(1,      1,      0.00001, 0.00001, 0.001,   1, -0.99, -0.99)
  ub1 <- c(1.9999, 1.9999, 100,     100,     0.9999, 100,  10,   10)
  lb2 <- c(-0.9999, 0.0001, -0.9999)
  ub2 <- c( 0.9999, 0.9999,  0.9999)

  para1 <- x0[1:8]
  para2 <- x0[9:11]

  stage1.fit <- nlminb(para1, Bmsm_scale_stage1_likelihood,
                        lower = lb1, upper = ub1,
                        dat = ret, kbar = kbar, n.vol = n,
                        control = list(eval.max = 500, iter.max = 300))

  para1 <- stage1.fit$par

  stage2.fit <- nlminb(para2, Bmsm_scale_stage2_likelihood,
                        lower = lb2, upper = ub2,
                        kbar = kbar, dat = ret, para1 = para1, n.vol = n,
                        control = list(eval.max = 500, iter.max = 300))

  para          <- matrix(c(para1, stage2.fit$par), ncol = 1)
  bmsm.estimate <- Bmsm_scale_filtered2(c(para1, stage2.fit$par), kbar, ret, n)

  se <- matrix(NA, 11, 1)
  if (isTRUE(s.err)) se <- Bmsm_scale_std_err(c(para1, stage2.fit$par), kbar, ret, n)

  if (kbar == 1) para[6] <- NA
  coef      <- para
  coef[3:4] <- coef[3:4] / sqrt(n)
  se[3:4]   <- se[3:4] / sqrt(n)

  rownames(coef) <- c("m01","m02","sigma1","sigma2","gammak","b","lev1","lev2",
                       "rhoe","lambda","rhom")
  colnames(coef) <- "Estimate"
  colnames(se)   <- "Std. Error"

  bmsm.estimate$optim.msg <- c(stage1.message = stage1.fit$message,
                                stage2.message = stage2.fit$message)
  bmsm.estimate$optim.convergence <- c(stage1.convergence = stage1.fit$convergence,
                                        stage2.convergence = stage2.fit$convergence)
  bmsm.estimate$optim.iter <- c(stage1.iteration = stage1.fit$iterations,
                                  stage2.iteration = stage2.fit$iterations)
  bmsm.estimate$coefficients <- coef
  bmsm.estimate$call         <- match.call()
  bmsm.estimate$ret          <- ret
  bmsm.estimate$LL1          <- stage1.fit$objective
  bmsm.estimate$se           <- se

  class(bmsm.estimate) <- "bmsmmodel"
  bmsm.estimate
}

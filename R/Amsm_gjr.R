#' Asymmetric MSM with GJR-type additive leverage effect.
#'
#' Estimates univariate MSM(k) with GJR leverage:
#' Var(r_t | M_t=g_m) = sigma^2*g_m + lev*I(r_{t-1}<0)*r_{t-1}^2.
#' Negative lagged returns add to state-conditional variance.
#'
#' @param ret column matrix of returns.
#' @param kbar number of frequency components. Default 1.
#' @param n.vol trading days per year. Default 252.
#' @param para0 optional starting values c(m0,b,gammak,sigma,lev). Default NULL.
#' @param nw.lag Newey-West lags for standard errors. Default 0.
#'
#' @return an \code{amsmmodel} object with type "gjr".
#'
#' @export
Amsm_gjr <- function(ret, kbar = 1, n.vol = 252, para0 = NULL, nw.lag = 0) {

  msm.check <- Msm_parameter_check(ret, kbar,
                                    if (!is.null(para0)) para0[1:4] else NULL)
  ret  <- msm.check$dat
  kbar <- msm.check$kbar
  x0   <- c(msm.check$start.value, 0)
  lb   <- c(msm.check$lb, 0)
  ub   <- c(msm.check$ub, 10)

  if (is.null(para0)) x0[4] <- x0[4] * sqrt(n.vol)

  msm.fit <- nlminb(x0, Amsm_gjr_ll, lower = lb, upper = ub,
                    kbar = kbar, dat = ret, n.vol = n.vol,
                    control = list(eval.max = 500, iter.max = 300))

  msm.estimate <- Amsm_gjr_likelihood(msm.fit$par, kbar, ret, n.vol)
  se           <- Amsm_gjr_std_err(msm.fit$par, kbar, ret, n.vol, nw.lag)

  para <- matrix(msm.fit$par, ncol = 1)
  if (kbar == 1) para[2] <- NA

  coef    <- para
  coef[4] <- coef[4] / sqrt(n.vol)
  se[4]   <- se[4] / sqrt(n.vol)

  rownames(coef) <- c("m0", "b", "gammak", "sigma", "lev")
  colnames(coef) <- "Estimate"
  colnames(se)   <- "Std. Error"

  msm.estimate$optim.msg         <- msm.fit$message
  msm.estimate$optim.convergence <- msm.fit$convergence
  msm.estimate$optim.iter        <- msm.fit$iterations
  msm.estimate$para              <- para
  msm.estimate$se                <- se
  msm.estimate$kbar              <- kbar
  msm.estimate$n                 <- n.vol
  msm.estimate$coefficients      <- coef
  msm.estimate$call              <- match.call()
  msm.estimate$ret               <- ret
  msm.estimate$type              <- "gjr"

  class(msm.estimate) <- "amsmmodel"
  msm.estimate
}

# Objective function for nlminb.
Amsm_gjr_ll <- function(para, kbar, dat, n.vol) {
  m0     <- para[1]
  b      <- para[2]
  gama.k <- para[3]
  sigma  <- para[4] / sqrt(n.vol)
  lev    <- para[5]

  g.m      <- Msm_states(m0, kbar)
  sigma_gm <- as.vector(sigma * g.m)

  if (kbar >= 8) {
    LL <- Amsm_gjr_ll_kron_cpp(dat, sigma_gm, b, gama.k, kbar, lev)
  } else {
    A  <- Msm_A(b, gama.k, kbar)
    LL <- Amsm_gjr_ll_fast_cpp(dat, sigma_gm, A, lev)
  }

  if (!is.finite(LL)) warning("Log-likelihood is inf.")
  LL
}

# Full output: filtered probabilities + LL + LLs.
#' @export
Amsm_gjr_likelihood <- function(para, kbar, dat, n.vol) {
  m0     <- para[1]
  b      <- para[2]
  gama.k <- para[3]
  sigma  <- para[4] / sqrt(n.vol)
  lev    <- para[5]

  A   <- Msm_A(b, gama.k, kbar)
  g.m <- Msm_states(m0, kbar)
  sigma_gm <- as.vector(sigma * g.m)

  if (kbar >= 8) {
    likelihood <- Amsm_gjr_kron_cpp(dat, sigma_gm, b, gama.k, kbar, lev)
  } else {
    likelihood <- Amsm_gjr_fast_cpp(dat, sigma_gm, A, lev)
  }

  likelihood$A   <- A
  likelihood$g.m <- g.m
  likelihood
}

Amsm_gjr_grad <- function(para, kbar, ret, n.vol) {
  check_para <- function(x) {
    if (x[1] >= 2) x[1] <- 1.9999
    if (x[3] >= 1) x[3] <- 0.9999
    x
  }

  para.size <- length(para)
  para      <- as.matrix(para)
  para.abs  <- abs(para)
  para2     <- if (!all(para == 0)) para / para.abs else matrix(1, para.size, 1)
  h1        <- cbind(para.abs, matrix(1, para.size, 1) * 1e-2)
  h         <- 1e-8 * matrix(apply(h1, 1, max), ncol = 1) * para2
  para.temp <- para + h
  h         <- para.temp - para

  para_list <- c(
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] + h[i, 1]; check_para(x) }),
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] - h[i, 1]; check_para(x) })
  )

  res <- parallel::mclapply(para_list, Amsm_gjr_likelihood,
                            kbar = kbar, dat = ret, n.vol = n.vol,
                            mc.cores = getOption("mc.cores", 1L))

  ll.1 <- matrix(0, nrow(ret), para.size)
  ll.2 <- matrix(0, nrow(ret), para.size)
  for (i in seq_len(para.size)) {
    ll.1[, i] <- res[[i]]$LLs
    ll.2[, i] <- res[[para.size + i]]$LLs
  }
  (ll.1 - ll.2) / (2 * t(matrix(rep(h, nrow(ret)), ncol = nrow(ret))))
}

Amsm_gjr_hessian <- function(para, kbar, ret, n.vol) {
  check_para <- function(x) {
    if (x[1] >= 2) x[1] <- 1.9999
    if (x[3] >= 1) x[3] <- 0.9999
    x
  }

  eps       <- .Machine$double.eps
  para.size <- length(para)
  para      <- as.matrix(para)
  f.ll      <- Amsm_gjr_likelihood(check_para(para), kbar, ret, n.vol)$LL
  h         <- matrix(eps^(1/3) * apply(cbind(abs(para), 1e-8), 1, max))
  para.h    <- para + h
  h         <- para.h - para
  ee        <- diag(h[, 1], para.size)
  ij        <- expand.grid(i = seq_len(para.size), j = seq_len(para.size))

  para_list <- c(
    lapply(seq_len(para.size),  function(i) check_para(para + ee[, i])),
    lapply(seq_len(para.size),  function(i) check_para(para - ee[, i])),
    lapply(seq_len(nrow(ij)),   function(k) check_para(para + ee[, ij$i[k]] + ee[, ij$j[k]])),
    lapply(seq_len(nrow(ij)),   function(k) check_para(para - ee[, ij$i[k]] - ee[, ij$j[k]]))
  )

  res <- parallel::mclapply(para_list,
                            function(p) Amsm_gjr_likelihood(p, kbar, ret, n.vol)$LL,
                            mc.cores = getOption("mc.cores", 1L))
  res <- unlist(res)

  n2  <- para.size
  gp  <- res[seq_len(n2)]
  gm  <- res[n2 + seq_len(n2)]
  hp  <- matrix(res[2*n2 + seq_len(n2^2)], n2, n2)
  hm  <- matrix(res[2*n2 + n2^2 + seq_len(n2^2)], n2, n2)
  hh  <- h %*% t(h)
  H   <- matrix(0, para.size, para.size)

  for (i in seq_len(para.size)) {
    for (j in seq_len(para.size)) {
      H[i, j] <- (hp[i,j] - gp[i] - gp[j] + f.ll + f.ll - gm[i] - gm[j] + hm[i,j]) / hh[i,j] / 2
      H[j, i] <- H[i, j]
    }
  }
  H
}

#' @export
Amsm_gjr_std_err <- function(para, kbar, ret, n.vol, lag = 0) {
  if (kbar == 1) {
    grad <- Amsm_gjr_grad(para, kbar, ret, n.vol)
    grad <- grad[, -2]
    J    <- t(grad) %*% grad
    s    <- sqrt(diag(solve(J)))
    se   <- matrix(c(s[1], NA, s[2:4]), ncol = 1)
  } else {
    H  <- Amsm_gjr_hessian(para, kbar, ret, n.vol)
    se <- matrix(sqrt(abs(diag(solve(H)))), ncol = 1)
  }
  se
}

#' Asymmetric MSM with sigma-scaling leverage effect.
#'
#' Estimates univariate MSM(k) with leverage: sigma_eff_t = sigma*(1 + lev*I(r_{t-1}<0)).
#' Negative lagged returns scale next-period conditional sigma by (1+lev).
#'
#' @param ret column matrix of returns.
#' @param kbar number of frequency components. Default 1.
#' @param n.vol trading days per year. Default 252.
#' @param para0 optional starting values c(m0,b,gammak,sigma,lev) or c(m0,b,gammak,sigma). Default NULL.
#' @param nw.lag Newey-West lags for standard errors. Default 0.
#'
#' @return an \code{amsmmodel} object.
#'
#' @export
Amsm_scale <- function(ret, kbar = 1, n.vol = 252, para0 = NULL, nw.lag = 0) {

  msm.check <- Msm_parameter_check(ret, kbar,
                                    if (!is.null(para0)) para0[1:4] else NULL)
  ret  <- msm.check$dat
  kbar <- msm.check$kbar
  x0   <- c(msm.check$start.value, 0)
  lb   <- c(msm.check$lb, -0.99)
  ub   <- c(msm.check$ub, 10)

  if (is.null(para0)) x0[4] <- x0[4] * sqrt(n.vol)

  msm.fit <- nlminb(x0, Amsm_scale_ll, lower = lb, upper = ub,
                    kbar = kbar, dat = ret, n.vol = n.vol,
                    control = list(eval.max = 500, iter.max = 300))

  msm.estimate <- Amsm_scale_likelihood(msm.fit$par, kbar, ret, n.vol)
  se           <- Amsm_scale_std_err(msm.fit$par, kbar, ret, n.vol, nw.lag)

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
  msm.estimate$type              <- "scale"

  class(msm.estimate) <- "amsmmodel"
  msm.estimate
}

# Objective function (returns scalar negative LL) for nlminb.
Amsm_scale_ll <- function(para, kbar, dat, n.vol) {
  m0     <- para[1]
  b      <- para[2]
  gama.k <- para[3]
  sigma  <- para[4] / sqrt(n.vol)
  lev    <- para[5]

  g.m      <- Msm_states(m0, kbar)
  sigma_gm <- as.vector(sigma * g.m)

  if (kbar >= 8) {
    LL <- Amsm_scale_ll_kron_cpp(dat, sigma_gm, b, gama.k, kbar, lev)
  } else {
    A  <- Msm_A(b, gama.k, kbar)
    LL <- Amsm_scale_ll_fast_cpp(dat, sigma_gm, A, lev)
  }

  if (!is.finite(LL)) warning("Log-likelihood is inf.")
  LL
}

# Full output: filtered probabilities + LL + LLs.
#' @export
Amsm_scale_likelihood <- function(para, kbar, dat, n.vol) {
  m0     <- para[1]
  b      <- para[2]
  gama.k <- para[3]
  sigma  <- para[4] / sqrt(n.vol)
  lev    <- para[5]

  A   <- Msm_A(b, gama.k, kbar)
  g.m <- Msm_states(m0, kbar)
  sigma_gm <- as.vector(sigma * g.m)

  if (kbar >= 8) {
    likelihood <- Amsm_scale_kron_cpp(dat, sigma_gm, b, gama.k, kbar, lev)
  } else {
    likelihood <- Amsm_scale_fast_cpp(dat, sigma_gm, A, lev)
  }

  likelihood$A   <- A
  likelihood$g.m <- g.m
  likelihood
}

# Numerical gradient of per-observation LLs w.r.t. all 5 parameters.
Amsm_scale_grad <- function(para, kbar, ret, n.vol) {
  check_para <- function(x) {
    if (x[1] >= 2)  x[1] <- 1.9999
    if (x[3] >= 1)  x[3] <- 0.9999
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

  res <- parallel::mclapply(para_list, Amsm_scale_likelihood,
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

# Numerical Hessian of scalar LL w.r.t. all 5 parameters (for kbar > 1 SE).
Amsm_scale_hessian <- function(para, kbar, ret, n.vol) {
  check_para <- function(x) {
    if (x[1] >= 2) x[1] <- 1.9999
    if (x[3] >= 1) x[3] <- 0.9999
    x
  }

  eps       <- .Machine$double.eps
  para.size <- length(para)
  para      <- as.matrix(para)
  f.ll      <- Amsm_scale_likelihood(check_para(para), kbar, ret, n.vol)$LL
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
                            function(p) Amsm_scale_likelihood(p, kbar, ret, n.vol)$LL,
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

# Standard errors for Amsm_scale model.
#' @export
Amsm_scale_std_err <- function(para, kbar, ret, n.vol, lag = 0) {
  if (kbar == 1) {
    grad <- Amsm_scale_grad(para, kbar, ret, n.vol)
    grad <- grad[, -2]  # drop b (unidentified at kbar=1)
    J    <- t(grad) %*% grad
    s    <- sqrt(diag(solve(J)))
    se   <- matrix(c(s[1], NA, s[2:4]), ncol = 1)
  } else {
    H  <- Amsm_scale_hessian(para, kbar, ret, n.vol)
    se <- matrix(sqrt(abs(diag(solve(H)))), ncol = 1)
  }
  se
}

# --- S3 methods ---

#' @export
print.amsmmodel <- function(x, ...) {
  cat("Call:\n")
  print(x$call)
  cat("\nCoefficients:\n")
  print(x$coefficients, digits = 4)
  cat("\n")
  cat("LogLikelihood:", x$LL, "\n")
  cat("Type: MSM with", x$type, "leverage\n")
}

#' @export
summary.amsmmodel <- function(object, ...) {
  se      <- object$se
  tval    <- coef(object) / se
  p.value <- 2 * pt(-abs(tval), df = nrow(object$ret) - 5)

  colnames(tval)    <- "t.value"
  colnames(p.value) <- "p.value"

  TAB <- cbind(Estimate = round(coef(object), 4),
               StdErr   = round(se, 4),
               t.value  = round(tval, 4),
               p.value  = round(p.value, 4))

  res <- list(call         = object$call,
              coefficients = TAB,
              kbar         = object$kbar,
              LL           = object$LL,
              type         = object$type)
  class(res) <- "summary.amsmmodel"
  res
}

#' @export
print.summary.amsmmodel <- function(x, ...) {
  cat("*----------------------------------------------------------------------------*\n")
  cat("  Asymmetric MSM (", x$type, " leverage) with", x$kbar, "Volatility Component(s)\n")
  cat("*----------------------------------------------------------------------------*\n\n")
  printCoefmat(x$coefficients, digits = 4, P.value = TRUE, has.Pvalue = TRUE)
  cat("\nLogLikelihood:", x$LL, "\n")
}

#' @export
coef.amsmmodel <- function(object, ...) {
  co <- as.numeric(object$coefficients)
  names(co) <- rownames(object$coefficients)
  co
}

#' @export
predict.amsmmodel <- function(object, h = NULL, ...) {
  if (!is.null(h) && length(h) > 1) {
    if (any(h < 1))          stop("h must be >= 1")
    if (!all(diff(h) == 1L)) stop("h must be consecutive integers, e.g. h=1:10")
    smoothed.p <- object$filtered
    results <- lapply(h, function(hi)
      Msm_predict(object$g.m, object$para[4], object$n, smoothed.p, object$A, hi))
    return(list(
      vol    = do.call(c, lapply(results, `[[`, "vol")),
      vol.sq = do.call(c, lapply(results, `[[`, "vol.sq"))
    ))
  }

  if (!is.null(h)) {
    smoothed.p <- object$filtered
  } else {
    smoothed.p <- Msm_smooth_cpp(object$A, object$filtered)
  }
  Msm_predict(object$g.m, object$para[4], object$n, smoothed.p, object$A, h)
}

#' @export
plot.amsmmodel <- function(object, what = "vol", ...) {
  smoothed.p <- Msm_smooth_cpp(object$A, object$filtered)
  pred       <- Msm_predict(object$g.m, object$para[4], object$n, smoothed.p, object$A)

  if (what == "vol") {
    plot.df <- matrix(cbind(pred$vol, abs(object$ret)), ncol = 2)
    colnames(plot.df) <- c("Conditional Volatility", "Absolute Returns")
    plot.df <- reshape2::melt(plot.df)
    msm_plot <- ggplot2::ggplot(plot.df, ggplot2::aes(x = Var1, y = value, colour = Var2)) +
      ggplot2::geom_line() + ggplot2::xlab("Time") + ggplot2::ylab("Volatility") +
      ggplot2::ggtitle("Conditional Volatility vs Absolute Returns") +
      ggplot2::theme(legend.title = ggplot2::element_blank(), legend.position = "bottom")
  } else if (what == "volsq") {
    plot.df <- matrix(cbind(pred$vol.sq, object$ret^2), ncol = 2)
    colnames(plot.df) <- c("Conditional Variance", "Squared Returns")
    plot.df <- reshape2::melt(plot.df)
    msm_plot <- ggplot2::ggplot(plot.df, ggplot2::aes(x = Var1, y = value, colour = Var2)) +
      ggplot2::geom_line() + ggplot2::xlab("Time") + ggplot2::ylab("Variance") +
      ggplot2::ggtitle("Conditional Variance vs Squared Returns") +
      ggplot2::theme(legend.title = ggplot2::element_blank(), legend.position = "bottom")
  } else {
    stop("what must be either 'vol' or 'volsq'")
  }
  print(msm_plot)
}

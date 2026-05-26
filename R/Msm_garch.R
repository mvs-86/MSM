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

# Numerical gradient of per-obs LLs w.r.t. all 7 parameters.
Msm_garch_grad <- function(para, kbar, ret, n.vol) {
  check_para <- function(x) {
    if (x[1] >= 2) x[1] <- 1.9999
    if (x[3] >= 1) x[3] <- 0.9999
    x
  }

  para.size <- length(para)
  para      <- as.matrix(para)
  para.abs  <- abs(para)
  # For zero parameters use sign +1 (step forward); avoids 0/0 = NaN.
  para2     <- ifelse(para == 0, 1, para / para.abs)
  h1        <- cbind(para.abs, matrix(1, para.size, 1) * 1e-2)
  h         <- 1e-8 * matrix(apply(h1, 1, max), ncol = 1) * para2
  para.temp <- para + h
  h         <- para.temp - para

  para_list <- c(
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] + h[i, 1]; check_para(x) }),
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] - h[i, 1]; check_para(x) })
  )

  res <- parallel::mclapply(para_list, Msm_garch_likelihood,
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

# Numerical 2-sided Hessian of scalar LL w.r.t. all 7 parameters.
Msm_garch_hessian <- function(para, kbar, ret, n.vol) {
  check_para <- function(x) {
    if (x[1] >= 2) x[1] <- 1.9999
    if (x[3] >= 1) x[3] <- 0.9999
    x
  }

  eps       <- .Machine$double.eps
  para.size <- length(para)
  para      <- as.matrix(para)
  f.ll      <- Msm_garch_likelihood(check_para(para), kbar, ret, n.vol)$LL
  h         <- matrix(eps^(1/3) * apply(cbind(abs(para), 1e-8), 1, max))
  para.h    <- para + h
  h         <- para.h - para
  ee        <- diag(h[, 1], para.size)
  ij        <- expand.grid(i = seq_len(para.size), j = seq_len(para.size))

  para_list <- c(
    lapply(seq_len(para.size), function(i) check_para(para + ee[, i])),
    lapply(seq_len(para.size), function(i) check_para(para - ee[, i])),
    lapply(seq_len(nrow(ij)),  function(k) check_para(para + ee[, ij$i[k]] + ee[, ij$j[k]])),
    lapply(seq_len(nrow(ij)),  function(k) check_para(para - ee[, ij$i[k]] - ee[, ij$j[k]]))
  )

  res <- parallel::mclapply(para_list,
                            function(p) Msm_garch_likelihood(p, kbar, ret, n.vol)$LL,
                            mc.cores = getOption("mc.cores", 1L))
  res <- unlist(res)

  n2 <- para.size
  gp <- res[seq_len(n2)]
  gm <- res[n2 + seq_len(n2)]
  hp <- matrix(res[2*n2 + seq_len(n2^2)], n2, n2)
  hm <- matrix(res[2*n2 + n2^2 + seq_len(n2^2)], n2, n2)
  hh <- h %*% t(h)
  H  <- matrix(0, para.size, para.size)

  for (i in seq_len(para.size)) {
    for (j in seq_len(para.size)) {
      H[i, j] <- (hp[i,j] - gp[i] - gp[j] + f.ll + f.ll - gm[i] - gm[j] + hm[i,j]) / hh[i,j] / 2
      H[j, i] <- H[i, j]
    }
  }
  H
}

#' Hybrid MSM-GJR-GARCH Volatility Model.
#'
#' Estimates a univariate MSM(k) model with a multiplicative GJR-GARCH(1,1)
#' short-run component: sigma2_t = sigma^2 * g_m^2 * h_t, where h_t is a
#' unit-mean GJR-GARCH transitory factor.
#'
#' @param ret column matrix of returns.
#' @param kbar number of MSM frequency components. Default 1.
#' @param n.vol trading days per year. Default 252.
#' @param para0 optional 7-element starting vector c(m0,b,gammak,sigma,alpha,beta,gamma_gjr). Default NULL.
#' @param nw.lag Newey-West lags (kbar=1 only). Default 0.
#'
#' @return an \code{msmgarchmodel} object.
#'
#' @export
Msm_garch <- function(ret, kbar = 1, n.vol = 252, para0 = NULL, nw.lag = 0) {

  chk  <- Msm_garch_parameter_check(ret, kbar, para0, n.vol)
  ret  <- chk$dat
  kbar <- chk$kbar
  x0   <- chk$start.value
  lb   <- chk$lb
  ub   <- chk$ub

  # Msm_garch_parameter_check already returns sigma annualised (sd * sqrt(252)).
  # Do NOT multiply by sqrt(n.vol) again.

  fit <- nlminb(x0, Msm_garch_ll, lower = lb, upper = ub,
                kbar = kbar, dat = ret, n.vol = n.vol,
                control = list(eval.max = 1000, iter.max = 500))

  est <- Msm_garch_likelihood(fit$par, kbar, ret, n.vol)
  se  <- Msm_garch_std_err(fit$par, kbar, ret, n.vol, nw.lag)

  para <- matrix(fit$par, ncol = 1)
  if (kbar == 1) para[2] <- NA

  coef    <- para
  coef[4] <- coef[4] / sqrt(n.vol)
  se[4]   <- se[4] / sqrt(n.vol)

  rownames(coef) <- c("m0", "b", "gammak", "sigma", "alpha", "beta", "gamma_gjr")
  colnames(coef) <- "Estimate"
  colnames(se)   <- "Std. Error"

  est$optim.msg         <- fit$message
  est$optim.convergence <- fit$convergence
  est$optim.iter        <- fit$iterations
  est$para              <- para
  est$se                <- se
  est$kbar              <- kbar
  est$n                 <- n.vol
  est$coefficients      <- coef
  est$call              <- match.call()
  est$ret               <- ret

  class(est) <- "msmgarchmodel"
  est
}

#' @export
print.msmgarchmodel <- function(x, ...) {
  cat("Call:\n")
  print(x$call)
  cat("\nCoefficients:\n")
  print(x$coefficients, digits = 4)
  cat("\nLogLikelihood:", x$LL, "\n")
}

#' @export
summary.msmgarchmodel <- function(object, ...) {
  se      <- object$se
  tval    <- coef(object) / se
  p.value <- 2 * pt(-abs(tval), df = nrow(object$ret) - 7)

  colnames(tval)    <- "t.value"
  colnames(p.value) <- "p.value"

  TAB <- cbind(Estimate = round(coef(object), 4),
               StdErr   = round(se, 4),
               t.value  = round(tval, 4),
               p.value  = round(p.value, 4))

  res <- list(call         = object$call,
              coefficients = TAB,
              kbar         = object$kbar,
              LL           = object$LL)
  class(res) <- "summary.msmgarchmodel"
  res
}

#' @export
print.summary.msmgarchmodel <- function(x, ...) {
  cat("*------------------------------------------------------*\n")
  cat("  MSM-GJR-GARCH with", x$kbar, "MSM Volatility Component(s)\n")
  cat("*------------------------------------------------------*\n\n")
  printCoefmat(x$coefficients, digits = 4, P.value = TRUE, has.Pvalue = TRUE)
  cat("\nLogLikelihood:", x$LL, "\n")
}

#' @export
coef.msmgarchmodel <- function(object, ...) {
  co <- as.numeric(object$coefficients)
  names(co) <- rownames(object$coefficients)
  co
}

#' @export
predict.msmgarchmodel <- function(object, h = NULL, ...) {
  # Fitted conditional volatility: sqrt(sigma^2 * E[g_m^2 | filtered] * h_t)
  # h-step forecast: MSM propagation only (h_t -> 1 as h -> inf under stationarity)
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
    # h-step-ahead MSM forecast (h_t -> 1 under stationarity, MSM-only)
    smoothed.p <- object$filtered
    return(Msm_predict(object$g.m, object$para[4], object$n, smoothed.p, object$A, h))
  }
  # In-sample fitted values: full MSM-GARCH combined volatility
  sigma      <- object$para[4] / sqrt(object$n)
  smoothed.p <- Msm_smooth_cpp(object$A, object$filtered)
  e_gm2      <- as.vector(smoothed.p %*% t(object$g.m^2))
  vol        <- sigma * sqrt(e_gm2 * object$h)
  list(vol = vol, vol.sq = vol^2)
}

#' @export
plot.msmgarchmodel <- function(object, what = "vol", ...) {
  sigma    <- object$para[4] / sqrt(object$n)
  g.m      <- object$g.m
  smoothed <- Msm_smooth_cpp(object$A, object$filtered)
  h        <- object$h

  # Fitted conditional vol: sigma * sqrt(E[g_m^2 | smoothed] * h_t)
  e_gm2    <- as.vector(smoothed %*% t(g.m^2))
  cond_vol <- sigma * sqrt(e_gm2 * as.vector(h))

  if (what == "vol") {
    plot.df <- matrix(cbind(cond_vol, abs(object$ret)), ncol = 2)
    colnames(plot.df) <- c("Conditional Volatility", "Absolute Returns")
    plot.df <- reshape2::melt(plot.df)
    msm_plot <- ggplot2::ggplot(plot.df, ggplot2::aes(x = Var1, y = value, colour = Var2)) +
      ggplot2::geom_line() + ggplot2::xlab("Time") + ggplot2::ylab("Volatility") +
      ggplot2::ggtitle("MSM-GJR-GARCH: Conditional Volatility vs Absolute Returns") +
      ggplot2::theme(legend.title = ggplot2::element_blank(), legend.position = "bottom")
  } else if (what == "volsq") {
    plot.df <- matrix(cbind(cond_vol^2, object$ret^2), ncol = 2)
    colnames(plot.df) <- c("Conditional Variance", "Squared Returns")
    plot.df <- reshape2::melt(plot.df)
    msm_plot <- ggplot2::ggplot(plot.df, ggplot2::aes(x = Var1, y = value, colour = Var2)) +
      ggplot2::geom_line() + ggplot2::xlab("Time") + ggplot2::ylab("Variance") +
      ggplot2::ggtitle("MSM-GJR-GARCH: Conditional Variance vs Squared Returns") +
      ggplot2::theme(legend.title = ggplot2::element_blank(), legend.position = "bottom")
  } else {
    stop("what must be 'vol' or 'volsq'")
  }
  print(msm_plot)
}

# Standard errors.
#' @export
Msm_garch_std_err <- function(para, kbar, ret, n.vol, lag = 0) {
  if (kbar == 1) {
    grad <- Msm_garch_grad(para, kbar, ret, n.vol)
    grad <- grad[, -2]   # drop b column (unidentified at kbar=1)
    J    <- t(grad) %*% grad
    Ji   <- tryCatch(solve(J), error = function(e) {
      warning("OPG matrix singular; using MASS::ginv fallback.")
      MASS::ginv(J)
    })
    s    <- sqrt(diag(Ji))
    se   <- matrix(c(s[1], NA, s[2:6]), ncol = 1)
  } else {
    H  <- Msm_garch_hessian(para, kbar, ret, n.vol)
    Hi <- tryCatch(solve(H), error = function(e) {
      warning("Hessian singular; using MASS::ginv fallback.")
      MASS::ginv(H)
    })
    se <- matrix(sqrt(abs(diag(Hi))), ncol = 1)
  }
  return(se)
}

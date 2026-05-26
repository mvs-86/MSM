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
  para2     <- if (!all(para == 0)) para / para.abs else matrix(1, para.size, 1)
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

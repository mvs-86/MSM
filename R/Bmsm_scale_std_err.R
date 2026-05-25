#' Sandwich standard errors for Bmsm_scale model.
#'
#' @param para length-11 param vector [m01,m02,sigma1,sigma2,gammak,b,lev1,lev2,rhoe,lambda,rhom].
#' @param kbar number of frequency components.
#' @param ret N x 2 return matrix.
#' @param n.vol trading days per year.
#' @return 11 x 1 matrix of standard errors (NA for b when kbar=1).
#' @export
Bmsm_scale_std_err <- function(para, kbar, ret, n.vol) {
  grad1 <- Bmsm_grad("Bmsm_scale_stage1_LLs",
                      arg.list = list(para = para[1:8], kbar = kbar,
                                      dat = ret, n.vol = n.vol))
  grad2 <- Bmsm_grad("Bmsm_scale_stage2_LLs",
                      arg.list = list(para = para[9:11], kbar = kbar,
                                      dat = ret, n.vol = n.vol, para1 = para[1:8]))

  H1 <- Bmsm_hessian_2_sided("Bmsm_scale_stage1hess_LL",
                               arg.list = list(para = para[1:8], kbar = kbar,
                                               dat = ret, n.vol = n.vol))
  H2 <- Bmsm_hessian_2_sided("Bmsm_scale_stage2hess_LL",
                               arg.list = list(para = para[9:11], kbar = kbar,
                                               dat = ret, n.vol = n.vol, para1 = para[1:8]))

  if (kbar == 1) {
    grad1 <- grad1[, -6]   # drop b column
    H1    <- H1[-6, -6]    # drop b row/col
  }

  N  <- nrow(ret)
  J1 <- t(grad1) %*% grad1
  J2 <- t(grad2) %*% grad2

  safe_solve <- function(M) {
    tryCatch(solve(M), error = function(e) {
      warning("Bmsm_scale_std_err: Hessian is singular; using pseudo-inverse. SEs may be unreliable.")
      MASS::ginv(M)
    })
  }

  d1 <- diag(safe_solve(H1/N) %*% (J1/N^2) %*% safe_solve(H1/N))
  if (any(d1 < 0)) warning("Bmsm_scale_std_err: negative sandwich variance; SE may be unreliable.")
  se1 <- sqrt(abs(d1))

  d2 <- diag(safe_solve(H2/N) %*% (J2/N^2) %*% safe_solve(H2/N))
  if (any(d2 < 0)) warning("Bmsm_scale_std_err: negative sandwich variance; SE may be unreliable.")
  se2 <- sqrt(abs(d2))

  if (kbar == 1)
    se1 <- c(se1[1:5], NA, se1[6:7])  # re-insert NA for b at position 6

  se <- matrix(c(se1, se2), ncol = 1)
  return(se)
}

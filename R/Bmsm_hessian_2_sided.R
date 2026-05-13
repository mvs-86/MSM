#' Finite Difference Hessian for \code{\link{Bmsm}}(k) model.
#'
#' Calculates 2-sided finite difference hessian for \code{\link{Bmsm}}(k) model.
#'
#' @param func is a function handle.
#' @param arg.list is a list of arguments for function "func".
#'
#' @return a  matrix of numerical hessians.
#'
#'
#' @export
Bmsm_hessian_2_sided <-function (func, arg.list){

  para <- arg.list$para
  kbar <- arg.list$kbar
  dat  <- arg.list$dat
  n.vol<- arg.list$n.vol

  eps <- .Machine$double.eps

  para.size <- length(para)
  para      <- as.matrix(para)

  #f.ll <- -Msm_likelihood2(check_para(para), kbar, dat, n.vol)$LL
  f.ll <- do.call(func, arg.list)

  h <- matrix(eps^(1/3)*apply(cbind(abs(para),1e-8),1,max))

  para.h <- para+h

  h <- para.h-para

  ee <- diag(h[,1], para.size)

  hh <- h %*% t(h)

  ij <- expand.grid(i = seq_len(para.size), j = seq_len(para.size))

  para_list <- c(
    lapply(seq_len(para.size),  function(i) para + ee[,i]),
    lapply(seq_len(para.size),  function(i) para - ee[,i]),
    lapply(seq_len(nrow(ij)),   function(k) para + ee[,ij$i[k]] + ee[,ij$j[k]]),
    lapply(seq_len(nrow(ij)),   function(k) para - ee[,ij$i[k]] - ee[,ij$j[k]])
  )

  res <- parallel::mclapply(para_list, function(p) {
    arg <- arg.list; arg$para <- p; do.call(func, arg)
  }, mc.cores = getOption("mc.cores", 1L))

  res <- unlist(res)
  n2  <- para.size
  gp  <- res[seq_len(n2)]
  gm  <- res[n2 + seq_len(n2)]
  hp  <- matrix(res[2*n2 + seq_len(n2^2)], n2, n2)
  hm  <- matrix(res[2*n2 + n2^2 + seq_len(n2^2)], n2, n2)

  H <- matrix(0, para.size, para.size)

  for (i in seq_len(para.size)) {
    for (j in seq_len(para.size)) {
      H[i,j] <- (hp[i,j] - gp[i] - gp[j] + f.ll + f.ll - gm[i] - gm[j] + hm[i,j]) / hh[i,j] / 2
      H[j,i] <- H[i,j]
    }
  }

  return(H)
}

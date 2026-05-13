#' Numerical Derivative Function for \code{\link{Msm}}(k) model.
#'
#' Calculates the numerical derivative for \code{\link{Msm}}(k) model.
#'
#' @param para is a vector of parameters returned by \code{\link{Msm}}.
#' @param kbar is the number of frequency components in the \code{\link{Msm}}(k) model.
#' @param ret is a column matrix of returns.
#' @param n.vol is the number of trading days in a year, if data is on a daily frequency. Default is 252.
#'
#' @return a \eqn{N=nrow(ret)} by \eqn{4} matrix of numerical derivatives.
#'
#'
#' @export

Msm_grad <- function(para, kbar, ret,n.vol){


  check_para <- function(x){
    if (x[1]>=2) x[1] <- 1.9999
    if (x[3]>=1) x[3] <- 0.9999

    return(x)
  }

  para.size <- length(para)
  para      <- as.matrix(para)

  para.abs <- abs(para)

  if (!all(para==0)) {
    para2 <- para/para.abs
  } else {
    para2 <- 1
  }

  h1 <- cbind(para.abs, matrix(1,para.size,1)*1e-2)
  h  <- 1e-8*matrix(apply(h1,1,max),ncol=1)*para2

  para.temp <- para + h

  h <- para.temp - para

  ll.1 <- matrix(0,nrow(ret),para.size)
  ll.2 <- matrix(0,nrow(ret),para.size)

  para_list <- c(
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] + h[i,1]; check_para(x) }),
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] - h[i,1]; check_para(x) })
  )

  res <- parallel::mclapply(para_list, Msm_likelihood2,
                            kbar = kbar, dat = ret, n.vol = n.vol,
                            mc.cores = getOption("mc.cores", 1L))

  for (i in seq_len(para.size)) {
    ll.1[,i] <- res[[i]]$LLs
    ll.2[,i] <- res[[para.size + i]]$LLs
  }

  der <- (ll.1-ll.2)/(2*t(matrix(rep(h,nrow(ret)),ncol=nrow(ret))))
  return(der)

}

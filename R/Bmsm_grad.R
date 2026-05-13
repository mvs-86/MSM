#' Numerical Derivative Function for \code{\link{Bmsm}}(k) model.
#'
#' Calculates the numerical derivative for \code{\link{Bmsm}}(k) model.
#'
#' @param func is a function handle.
#' @param arg.list is a list of arguments for function "func".
#'
#' @return a  matrix of numerical derivatives.
#'
#'
#' @export
Bmsm_grad <- function(func, arg.list){

  #function(func, para, kbar, dat,n.vol)
  para <- arg.list$para
  kbar <- arg.list$kbar
  dat  <- arg.list$dat
  n.vol<- arg.list$n.vol

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

  ll.1 <- matrix(0,nrow(dat),para.size)
  ll.2 <- matrix(0,nrow(dat),para.size)

  para_list <- c(
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] + h[i,1]; x }),
    lapply(seq_len(para.size), function(i) { x <- para; x[i] <- x[i] - h[i,1]; x })
  )

  res <- parallel::mclapply(para_list, function(p) {
    arg <- arg.list; arg$para <- p; do.call(func, arg)
  }, mc.cores = getOption("mc.cores", 1L))

  for (i in seq_len(para.size)) {
    ll.1[,i] <- res[[i]]
    ll.2[,i] <- res[[para.size + i]]
  }

  der <- sweep(ll.1 - ll.2, 2, 2 * h[,1], "/")
  return(der)

}

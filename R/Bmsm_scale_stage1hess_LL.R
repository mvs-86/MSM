Bmsm_scale_stage1hess_LL <- function(para, kbar, dat, n.vol) {
  if (para[1] >= 2) para[1] <- 1.9999
  if (para[2] >= 2) para[2] <- 1.9999
  if (para[5] >= 1) para[5] <- 0.9999

  ret1 <- matrix(dat[, 1], ncol = 1)
  ret2 <- matrix(dat[, 2], ncol = 1)
  par1 <- c(para[1], para[6], para[5], para[3], para[7])
  par2 <- c(para[2], para[6], para[5], para[4], para[8])

  LL <- Amsm_scale_likelihood(par1, kbar, ret1, n.vol)$LL +
        Amsm_scale_likelihood(par2, kbar, ret2, n.vol)$LL
  return(LL)
}

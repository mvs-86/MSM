Bmsm_scale_stage2_LLs <- function(para, kbar, dat, n.vol, para1) {
  if (para[1] >=  1) para[1] <-  0.9999
  if (para[1] <= -1) para[1] <- -0.9999
  if (para[2] >=  1) para[2] <-  0.9999
  if (para[3] >=  1) para[3] <-  0.9999
  if (para[3] <= -1) para[3] <- -0.9999

  rhoe  <- para[1]; lamda <- para[2]; rho.m <- para[3]
  m01     <- para1[1]; m02     <- para1[2]
  sigma1  <- para1[3] / sqrt(n.vol); sigma2  <- para1[4] / sqrt(n.vol)
  gamma.k <- para1[5]; b <- para1[6]; lev1 <- para1[7]; lev2 <- para1[8]

  gm <- Bmsm_states(m01, m02, kbar)

  if (kbar >= 4) {
    LLs <- Bmsm_scale_filtered_kron(dat, gm, rhoe, sigma1, sigma2,
                                     b, gamma.k, lamda, rho.m, kbar, lev1, lev2)$LLs
  } else {
    A   <- Bmsm_A(kbar, b, gamma.k, lamda, rho.m)
    LLs <- Bmsm_scale_filtered_cpp(dat, A, gm, rhoe, sigma1, sigma2, lev1, lev2)$LLs
  }
  return(LLs)
}

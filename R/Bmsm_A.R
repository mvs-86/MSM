#' Transition Probability Matrix for \code{\link{Bmsm}}(k) Model.
#'
#' Calculates transition matrix for \code{\link{Bmsm}}(k) model.
#
#' @param kbar is the number of frequency components in the \code{\link{Bmsm}}(k) model.
#' @param b is the  growth rate of the switching probabilities of the volatility components and
#' must be in the range (1, inf).
#' @param gamma.kbar is the transition probability of the highest frequency component
#' and must be in the range (0,1).
#' @param lamda in [0, 1] is the unconditional correlation between volatility arrivals.
#' @param rho.m in [-1, 1] is the correlation between volatility components \eqn{M_a} and \eqn{M_b}

#'
#' @return a \eqn{4^k} by \eqn{4^k} transition matrix
#'
#' @examples
#' Bmsm_A(2, 12.33, 0.151, 0.818, 0.9999)
#'
#' @export
Bmsm_A <- function(kbar, b, gamma.kbar, lamda, rho.m){
  Bmsm_A_cpp(kbar, b, gamma.kbar, lamda, rho.m)
}

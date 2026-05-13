#' Volatility State Vector For \code{\link{Msm}} model.
#'
#' Calculates all possible \eqn{2^k} state values for an  \code{\link{Msm}}(k) model.
#'
#' @param m0 is the state variable value.
#' @param kbar is the number of frequency components in the \code{\link{Msm}}(k) model.
#'
#' @return a \eqn{2^k} state vector.
#'
#' @examples
#' Msm_states(1.5, 2)
#'
#' @export
Msm_states <- function(m0, kbar){
  Msm_states_cpp(m0, kbar)
}

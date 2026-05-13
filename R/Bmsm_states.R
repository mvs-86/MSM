#' Volatility State Vector For \code{\link{Bmsm}} model.
#'
#' Calculates all possible \eqn{4^k} state values for an  \code{\link{Bmsm}}(k) model.
#'
#' @param m01 is the state variable value for series 1.
#' @param m02 is the state variable value for series 2.
#' @param kbar is the number of frequency components in the \code{\link{Bmsm}}(k) model.
#'
#' @return a  2-by-\eqn{4^k} state matrix.
#'
#' @examples
#' s <- Bmsm_states(1.5, 1.7, 2)
#'
#' @export
Bmsm_states <- function(m01, m02, kbar){
  Bmsm_states_cpp(m01, m02, kbar)
}

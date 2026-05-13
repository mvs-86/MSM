#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace Rcpp;

// [[Rcpp::export]]
List Msm_likelihood_fast(const arma::vec& dat, const arma::rowvec& sigma_gm, const arma::mat& A) {
	int N = dat.n_rows, k = sigma_gm.n_elem;
	const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
	double pidenom, ll = 0.0;

	arma::mat filtered(N, k, arma::fill::zeros);
	arma::rowvec pi_t(k), piA(k), omega_t(k), pinum(k);
	arma::colvec LLs(N, arma::fill::zeros);

	pi_t.fill(1.0 / k);

	for (int i = 0; i < N; i++) {
		piA = pi_t * A;
		omega_t = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_gm)) / sigma_gm + 1e-16;
		pinum = omega_t % piA;
		pidenom = arma::accu(pinum);
		LLs(i) = std::log(pidenom);
		if (pidenom == 0.0) {
			pi_t.zeros();
			pi_t(0) = 1.0;
		} else {
			pi_t = pinum / pidenom;
		}
		filtered.row(i) = pi_t;
	}

	ll = -arma::accu(LLs);
	return List::create(
		Named("filtered") = filtered,
		Named("LL") = ll,
		Named("LLs") = LLs);
}

// [[Rcpp::export]]
List Msm_likelihood_cpp(const arma::mat& pimat0, const arma::mat& omegat, const arma::mat& A) {
	int T = omegat.n_rows+1, k = omegat.n_cols;
	double pidenom = 0.0, ll=0.0;
	arma::mat piA(1, k);
	arma::mat pinum(1, k);
	arma::mat pmat(T, k);
	arma::colvec LLs(T - 1);

	LLs.zeros();
	piA.zeros();
	pinum.zeros();
	pmat.zeros();
	pmat.row(0) = pimat0;


	for (int i = 1; i<T; i++) {
		piA = pmat.row(i - 1)*A;
		pinum = omegat.row(i - 1) % piA;
		pidenom = arma::as_scalar(omegat.row(i - 1)*piA.t());

		if (pidenom == 0) {
			pmat(i, 0) = 1;
		}
		else {
			pmat.row(i) = pinum / pidenom;
		}

		LLs(i - 1) = log(pidenom);
	}

	ll = accu(LLs);
	ll = -1 * ll;
	return List::create(Named("pmat") = pmat,
		Named("LL") = ll,
		Named("LLs") = LLs);
}
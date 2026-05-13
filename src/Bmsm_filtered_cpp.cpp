#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace Rcpp;

// [[Rcpp::export]]
List Bmsm_filtered_cpp(const arma::mat& dat, const arma::mat& A, const arma::mat& gm,
	const double& rhoe, const double& sigma1, const double& sigma2) {
	int T = dat.n_rows, k = A.n_cols;
	double pa       = 1.0 / (2.0 * M_PI * std::sqrt(1.0 - rhoe * rhoe));
	double inv_2var = 1.0 / (2.0 * (1.0 - rhoe * rhoe));

	const arma::rowvec sg1  = sigma1 * gm.row(0);
	const arma::rowvec sg2  = sigma2 * gm.row(1);
	const arma::rowvec sg12 = sg1 % sg2;

	arma::mat P(T, k, arma::fill::zeros);
	arma::rowvec piA(k), C(k), r1(k), r2(k), w(k);
	arma::rowvec pia(k, arma::fill::zeros);
	arma::colvec LLs(T, arma::fill::zeros);

	piA.fill(1.0 / k);
	piA = piA * A;
	pia(0) = 1.0;

	for (int t = 0; t < T; t++) {
		r1 = dat(t, 0) / sg1;
		r2 = dat(t, 1) / sg2;
		w  = pa * arma::exp(-(r1 % r1 + r2 % r2 - 2.0 * rhoe * (r1 % r2)) * inv_2var) / sg12 + 1e-16;

		C = w % piA;
		double ft = arma::accu(C);
		LLs(t) = std::log(ft);

		if (ft == 0.0) { piA = pia * A; } else { piA = (C / ft) * A; }
		P.row(t) = piA;
	}

	return List::create(
		Named("filtered.P") = P,
		Named("LL")         = -arma::accu(LLs),
		Named("LLs")        = LLs);
}

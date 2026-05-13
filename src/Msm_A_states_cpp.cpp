#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat Msm_A_cpp(double b, double gamma_kbar, int kbar) {
	arma::vec gamma_k(kbar, arma::fill::zeros);
	gamma_k(0) = 1.0 - std::pow(1.0 - gamma_kbar, 1.0 / std::pow(b, kbar - 1));

	double g = gamma_k(0);
	arma::mat A = { {1.0 - 0.5*g, 0.5*g}, {0.5*g, 1.0 - 0.5*g} };

	for (int i = 1; i < kbar; i++) {
		gamma_k(i) = 1.0 - std::pow(1.0 - gamma_k(0), std::pow(b, i));
		double gi = gamma_k(i);
		arma::mat a = { {1.0 - 0.5*gi, 0.5*gi}, {0.5*gi, 1.0 - 0.5*gi} };
		A = arma::kron(A, a);
	}

	return A;
}

// [[Rcpp::export]]
arma::rowvec Msm_states_cpp(double m0, int kbar) {
	double m1 = 2.0 - m0;
	int k2 = 1 << kbar;
	arma::rowvec gm(k2);

	for (int i = 0; i < k2; i++) {
		double g = 1.0;
		for (int j = 0; j < kbar; j++) {
			g *= ((i & (1 << j)) != 0) ? m1 : m0;
		}
		gm(i) = std::sqrt(g);
	}

	return gm;
}

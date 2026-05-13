#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace Rcpp;

static arma::mat bmsm_transition(double gama, double lamda, double rho_m) {
	double base = gama * ((1.0 - lamda) * gama + lamda);
	double p  = 1.0 - gama + base * ((1.0 + rho_m) / 4.0);
	double q  = 1.0 - gama + base * ((1.0 - rho_m) / 4.0);
	double g2 = gama / 2.0;

	arma::mat T(4, 4);
	T(0,0)=p;        T(0,1)=1-g2-p;   T(0,2)=1-g2-p;   T(0,3)=gama-1+p;
	T(1,0)=1-g2-q;   T(1,1)=q;        T(1,2)=gama-1+q;  T(1,3)=1-g2-q;
	T(2,0)=1-g2-q;   T(2,1)=gama-1+q; T(2,2)=q;         T(2,3)=1-g2-q;
	T(3,0)=gama-1+p; T(3,1)=1-g2-p;   T(3,2)=1-g2-p;    T(3,3)=p;
	return T;
}

// [[Rcpp::export]]
arma::mat Bmsm_A_cpp(int kbar, double b, double gamma_kbar, double lamda, double rho_m) {
	arma::vec gamma_k(kbar, arma::fill::zeros);
	gamma_k(0) = 1.0 - std::pow(1.0 - gamma_kbar, 1.0 / std::pow(b, kbar - 1));

	arma::mat A = bmsm_transition(gamma_k(0), lamda, rho_m);

	for (int i = 1; i < kbar; i++) {
		gamma_k(i) = 1.0 - std::pow(1.0 - gamma_k(0), std::pow(b, i));
		arma::mat a = bmsm_transition(gamma_k(i), lamda, rho_m);
		A = arma::kron(a, A);
	}

	return A;
}

// [[Rcpp::export]]
arma::mat Bmsm_states_cpp(double m01, double m02, int kbar) {
	int k4 = 1 << (2 * kbar);  // 4^kbar
	arma::mat gm(2, k4);

	for (int i = 0; i < k4; i++) {
		double g1 = 1.0, g2 = 1.0;
		for (int j = 0; j < kbar; j++) {
			g1 *= ((i >> (2*j + 1)) & 1) ? (2.0 - m01) : m01;
			g2 *= ((i >> (2*j))     & 1) ? (2.0 - m02) : m02;
		}
		gm(0, i) = std::sqrt(g1);
		gm(1, i) = std::sqrt(g2);
	}

	return gm;
}

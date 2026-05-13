#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace Rcpp;

static inline void bkron_apply(arma::rowvec& pi, const arma::vec& gamma_k,
                                double lamda, double rho_m) {
	int kbar = gamma_k.n_elem;
	int n    = pi.n_elem;
	for (int k = 0; k < kbar; k++) {
		double g    = gamma_k(k);
		double bv   = g * ((1.0 - lamda) * g + lamda);
		double g2   = 0.5 * g;
		double p    = 1.0 - g  + bv * (1.0 + rho_m) * 0.25;
		double q    = 1.0 - g  + bv * (1.0 - rho_m) * 0.25;
		double t00  = p,         t01 = 1.0 - g2 - p, t03 = g - 1.0 + p;
		double t11  = q,         t10 = 1.0 - g2 - q, t12 = g - 1.0 + q;
		int stride = 1;
		for (int s = 0; s < k; s++) stride *= 4;
		int block = stride * 4;
		for (int base2 = 0; base2 < n; base2 += block) {
			for (int i = 0; i < stride; i++) {
				double v0 = pi(base2 + i);
				double v1 = pi(base2 + i + stride);
				double v2 = pi(base2 + i + 2*stride);
				double v3 = pi(base2 + i + 3*stride);
				double s03 = v0 + v3, s12 = v1 + v2;
				pi(base2 + i)            = t00*v0 + t10*s12 + t03*v3;
				pi(base2 + i + stride)   = t01*s03 + t11*v1 + t12*v2;
				pi(base2 + i + 2*stride) = t01*s03 + t12*v1 + t11*v2;
				pi(base2 + i + 3*stride) = t03*v0 + t10*s12 + t00*v3;
			}
		}
	}
}

static inline arma::vec bmsm_make_gamma_k(double b, double gamma_kbar, int kbar) {
	arma::vec gamma_k(kbar);
	gamma_k(0) = 1.0 - std::pow(1.0 - gamma_kbar, 1.0 / std::pow(b, kbar - 1));
	for (int i = 1; i < kbar; i++)
		gamma_k(i) = 1.0 - std::pow(1.0 - gamma_k(0), std::pow(b, i));
	return gamma_k;
}

// [[Rcpp::export]]
NumericVector Bmsm_stage2_ll_kron(const arma::mat& dat, const arma::mat& gm,
	const double& rhoe, const double& sigma1, const double& sigma2,
	double b, double gamma_kbar, double lamda, double rho_m, int kbar) {
	int T = dat.n_rows, k = 1 << (2 * kbar);
	double pa       = 1.0 / (2.0 * M_PI * std::sqrt(1.0 - rhoe * rhoe));
	double inv_2var = 1.0 / (2.0 * (1.0 - rhoe * rhoe));

	const arma::vec  gamma_k = bmsm_make_gamma_k(b, gamma_kbar, kbar);
	const arma::rowvec sg1  = sigma1 * gm.row(0);
	const arma::rowvec sg2  = sigma2 * gm.row(1);
	const arma::rowvec sg12 = sg1 % sg2;

	arma::rowvec pi_t(k), C(k), r1(k), r2(k), w(k);
	arma::colvec LLs(T, arma::fill::zeros);

	pi_t.fill(1.0 / k);
	bkron_apply(pi_t, gamma_k, lamda, rho_m);

	for (int t = 0; t < T; t++) {
		r1 = dat(t, 0) / sg1;
		r2 = dat(t, 1) / sg2;
		w  = pa * arma::exp(-(r1 % r1 + r2 % r2 - 2.0 * rhoe * (r1 % r2)) * inv_2var) / sg12 + 1e-16;

		C = w % pi_t;
		double ft = arma::accu(C);
		LLs(t) = std::log(ft);

		if (ft == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
		else           { pi_t = C / ft; }
		bkron_apply(pi_t, gamma_k, lamda, rho_m);
	}

	return NumericVector::create(Named("LL") = -arma::accu(LLs));
}

// [[Rcpp::export]]
NumericVector Bmsm_stage2_ll_cpp(const arma::mat& dat, const arma::mat& A, const arma::mat& gm,
	const double& rhoe, const double& sigma1, const double& sigma2) {
	int T = dat.n_rows, k = A.n_cols;
	double pa       = 1.0 / (2.0 * M_PI * std::sqrt(1.0 - rhoe * rhoe));
	double inv_2var = 1.0 / (2.0 * (1.0 - rhoe * rhoe));

	const arma::rowvec sg1  = sigma1 * gm.row(0);
	const arma::rowvec sg2  = sigma2 * gm.row(1);
	const arma::rowvec sg12 = sg1 % sg2;

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
	}

	return NumericVector::create(Named("LL") = -arma::accu(LLs));
}

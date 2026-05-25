#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace Rcpp;

static inline void kron_apply_a(arma::rowvec& pi, const arma::vec& gamma_k) {
	int kbar = gamma_k.n_elem;
	int n    = pi.n_elem;
	for (int k = 0; k < kbar; k++) {
		double g  = gamma_k(k);
		double ad = 1.0 - 0.5 * g;
		double ao = 0.5 * g;
		int stride = 1 << (kbar - 1 - k);
		int block  = stride << 1;
		for (int base = 0; base < n; base += block) {
			for (int i = 0; i < stride; i++) {
				double p0 = pi(base + i);
				double p1 = pi(base + i + stride);
				pi(base + i)          = ad * p0 + ao * p1;
				pi(base + i + stride) = ao * p0 + ad * p1;
			}
		}
	}
}

static inline arma::vec make_gamma_k_a(double b, double gamma_kbar, int kbar) {
	arma::vec gamma_k(kbar);
	gamma_k(0) = 1.0 - std::pow(1.0 - gamma_kbar, 1.0 / std::pow(b, kbar - 1));
	for (int i = 1; i < kbar; i++)
		gamma_k(i) = 1.0 - std::pow(1.0 - gamma_k(0), std::pow(b, i));
	return gamma_k;
}

// Scalar LL for optimization, kbar < 8.
// A_pos: transition matrix when r_{t-1} >= 0.
// A_neg: transition matrix when r_{t-1} <  0 (switching accelerated by lev).
// [[Rcpp::export]]
double Amsm_atp_ll_fast_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                              const arma::mat& A_pos, const arma::mat& A_neg) {
	int N = dat.n_rows, k = sigma_gm.n_elem;
	const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
	double pidenom, ll = 0.0;
	arma::rowvec pi_t(k), piA(k), omega_t(k), pinum(k);
	pi_t.fill(1.0 / k);
	for (int i = 0; i < N; i++) {
		const arma::mat& A = (i > 0 && dat(i - 1) < 0.0) ? A_neg : A_pos;
		piA     = pi_t * A;
		omega_t = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_gm)) / sigma_gm + 1e-16;
		pinum   = omega_t % piA;
		pidenom = arma::accu(pinum);
		ll     += std::log(pidenom);
		if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
		else                { pi_t = pinum / pidenom; }
	}
	return -ll;
}

// Scalar LL for optimization, kbar >= 8.
// gamma_kbar_neg = min(gamma_kbar*(1+lev), 0.9999) — switching rate after negative returns.
// [[Rcpp::export]]
double Amsm_atp_ll_kron_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                              double b, double gamma_kbar, int kbar, double lev) {
	double gk_neg = std::min(gamma_kbar * (1.0 + lev), 0.9999);
	const arma::vec gamma_k_pos = make_gamma_k_a(b, gamma_kbar, kbar);
	const arma::vec gamma_k_neg = make_gamma_k_a(b, gk_neg, kbar);

	int N = dat.n_rows, k = sigma_gm.n_elem;
	const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
	double pidenom, ll = 0.0;
	arma::rowvec pi_t(k), omega_t(k), pinum(k);
	pi_t.fill(1.0 / k);
	for (int i = 0; i < N; i++) {
		const arma::vec& gk = (i > 0 && dat(i - 1) < 0.0) ? gamma_k_neg : gamma_k_pos;
		kron_apply_a(pi_t, gk);
		omega_t = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_gm)) / sigma_gm + 1e-16;
		pinum   = omega_t % pi_t;
		pidenom = arma::accu(pinum);
		ll     += std::log(pidenom);
		if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
		else                { pi_t = pinum / pidenom; }
	}
	return -ll;
}

// Full output (filtered + LL + LLs), kbar < 8.
// [[Rcpp::export]]
List Amsm_atp_fast_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                        const arma::mat& A_pos, const arma::mat& A_neg) {
	int N = dat.n_rows, k = sigma_gm.n_elem;
	const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
	double pidenom, ll = 0.0;
	arma::mat filtered(N, k, arma::fill::zeros);
	arma::rowvec pi_t(k), piA(k), omega_t(k), pinum(k);
	arma::colvec LLs(N, arma::fill::zeros);
	pi_t.fill(1.0 / k);
	for (int i = 0; i < N; i++) {
		const arma::mat& A = (i > 0 && dat(i - 1) < 0.0) ? A_neg : A_pos;
		piA     = pi_t * A;
		omega_t = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_gm)) / sigma_gm + 1e-16;
		pinum   = omega_t % piA;
		pidenom = arma::accu(pinum);
		LLs(i)  = std::log(pidenom);
		if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
		else                { pi_t = pinum / pidenom; }
		filtered.row(i) = pi_t;
	}
	ll = -arma::accu(LLs);
	return List::create(Named("filtered") = filtered, Named("LL") = ll, Named("LLs") = LLs);
}

// Full output (filtered + LL + LLs), kbar >= 8.
// [[Rcpp::export]]
List Amsm_atp_kron_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                        double b, double gamma_kbar, int kbar, double lev) {
	double gk_neg = std::min(gamma_kbar * (1.0 + lev), 0.9999);
	const arma::vec gamma_k_pos = make_gamma_k_a(b, gamma_kbar, kbar);
	const arma::vec gamma_k_neg = make_gamma_k_a(b, gk_neg, kbar);

	int N = dat.n_rows, k = sigma_gm.n_elem;
	const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
	double pidenom, ll = 0.0;
	arma::mat filtered(N, k, arma::fill::zeros);
	arma::rowvec pi_t(k), omega_t(k), pinum(k);
	arma::colvec LLs(N, arma::fill::zeros);
	pi_t.fill(1.0 / k);
	for (int i = 0; i < N; i++) {
		const arma::vec& gk = (i > 0 && dat(i - 1) < 0.0) ? gamma_k_neg : gamma_k_pos;
		kron_apply_a(pi_t, gk);
		omega_t = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_gm)) / sigma_gm + 1e-16;
		pinum   = omega_t % pi_t;
		pidenom = arma::accu(pinum);
		LLs(i)  = std::log(pidenom);
		if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
		else                { pi_t = pinum / pidenom; }
		filtered.row(i) = pi_t;
	}
	ll = -arma::accu(LLs);
	return List::create(Named("filtered") = filtered, Named("LL") = ll, Named("LLs") = LLs);
}

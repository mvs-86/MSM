#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace Rcpp;

// In-place application of A = A_0 ⊗ A_1 ⊗ ... ⊗ A_{kbar-1} to row vector pi.
// Each A_k = [[1-g/2, g/2],[g/2, 1-g/2]] is symmetric, so pi*A = A*pi.
// A_0 is outermost (kron(A,a) pattern from Msm_A_cpp): stride of A_k = 2^(kbar-1-k).
// Cost: O(kbar * 2^kbar) vs O(4^kbar) for dense multiply.
static inline void kron_apply(arma::rowvec& pi, const arma::vec& gamma_k) {
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

// Precompute gamma_k from (b, gamma_kbar, kbar) — same formula as Msm_A_cpp.
static inline arma::vec make_gamma_k(double b, double gamma_kbar, int kbar) {
	arma::vec gamma_k(kbar);
	gamma_k(0) = 1.0 - std::pow(1.0 - gamma_kbar, 1.0 / std::pow(b, kbar - 1));
	for (int i = 1; i < kbar; i++)
		gamma_k(i) = 1.0 - std::pow(1.0 - gamma_k(0), std::pow(b, i));
	return gamma_k;
}

// [[Rcpp::export]]
NumericVector Msm_ll_kron(const arma::vec& dat, const arma::rowvec& sigma_gm,
                          double b, double gamma_kbar, int kbar) {
	int N = dat.n_rows, k = sigma_gm.n_elem;
	const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
	double pidenom, ll = 0.0;

	const arma::vec gamma_k = make_gamma_k(b, gamma_kbar, kbar);
	arma::rowvec pi_t(k), omega_t(k), pinum(k);
	arma::colvec LLs(N, arma::fill::zeros);

	pi_t.fill(1.0 / k);

	for (int i = 0; i < N; i++) {
		kron_apply(pi_t, gamma_k);
		omega_t = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_gm)) / sigma_gm + 1e-16;
		pinum   = omega_t % pi_t;
		pidenom = arma::accu(pinum);
		LLs(i)  = std::log(pidenom);
		if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
		else                { pi_t = pinum / pidenom; }
	}

	ll = -arma::accu(LLs);
	return NumericVector::create(Named("LL") = ll);
}

// [[Rcpp::export]]
NumericVector Msm_ll_fast(const arma::vec& dat, const arma::rowvec& sigma_gm, const arma::mat& A) {
	int N = dat.n_rows, k = sigma_gm.n_elem;
	const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
	double pidenom, ll = 0.0;

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
	}

	ll = -arma::accu(LLs);
	return NumericVector::create(Named("LL") = ll);
}

// [[Rcpp::export]]
NumericVector Msm_ll_cpp(const arma::mat& pimat0, const arma::mat& omegat, const arma::mat& A) {
	int T = omegat.n_rows + 1, k = omegat.n_cols;
	double pidenom = 0.0, ll = 0.0;
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
	return NumericVector::create(Named("LL") = ll);
}
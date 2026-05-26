#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace Rcpp;

static inline void kron_apply_g(arma::rowvec& pi, const arma::vec& gamma_k) {
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

static inline arma::vec make_gamma_k_g(double b, double gamma_kbar, int kbar) {
    arma::vec gamma_k(kbar);
    gamma_k(0) = 1.0 - std::pow(1.0 - gamma_kbar, 1.0 / std::pow(b, kbar - 1));
    for (int i = 1; i < kbar; i++)
        gamma_k(i) = 1.0 - std::pow(1.0 - gamma_k(0), std::pow(b, i));
    return gamma_k;
}

// kbar < 8, scalar LL
// [[Rcpp::export]]
double Msm_garch_ll_fast_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                              const arma::mat& A, double alpha, double beta,
                              double gamma_gjr) {
    int N = dat.n_rows, k = sigma_gm.n_elem;
    const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    const double omega_g = 1.0 - alpha - beta - 0.5 * gamma_gjr;

    arma::rowvec sigma_gm_sq = sigma_gm % sigma_gm;
    arma::rowvec pi_t(k), piA(k), omega_t(k), pinum(k), sigma_eff(k);
    pi_t.fill(1.0 / k);
    double h_prev    = 1.0;
    double Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
    double ll = 0.0, pidenom;

    for (int i = 0; i < N; i++) {
        piA = pi_t * A;

        double h_t;
        if (i == 0) {
            h_t = 1.0;
        } else {
            double r_prev = dat(i - 1);
            double eps2   = (Mhat_prev > 0.0) ? (r_prev * r_prev) / Mhat_prev : 0.0;
            double I_neg  = (r_prev < 0.0) ? 1.0 : 0.0;
            h_t = omega_g + (alpha + gamma_gjr * I_neg) * eps2 + beta * h_prev;
            if (h_t < 1e-16) h_t = 1e-16;
        }

        sigma_eff = sigma_gm * std::sqrt(h_t);
        omega_t   = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_eff)) / sigma_eff + 1e-16;
        pinum     = omega_t % piA;
        pidenom   = arma::accu(pinum);
        ll       += std::log(pidenom);

        if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else                { pi_t = pinum / pidenom; }

        Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
        h_prev    = h_t;
    }
    return -ll;
}

// kbar >= 8, scalar LL
// [[Rcpp::export]]
double Msm_garch_ll_kron_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                              double b, double gamma_kbar, int kbar,
                              double alpha, double beta, double gamma_gjr) {
    int N = dat.n_rows, k = sigma_gm.n_elem;
    const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    const double omega_g = 1.0 - alpha - beta - 0.5 * gamma_gjr;
    const arma::vec gamma_k = make_gamma_k_g(b, gamma_kbar, kbar);

    arma::rowvec sigma_gm_sq = sigma_gm % sigma_gm;
    arma::rowvec pi_t(k), omega_t(k), pinum(k), sigma_eff(k);
    pi_t.fill(1.0 / k);
    double h_prev    = 1.0;
    double Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
    double ll = 0.0, pidenom;

    for (int i = 0; i < N; i++) {
        kron_apply_g(pi_t, gamma_k);

        double h_t;
        if (i == 0) {
            h_t = 1.0;
        } else {
            double r_prev = dat(i - 1);
            double eps2   = (Mhat_prev > 0.0) ? (r_prev * r_prev) / Mhat_prev : 0.0;
            double I_neg  = (r_prev < 0.0) ? 1.0 : 0.0;
            h_t = omega_g + (alpha + gamma_gjr * I_neg) * eps2 + beta * h_prev;
            if (h_t < 1e-16) h_t = 1e-16;
        }

        sigma_eff = sigma_gm * std::sqrt(h_t);
        omega_t   = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_eff)) / sigma_eff + 1e-16;
        pinum     = omega_t % pi_t;
        pidenom   = arma::accu(pinum);
        ll       += std::log(pidenom);

        if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else                { pi_t = pinum / pidenom; }

        Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
        h_prev    = h_t;
    }
    return -ll;
}

// kbar < 8, full output
// [[Rcpp::export]]
List Msm_garch_fast_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                        const arma::mat& A, double alpha, double beta,
                        double gamma_gjr) {
    int N = dat.n_rows, k = sigma_gm.n_elem;
    const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    const double omega_g = 1.0 - alpha - beta - 0.5 * gamma_gjr;

    arma::rowvec sigma_gm_sq = sigma_gm % sigma_gm;
    arma::mat    filtered(N, k, arma::fill::zeros);
    arma::colvec LLs(N, arma::fill::zeros);
    arma::colvec h_vec(N, arma::fill::zeros);
    arma::rowvec pi_t(k), piA(k), omega_t(k), pinum(k), sigma_eff(k);
    pi_t.fill(1.0 / k);
    double h_prev    = 1.0;
    double Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
    double ll = 0.0, pidenom;

    for (int i = 0; i < N; i++) {
        piA = pi_t * A;

        double h_t;
        if (i == 0) {
            h_t = 1.0;
        } else {
            double r_prev = dat(i - 1);
            double eps2   = (Mhat_prev > 0.0) ? (r_prev * r_prev) / Mhat_prev : 0.0;
            double I_neg  = (r_prev < 0.0) ? 1.0 : 0.0;
            h_t = omega_g + (alpha + gamma_gjr * I_neg) * eps2 + beta * h_prev;
            if (h_t < 1e-16) h_t = 1e-16;
        }

        sigma_eff = sigma_gm * std::sqrt(h_t);
        omega_t   = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_eff)) / sigma_eff + 1e-16;
        pinum     = omega_t % piA;
        pidenom   = arma::accu(pinum);
        LLs(i)    = std::log(pidenom);
        ll       += LLs(i);

        if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else                { pi_t = pinum / pidenom; }

        filtered.row(i) = pi_t;
        h_vec(i)        = h_t;
        Mhat_prev       = arma::dot(pi_t, sigma_gm_sq);
        h_prev          = h_t;
    }
    return List::create(Named("filtered") = filtered,
                        Named("LL")       = -ll,
                        Named("LLs")      = LLs,
                        Named("h")        = h_vec);
}

// kbar >= 8, full output
// [[Rcpp::export]]
List Msm_garch_kron_cpp(const arma::vec& dat, const arma::rowvec& sigma_gm,
                        double b, double gamma_kbar, int kbar,
                        double alpha, double beta, double gamma_gjr) {
    int N = dat.n_rows, k = sigma_gm.n_elem;
    const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    const double omega_g = 1.0 - alpha - beta - 0.5 * gamma_gjr;
    const arma::vec gamma_k = make_gamma_k_g(b, gamma_kbar, kbar);

    arma::rowvec sigma_gm_sq = sigma_gm % sigma_gm;
    arma::mat    filtered(N, k, arma::fill::zeros);
    arma::colvec LLs(N, arma::fill::zeros);
    arma::colvec h_vec(N, arma::fill::zeros);
    arma::rowvec pi_t(k), omega_t(k), pinum(k), sigma_eff(k);
    pi_t.fill(1.0 / k);
    double h_prev    = 1.0;
    double Mhat_prev = arma::dot(pi_t, sigma_gm_sq);
    double ll = 0.0, pidenom;

    for (int i = 0; i < N; i++) {
        kron_apply_g(pi_t, gamma_k);

        double h_t;
        if (i == 0) {
            h_t = 1.0;
        } else {
            double r_prev = dat(i - 1);
            double eps2   = (Mhat_prev > 0.0) ? (r_prev * r_prev) / Mhat_prev : 0.0;
            double I_neg  = (r_prev < 0.0) ? 1.0 : 0.0;
            h_t = omega_g + (alpha + gamma_gjr * I_neg) * eps2 + beta * h_prev;
            if (h_t < 1e-16) h_t = 1e-16;
        }

        sigma_eff = sigma_gm * std::sqrt(h_t);
        omega_t   = inv_sqrt2pi * arma::exp(-0.5 * arma::square(dat(i) / sigma_eff)) / sigma_eff + 1e-16;
        pinum     = omega_t % pi_t;
        pidenom   = arma::accu(pinum);
        LLs(i)    = std::log(pidenom);
        ll       += LLs(i);

        if (pidenom == 0.0) { pi_t.zeros(); pi_t(0) = 1.0; }
        else                { pi_t = pinum / pidenom; }

        filtered.row(i) = pi_t;
        h_vec(i)        = h_t;
        Mhat_prev       = arma::dot(pi_t, sigma_gm_sq);
        h_prev          = h_t;
    }
    return List::create(Named("filtered") = filtered,
                        Named("LL")       = -ll,
                        Named("LLs")      = LLs,
                        Named("h")        = h_vec);
}

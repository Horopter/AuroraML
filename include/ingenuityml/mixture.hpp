#pragma once

#include "base.hpp"
#include <vector>
#include <random>

namespace ingenuityml {
namespace mixture {

/**
 * GaussianMixture - Gaussian Mixture Model
 * 
 * Similar to scikit-learn's GaussianMixture, fits a mixture of Gaussian
 * distributions using the EM algorithm.
 */
class GaussianMixture : public Estimator {
private:
    int n_components_;
    int max_iter_;
    double tol_;
    int random_state_;
    bool fitted_;
    int n_features_;
    std::vector<VectorXd> means_;
    std::vector<MatrixXd> covariances_;
    VectorXd weights_;
    MatrixXd responsibilities_;

public:
    GaussianMixture(
        int n_components = 1,
        int max_iter = 100,
        double tol = 1e-3,
        int random_state = -1
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict(const MatrixXd& X) const;
    MatrixXd predict_proba(const MatrixXd& X) const;
    VectorXd score_samples(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    std::vector<VectorXd> means() const { return means_; }
    std::vector<MatrixXd> covariances() const { return covariances_; }
    VectorXd weights() const { return weights_; }

private:
    void initialize_parameters(const MatrixXd& X, std::mt19937& rng);
    double e_step(const MatrixXd& X);
    void m_step(const MatrixXd& X);
    double log_likelihood(const MatrixXd& X) const;
};

/**
 * BayesianGaussianMixture - Bayesian Gaussian Mixture Model
 *
 * Simplified wrapper around GaussianMixture with a Bayesian-style interface.
 */
class BayesianGaussianMixture : public Estimator {
private:
    int n_components_;
    int max_iter_;
    double tol_;
    double weight_concentration_prior_;
    int random_state_;
    GaussianMixture impl_;

public:
    BayesianGaussianMixture(
        int n_components = 1,
        int max_iter = 100,
        double tol = 1e-3,
        double weight_concentration_prior = 1.0,
        int random_state = -1
    );

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict(const MatrixXd& X) const;
    MatrixXd predict_proba(const MatrixXd& X) const;
    VectorXd score_samples(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return impl_.is_fitted(); }

    std::vector<VectorXd> means() const { return impl_.means(); }
    std::vector<MatrixXd> covariances() const { return impl_.covariances(); }
    VectorXd weights() const { return impl_.weights(); }
};

} // namespace mixture
} // namespace ingenuityml

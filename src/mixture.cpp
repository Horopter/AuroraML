#include "ingenuityml/mixture.hpp"
#include "ingenuityml/base.hpp"
#include <cmath>
#include <random>

namespace ingenuityml {
namespace mixture {

GaussianMixture::GaussianMixture(
    int n_components,
    int max_iter,
    double tol,
    int random_state
) : n_components_(n_components), max_iter_(max_iter), tol_(tol),
    random_state_(random_state), fitted_(false), n_features_(0) {
    if (n_components <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }
}

void GaussianMixture::initialize_parameters(const MatrixXd& X, std::mt19937& rng) {
    means_.clear();
    covariances_.clear();
    weights_ = VectorXd::Constant(n_components_, 1.0 / n_components_);
    
    // Initialize means randomly
    std::uniform_int_distribution<int> sample_dist(0, X.rows() - 1);
    for (int k = 0; k < n_components_; ++k) {
        int idx = sample_dist(rng);
        means_.push_back(X.row(idx).transpose());  // Transpose row to column vector
    }
    
    // Initialize covariances as identity matrices
    for (int k = 0; k < n_components_; ++k) {
        covariances_.push_back(MatrixXd::Identity(n_features_, n_features_));
    }
}

double GaussianMixture::log_likelihood(const MatrixXd& X) const {
    double log_likelihood = 0.0;
    
    for (int i = 0; i < X.rows(); ++i) {
        double sample_log_prob = 0.0;
        
        for (int k = 0; k < n_components_; ++k) {
            VectorXd diff = X.row(i).transpose() - means_[k];  // Transpose row to column vector
            MatrixXd cov_inv = covariances_[k].inverse();
            double quad_form = diff.transpose() * cov_inv * diff;
            double log_det = std::log(covariances_[k].determinant());
            
            double component_log_prob = -0.5 * quad_form - 0.5 * log_det - 
                                        0.5 * n_features_ * std::log(2.0 * M_PI) + 
                                        std::log(weights_(k));
            
            sample_log_prob += std::exp(component_log_prob);
        }
        
        log_likelihood += std::log(sample_log_prob + 1e-10);
    }
    
    return log_likelihood;
}

double GaussianMixture::e_step(const MatrixXd& X) {
    double log_likelihood = 0.0;
    responsibilities_ = MatrixXd::Zero(X.rows(), n_components_);
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd log_probs = VectorXd::Zero(n_components_);
        
        for (int k = 0; k < n_components_; ++k) {
            VectorXd diff = X.row(i).transpose() - means_[k];  // Transpose row to column vector
            MatrixXd cov_inv = covariances_[k].inverse();
            double quad_form = diff.transpose() * cov_inv * diff;
            double log_det = std::log(covariances_[k].determinant());
            
            log_probs(k) = -0.5 * quad_form - 0.5 * log_det - 
                           0.5 * n_features_ * std::log(2.0 * M_PI) + 
                           std::log(weights_(k));
        }
        
        // Convert to probabilities using log-sum-exp trick
        double max_log = log_probs.maxCoeff();
        VectorXd exp_log = (log_probs.array() - max_log).exp();
        double sum_exp = exp_log.sum();
        
        if (sum_exp > 0) {
            responsibilities_.row(i) = exp_log.transpose() / sum_exp;
            log_likelihood += max_log + std::log(sum_exp);
        }
    }
    
    return log_likelihood;
}

void GaussianMixture::m_step(const MatrixXd& X) {
    // Update weights
    VectorXd Nk = responsibilities_.colwise().sum();
    weights_ = Nk / X.rows();
    
    // Update means
    for (int k = 0; k < n_components_; ++k) {
        VectorXd weighted_sum = VectorXd::Zero(n_features_);
        for (int i = 0; i < X.rows(); ++i) {
            weighted_sum += responsibilities_(i, k) * X.row(i).transpose();  // Transpose row to column vector
        }
        if (Nk(k) > 0) {
            means_[k] = weighted_sum / Nk(k);
        }
    }
    
    // Update covariances
    for (int k = 0; k < n_components_; ++k) {
        MatrixXd cov = MatrixXd::Zero(n_features_, n_features_);
        for (int i = 0; i < X.rows(); ++i) {
            VectorXd diff = X.row(i).transpose() - means_[k];  // Transpose row to column vector
            cov += responsibilities_(i, k) * diff * diff.transpose();
        }
        if (Nk(k) > 0) {
            covariances_[k] = cov / Nk(k);
            // Add small regularization
            covariances_[k] += MatrixXd::Identity(n_features_, n_features_) * 1e-6;
        }
    }
}

Estimator& GaussianMixture::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    n_features_ = X.cols();
    
    std::mt19937 rng;
    if (random_state_ >= 0) {
        rng.seed(random_state_);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    initialize_parameters(X, rng);
    
    double prev_log_likelihood = -std::numeric_limits<double>::infinity();
    
    for (int iter = 0; iter < max_iter_; ++iter) {
        double log_likelihood = e_step(X);
        m_step(X);
        
        if (std::abs(log_likelihood - prev_log_likelihood) < tol_) {
            break;
        }
        
        prev_log_likelihood = log_likelihood;
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd GaussianMixture::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("GaussianMixture must be fitted before predict_proba");
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), n_components_);
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd log_probs = VectorXd::Zero(n_components_);
        
        for (int k = 0; k < n_components_; ++k) {
            VectorXd diff = X.row(i).transpose() - means_[k];  // Transpose row to column vector
            MatrixXd cov_inv = covariances_[k].inverse();
            double quad_form = diff.transpose() * cov_inv * diff;
            double log_det = std::log(covariances_[k].determinant());
            
            log_probs(k) = -0.5 * quad_form - 0.5 * log_det - 
                           0.5 * n_features_ * std::log(2.0 * M_PI) + 
                           std::log(weights_(k));
        }
        
        double max_log = log_probs.maxCoeff();
        VectorXd exp_log = (log_probs.array() - max_log).exp();
        double sum_exp = exp_log.sum();
        
        if (sum_exp > 0) {
            proba.row(i) = exp_log.transpose() / sum_exp;
        } else {
            proba.row(i).fill(1.0 / n_components_);
        }
    }
    
    return proba;
}

VectorXi GaussianMixture::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("GaussianMixture must be fitted before predict");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        int max_idx = 0;
        for (int j = 1; j < n_components_; ++j) {
            if (proba(i, j) > proba(i, max_idx)) {
                max_idx = j;
            }
        }
        predictions(i) = max_idx;
    }
    
    return predictions;
}

VectorXd GaussianMixture::score_samples(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("GaussianMixture must be fitted before score_samples");
    }
    
    VectorXd scores = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        double sample_log_prob = 0.0;
        
        for (int k = 0; k < n_components_; ++k) {
            VectorXd diff = X.row(i).transpose() - means_[k];  // Transpose row to column vector
            MatrixXd cov_inv = covariances_[k].inverse();
            double quad_form = diff.transpose() * cov_inv * diff;
            double log_det = std::log(covariances_[k].determinant());
            
            double component_log_prob = -0.5 * quad_form - 0.5 * log_det - 
                                        0.5 * n_features_ * std::log(2.0 * M_PI) + 
                                        std::log(weights_(k));
            
            sample_log_prob += std::exp(component_log_prob);
        }
        
        scores(i) = std::log(sample_log_prob + 1e-10);
    }
    
    return scores;
}

Params GaussianMixture::get_params() const {
    Params params;
    params["n_components"] = std::to_string(n_components_);
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& GaussianMixture::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// BayesianGaussianMixture implementation

BayesianGaussianMixture::BayesianGaussianMixture(
    int n_components,
    int max_iter,
    double tol,
    double weight_concentration_prior,
    int random_state
) : n_components_(n_components),
    max_iter_(max_iter),
    tol_(tol),
    weight_concentration_prior_(weight_concentration_prior),
    random_state_(random_state),
    impl_(n_components, max_iter, tol, random_state) {
}

Estimator& BayesianGaussianMixture::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    impl_.set_params({{"n_components", std::to_string(n_components_)},
                      {"max_iter", std::to_string(max_iter_)},
                      {"tol", std::to_string(tol_)},
                      {"random_state", std::to_string(random_state_)}});
    impl_.fit(X, VectorXd());
    return *this;
}

VectorXi BayesianGaussianMixture::predict(const MatrixXd& X) const {
    return impl_.predict(X);
}

MatrixXd BayesianGaussianMixture::predict_proba(const MatrixXd& X) const {
    return impl_.predict_proba(X);
}

VectorXd BayesianGaussianMixture::score_samples(const MatrixXd& X) const {
    return impl_.score_samples(X);
}

Params BayesianGaussianMixture::get_params() const {
    Params params;
    params["n_components"] = std::to_string(n_components_);
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    params["weight_concentration_prior"] = std::to_string(weight_concentration_prior_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& BayesianGaussianMixture::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    weight_concentration_prior_ = utils::get_param_double(params, "weight_concentration_prior", weight_concentration_prior_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    impl_.set_params({{"n_components", std::to_string(n_components_)},
                      {"max_iter", std::to_string(max_iter_)},
                      {"tol", std::to_string(tol_)},
                      {"random_state", std::to_string(random_state_)}});
    return *this;
}

} // namespace mixture
} // namespace ingenuityml

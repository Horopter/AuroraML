#include "ingenuityml/discriminant_analysis.hpp"
#include "ingenuityml/base.hpp"
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>

namespace ingenuityml {
namespace discriminant_analysis {

QuadraticDiscriminantAnalysis::QuadraticDiscriminantAnalysis(double regularization)
    : regularization_(regularization), fitted_(false), n_features_(0), n_classes_(0) {
}

Estimator& QuadraticDiscriminantAnalysis::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    
    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    if (unique_classes_set.size() < 2) {
        throw std::invalid_argument("QuadraticDiscriminantAnalysis requires at least 2 classes");
    }
    
    classes_.resize(unique_classes_set.size());
    n_classes_ = unique_classes_set.size();
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }
    
    means_.clear();
    covariances_.clear();
    priors_ = VectorXd::Zero(n_classes_);
    
    // Compute class means, covariances, and priors
    for (int c = 0; c < n_classes_; ++c) {
        int class_label = classes_(c);
        
        // Collect samples for this class
        std::vector<int> class_indices;
        for (int i = 0; i < y.size(); ++i) {
            if (static_cast<int>(y(i)) == class_label) {
                class_indices.push_back(i);
            }
        }
        
        priors_(c) = static_cast<double>(class_indices.size()) / y.size();
        
        // Compute mean
        VectorXd mean = VectorXd::Zero(n_features_);
        for (int idx : class_indices) {
            mean += X.row(idx);
        }
        mean /= class_indices.size();
        means_.push_back(mean);
        
        // Compute covariance
        MatrixXd cov = MatrixXd::Zero(n_features_, n_features_);
        for (int idx : class_indices) {
            VectorXd diff = X.row(idx).transpose() - mean;  // Transpose row to column vector
            cov += diff * diff.transpose();
        }
        cov /= (class_indices.size() - 1);
        
        // Add regularization
        cov += MatrixXd::Identity(n_features_, n_features_) * regularization_;
        covariances_.push_back(cov);
    }
    
    fitted_ = true;
    return *this;
}

VectorXi QuadraticDiscriminantAnalysis::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("QuadraticDiscriminantAnalysis must be fitted before predict");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        int max_idx = 0;
        for (int j = 1; j < proba.cols(); ++j) {
            if (proba(i, j) > proba(i, max_idx)) {
                max_idx = j;
            }
        }
        predictions(i) = classes_(max_idx);
    }
    
    return predictions;
}

MatrixXd QuadraticDiscriminantAnalysis::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("QuadraticDiscriminantAnalysis must be fitted before predict_proba");
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), n_classes_);
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd x = X.row(i);
        VectorXd log_proba = VectorXd::Zero(n_classes_);
        
        for (int c = 0; c < n_classes_; ++c) {
            VectorXd diff = x - means_[c];
            MatrixXd cov_inv = covariances_[c].inverse();
            
            // Quadratic discriminant: -0.5 * (x - mu)^T * Sigma^-1 * (x - mu) - 0.5 * log|Sigma| + log(prior)
            double quad_form = diff.transpose() * cov_inv * diff;
            double log_det = std::log(covariances_[c].determinant());
            
            log_proba(c) = -0.5 * quad_form - 0.5 * log_det + std::log(priors_(c));
        }
        
        // Convert to probabilities using softmax
        double max_log = log_proba.maxCoeff();
        VectorXd exp_log = (log_proba.array() - max_log).exp();
        double sum_exp = exp_log.sum();
        
        if (sum_exp > 0) {
            proba.row(i) = exp_log.transpose() / sum_exp;
        } else {
            proba.row(i).fill(1.0 / n_classes_);
        }
    }
    
    return proba;
}

VectorXd QuadraticDiscriminantAnalysis::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("QuadraticDiscriminantAnalysis must be fitted before decision_function");
    }
    
    // Return log probabilities
    VectorXd decision = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd x = X.row(i);
        double max_log_proba = -std::numeric_limits<double>::infinity();
        
        for (int c = 0; c < n_classes_; ++c) {
            VectorXd diff = x - means_[c];
            MatrixXd cov_inv = covariances_[c].inverse();
            double quad_form = diff.transpose() * cov_inv * diff;
            double log_det = std::log(covariances_[c].determinant());
            double log_proba = -0.5 * quad_form - 0.5 * log_det + std::log(priors_(c));
            
            if (log_proba > max_log_proba) {
                max_log_proba = log_proba;
            }
        }
        
        decision(i) = max_log_proba;
    }
    
    return decision;
}

Params QuadraticDiscriminantAnalysis::get_params() const {
    Params params;
    params["regularization"] = std::to_string(regularization_);
    return params;
}

Estimator& QuadraticDiscriminantAnalysis::set_params(const Params& params) {
    regularization_ = utils::get_param_double(params, "regularization", regularization_);
    return *this;
}

// LinearDiscriminantAnalysis implementation

LinearDiscriminantAnalysis::LinearDiscriminantAnalysis(double regularization)
    : regularization_(regularization), fitted_(false), n_features_(0), n_classes_(0) {
}

Estimator& LinearDiscriminantAnalysis::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);

    n_features_ = X.cols();

    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    if (unique_classes_set.size() < 2) {
        throw std::invalid_argument("LinearDiscriminantAnalysis requires at least 2 classes");
    }

    classes_.resize(unique_classes_set.size());
    n_classes_ = unique_classes_set.size();
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }

    means_.clear();
    priors_ = VectorXd::Zero(n_classes_);

    for (int c = 0; c < n_classes_; ++c) {
        int class_label = classes_(c);
        std::vector<int> class_indices;
        for (int i = 0; i < y.size(); ++i) {
            if (static_cast<int>(y(i)) == class_label) {
                class_indices.push_back(i);
            }
        }

        priors_(c) = static_cast<double>(class_indices.size()) / y.size();

        VectorXd mean = VectorXd::Zero(n_features_);
        for (int i : class_indices) {
            mean += X.row(i);
        }
        mean /= class_indices.size();
        means_.push_back(mean);
    }

    covariance_ = MatrixXd::Zero(n_features_, n_features_);
    for (int c = 0; c < n_classes_; ++c) {
        int class_label = classes_(c);
        for (int i = 0; i < y.size(); ++i) {
            if (static_cast<int>(y(i)) == class_label) {
                VectorXd diff = X.row(i).transpose() - means_[c];
                covariance_ += diff * diff.transpose();
            }
        }
    }

    int denom = std::max(1, static_cast<int>(y.size()) - n_classes_);
    covariance_ /= static_cast<double>(denom);
    covariance_ += MatrixXd::Identity(n_features_, n_features_) * regularization_;
    covariance_inv_ = covariance_.inverse();

    fitted_ = true;
    return *this;
}

VectorXi LinearDiscriminantAnalysis::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LinearDiscriminantAnalysis must be fitted before predict");
    }

    MatrixXd proba = predict_proba(X);
    VectorXi predictions = VectorXi::Zero(X.rows());

    for (int i = 0; i < X.rows(); ++i) {
        int max_idx = 0;
        for (int j = 1; j < proba.cols(); ++j) {
            if (proba(i, j) > proba(i, max_idx)) {
                max_idx = j;
            }
        }
        predictions(i) = classes_(max_idx);
    }

    return predictions;
}

MatrixXd LinearDiscriminantAnalysis::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LinearDiscriminantAnalysis must be fitted before predict_proba");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd proba = MatrixXd::Zero(X.rows(), n_classes_);

    for (int i = 0; i < X.rows(); ++i) {
        VectorXd x = X.row(i).transpose();
        VectorXd log_proba = VectorXd::Zero(n_classes_);

        for (int c = 0; c < n_classes_; ++c) {
            const VectorXd& mean = means_[c];
            double term1 = x.dot(covariance_inv_ * mean);
            double term2 = 0.5 * mean.dot(covariance_inv_ * mean);
            log_proba(c) = term1 - term2 + std::log(priors_(c));
        }

        double max_log = log_proba.maxCoeff();
        VectorXd exp_log = (log_proba.array() - max_log).exp();
        double sum_exp = exp_log.sum();

        if (sum_exp > 0) {
            proba.row(i) = exp_log.transpose() / sum_exp;
        } else {
            proba.row(i).fill(1.0 / n_classes_);
        }
    }

    return proba;
}

VectorXd LinearDiscriminantAnalysis::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LinearDiscriminantAnalysis must be fitted before decision_function");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    VectorXd decision = VectorXd::Zero(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd x = X.row(i).transpose();
        double max_log_proba = -std::numeric_limits<double>::infinity();

        for (int c = 0; c < n_classes_; ++c) {
            const VectorXd& mean = means_[c];
            double term1 = x.dot(covariance_inv_ * mean);
            double term2 = 0.5 * mean.dot(covariance_inv_ * mean);
            double log_proba = term1 - term2 + std::log(priors_(c));
            if (log_proba > max_log_proba) {
                max_log_proba = log_proba;
            }
        }

        decision(i) = max_log_proba;
    }

    return decision;
}

Params LinearDiscriminantAnalysis::get_params() const {
    Params params;
    params["regularization"] = std::to_string(regularization_);
    return params;
}

Estimator& LinearDiscriminantAnalysis::set_params(const Params& params) {
    regularization_ = utils::get_param_double(params, "regularization", regularization_);
    return *this;
}

} // namespace discriminant_analysis
} // namespace ingenuityml

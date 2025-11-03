#include "auroraml/adaboost.hpp"
#include "auroraml/base.hpp"
#include "auroraml/tree.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <cmath>

namespace auroraml {
namespace ensemble {

// AdaBoost Classifier Implementation
Estimator& AdaBoostClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    int n_samples = X.rows();
    
    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    classes_.assign(unique_classes_set.begin(), unique_classes_set.end());
    n_classes_ = classes_.size();
    
    if (n_classes_ < 2) {
        throw std::invalid_argument("AdaBoostClassifier requires at least 2 classes");
    }
    
    // Convert labels to {0, 1} for binary classification (or extend for multi-class)
    VectorXd y_binary = VectorXd::Zero(n_samples);
    int class0 = classes_[0];
    for (int i = 0; i < n_samples; ++i) {
        y_binary(i) = (static_cast<int>(y(i)) == class0) ? -1.0 : 1.0;
    }
    
    // Initialize sample weights uniformly
    VectorXd sample_weights = VectorXd::Constant(n_samples, 1.0 / n_samples);
    
    estimators_.clear();
    estimator_weights_.clear();
    estimators_.reserve(n_estimators_);
    estimator_weights_.reserve(n_estimators_);
    
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    
    for (int t = 0; t < n_estimators_; ++t) {
        // Fit a weak learner (decision stump - depth 1 tree)
        tree::DecisionTreeClassifier stump("gini", 1, 2, 1, 0.0);
        stump.fit(X, y);
        
        // Predict with current stump
        VectorXi predictions = stump.predict_classes(X);
        VectorXd predictions_double = predictions.cast<double>();
        
        // Convert predictions to {-1, 1}
        for (int i = 0; i < predictions_double.size(); ++i) {
            predictions_double(i) = (predictions_double(i) == class0) ? -1.0 : 1.0;
        }
        
        // Calculate weighted error
        double weighted_error = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            if (predictions_double(i) != y_binary(i)) {
                weighted_error += sample_weights(i);
            }
        }
        
        // Avoid perfect classifier (would lead to infinite weight)
        weighted_error = std::max(weighted_error, 1e-10);
        weighted_error = std::min(weighted_error, 1.0 - 1e-10);
        
        // Calculate estimator weight
        double alpha = learning_rate_ * 0.5 * std::log((1.0 - weighted_error) / weighted_error);
        estimator_weights_.push_back(alpha);
        
        // Update sample weights
        for (int i = 0; i < n_samples; ++i) {
            if (predictions_double(i) != y_binary(i)) {
                sample_weights(i) *= std::exp(alpha);
            } else {
                sample_weights(i) *= std::exp(-alpha);
            }
        }
        
        // Normalize weights
        double sum_weights = sample_weights.sum();
        sample_weights /= sum_weights;
        
        estimators_.push_back(std::move(stump));
    }
    
    fitted_ = true;
    return static_cast<Estimator&>(*this);
}

VectorXi AdaBoostClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("AdaBoostClassifier must be fitted before predict");
    }
    validation::check_X(X);
    
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    int n_samples = X.rows();
    VectorXd weighted_sum = VectorXd::Zero(n_samples);
    int class0 = classes_[0];
    int class1 = classes_[1];
    
    // Sum weighted predictions from all estimators
    for (size_t t = 0; t < estimators_.size(); ++t) {
        VectorXi predictions = estimators_[t].predict_classes(X);
        VectorXd predictions_double = predictions.cast<double>();
        
        // Convert to {-1, 1} and weight
        for (int i = 0; i < n_samples; ++i) {
            double pred_val = (predictions_double(i) == class0) ? -1.0 : 1.0;
            weighted_sum(i) += estimator_weights_[t] * pred_val;
        }
    }
    
    // Convert back to class labels
    VectorXi final_predictions(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        final_predictions(i) = (weighted_sum(i) >= 0.0) ? class1 : class0;
    }
    
    return final_predictions;
}

MatrixXd AdaBoostClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("AdaBoostClassifier must be fitted before predict_proba");
    }
    
    VectorXd decision = decision_function(X);
    MatrixXd proba(X.rows(), n_classes_);
    
    // Convert decision function to probabilities using sigmoid
    for (int i = 0; i < decision.size(); ++i) {
        double prob_pos = 1.0 / (1.0 + std::exp(-2.0 * decision(i)));
        proba(i, 0) = 1.0 - prob_pos;
        proba(i, 1) = prob_pos;
    }
    
    return proba;
}

VectorXd AdaBoostClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("AdaBoostClassifier must be fitted before decision_function");
    }
    
    int n_samples = X.rows();
    VectorXd weighted_sum = VectorXd::Zero(n_samples);
    int class0 = classes_[0];
    int class1 = classes_[1];
    
    for (size_t t = 0; t < estimators_.size(); ++t) {
        VectorXi predictions = estimators_[t].predict_classes(X);
        VectorXd predictions_double = predictions.cast<double>();
        
        for (int i = 0; i < n_samples; ++i) {
            double pred_val = (predictions_double(i) == class0) ? -1.0 : 1.0;
            weighted_sum(i) += estimator_weights_[t] * pred_val;
        }
    }
    
    return weighted_sum;
}

Params AdaBoostClassifier::get_params() const {
    return {
        {"n_estimators", std::to_string(n_estimators_)},
        {"learning_rate", std::to_string(learning_rate_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& AdaBoostClassifier::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    learning_rate_ = utils::get_param_double(params, "learning_rate", learning_rate_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// AdaBoost Regressor Implementation (AdaBoost.R2 algorithm)
Estimator& AdaBoostRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    n_features_ = X.cols();
    int n_samples = X.rows();

    // Initialize sample weights uniformly
    VectorXd sample_weights = VectorXd::Constant(n_samples, 1.0 / n_samples);

    estimators_.clear();
    estimator_weights_.clear();
    estimators_.reserve(n_estimators_);
    estimator_weights_.reserve(n_estimators_);

    // Initialize predictions to mean
    double init = y.mean();
    VectorXd predictions = VectorXd::Constant(n_samples, init);

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());

    for (int t = 0; t < n_estimators_; ++t) {
        // Fit a weak learner (decision stump - depth 1 tree) to weighted residuals
        VectorXd residuals = y - predictions;
        
        tree::DecisionTreeRegressor stump("mse", 1, 2, 1, 0.0);
        stump.fit(X, residuals);
        
        // Predict residuals
        VectorXd stump_pred = stump.predict(X);
        
        // Calculate errors (absolute errors)
        VectorXd errors = (residuals - stump_pred).cwiseAbs();
        
        // Calculate max error for normalization
        double max_error = errors.maxCoeff();
        if (max_error < 1e-10) {
            max_error = 1.0;
        }
        
        // Normalize errors to [0, 1]
        VectorXd normalized_errors = errors / max_error;
        
        // Calculate loss-dependent errors
        VectorXd loss_errors = normalized_errors;
        if (loss_ == "square") {
            loss_errors = normalized_errors.cwiseProduct(normalized_errors);
        } else if (loss_ == "exponential") {
            loss_errors = 1.0 - (-normalized_errors).array().exp();
        }
        // "linear" is default
        
        // Calculate weighted error from loss
        double weighted_loss_error = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            weighted_loss_error += sample_weights(i) * loss_errors(i);
        }
        
        // Avoid perfect predictor (would lead to infinite weight)
        weighted_loss_error = std::max(weighted_loss_error, 1e-10);
        weighted_loss_error = std::min(weighted_loss_error, 1.0 - 1e-10);
        
        // Calculate estimator weight using AdaBoost.R2 formula
        double beta = weighted_loss_error / (1.0 - weighted_loss_error);
        double alpha = learning_rate_ * std::log(1.0 / beta);
        estimator_weights_.push_back(alpha);
        
        // Update sample weights
        for (int i = 0; i < n_samples; ++i) {
            sample_weights(i) *= std::pow(beta, 1.0 - loss_errors(i));
        }
        
        // Normalize weights
        double sum_weights = sample_weights.sum();
        if (sum_weights > 1e-10) {
            sample_weights /= sum_weights;
        } else {
            sample_weights.fill(1.0 / n_samples);
        }
        
        // Update predictions
        predictions += learning_rate_ * stump_pred;
        
        estimators_.push_back(std::move(stump));
    }

    fitted_ = true;
    return *this;
}

VectorXd AdaBoostRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("AdaBoostRegressor must be fitted before predict");
    }
    validation::check_X(X);
    
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    int n_samples = X.rows();
    VectorXd predictions = VectorXd::Zero(n_samples);
    
    // Sum weighted predictions from all estimators
    for (size_t t = 0; t < estimators_.size(); ++t) {
        VectorXd tree_pred = estimators_[t].predict(X);
        predictions += estimator_weights_[t] * tree_pred;
    }
    
    return predictions;
}

Params AdaBoostRegressor::get_params() const {
    return {
        {"n_estimators", std::to_string(n_estimators_)},
        {"learning_rate", std::to_string(learning_rate_)},
        {"loss", loss_},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& AdaBoostRegressor::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    learning_rate_ = utils::get_param_double(params, "learning_rate", learning_rate_);
    loss_ = utils::get_param_string(params, "loss", loss_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

} // namespace ensemble
} // namespace cxml


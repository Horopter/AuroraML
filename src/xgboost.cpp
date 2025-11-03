#include "auroraml/xgboost.hpp"
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

// Helper function for sigmoid
static double sigmoid(double x) {
    x = std::max(-500.0, std::min(500.0, x));
    return 1.0 / (1.0 + std::exp(-x));
}

// XGBoost Classifier Implementation
Estimator& XGBClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    int n_samples = X.rows();
    
    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    classes_.assign(unique_classes_set.begin(), unique_classes_set.end());
    n_classes_ = static_cast<int>(classes_.size());
    
    if (n_classes_ < 2) {
        throw std::invalid_argument("XGBClassifier requires at least 2 classes");
    }
    
    // Initialize base score (log-odds)
    base_score_ = VectorXd::Zero(n_classes_);
    VectorXd class_counts = VectorXd::Zero(n_classes_);
    for (int i = 0; i < n_samples; ++i) {
        int class_idx = std::find(classes_.begin(), classes_.end(), static_cast<int>(y(i))) - classes_.begin();
        class_counts(class_idx) += 1.0;
    }
    class_counts /= n_samples;
    for (int k = 0; k < n_classes_; ++k) {
        double prob = class_counts(k);
        base_score_(k) = std::log(prob / (1.0 - prob + 1e-10) + 1e-10);
    }
    
    // Initialize predictions
    MatrixXd predictions = MatrixXd::Constant(n_samples, n_classes_, 0.0);
    for (int i = 0; i < n_samples; ++i) {
        predictions.row(i) = base_score_.transpose();
    }
    
    estimators_.clear();
    estimators_.reserve(n_estimators_ * n_classes_);
    
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    
    // Feature subsampling for colsample_bytree
    std::vector<int> feature_indices(n_features_);
    for (int i = 0; i < n_features_; ++i) {
        feature_indices[i] = i;
    }
    
    // Sample indices for subsample
    std::vector<int> sample_indices(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        sample_indices[i] = i;
    }
    
    for (int t = 0; t < n_estimators_; ++t) {
        // Subsample data if needed
        std::vector<int> current_samples = sample_indices;
        if (subsample_ < 1.0) {
            std::shuffle(current_samples.begin(), current_samples.end(), rng);
            int n_subsample = static_cast<int>(subsample_ * n_samples);
            current_samples.resize(n_subsample);
        }
        
        // Compute probabilities using softmax
        MatrixXd probabilities = MatrixXd::Zero(n_samples, n_classes_);
        for (int i = 0; i < n_samples; ++i) {
            VectorXd exp_preds = predictions.row(i).array().exp();
            double sum_exp = exp_preds.sum();
            if (sum_exp > 1e-10) {
                probabilities.row(i) = exp_preds.transpose() / sum_exp;
            } else {
                probabilities.row(i).fill(1.0 / n_classes_);
            }
        }
        
        // For each class, fit a tree to gradients
        for (int k = 0; k < n_classes_; ++k) {
            // Compute gradients and hessians for logistic loss
            VectorXd gradients = VectorXd::Zero(n_samples);
            VectorXd hessians = VectorXd::Zero(n_samples);
            
            for (int i = 0; i < n_samples; ++i) {
                int y_class = static_cast<int>(y(i));
                int class_idx = std::find(classes_.begin(), classes_.end(), y_class) - classes_.begin();
                
                // Gradient: indicator - prob (negative gradient for minimizing loss)
                // This matches gradient boosting: y_onehot - probabilities
                gradients(i) = (class_idx == k ? 1.0 : 0.0) - probabilities(i, k);
                
                // Hessian: prob * (1 - prob) for logistic loss
                hessians(i) = probabilities(i, k) * (1.0 - probabilities(i, k));
                hessians(i) = std::max(hessians(i), 1e-6); // Prevent division by zero
            }
            
            // Create subsampled dataset
            MatrixXd X_subsample(current_samples.size(), n_features_);
            VectorXd gradients_subsample(current_samples.size());
            
            for (size_t i = 0; i < current_samples.size(); ++i) {
                int idx = current_samples[i];
                X_subsample.row(i) = X.row(idx);
                gradients_subsample(i) = gradients(idx);
            }
            
            // Fit tree to gradients (XGBoost fits to first-order gradients)
            tree::DecisionTreeRegressor tree("mse", max_depth_, 2, min_child_weight_, gamma_);
            tree.fit(X_subsample, gradients_subsample);
            
            // Predict and update
            VectorXd tree_pred = tree.predict(X);
            
            // Update predictions with learning rate
            // Standard gradient boosting update: tree is fitted to gradients, so we add tree_pred
            for (int i = 0; i < n_samples; ++i) {
                predictions(i, k) += learning_rate_ * tree_pred(i);
            }
            
            estimators_.push_back(std::move(tree));
        }
    }
    
    fitted_ = true;
    return static_cast<Estimator&>(*this);
}

VectorXi XGBClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("XGBClassifier must be fitted before predict");
    }
    validation::check_X(X);
    
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXi predictions(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        int max_idx = 0;
        double max_prob = proba(i, 0);
        for (int k = 1; k < n_classes_; ++k) {
            if (proba(i, k) > max_prob) {
                max_prob = proba(i, k);
                max_idx = k;
            }
        }
        predictions(i) = classes_[max_idx];
    }
    
    return predictions;
}

MatrixXd XGBClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("XGBClassifier must be fitted before predict_proba");
    }
    
    int n_samples = X.rows();
    MatrixXd predictions = MatrixXd::Constant(n_samples, n_classes_, 0.0);
    
    // Start with base scores
    for (int i = 0; i < n_samples; ++i) {
        predictions.row(i) = base_score_.transpose();
    }
    
    // Add predictions from all trees
    int tree_idx = 0;
    for (int t = 0; t < n_estimators_; ++t) {
        for (int k = 0; k < n_classes_; ++k) {
            VectorXd tree_pred = estimators_[tree_idx].predict(X);
            predictions.col(k) += learning_rate_ * tree_pred;
            tree_idx++;
        }
    }
    
    // Apply softmax to get probabilities
    MatrixXd proba(n_samples, n_classes_);
    for (int i = 0; i < n_samples; ++i) {
        VectorXd exp_preds = predictions.row(i).array().exp();
        double sum_exp = exp_preds.sum();
        if (sum_exp > 1e-10) {
            proba.row(i) = exp_preds.transpose() / sum_exp;
        } else {
            proba.row(i).fill(1.0 / n_classes_);
        }
    }
    
    return proba;
}

VectorXd XGBClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("XGBClassifier must be fitted before decision_function");
    }
    
    int n_samples = X.rows();
    MatrixXd predictions = MatrixXd::Constant(n_samples, n_classes_, 0.0);
    
    for (int i = 0; i < n_samples; ++i) {
        predictions.row(i) = base_score_.transpose();
    }
    
    int tree_idx = 0;
    for (int t = 0; t < n_estimators_; ++t) {
        for (int k = 0; k < n_classes_; ++k) {
            VectorXd tree_pred = estimators_[tree_idx].predict(X);
            predictions.col(k) += learning_rate_ * tree_pred;
            tree_idx++;
        }
    }
    
    // For binary classification, return the difference
    if (n_classes_ == 2) {
        return predictions.col(1) - predictions.col(0);
    }
    
    // For multi-class, return max
    VectorXd decision(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        decision(i) = predictions.row(i).maxCoeff();
    }
    return decision;
}

Params XGBClassifier::get_params() const {
    return {
        {"n_estimators", std::to_string(n_estimators_)},
        {"learning_rate", std::to_string(learning_rate_)},
        {"max_depth", std::to_string(max_depth_)},
        {"gamma", std::to_string(gamma_)},
        {"reg_alpha", std::to_string(reg_alpha_)},
        {"reg_lambda", std::to_string(reg_lambda_)},
        {"min_child_weight", std::to_string(min_child_weight_)},
        {"subsample", std::to_string(subsample_)},
        {"colsample_bytree", std::to_string(colsample_bytree_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& XGBClassifier::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    learning_rate_ = utils::get_param_double(params, "learning_rate", learning_rate_);
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    gamma_ = utils::get_param_double(params, "gamma", gamma_);
    reg_alpha_ = utils::get_param_double(params, "reg_alpha", reg_alpha_);
    reg_lambda_ = utils::get_param_double(params, "reg_lambda", reg_lambda_);
    min_child_weight_ = utils::get_param_int(params, "min_child_weight", min_child_weight_);
    subsample_ = utils::get_param_double(params, "subsample", subsample_);
    colsample_bytree_ = utils::get_param_double(params, "colsample_bytree", colsample_bytree_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// XGBoost Regressor Implementation
Estimator& XGBRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    int n_samples = X.rows();
    
    // Initialize base score (mean of target)
    base_score_ = y.mean();
    
    // Initialize predictions
    VectorXd predictions = VectorXd::Constant(n_samples, base_score_);
    
    estimators_.clear();
    estimators_.reserve(n_estimators_);
    
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    
    // Sample indices for subsample
    std::vector<int> sample_indices(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        sample_indices[i] = i;
    }
    
    for (int t = 0; t < n_estimators_; ++t) {
        // Subsample data if needed
        std::vector<int> current_samples = sample_indices;
        if (subsample_ < 1.0) {
            std::shuffle(current_samples.begin(), current_samples.end(), rng);
            int n_subsample = static_cast<int>(subsample_ * n_samples);
            current_samples.resize(n_subsample);
        }
        
        // Compute gradients (negative residuals for squared loss)
        VectorXd gradients = y - predictions;
        
        // Hessians for squared loss are constant (2.0)
        VectorXd hessians = VectorXd::Constant(n_samples, 2.0);
        
        // Create subsampled dataset
        MatrixXd X_subsample(current_samples.size(), n_features_);
        VectorXd gradients_subsample(current_samples.size());
        
        for (size_t i = 0; i < current_samples.size(); ++i) {
            int idx = current_samples[i];
            X_subsample.row(i) = X.row(idx);
            gradients_subsample(i) = gradients(idx);
        }
        
        // Fit tree to gradients
        tree::DecisionTreeRegressor tree("mse", max_depth_, 2, min_child_weight_, gamma_);
        tree.fit(X_subsample, gradients_subsample);
        
        // Predict and update
        VectorXd tree_pred = tree.predict(X);
        
        // Update predictions with learning rate
        // Standard gradient boosting update: tree is fitted to gradients
        for (int i = 0; i < n_samples; ++i) {
            predictions(i) += learning_rate_ * tree_pred(i);
        }
        
        estimators_.push_back(std::move(tree));
    }
    
    fitted_ = true;
    return static_cast<Estimator&>(*this);
}

VectorXd XGBRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("XGBRegressor must be fitted before predict");
    }
    validation::check_X(X);
    
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    int n_samples = X.rows();
    VectorXd predictions = VectorXd::Constant(n_samples, base_score_);
    
    // Add predictions from all trees
    for (const auto& estimator : estimators_) {
        VectorXd tree_pred = estimator.predict(X);
        predictions += learning_rate_ * tree_pred;
    }
    
    return predictions;
}

Params XGBRegressor::get_params() const {
    return {
        {"n_estimators", std::to_string(n_estimators_)},
        {"learning_rate", std::to_string(learning_rate_)},
        {"max_depth", std::to_string(max_depth_)},
        {"gamma", std::to_string(gamma_)},
        {"reg_alpha", std::to_string(reg_alpha_)},
        {"reg_lambda", std::to_string(reg_lambda_)},
        {"min_child_weight", std::to_string(min_child_weight_)},
        {"subsample", std::to_string(subsample_)},
        {"colsample_bytree", std::to_string(colsample_bytree_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& XGBRegressor::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    learning_rate_ = utils::get_param_double(params, "learning_rate", learning_rate_);
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    gamma_ = utils::get_param_double(params, "gamma", gamma_);
    reg_alpha_ = utils::get_param_double(params, "reg_alpha", reg_alpha_);
    reg_lambda_ = utils::get_param_double(params, "reg_lambda", reg_lambda_);
    min_child_weight_ = utils::get_param_int(params, "min_child_weight", min_child_weight_);
    subsample_ = utils::get_param_double(params, "subsample", subsample_);
    colsample_bytree_ = utils::get_param_double(params, "colsample_bytree", colsample_bytree_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

} // namespace ensemble
} // namespace cxml

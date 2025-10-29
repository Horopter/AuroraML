#include "auroraml/gradient_boosting.hpp"
#include "auroraml/base.hpp"
#include "auroraml/tree.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <set>
#include <algorithm>

namespace auroraml {
namespace ensemble {

// Gradient Boosting Classifier Implementation
Estimator& GradientBoostingClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    int n_samples = X.rows();
    
    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    classes_.assign(unique_classes_set.begin(), unique_classes_set.end());
    int n_classes = classes_.size();
    
    if (n_classes < 2) {
        throw std::invalid_argument("GradientBoostingClassifier requires at least 2 classes");
    }
    
    // Initialize predictions to log-odds
    init_prediction_ = VectorXd::Zero(n_classes);
    estimators_.clear();
    estimators_.reserve(n_estimators_ * n_classes);
    
    // Convert labels to one-hot encoding
    MatrixXd y_onehot = MatrixXd::Zero(n_samples, n_classes);
    for (int i = 0; i < n_samples; ++i) {
        int class_idx = std::find(classes_.begin(), classes_.end(), static_cast<int>(y(i))) - classes_.begin();
        y_onehot(i, class_idx) = 1.0;
    }
    
    // Initialize predictions with class probabilities
    VectorXd class_counts = y_onehot.colwise().sum();
    VectorXd class_probs = class_counts / class_counts.sum();
    
    // Convert probabilities to log-odds
    for (int k = 0; k < n_classes; ++k) {
        init_prediction_(k) = std::log(class_probs(k) / (1.0 - class_probs(k) + 1e-15));
    }
    
    // Initialize predictions matrix
    MatrixXd predictions = MatrixXd::Constant(n_samples, n_classes, 0.0);
    for (int i = 0; i < n_samples; ++i) {
        predictions.row(i) = init_prediction_.transpose();
    }
    
    // Gradient boosting iterations
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    
    for (int t = 0; t < n_estimators_; ++t) {
        // Compute probabilities using softmax
        MatrixXd probabilities = MatrixXd::Zero(n_samples, n_classes);
        for (int i = 0; i < n_samples; ++i) {
            VectorXd exp_preds = predictions.row(i).array().exp();
            double sum_exp = exp_preds.sum();
            probabilities.row(i) = exp_preds.transpose() / sum_exp;
        }
        
        // Compute negative gradients (residuals)
        MatrixXd gradients = y_onehot - probabilities;
        
        // Fit a tree for each class
        for (int k = 0; k < n_classes; ++k) {
            VectorXd gradient_k = gradients.col(k);
            
            tree::DecisionTreeRegressor tree("mse", max_depth_, min_samples_split_, 
                                           min_samples_leaf_, min_impurity_decrease_);
            tree.fit(X, gradient_k);
            
            // Update predictions
            VectorXd tree_pred = tree.predict(X);
            predictions.col(k) += learning_rate_ * tree_pred;
            
            estimators_.push_back(std::move(tree));
        }
    }
    
    fitted_ = true;
    return static_cast<Estimator&>(*this);
}

VectorXi GradientBoostingClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("GradientBoostingClassifier must be fitted before predict");
    validation::check_X(X);
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    int n_samples = X.rows();
    int n_classes = classes_.size();
    
    // Initialize predictions
    MatrixXd predictions = MatrixXd::Constant(n_samples, n_classes, 0.0);
    for (int i = 0; i < n_samples; ++i) {
        predictions.row(i) = init_prediction_.transpose();
    }
    
    // Apply all estimators
    int estimator_idx = 0;
    for (int t = 0; t < n_estimators_; ++t) {
        for (int k = 0; k < n_classes; ++k) {
            VectorXd tree_pred = estimators_[estimator_idx].predict(X);
            predictions.col(k) += learning_rate_ * tree_pred;
            estimator_idx++;
        }
    }
    
    // Convert to probabilities and predict classes
    VectorXi y_pred(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        VectorXd exp_preds = predictions.row(i).array().exp();
        double sum_exp = exp_preds.sum();
        VectorXd probabilities = exp_preds / sum_exp;
        
        int best_class = 0;
        double best_prob = probabilities(0);
        for (int k = 1; k < n_classes; ++k) {
            if (probabilities(k) > best_prob) {
                best_prob = probabilities(k);
                best_class = k;
            }
        }
        y_pred(i) = classes_[best_class];
    }
    
    return y_pred;
}

MatrixXd GradientBoostingClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("GradientBoostingClassifier must be fitted before predict_proba");
    validation::check_X(X);
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    int n_samples = X.rows();
    int n_classes = classes_.size();
    
    // Initialize predictions
    MatrixXd predictions = MatrixXd::Constant(n_samples, n_classes, 0.0);
    for (int i = 0; i < n_samples; ++i) {
        predictions.row(i) = init_prediction_.transpose();
    }
    
    // Apply all estimators
    int estimator_idx = 0;
    for (int t = 0; t < n_estimators_; ++t) {
        for (int k = 0; k < n_classes; ++k) {
            VectorXd tree_pred = estimators_[estimator_idx].predict(X);
            predictions.col(k) += learning_rate_ * tree_pred;
            estimator_idx++;
        }
    }
    
    // Convert to probabilities
    MatrixXd probabilities = MatrixXd::Zero(n_samples, n_classes);
    for (int i = 0; i < n_samples; ++i) {
        VectorXd exp_preds = predictions.row(i).array().exp();
        double sum_exp = exp_preds.sum();
        probabilities.row(i) = exp_preds.transpose() / sum_exp;
    }
    
    return probabilities;
}

VectorXd GradientBoostingClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("GradientBoostingClassifier must be fitted before decision_function");
    validation::check_X(X);
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    int n_samples = X.rows();
    int n_classes = classes_.size();
    
    // Initialize predictions
    MatrixXd predictions = MatrixXd::Constant(n_samples, n_classes, 0.0);
    for (int i = 0; i < n_samples; ++i) {
        predictions.row(i) = init_prediction_.transpose();
    }
    
    // Apply all estimators
    int estimator_idx = 0;
    for (int t = 0; t < n_estimators_; ++t) {
        for (int k = 0; k < n_classes; ++k) {
            VectorXd tree_pred = estimators_[estimator_idx].predict(X);
            predictions.col(k) += learning_rate_ * tree_pred;
            estimator_idx++;
        }
    }
    
    // For binary classification, return the decision function for the positive class
    if (n_classes == 2) {
        return predictions.col(1) - predictions.col(0);
    } else {
        // For multiclass, return the maximum decision function
        VectorXd max_decisions = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            max_decisions(i) = predictions.row(i).maxCoeff();
        }
        return max_decisions;
    }
}

Params GradientBoostingClassifier::get_params() const {
    return {{"n_estimators", std::to_string(n_estimators_)},
            {"learning_rate", std::to_string(learning_rate_)},
            {"max_depth", std::to_string(max_depth_)},
            {"min_samples_split", std::to_string(min_samples_split_)},
            {"min_samples_leaf", std::to_string(min_samples_leaf_)},
            {"min_impurity_decrease", std::to_string(min_impurity_decrease_)},
            {"random_state", std::to_string(random_state_)}};
}

Estimator& GradientBoostingClassifier::set_params(const Params& params) {
    if (params.find("n_estimators") != params.end()) {
        n_estimators_ = std::stoi(params.at("n_estimators"));
    }
    if (params.find("learning_rate") != params.end()) {
        learning_rate_ = std::stod(params.at("learning_rate"));
    }
    if (params.find("max_depth") != params.end()) {
        max_depth_ = std::stoi(params.at("max_depth"));
    }
    if (params.find("min_samples_split") != params.end()) {
        min_samples_split_ = std::stoi(params.at("min_samples_split"));
    }
    if (params.find("min_samples_leaf") != params.end()) {
        min_samples_leaf_ = std::stoi(params.at("min_samples_leaf"));
    }
    if (params.find("min_impurity_decrease") != params.end()) {
        min_impurity_decrease_ = std::stod(params.at("min_impurity_decrease"));
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
    return static_cast<Estimator&>(*this);
}

// Gradient Boosting Regressor Implementation
Estimator& GradientBoostingRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    int n_samples = X.rows();
    
    // Initialize prediction with mean of target
    init_prediction_ = y.mean();
    
    estimators_.clear();
    estimators_.reserve(n_estimators_);
    
    // Initialize predictions
    VectorXd predictions = VectorXd::Constant(n_samples, init_prediction_);
    
    // Gradient boosting iterations
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    
    for (int t = 0; t < n_estimators_; ++t) {
        // Compute negative gradients (residuals)
        VectorXd gradients = y - predictions;
        
        // Fit a tree to the gradients
        tree::DecisionTreeRegressor tree("mse", max_depth_, min_samples_split_, 
                                       min_samples_leaf_, min_impurity_decrease_);
        tree.fit(X, gradients);
        
        // Update predictions
        VectorXd tree_pred = tree.predict(X);
        predictions += learning_rate_ * tree_pred;
        
        estimators_.push_back(std::move(tree));
    }
    
    fitted_ = true;
    return static_cast<Estimator&>(*this);
}

VectorXd GradientBoostingRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("GradientBoostingRegressor must be fitted before predict");
    validation::check_X(X);
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    int n_samples = X.rows();
    
    // Initialize predictions
    VectorXd predictions = VectorXd::Constant(n_samples, init_prediction_);
    
    // Apply all estimators
    for (const auto& estimator : estimators_) {
        VectorXd tree_pred = estimator.predict(X);
        predictions += learning_rate_ * tree_pred;
    }
    
    return predictions;
}

Params GradientBoostingRegressor::get_params() const {
    return {{"n_estimators", std::to_string(n_estimators_)},
            {"learning_rate", std::to_string(learning_rate_)},
            {"max_depth", std::to_string(max_depth_)},
            {"min_samples_split", std::to_string(min_samples_split_)},
            {"min_samples_leaf", std::to_string(min_samples_leaf_)},
            {"min_impurity_decrease", std::to_string(min_impurity_decrease_)},
            {"random_state", std::to_string(random_state_)}};
}

Estimator& GradientBoostingRegressor::set_params(const Params& params) {
    if (params.find("n_estimators") != params.end()) {
        n_estimators_ = std::stoi(params.at("n_estimators"));
    }
    if (params.find("learning_rate") != params.end()) {
        learning_rate_ = std::stod(params.at("learning_rate"));
    }
    if (params.find("max_depth") != params.end()) {
        max_depth_ = std::stoi(params.at("max_depth"));
    }
    if (params.find("min_samples_split") != params.end()) {
        min_samples_split_ = std::stoi(params.at("min_samples_split"));
    }
    if (params.find("min_samples_leaf") != params.end()) {
        min_samples_leaf_ = std::stoi(params.at("min_samples_leaf"));
    }
    if (params.find("min_impurity_decrease") != params.end()) {
        min_impurity_decrease_ = std::stod(params.at("min_impurity_decrease"));
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
    return static_cast<Estimator&>(*this);
}

// GradientBoostingClassifier save/load implementation
void GradientBoostingClassifier::save(const std::string& filepath) const {
    if (!fitted_) {
        throw std::runtime_error("GradientBoostingClassifier must be fitted before saving");
    }
    
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&n_estimators_), sizeof(n_estimators_));
    ofs.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
    ofs.write(reinterpret_cast<const char*>(&max_depth_), sizeof(max_depth_));
    ofs.write(reinterpret_cast<const char*>(&min_samples_split_), sizeof(min_samples_split_));
    ofs.write(reinterpret_cast<const char*>(&min_samples_leaf_), sizeof(min_samples_leaf_));
    ofs.write(reinterpret_cast<const char*>(&min_impurity_decrease_), sizeof(min_impurity_decrease_));
    ofs.write(reinterpret_cast<const char*>(&random_state_), sizeof(random_state_));
    ofs.write(reinterpret_cast<const char*>(&n_features_), sizeof(n_features_));
    
    // Save classes
    int classes_size = classes_.size();
    ofs.write(reinterpret_cast<const char*>(&classes_size), sizeof(classes_size));
    if (classes_size > 0) {
        ofs.write(reinterpret_cast<const char*>(classes_.data()), classes_size * sizeof(int));
    }
    
    // Save initial predictions
    int init_pred_size = init_prediction_.size();
    ofs.write(reinterpret_cast<const char*>(&init_pred_size), sizeof(init_pred_size));
    if (init_pred_size > 0) {
        ofs.write(reinterpret_cast<const char*>(init_prediction_.data()), init_pred_size * sizeof(double));
    }
    
    // Save number of estimators
    int estimators_size = estimators_.size();
    ofs.write(reinterpret_cast<const char*>(&estimators_size), sizeof(estimators_size));
    
    ofs.close();
    
    // Save each estimator to temporary files
    for (int i = 0; i < estimators_size; ++i) {
        std::string temp_file = filepath + ".estimator_" + std::to_string(i);
        estimators_[i].save(temp_file);
    }
}

void GradientBoostingClassifier::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&n_estimators_), sizeof(n_estimators_));
    ifs.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));
    ifs.read(reinterpret_cast<char*>(&max_depth_), sizeof(max_depth_));
    ifs.read(reinterpret_cast<char*>(&min_samples_split_), sizeof(min_samples_split_));
    ifs.read(reinterpret_cast<char*>(&min_samples_leaf_), sizeof(min_samples_leaf_));
    ifs.read(reinterpret_cast<char*>(&min_impurity_decrease_), sizeof(min_impurity_decrease_));
    ifs.read(reinterpret_cast<char*>(&random_state_), sizeof(random_state_));
    ifs.read(reinterpret_cast<char*>(&n_features_), sizeof(n_features_));
    
    // Load classes
    int classes_size;
    ifs.read(reinterpret_cast<char*>(&classes_size), sizeof(classes_size));
    classes_.resize(classes_size);
    if (classes_size > 0) {
        ifs.read(reinterpret_cast<char*>(classes_.data()), classes_size * sizeof(int));
    }
    
    // Load initial predictions
    int init_pred_size;
    ifs.read(reinterpret_cast<char*>(&init_pred_size), sizeof(init_pred_size));
    init_prediction_.resize(init_pred_size);
    if (init_pred_size > 0) {
        ifs.read(reinterpret_cast<char*>(init_prediction_.data()), init_pred_size * sizeof(double));
    }
    
    // Load number of estimators
    int estimators_size;
    ifs.read(reinterpret_cast<char*>(&estimators_size), sizeof(estimators_size));
    
    ifs.close();
    
    // Load each estimator from temporary files
    estimators_.clear();
    estimators_.reserve(estimators_size);
    for (int i = 0; i < estimators_size; ++i) {
        std::string temp_file = filepath + ".estimator_" + std::to_string(i);
        tree::DecisionTreeRegressor estimator("mse", max_depth_, min_samples_split_, min_samples_leaf_, min_impurity_decrease_);
        estimator.load(temp_file);
        estimators_.push_back(std::move(estimator));
        // Clean up temporary file
        std::remove(temp_file.c_str());
    }
}

// GradientBoostingRegressor save/load implementation
void GradientBoostingRegressor::save(const std::string& filepath) const {
    if (!fitted_) {
        throw std::runtime_error("GradientBoostingRegressor must be fitted before saving");
    }
    
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&n_estimators_), sizeof(n_estimators_));
    ofs.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
    ofs.write(reinterpret_cast<const char*>(&max_depth_), sizeof(max_depth_));
    ofs.write(reinterpret_cast<const char*>(&min_samples_split_), sizeof(min_samples_split_));
    ofs.write(reinterpret_cast<const char*>(&min_samples_leaf_), sizeof(min_samples_leaf_));
    ofs.write(reinterpret_cast<const char*>(&min_impurity_decrease_), sizeof(min_impurity_decrease_));
    ofs.write(reinterpret_cast<const char*>(&random_state_), sizeof(random_state_));
    ofs.write(reinterpret_cast<const char*>(&n_features_), sizeof(n_features_));
    
    // Save initial prediction
    ofs.write(reinterpret_cast<const char*>(&init_prediction_), sizeof(init_prediction_));
    
    // Save number of estimators
    int estimators_size = estimators_.size();
    ofs.write(reinterpret_cast<const char*>(&estimators_size), sizeof(estimators_size));
    
    ofs.close();
    
    // Save each estimator to temporary files
    for (int i = 0; i < estimators_size; ++i) {
        std::string temp_file = filepath + ".estimator_" + std::to_string(i);
        estimators_[i].save(temp_file);
    }
}

void GradientBoostingRegressor::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&n_estimators_), sizeof(n_estimators_));
    ifs.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));
    ifs.read(reinterpret_cast<char*>(&max_depth_), sizeof(max_depth_));
    ifs.read(reinterpret_cast<char*>(&min_samples_split_), sizeof(min_samples_split_));
    ifs.read(reinterpret_cast<char*>(&min_samples_leaf_), sizeof(min_samples_leaf_));
    ifs.read(reinterpret_cast<char*>(&min_impurity_decrease_), sizeof(min_impurity_decrease_));
    ifs.read(reinterpret_cast<char*>(&random_state_), sizeof(random_state_));
    ifs.read(reinterpret_cast<char*>(&n_features_), sizeof(n_features_));
    
    // Load initial prediction
    ifs.read(reinterpret_cast<char*>(&init_prediction_), sizeof(init_prediction_));
    
    // Load number of estimators
    int estimators_size;
    ifs.read(reinterpret_cast<char*>(&estimators_size), sizeof(estimators_size));
    
    ifs.close();
    
    // Load each estimator from temporary files
    estimators_.clear();
    estimators_.reserve(estimators_size);
    for (int i = 0; i < estimators_size; ++i) {
        std::string temp_file = filepath + ".estimator_" + std::to_string(i);
        tree::DecisionTreeRegressor estimator("mse", max_depth_, min_samples_split_, min_samples_leaf_, min_impurity_decrease_);
        estimator.load(temp_file);
        estimators_.push_back(std::move(estimator));
        // Clean up temporary file
        std::remove(temp_file.c_str());
    }
}

} // namespace ensemble
} // namespace cxml

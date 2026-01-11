#include "ingenuityml/tree.hpp"
#include "ingenuityml/base.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <map>
#include <set>
#include <fstream>

namespace ingenuityml {
namespace tree {

// DecisionTreeClassifier implementation
DecisionTreeClassifier::DecisionTreeClassifier(const std::string& criterion, int max_depth,
                                               int min_samples_split, int min_samples_leaf,
                                               double min_impurity_decrease)
    : root_(nullptr), fitted_(false), criterion_(criterion), max_depth_(max_depth),
      min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf),
      min_impurity_decrease_(min_impurity_decrease), n_classes_(0), n_features_(0) {
    if (criterion != "gini" && criterion != "entropy") {
        throw std::invalid_argument("Criterion must be 'gini' or 'entropy'");
    }
    if (min_samples_split < 2) {
        throw std::invalid_argument("min_samples_split must be at least 2");
    }
}

Estimator& DecisionTreeClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    
    // Convert y to integer vector
    VectorXi y_int = y.cast<int>();
    
    // Find unique classes
    std::set<int> unique_classes;
    for (int i = 0; i < y_int.size(); ++i) {
        unique_classes.insert(y_int(i));
    }
    
    classes_ = std::vector<int>(unique_classes.begin(), unique_classes.end());
    n_classes_ = classes_.size();
    X_train_ = X;
    
    // Build tree
    build_tree(X, y_int);
    
    fitted_ = true;
    return *this;
}

VectorXi DecisionTreeClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("DecisionTreeClassifier must be fitted before predict");
    }
    
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXi predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = predict_single(X.row(i));
    }
    
    return predictions;
}

MatrixXd DecisionTreeClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("DecisionTreeClassifier must be fitted before predict_proba");
    }
    
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd probabilities(X.rows(), n_classes_);
    for (int i = 0; i < X.rows(); ++i) {
        probabilities.row(i) = predict_proba_single(X.row(i));
    }
    
    return probabilities;
}

VectorXd DecisionTreeClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("DecisionTreeClassifier must be fitted before decision_function");
    }
    
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXd decision_values(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        auto probs = predict_proba_single(X.row(i));
        // Simple decision function: probability of positive class
        decision_values(i) = probs(1);
    }
    
    return decision_values;
}

Params DecisionTreeClassifier::get_params() const {
    return {
        {"criterion", criterion_},
        {"max_depth", std::to_string(max_depth_)},
        {"min_samples_split", std::to_string(min_samples_split_)},
        {"min_samples_leaf", std::to_string(min_samples_leaf_)},
        {"min_impurity_decrease", std::to_string(min_impurity_decrease_)}
    };
}

Estimator& DecisionTreeClassifier::set_params(const Params& params) {
    criterion_ = utils::get_param_string(params, "criterion", criterion_);
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    min_samples_split_ = utils::get_param_int(params, "min_samples_split", min_samples_split_);
    min_samples_leaf_ = utils::get_param_int(params, "min_samples_leaf", min_samples_leaf_);
    min_impurity_decrease_ = utils::get_param_double(params, "min_impurity_decrease", min_impurity_decrease_);
    return *this;
}

bool DecisionTreeClassifier::is_fitted() const {
    return fitted_;
}

void DecisionTreeClassifier::build_tree(const MatrixXd& X, const VectorXi& y) {
    std::vector<int> samples(X.rows());
    std::iota(samples.begin(), samples.end(), 0);
    
    root_ = build_tree_recursive(X, y, samples, 0);
}

std::unique_ptr<TreeNode> DecisionTreeClassifier::build_tree_recursive(
    const MatrixXd& X, const VectorXi& y, const std::vector<int>& samples, int depth) {
    
    // Check stopping criteria
    if (samples.size() < min_samples_split_) {
        return create_leaf(y, samples);
    }
    
    if (max_depth_ > 0 && depth >= max_depth_) {
        return create_leaf(y, samples);
    }
    
    // Find best split
    auto [feature, threshold] = find_best_split(X, y, samples);
    
    if (feature == -1) {
        return create_leaf(y, samples);
    }
    
    // Split samples
    auto [left_samples, right_samples] = split_samples(X, samples, feature, threshold);
    
    if (left_samples.size() < min_samples_leaf_ || right_samples.size() < min_samples_leaf_) {
        return create_leaf(y, samples);
    }
    
    // Create internal node
    auto node = std::make_unique<TreeNode>();
    node->is_leaf = false;
    node->feature_index = feature;
    node->threshold = threshold;
    
    // Recursively build children
    node->left = build_tree_recursive(X, y, left_samples, depth + 1);
    node->right = build_tree_recursive(X, y, right_samples, depth + 1);
    
    return node;
}

std::pair<int, double> DecisionTreeClassifier::find_best_split(
    const MatrixXd& X, const VectorXi& y, const std::vector<int>& samples) {
    
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_impurity = std::numeric_limits<double>::max();
    
    for (int feature = 0; feature < X.cols(); ++feature) {
        // Get unique values for this feature
        std::set<double> unique_values;
        for (int idx : samples) {
            unique_values.insert(X(idx, feature));
        }
        
        for (double value : unique_values) {
            auto [left_samples, right_samples] = split_samples(X, samples, feature, value);
            
            if (left_samples.empty() || right_samples.empty()) {
                continue;
            }
            
            double left_impurity = (criterion_ == "gini") ? 
                gini_impurity(y, left_samples) : entropy(y, left_samples);
            double right_impurity = (criterion_ == "gini") ? 
                gini_impurity(y, right_samples) : entropy(y, right_samples);
            
            double weighted_impurity = (left_samples.size() * left_impurity + 
                                      right_samples.size() * right_impurity) / samples.size();
            
            if (weighted_impurity < best_impurity) {
                best_impurity = weighted_impurity;
                best_feature = feature;
                best_threshold = value;
            }
        }
    }
    
    return {best_feature, best_threshold};
}

double DecisionTreeClassifier::gini_impurity(const VectorXi& y, const std::vector<int>& samples) {
    std::map<int, int> class_counts;
    for (int idx : samples) {
        class_counts[y(idx)]++;
    }
    
    double gini = 1.0;
    int total = samples.size();
    for (const auto& pair : class_counts) {
        double prob = static_cast<double>(pair.second) / total;
        gini -= prob * prob;
    }
    
    return gini;
}

double DecisionTreeClassifier::entropy(const VectorXi& y, const std::vector<int>& samples) {
    std::map<int, int> class_counts;
    for (int idx : samples) {
        class_counts[y(idx)]++;
    }
    
    double entropy = 0.0;
    int total = samples.size();
    for (const auto& pair : class_counts) {
        double prob = static_cast<double>(pair.second) / total;
        if (prob > 0) {
            entropy -= prob * std::log2(prob);
        }
    }
    
    return entropy;
}

std::pair<std::vector<int>, std::vector<int>> DecisionTreeClassifier::split_samples(
    const MatrixXd& X, const std::vector<int>& samples, int feature, double threshold) {
    
    std::vector<int> left_samples, right_samples;
    
    for (int idx : samples) {
        if (X(idx, feature) <= threshold) {
            left_samples.push_back(idx);
        } else {
            right_samples.push_back(idx);
        }
    }
    
    return {left_samples, right_samples};
}

std::unique_ptr<TreeNode> DecisionTreeClassifier::create_leaf(
    const VectorXi& y, const std::vector<int>& samples) {
    
    auto leaf = std::make_unique<TreeNode>();
    leaf->is_leaf = true;
    
    // Find most common class and store class counts
    std::unordered_map<int, int> class_counts;
    for (int idx : samples) {
        class_counts[y(idx)]++;
    }
    
    // Store class counts in the node
    for (const auto& pair : class_counts) {
        leaf->class_counts[pair.first] = pair.second;
    }
    
    int most_common_class = class_counts.begin()->first;
    int max_count = class_counts.begin()->second;
    
    for (const auto& pair : class_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common_class = pair.first;
        }
    }
    
    leaf->class_label = most_common_class;
    leaf->value = static_cast<double>(most_common_class);
    
    return leaf;
}

int DecisionTreeClassifier::predict_single(const VectorXd& x) const {
    TreeNode* current = root_.get();
    
    while (!current->is_leaf) {
        if (x(current->feature_index) <= current->threshold) {
            current = current->left.get();
        } else {
            current = current->right.get();
        }
    }
    
    return current->class_label;
}

VectorXd DecisionTreeClassifier::predict_proba_single(const VectorXd& x) const {
    TreeNode* current = root_.get();
    
    while (!current->is_leaf) {
        if (x(current->feature_index) <= current->threshold) {
            current = current->left.get();
        } else {
            current = current->right.get();
        }
    }
    
    // Return probability vector based on class distribution at leaf
    VectorXd probs = VectorXd::Zero(n_classes_);
    
    // Calculate total samples at this leaf
    int total_samples = 0;
    for (int cls : classes_) {
        auto it = current->class_counts.find(cls);
        if (it != current->class_counts.end()) {
            total_samples += it->second;
        }
    }
    
    // Set probabilities based on class distribution
    for (int i = 0; i < n_classes_; ++i) {
        int class_label = classes_[i];
        // Use find to check if class exists, default to 0 if not found
        auto it = current->class_counts.find(class_label);
        int count = (it != current->class_counts.end()) ? it->second : 0;
        probs(i) = static_cast<double>(count) / total_samples;
    }
    
    return probs;
}

// DecisionTreeRegressor implementation
DecisionTreeRegressor::DecisionTreeRegressor(const std::string& criterion, int max_depth,
                                           int min_samples_split, int min_samples_leaf,
                                           double min_impurity_decrease)
    : root_(nullptr), fitted_(false), criterion_(criterion), max_depth_(max_depth),
      min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf),
      min_impurity_decrease_(min_impurity_decrease), n_features_(0) {
    if (criterion != "mse") {
        throw std::invalid_argument("Criterion must be 'mse'");
    }
    if (min_samples_split < 2) {
        throw std::invalid_argument("min_samples_split must be at least 2");
    }
}

Estimator& DecisionTreeRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    X_train_ = X;
    
    // Build tree
    build_tree(X, y);
    
    fitted_ = true;
    return *this;
}

VectorXd DecisionTreeRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("DecisionTreeRegressor must be fitted before predict");
    }
    
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = predict_single(X.row(i));
    }
    
    return predictions;
}

Params DecisionTreeRegressor::get_params() const {
    return {
        {"criterion", criterion_},
        {"max_depth", std::to_string(max_depth_)},
        {"min_samples_split", std::to_string(min_samples_split_)},
        {"min_samples_leaf", std::to_string(min_samples_leaf_)},
        {"min_impurity_decrease", std::to_string(min_impurity_decrease_)}
    };
}

Estimator& DecisionTreeRegressor::set_params(const Params& params) {
    criterion_ = utils::get_param_string(params, "criterion", criterion_);
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    min_samples_split_ = utils::get_param_int(params, "min_samples_split", min_samples_split_);
    min_samples_leaf_ = utils::get_param_int(params, "min_samples_leaf", min_samples_leaf_);
    min_impurity_decrease_ = utils::get_param_double(params, "min_impurity_decrease", min_impurity_decrease_);
    return *this;
}

bool DecisionTreeRegressor::is_fitted() const {
    return fitted_;
}

void DecisionTreeRegressor::build_tree(const MatrixXd& X, const VectorXd& y) {
    std::vector<int> samples(X.rows());
    std::iota(samples.begin(), samples.end(), 0);
    
    root_ = build_tree_recursive(X, y, samples, 0);
}

std::unique_ptr<TreeNode> DecisionTreeRegressor::build_tree_recursive(
    const MatrixXd& X, const VectorXd& y, const std::vector<int>& samples, int depth) {
    
    // Check stopping criteria
    if (samples.size() < min_samples_split_) {
        return create_leaf(y, samples);
    }
    
    if (max_depth_ > 0 && depth >= max_depth_) {
        return create_leaf(y, samples);
    }
    
    // Find best split
    auto [feature, threshold] = find_best_split(X, y, samples);
    
    if (feature == -1) {
        return create_leaf(y, samples);
    }
    
    // Split samples
    auto [left_samples, right_samples] = split_samples(X, samples, feature, threshold);
    
    if (left_samples.size() < min_samples_leaf_ || right_samples.size() < min_samples_leaf_) {
        return create_leaf(y, samples);
    }
    
    // Create internal node
    auto node = std::make_unique<TreeNode>();
    node->is_leaf = false;
    node->feature_index = feature;
    node->threshold = threshold;
    
    // Recursively build children
    node->left = build_tree_recursive(X, y, left_samples, depth + 1);
    node->right = build_tree_recursive(X, y, right_samples, depth + 1);
    
    return node;
}

std::pair<int, double> DecisionTreeRegressor::find_best_split(
    const MatrixXd& X, const VectorXd& y, const std::vector<int>& samples) {
    
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_mse = std::numeric_limits<double>::max();
    
    for (int feature = 0; feature < X.cols(); ++feature) {
        // Get unique values for this feature
        std::set<double> unique_values;
        for (int idx : samples) {
            unique_values.insert(X(idx, feature));
        }
        
        for (double value : unique_values) {
            auto [left_samples, right_samples] = split_samples(X, samples, feature, value);
            
            if (left_samples.empty() || right_samples.empty()) {
                continue;
            }
            
            double left_mse = calculate_mse(y, left_samples);
            double right_mse = calculate_mse(y, right_samples);
            
            double weighted_mse = (left_samples.size() * left_mse + 
                                 right_samples.size() * right_mse) / samples.size();
            
            if (weighted_mse < best_mse) {
                best_mse = weighted_mse;
                best_feature = feature;
                best_threshold = value;
            }
        }
    }
    
    return {best_feature, best_threshold};
}

double DecisionTreeRegressor::calculate_mse(const VectorXd& y, const std::vector<int>& samples) {
    if (samples.empty()) return 0.0;
    
    double mean = 0.0;
    for (int idx : samples) {
        mean += y(idx);
    }
    mean /= samples.size();
    
    double mse = 0.0;
    for (int idx : samples) {
        double diff = y(idx) - mean;
        mse += diff * diff;
    }
    
    return mse / samples.size();
}

std::pair<std::vector<int>, std::vector<int>> DecisionTreeRegressor::split_samples(
    const MatrixXd& X, const std::vector<int>& samples, int feature, double threshold) {
    
    std::vector<int> left_samples, right_samples;
    
    for (int idx : samples) {
        if (X(idx, feature) <= threshold) {
            left_samples.push_back(idx);
        } else {
            right_samples.push_back(idx);
        }
    }
    
    return {left_samples, right_samples};
}

std::unique_ptr<TreeNode> DecisionTreeRegressor::create_leaf(
    const VectorXd& y, const std::vector<int>& samples) {
    
    auto leaf = std::make_unique<TreeNode>();
    leaf->is_leaf = true;
    
    // Calculate mean value
    double mean = 0.0;
    for (int idx : samples) {
        mean += y(idx);
    }
    mean /= samples.size();
    
    leaf->value = mean;
    
    return leaf;
}

double DecisionTreeRegressor::predict_single(const VectorXd& x) const {
    TreeNode* current = root_.get();
    
    while (!current->is_leaf) {
        if (x(current->feature_index) <= current->threshold) {
            current = current->left.get();
        } else {
            current = current->right.get();
        }
    }
    
    return current->value;
}

// Helper functions for tree serialization
void save_tree_recursive(TreeNode* node, std::ofstream& ofs) {
    if (!node) {
        bool is_null = true;
        ofs.write(reinterpret_cast<const char*>(&is_null), sizeof(is_null));
        return;
    }
    
    bool is_null = false;
    ofs.write(reinterpret_cast<const char*>(&is_null), sizeof(is_null));
    
    // Save node data
    ofs.write(reinterpret_cast<const char*>(&node->is_leaf), sizeof(node->is_leaf));
    ofs.write(reinterpret_cast<const char*>(&node->feature_index), sizeof(node->feature_index));
    ofs.write(reinterpret_cast<const char*>(&node->threshold), sizeof(node->threshold));
    ofs.write(reinterpret_cast<const char*>(&node->value), sizeof(node->value));
    ofs.write(reinterpret_cast<const char*>(&node->class_label), sizeof(node->class_label));
    
    // Save class_counts
    int class_counts_size = node->class_counts.size();
    ofs.write(reinterpret_cast<const char*>(&class_counts_size), sizeof(class_counts_size));
    for (const auto& pair : node->class_counts) {
        ofs.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        ofs.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }
    
    // Recursively save children
    save_tree_recursive(node->left.get(), ofs);
    save_tree_recursive(node->right.get(), ofs);
}

std::unique_ptr<TreeNode> load_tree_recursive(std::ifstream& ifs) {
    bool is_null;
    ifs.read(reinterpret_cast<char*>(&is_null), sizeof(is_null));
    if (is_null) {
        return nullptr;
    }
    
    auto node = std::make_unique<TreeNode>();
    
    // Load node data
    ifs.read(reinterpret_cast<char*>(&node->is_leaf), sizeof(node->is_leaf));
    ifs.read(reinterpret_cast<char*>(&node->feature_index), sizeof(node->feature_index));
    ifs.read(reinterpret_cast<char*>(&node->threshold), sizeof(node->threshold));
    ifs.read(reinterpret_cast<char*>(&node->value), sizeof(node->value));
    ifs.read(reinterpret_cast<char*>(&node->class_label), sizeof(node->class_label));
    
    // Load class_counts
    int class_counts_size;
    ifs.read(reinterpret_cast<char*>(&class_counts_size), sizeof(class_counts_size));
    for (int i = 0; i < class_counts_size; ++i) {
        int key, value;
        ifs.read(reinterpret_cast<char*>(&key), sizeof(key));
        ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
        node->class_counts[key] = value;
    }
    
    // Recursively load children
    node->left = load_tree_recursive(ifs);
    node->right = load_tree_recursive(ifs);
    
    return node;
}

// DecisionTreeClassifier save/load implementation
void DecisionTreeClassifier::save(const std::string& filepath) const {
    if (!fitted_) {
        throw std::runtime_error("DecisionTreeClassifier must be fitted before saving");
    }
    
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&n_classes_), sizeof(n_classes_));
    ofs.write(reinterpret_cast<const char*>(&n_features_), sizeof(n_features_));
    ofs.write(reinterpret_cast<const char*>(&max_depth_), sizeof(max_depth_));
    ofs.write(reinterpret_cast<const char*>(&min_samples_split_), sizeof(min_samples_split_));
    ofs.write(reinterpret_cast<const char*>(&min_samples_leaf_), sizeof(min_samples_leaf_));
    ofs.write(reinterpret_cast<const char*>(&min_impurity_decrease_), sizeof(min_impurity_decrease_));
    
    // Save criterion string
    size_t criterion_len = criterion_.length();
    ofs.write(reinterpret_cast<const char*>(&criterion_len), sizeof(criterion_len));
    ofs.write(criterion_.c_str(), criterion_len);
    
    // Save classes
    int classes_size = classes_.size();
    ofs.write(reinterpret_cast<const char*>(&classes_size), sizeof(classes_size));
    if (classes_size > 0) {
        ofs.write(reinterpret_cast<const char*>(classes_.data()), classes_size * sizeof(int));
    }
    
    // Save tree structure recursively
    save_tree_recursive(root_.get(), ofs);
    
    ofs.close();
}

void DecisionTreeClassifier::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&n_classes_), sizeof(n_classes_));
    ifs.read(reinterpret_cast<char*>(&n_features_), sizeof(n_features_));
    ifs.read(reinterpret_cast<char*>(&max_depth_), sizeof(max_depth_));
    ifs.read(reinterpret_cast<char*>(&min_samples_split_), sizeof(min_samples_split_));
    ifs.read(reinterpret_cast<char*>(&min_samples_leaf_), sizeof(min_samples_leaf_));
    ifs.read(reinterpret_cast<char*>(&min_impurity_decrease_), sizeof(min_impurity_decrease_));
    
    // Load criterion string
    size_t criterion_len;
    ifs.read(reinterpret_cast<char*>(&criterion_len), sizeof(criterion_len));
    criterion_.resize(criterion_len);
    ifs.read(&criterion_[0], criterion_len);
    
    // Load classes
    int classes_size;
    ifs.read(reinterpret_cast<char*>(&classes_size), sizeof(classes_size));
    classes_.resize(classes_size);
    if (classes_size > 0) {
        ifs.read(reinterpret_cast<char*>(classes_.data()), classes_size * sizeof(int));
    }
    
    // Load tree structure recursively
    root_ = load_tree_recursive(ifs);
    
    ifs.close();
}

// DecisionTreeRegressor save/load implementation
void DecisionTreeRegressor::save(const std::string& filepath) const {
    if (!fitted_) {
        throw std::runtime_error("DecisionTreeRegressor must be fitted before saving");
    }
    
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&n_features_), sizeof(n_features_));
    ofs.write(reinterpret_cast<const char*>(&max_depth_), sizeof(max_depth_));
    ofs.write(reinterpret_cast<const char*>(&min_samples_split_), sizeof(min_samples_split_));
    ofs.write(reinterpret_cast<const char*>(&min_samples_leaf_), sizeof(min_samples_leaf_));
    ofs.write(reinterpret_cast<const char*>(&min_impurity_decrease_), sizeof(min_impurity_decrease_));
    
    // Save criterion string
    size_t criterion_len = criterion_.length();
    ofs.write(reinterpret_cast<const char*>(&criterion_len), sizeof(criterion_len));
    ofs.write(criterion_.c_str(), criterion_len);
    
    // Save tree structure recursively
    save_tree_recursive(root_.get(), ofs);
    
    ofs.close();
}

void DecisionTreeRegressor::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&n_features_), sizeof(n_features_));
    ifs.read(reinterpret_cast<char*>(&max_depth_), sizeof(max_depth_));
    ifs.read(reinterpret_cast<char*>(&min_samples_split_), sizeof(min_samples_split_));
    ifs.read(reinterpret_cast<char*>(&min_samples_leaf_), sizeof(min_samples_leaf_));
    ifs.read(reinterpret_cast<char*>(&min_impurity_decrease_), sizeof(min_impurity_decrease_));
    
    // Load criterion string
    size_t criterion_len;
    ifs.read(reinterpret_cast<char*>(&criterion_len), sizeof(criterion_len));
    criterion_.resize(criterion_len);
    ifs.read(&criterion_[0], criterion_len);
    
    // Load tree structure recursively
    root_ = load_tree_recursive(ifs);
    
    ifs.close();
}

} // namespace tree
} // namespace ingenuityml
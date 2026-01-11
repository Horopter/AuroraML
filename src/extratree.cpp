#include "ingenuityml/extratree.hpp"
#include "ingenuityml/base.hpp"
#include <set>
#include <random>
#include <algorithm>

namespace ingenuityml {
namespace tree {

// ExtraTreeClassifier implementation

ExtraTreeClassifier::ExtraTreeClassifier(
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int random_state
) : max_depth_(max_depth), min_samples_split_(min_samples_split),
    min_samples_leaf_(min_samples_leaf), max_features_(max_features),
    random_state_(random_state), fitted_(false), n_features_(0), n_classes_(0) {
}

Estimator& ExtraTreeClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    
    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    
    classes_.resize(unique_classes_set.size());
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }
    n_classes_ = classes_.size();
    
    // Convert y to integer vector
    VectorXi y_int = y.cast<int>();
    
    // Create all sample indices
    std::vector<int> samples;
    for (int i = 0; i < X.rows(); ++i) {
        samples.push_back(i);
    }
    
    // Initialize random number generator
    std::mt19937 rng;
    if (random_state_ >= 0) {
        rng.seed(random_state_);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    // Build tree
    root_ = build_tree_recursive(X, y_int, samples, 0, rng);
    
    fitted_ = true;
    return *this;
}

std::unique_ptr<TreeNode> ExtraTreeClassifier::build_tree_recursive(
    const MatrixXd& X, const VectorXi& y,
    const std::vector<int>& samples, int depth,
    std::mt19937& rng) {
    
    if (samples.empty()) {
        return nullptr;
    }
    
    // Check stopping criteria
    bool same_class = true;
    int first_class = y(samples[0]);
    for (int idx : samples) {
        if (y(idx) != first_class) {
            same_class = false;
            break;
        }
    }
    
    if (same_class || samples.size() < min_samples_split_ ||
        (max_depth_ > 0 && depth >= max_depth_)) {
        auto leaf = std::make_unique<TreeNode>();
        leaf->is_leaf = true;
        
        // Count classes
        std::map<int, int> class_counts;
        for (int idx : samples) {
            class_counts[y(idx)]++;
        }
        
        // Find most common class
        int max_count = 0;
        int majority_class = first_class;
        for (const auto& [cls, count] : class_counts) {
            if (count > max_count) {
                max_count = count;
                majority_class = cls;
            }
        }
        
        leaf->class_label = majority_class;
        leaf->value = static_cast<double>(max_count) / samples.size();
        leaf->class_counts = class_counts;
        return leaf;
    }
    
    // Find random split
    auto [feature, threshold] = find_random_split(X, y, samples, rng);
    
    if (feature < 0) {
        // Cannot split further
        auto leaf = std::make_unique<TreeNode>();
        leaf->is_leaf = true;
        std::map<int, int> class_counts;
        for (int idx : samples) {
            class_counts[y(idx)]++;
        }
        int max_count = 0;
        int majority_class = first_class;
        for (const auto& [cls, count] : class_counts) {
            if (count > max_count) {
                max_count = count;
                majority_class = cls;
            }
        }
        leaf->class_label = majority_class;
        leaf->value = static_cast<double>(max_count) / samples.size();
        leaf->class_counts = class_counts;
        return leaf;
    }
    
    // Split samples
    std::vector<int> left_samples, right_samples;
    for (int idx : samples) {
        if (X(idx, feature) <= threshold) {
            left_samples.push_back(idx);
        } else {
            right_samples.push_back(idx);
        }
    }
    
    // Check minimum samples per leaf
    if (left_samples.size() < min_samples_leaf_ || right_samples.size() < min_samples_leaf_) {
        auto leaf = std::make_unique<TreeNode>();
        leaf->is_leaf = true;
        std::map<int, int> class_counts;
        for (int idx : samples) {
            class_counts[y(idx)]++;
        }
        int max_count = 0;
        int majority_class = first_class;
        for (const auto& [cls, count] : class_counts) {
            if (count > max_count) {
                max_count = count;
                majority_class = cls;
            }
        }
        leaf->class_label = majority_class;
        leaf->value = static_cast<double>(max_count) / samples.size();
        leaf->class_counts = class_counts;
        return leaf;
    }
    
    // Create node
    auto node = std::make_unique<TreeNode>();
    node->is_leaf = false;
    node->feature_index = feature;
    node->threshold = threshold;
    
    // Recursively build children
    node->left = build_tree_recursive(X, y, left_samples, depth + 1, rng);
    node->right = build_tree_recursive(X, y, right_samples, depth + 1, rng);
    
    return node;
}

std::pair<int, double> ExtraTreeClassifier::find_random_split(
    const MatrixXd& X, const VectorXi& y,
    const std::vector<int>& samples, std::mt19937& rng) {
    
    if (samples.empty()) {
        return {-1, 0.0};
    }
    
    // Randomly select features to consider
    int n_features_to_try = max_features_;
    if (n_features_to_try <= 0 || n_features_to_try > n_features_) {
        n_features_to_try = static_cast<int>(std::sqrt(n_features_));
    }
    
    std::vector<int> feature_indices(n_features_);
    std::iota(feature_indices.begin(), feature_indices.end(), 0);
    std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
    
    // Try random splits
    for (int f = 0; f < std::min(n_features_to_try, n_features_); ++f) {
        int feature = feature_indices[f];
        
        // Find min and max values for this feature
        double min_val = X(samples[0], feature);
        double max_val = X(samples[0], feature);
        for (int idx : samples) {
            if (X(idx, feature) < min_val) min_val = X(idx, feature);
            if (X(idx, feature) > max_val) max_val = X(idx, feature);
        }
        
        if (min_val >= max_val) {
            continue;
        }
        
        // Random threshold
        std::uniform_real_distribution<double> dist(min_val, max_val);
        double threshold = dist(rng);
        
        return {feature, threshold};
    }
    
    return {-1, 0.0};
}

VectorXi ExtraTreeClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ExtraTreeClassifier must be fitted before predict");
    }
    
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        TreeNode* node = root_.get();
        while (!node->is_leaf) {
            if (X(i, node->feature_index) <= node->threshold) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        predictions(i) = node->class_label;
    }
    
    return predictions;
}

MatrixXd ExtraTreeClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ExtraTreeClassifier must be fitted before predict_proba");
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), n_classes_);
    
    for (int i = 0; i < X.rows(); ++i) {
        TreeNode* node = root_.get();
        while (!node->is_leaf) {
            if (X(i, node->feature_index) <= node->threshold) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        
        // Convert class counts to probabilities
        int total = 0;
        for (const auto& [cls, count] : node->class_counts) {
            total += count;
        }
        
        for (int c = 0; c < n_classes_; ++c) {
            int class_label = classes_[c];
            auto it = node->class_counts.find(class_label);
            if (it != node->class_counts.end()) {
                proba(i, c) = static_cast<double>(it->second) / total;
            }
        }
    }
    
    return proba;
}

VectorXd ExtraTreeClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ExtraTreeClassifier must be fitted before decision_function");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXd decision = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        decision(i) = proba.row(i).maxCoeff();
    }
    
    return decision;
}

Params ExtraTreeClassifier::get_params() const {
    Params params;
    params["max_depth"] = std::to_string(max_depth_);
    params["min_samples_split"] = std::to_string(min_samples_split_);
    params["min_samples_leaf"] = std::to_string(min_samples_leaf_);
    params["max_features"] = std::to_string(max_features_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& ExtraTreeClassifier::set_params(const Params& params) {
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    min_samples_split_ = utils::get_param_int(params, "min_samples_split", min_samples_split_);
    min_samples_leaf_ = utils::get_param_int(params, "min_samples_leaf", min_samples_leaf_);
    max_features_ = utils::get_param_int(params, "max_features", max_features_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// ExtraTreeRegressor implementation

ExtraTreeRegressor::ExtraTreeRegressor(
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int random_state
) : max_depth_(max_depth), min_samples_split_(min_samples_split),
    min_samples_leaf_(min_samples_leaf), max_features_(max_features),
    random_state_(random_state), fitted_(false), n_features_(0) {
}

Estimator& ExtraTreeRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    
    std::vector<int> samples;
    for (int i = 0; i < X.rows(); ++i) {
        samples.push_back(i);
    }
    
    std::mt19937 rng;
    if (random_state_ >= 0) {
        rng.seed(random_state_);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    root_ = build_tree_recursive(X, y, samples, 0, rng);
    
    fitted_ = true;
    return *this;
}

std::unique_ptr<TreeNode> ExtraTreeRegressor::build_tree_recursive(
    const MatrixXd& X, const VectorXd& y,
    const std::vector<int>& samples, int depth,
    std::mt19937& rng) {
    
    if (samples.empty()) {
        return nullptr;
    }
    
    // Check stopping criteria
    if (samples.size() < min_samples_split_ || (max_depth_ > 0 && depth >= max_depth_)) {
        auto leaf = std::make_unique<TreeNode>();
        leaf->is_leaf = true;
        
        // Compute mean value
        double mean = 0.0;
        for (int idx : samples) {
            mean += y(idx);
        }
        mean /= samples.size();
        
        leaf->value = mean;
        return leaf;
    }
    
    // Find random split
    auto [feature, threshold] = find_random_split(X, y, samples, rng);
    
    if (feature < 0) {
        auto leaf = std::make_unique<TreeNode>();
        leaf->is_leaf = true;
        double mean = 0.0;
        for (int idx : samples) {
            mean += y(idx);
        }
        mean /= samples.size();
        leaf->value = mean;
        return leaf;
    }
    
    // Split samples
    std::vector<int> left_samples, right_samples;
    for (int idx : samples) {
        if (X(idx, feature) <= threshold) {
            left_samples.push_back(idx);
        } else {
            right_samples.push_back(idx);
        }
    }
    
    if (left_samples.size() < min_samples_leaf_ || right_samples.size() < min_samples_leaf_) {
        auto leaf = std::make_unique<TreeNode>();
        leaf->is_leaf = true;
        double mean = 0.0;
        for (int idx : samples) {
            mean += y(idx);
        }
        mean /= samples.size();
        leaf->value = mean;
        return leaf;
    }
    
    auto node = std::make_unique<TreeNode>();
    node->is_leaf = false;
    node->feature_index = feature;
    node->threshold = threshold;
    
    node->left = build_tree_recursive(X, y, left_samples, depth + 1, rng);
    node->right = build_tree_recursive(X, y, right_samples, depth + 1, rng);
    
    return node;
}

std::pair<int, double> ExtraTreeRegressor::find_random_split(
    const MatrixXd& X, const VectorXd& y,
    const std::vector<int>& samples, std::mt19937& rng) {
    
    if (samples.empty()) {
        return {-1, 0.0};
    }
    
    int n_features_to_try = max_features_;
    if (n_features_to_try <= 0 || n_features_to_try > n_features_) {
        n_features_to_try = static_cast<int>(std::sqrt(n_features_));
    }
    
    std::vector<int> feature_indices(n_features_);
    std::iota(feature_indices.begin(), feature_indices.end(), 0);
    std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
    
    for (int f = 0; f < std::min(n_features_to_try, n_features_); ++f) {
        int feature = feature_indices[f];
        
        double min_val = X(samples[0], feature);
        double max_val = X(samples[0], feature);
        for (int idx : samples) {
            if (X(idx, feature) < min_val) min_val = X(idx, feature);
            if (X(idx, feature) > max_val) max_val = X(idx, feature);
        }
        
        if (min_val >= max_val) {
            continue;
        }
        
        std::uniform_real_distribution<double> dist(min_val, max_val);
        double threshold = dist(rng);
        
        return {feature, threshold};
    }
    
    return {-1, 0.0};
}

VectorXd ExtraTreeRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ExtraTreeRegressor must be fitted before predict");
    }
    
    VectorXd predictions = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        TreeNode* node = root_.get();
        while (!node->is_leaf) {
            if (X(i, node->feature_index) <= node->threshold) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        predictions(i) = node->value;
    }
    
    return predictions;
}

Params ExtraTreeRegressor::get_params() const {
    Params params;
    params["max_depth"] = std::to_string(max_depth_);
    params["min_samples_split"] = std::to_string(min_samples_split_);
    params["min_samples_leaf"] = std::to_string(min_samples_leaf_);
    params["max_features"] = std::to_string(max_features_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& ExtraTreeRegressor::set_params(const Params& params) {
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    min_samples_split_ = utils::get_param_int(params, "min_samples_split", min_samples_split_);
    min_samples_leaf_ = utils::get_param_int(params, "min_samples_leaf", min_samples_leaf_);
    max_features_ = utils::get_param_int(params, "max_features", max_features_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

} // namespace tree
} // namespace ingenuityml


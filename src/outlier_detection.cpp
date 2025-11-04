#include "auroraml/outlier_detection.hpp"
#include "auroraml/base.hpp"
#include <random>
#include <algorithm>
#include <cmath>

namespace auroraml {
namespace outlier_detection {

// IsolationForest implementation

IsolationForest::IsolationForest(
    int n_estimators,
    int max_samples,
    double contamination,
    int random_state
) : n_estimators_(n_estimators), max_samples_(max_samples),
    contamination_(contamination), random_state_(random_state), fitted_(false) {
}

std::unique_ptr<tree::TreeNode> IsolationForest::build_isolation_tree(
    const MatrixXd& X, const std::vector<int>& samples, int depth, int max_depth, std::mt19937& rng) {
    
    if (samples.empty() || depth >= max_depth) {
        auto leaf = std::make_unique<tree::TreeNode>();
        leaf->is_leaf = true;
        leaf->value = static_cast<double>(depth);
        return leaf;
    }
    
    // Randomly select a feature
    std::uniform_int_distribution<int> feature_dist(0, X.cols() - 1);
    int feature = feature_dist(rng);
    
    // Find min and max for this feature
    double min_val = X(samples[0], feature);
    double max_val = X(samples[0], feature);
    for (int idx : samples) {
        if (X(idx, feature) < min_val) min_val = X(idx, feature);
        if (X(idx, feature) > max_val) max_val = X(idx, feature);
    }
    
    if (min_val >= max_val) {
        auto leaf = std::make_unique<tree::TreeNode>();
        leaf->is_leaf = true;
        leaf->value = static_cast<double>(depth);
        return leaf;
    }
    
    // Random threshold
    std::uniform_real_distribution<double> threshold_dist(min_val, max_val);
    double threshold = threshold_dist(rng);
    
    // Split samples
    std::vector<int> left_samples, right_samples;
    for (int idx : samples) {
        if (X(idx, feature) <= threshold) {
            left_samples.push_back(idx);
        } else {
            right_samples.push_back(idx);
        }
    }
    
    auto node = std::make_unique<tree::TreeNode>();
    node->is_leaf = false;
    node->feature_index = feature;
    node->threshold = threshold;
    
    node->left = build_isolation_tree(X, left_samples, depth + 1, max_depth, rng);
    node->right = build_isolation_tree(X, right_samples, depth + 1, max_depth, rng);
    
    return node;
}

double IsolationForest::path_length(const tree::TreeNode* node, const VectorXd& sample, int depth) const {
    if (node->is_leaf) {
        return static_cast<double>(depth);
    }
    
    if (sample(node->feature_index) <= node->threshold) {
        return path_length(node->left.get(), sample, depth + 1);
    } else {
        return path_length(node->right.get(), sample, depth + 1);
    }
}

Estimator& IsolationForest::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    int n_samples = X.rows();
    int max_samples_actual = (max_samples_ > 0 && max_samples_ < n_samples) ? max_samples_ : n_samples;
    int max_depth = static_cast<int>(std::ceil(std::log2(max_samples_actual)));
    
    trees_.clear();
    trees_.reserve(n_estimators_);
    
    std::mt19937 rng;
    if (random_state_ >= 0) {
        rng.seed(random_state_);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    for (int i = 0; i < n_estimators_; ++i) {
        // Sample random subset
        std::vector<int> all_indices(n_samples);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        
        std::vector<int> sample_indices(max_samples_actual);
        for (int j = 0; j < max_samples_actual; ++j) {
            sample_indices[j] = all_indices[j];
        }
        
        auto tree = build_isolation_tree(X, sample_indices, 0, max_depth, rng);
        trees_.push_back(std::move(tree));
    }
    
    fitted_ = true;
    return *this;
}

VectorXd IsolationForest::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("IsolationForest must be fitted before decision_function");
    }
    
    VectorXd scores = VectorXd::Zero(X.rows());
    
    // Average path lengths across all trees
    for (int i = 0; i < X.rows(); ++i) {
        double avg_path_length = 0.0;
        for (const auto& tree : trees_) {
            avg_path_length += path_length(tree.get(), X.row(i), 0);
        }
        avg_path_length /= trees_.size();
        
        // Convert to anomaly score
        double n_samples = X.rows();
        double c = 2.0 * (std::log(n_samples - 1) + 0.5772156649) - 2.0 * (n_samples - 1) / n_samples;
        scores(i) = std::pow(2.0, -avg_path_length / c);
    }
    
    return scores;
}

VectorXi IsolationForest::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("IsolationForest must be fitted before predict");
    }
    
    VectorXd scores = decision_function(X);
    
    // Determine threshold based on contamination
    VectorXd scores_sorted = scores;
    std::sort(scores_sorted.data(), scores_sorted.data() + scores_sorted.size());
    int threshold_idx = static_cast<int>((1.0 - contamination_) * scores_sorted.size());
    double threshold = scores_sorted(threshold_idx);
    
    VectorXi predictions = VectorXi::Zero(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = (scores(i) > threshold) ? -1 : 1; // -1 = outlier, 1 = inlier
    }
    
    return predictions;
}

VectorXi IsolationForest::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return predict(X);
}

Params IsolationForest::get_params() const {
    Params params;
    params["n_estimators"] = std::to_string(n_estimators_);
    params["max_samples"] = std::to_string(max_samples_);
    params["contamination"] = std::to_string(contamination_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& IsolationForest::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    max_samples_ = utils::get_param_int(params, "max_samples", max_samples_);
    contamination_ = utils::get_param_double(params, "contamination", contamination_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// LocalOutlierFactor implementation

LocalOutlierFactor::LocalOutlierFactor(
    int n_neighbors,
    const std::string& metric,
    double contamination
) : n_neighbors_(n_neighbors), metric_(metric), contamination_(contamination), fitted_(false) {
}

double LocalOutlierFactor::reachability_distance(const MatrixXd& X, int a, int b, int k) const {
    // k-distance of b
    std::vector<std::pair<double, int>> distances;
    for (int i = 0; i < X.rows(); ++i) {
        if (i == b) continue;
        double dist = (X.row(a) - X.row(i)).norm();
        distances.push_back({dist, i});
    }
    std::sort(distances.begin(), distances.end());
    
    double k_dist = distances[std::min(k - 1, static_cast<int>(distances.size()) - 1)].first;
    double dist_ab = (X.row(a) - X.row(b)).norm();
    
    return std::max(dist_ab, k_dist);
}

double LocalOutlierFactor::local_reachability_density(const MatrixXd& X, int sample_idx, int n_neighbors) const {
    // Find k nearest neighbors
    std::vector<std::pair<double, int>> distances;
    for (int i = 0; i < X.rows(); ++i) {
        if (i == sample_idx) continue;
        double dist = (X.row(sample_idx) - X.row(i)).norm();
        distances.push_back({dist, i});
    }
    std::sort(distances.begin(), distances.end());
    
    int k = std::min(n_neighbors, static_cast<int>(distances.size()));
    double sum_reach_dist = 0.0;
    
    for (int i = 0; i < k; ++i) {
        int neighbor_idx = distances[i].second;
        sum_reach_dist += reachability_distance(X, sample_idx, neighbor_idx, n_neighbors);
    }
    
    if (sum_reach_dist == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    return static_cast<double>(k) / sum_reach_dist;
}

Estimator& LocalOutlierFactor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    X_fitted_ = X;
    lrd_scores_.clear();
    lrd_scores_.resize(X.rows());
    
    // Compute local reachability density for each sample
    for (int i = 0; i < X.rows(); ++i) {
        lrd_scores_[i] = local_reachability_density(X, i, n_neighbors_);
    }
    
    fitted_ = true;
    return *this;
}

VectorXd LocalOutlierFactor::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LocalOutlierFactor must be fitted before decision_function");
    }
    
    VectorXd scores = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        // Find k nearest neighbors in fitted data
        std::vector<std::pair<double, int>> distances;
        for (int j = 0; j < X_fitted_.rows(); ++j) {
            double dist = (X.row(i) - X_fitted_.row(j)).norm();
            distances.push_back({dist, j});
        }
        std::sort(distances.begin(), distances.end());
        
        int k = std::min(n_neighbors_, static_cast<int>(distances.size()));
        double lrd_query = local_reachability_density(X_fitted_, i, n_neighbors_);
        
        // Average LRD of neighbors
        double avg_lrd_neighbors = 0.0;
        for (int j = 0; j < k; ++j) {
            int neighbor_idx = distances[j].second;
            avg_lrd_neighbors += lrd_scores_[neighbor_idx];
        }
        avg_lrd_neighbors /= k;
        
        // LOF score
        scores(i) = avg_lrd_neighbors / (lrd_query + 1e-10);
    }
    
    return scores;
}

VectorXi LocalOutlierFactor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LocalOutlierFactor must be fitted before predict");
    }
    
    VectorXd scores = decision_function(X);
    
    // Determine threshold
    VectorXd scores_sorted = scores;
    std::sort(scores_sorted.data(), scores_sorted.data() + scores_sorted.size());
    int threshold_idx = static_cast<int>((1.0 - contamination_) * scores_sorted.size());
    double threshold = scores_sorted(threshold_idx);
    
    VectorXi predictions = VectorXi::Zero(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = (scores(i) > threshold) ? -1 : 1;
    }
    
    return predictions;
}

VectorXi LocalOutlierFactor::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return predict(X);
}

Params LocalOutlierFactor::get_params() const {
    Params params;
    params["n_neighbors"] = std::to_string(n_neighbors_);
    params["metric"] = metric_;
    params["contamination"] = std::to_string(contamination_);
    return params;
}

Estimator& LocalOutlierFactor::set_params(const Params& params) {
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", n_neighbors_);
    metric_ = utils::get_param_string(params, "metric", metric_);
    contamination_ = utils::get_param_double(params, "contamination", contamination_);
    return *this;
}

} // namespace outlier_detection
} // namespace auroraml


#include "auroraml/feature_selection.hpp"
#include "auroraml/base.hpp"
#include "auroraml/metrics.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace auroraml {
namespace feature_selection {

// VarianceThreshold implementation

VarianceThreshold::VarianceThreshold(double threshold)
    : threshold_(threshold), fitted_(false) {
}

Estimator& VarianceThreshold::fit(const MatrixXd& X, const VectorXd& y) {
    selected_features_.clear();
    
    // Calculate variance for each feature
    for (int col = 0; col < X.cols(); ++col) {
        VectorXd feature = X.col(col);
        double mean = feature.mean();
        double variance = (feature.array() - mean).square().mean();
        
        if (variance > threshold_) {
            selected_features_.push_back(col);
        }
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd VarianceThreshold::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("VarianceThreshold must be fitted before transform");
    }
    
    MatrixXd result(X.rows(), selected_features_.size());
    for (size_t i = 0; i < selected_features_.size(); ++i) {
        result.col(i) = X.col(selected_features_[i]);
    }
    
    return result;
}

MatrixXd VarianceThreshold::inverse_transform(const MatrixXd& X) const {
    // Not meaningful for feature selection, return original shape
    throw std::runtime_error("inverse_transform not supported for VarianceThreshold");
}

MatrixXd VarianceThreshold::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params VarianceThreshold::get_params() const {
    Params params;
    params["threshold"] = std::to_string(threshold_);
    return params;
}

Estimator& VarianceThreshold::set_params(const Params& params) {
    threshold_ = utils::get_param_double(params, "threshold", 0.0);
    return *this;
}

std::vector<int> VarianceThreshold::get_support() const {
    return selected_features_;
}

// SelectKBest implementation

SelectKBest::SelectKBest(std::function<double(const VectorXd&, const VectorXd&)> score_func, int k)
    : k_(k), score_func_(score_func), fitted_(false) {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
}

Estimator& SelectKBest::fit(const MatrixXd& X, const VectorXd& y) {
    selected_features_.clear();
    scores_.clear();
    
    // Calculate scores for each feature
    std::vector<std::pair<double, int>> feature_scores;
    for (int col = 0; col < X.cols(); ++col) {
        VectorXd feature = X.col(col);
        double score = score_func_(feature, y);
        scores_.push_back(score);
        feature_scores.push_back({score, col});
    }
    
    // Sort by score (descending)
    std::sort(feature_scores.begin(), feature_scores.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });
    
    // Select top k features
    int k_selected = std::min(k_, static_cast<int>(feature_scores.size()));
    for (int i = 0; i < k_selected; ++i) {
        selected_features_.push_back(feature_scores[i].second);
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd SelectKBest::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SelectKBest must be fitted before transform");
    }
    
    MatrixXd result(X.rows(), selected_features_.size());
    for (size_t i = 0; i < selected_features_.size(); ++i) {
        result.col(i) = X.col(selected_features_[i]);
    }
    
    return result;
}

MatrixXd SelectKBest::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for SelectKBest");
}

MatrixXd SelectKBest::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SelectKBest::get_params() const {
    Params params;
    params["k"] = std::to_string(k_);
    return params;
}

Estimator& SelectKBest::set_params(const Params& params) {
    k_ = utils::get_param_int(params, "k", 10);
    return *this;
}

std::vector<int> SelectKBest::get_support() const {
    return selected_features_;
}

// SelectPercentile implementation

SelectPercentile::SelectPercentile(std::function<double(const VectorXd&, const VectorXd&)> score_func, int percentile)
    : percentile_(percentile), score_func_(score_func), fitted_(false) {
    if (percentile < 0 || percentile > 100) {
        throw std::invalid_argument("percentile must be between 0 and 100");
    }
}

Estimator& SelectPercentile::fit(const MatrixXd& X, const VectorXd& y) {
    selected_features_.clear();
    scores_.clear();
    
    // Calculate scores for each feature
    std::vector<std::pair<double, int>> feature_scores;
    for (int col = 0; col < X.cols(); ++col) {
        VectorXd feature = X.col(col);
        double score = score_func_(feature, y);
        scores_.push_back(score);
        feature_scores.push_back({score, col});
    }
    
    // Sort by score (descending)
    std::sort(feature_scores.begin(), feature_scores.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });
    
    // Calculate number of features to keep based on percentile
    int n_features = feature_scores.size();
    int k_keep = std::max(1, static_cast<int>(std::ceil(n_features * percentile_ / 100.0)));
    
    // Select top features
    for (int i = 0; i < k_keep && i < n_features; ++i) {
        selected_features_.push_back(feature_scores[i].second);
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd SelectPercentile::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SelectPercentile must be fitted before transform");
    }
    
    MatrixXd result(X.rows(), selected_features_.size());
    for (size_t i = 0; i < selected_features_.size(); ++i) {
        result.col(i) = X.col(selected_features_[i]);
    }
    
    return result;
}

MatrixXd SelectPercentile::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for SelectPercentile");
}

MatrixXd SelectPercentile::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SelectPercentile::get_params() const {
    Params params;
    params["percentile"] = std::to_string(percentile_);
    return params;
}

Estimator& SelectPercentile::set_params(const Params& params) {
    percentile_ = utils::get_param_int(params, "percentile", 10);
    return *this;
}

std::vector<int> SelectPercentile::get_support() const {
    return selected_features_;
}

// Scoring functions implementation

namespace scores {

double f_classif(const VectorXd& X_feature, const VectorXd& y) {
    // Simplified F-statistic for classification
    // Convert y to integer labels if needed
    VectorXi y_int = y.cast<int>();
    
    // Calculate between-group and within-group variance
    std::unordered_map<int, std::vector<double>> groups;
    for (int i = 0; i < X_feature.size(); ++i) {
        groups[y_int(i)].push_back(X_feature(i));
    }
    
    if (groups.size() < 2) {
        return 0.0;
    }
    
    // Calculate group means
    double overall_mean = X_feature.mean();
    double between_var = 0.0;
    double within_var = 0.0;
    
    for (const auto& [label, values] : groups) {
        double group_mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        between_var += values.size() * (group_mean - overall_mean) * (group_mean - overall_mean);
        
        for (double val : values) {
            within_var += (val - group_mean) * (val - group_mean);
        }
    }
    
    between_var /= (groups.size() - 1);
    if (X_feature.size() > groups.size()) {
        within_var /= (X_feature.size() - groups.size());
    }
    
    if (within_var == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    return between_var / within_var;
}

double f_regression(const VectorXd& X_feature, const VectorXd& y) {
    // Simplified F-statistic for regression (uses R²)
    // Calculate correlation
    double x_mean = X_feature.mean();
    double y_mean = y.mean();
    
    double numerator = 0.0;
    double x_var = 0.0;
    double y_var = 0.0;
    
    for (int i = 0; i < X_feature.size(); ++i) {
        double x_diff = X_feature(i) - x_mean;
        double y_diff = y(i) - y_mean;
        numerator += x_diff * y_diff;
        x_var += x_diff * x_diff;
        y_var += y_diff * y_diff;
    }
    
    if (x_var == 0.0 || y_var == 0.0) {
        return 0.0;
    }
    
    double correlation = numerator / std::sqrt(x_var * y_var);
    return correlation * correlation; // Return R² as proxy for F-statistic
}

double mutual_info_classif(const VectorXd& X_feature, const VectorXd& y) {
    // Simplified mutual information - uses correlation as proxy
    return std::abs(f_regression(X_feature, y));
}

double mutual_info_regression(const VectorXd& X_feature, const VectorXd& y) {
    // Simplified mutual information - uses correlation as proxy
    return std::abs(f_regression(X_feature, y));
}

double chi2(const VectorXd& X_feature, const VectorXi& y) {
    // Simplified chi-squared statistic
    // Discretize X_feature into bins
    int n_bins = 10;
    double min_val = X_feature.minCoeff();
    double max_val = X_feature.maxCoeff();
    double bin_width = (max_val - min_val) / n_bins;
    
    if (bin_width == 0.0) {
        return 0.0;
    }
    
    // Count occurrences in each bin for each class
    std::unordered_map<int, std::unordered_map<int, int>> contingency;
    std::unordered_map<int, int> class_counts;
    
    for (int i = 0; i < X_feature.size(); ++i) {
        int bin = std::min(static_cast<int>((X_feature(i) - min_val) / bin_width), n_bins - 1);
        int label = y(i);
        contingency[bin][label]++;
        class_counts[label]++;
    }
    
    // Calculate chi-squared statistic
    double chi2_stat = 0.0;
    int n = X_feature.size();
    
    for (const auto& [bin, class_map] : contingency) {
        int bin_total = 0;
        for (const auto& [label, count] : class_map) {
            bin_total += count;
        }
        
        for (const auto& [label, observed] : class_map) {
            int class_total = class_counts[label];
            double expected = (static_cast<double>(bin_total) * class_total) / n;
            if (expected > 0) {
                double diff = observed - expected;
                chi2_stat += (diff * diff) / expected;
            }
        }
    }
    
    return chi2_stat;
}

} // namespace scores

} // namespace feature_selection
} // namespace auroraml


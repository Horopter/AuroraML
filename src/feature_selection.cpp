#include "ingenuityml/feature_selection.hpp"
#include "ingenuityml/base.hpp"
#include "ingenuityml/metrics.hpp"
#include "ingenuityml/model_selection.hpp"
#include "ingenuityml/linear_model.hpp"
#include "ingenuityml/svm.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <map>
#include <set>
#include <limits>

namespace ingenuityml {
namespace feature_selection {

namespace {
std::vector<double> compute_scores(const MatrixXd& X, const VectorXd& y,
                                   const std::function<double(const VectorXd&, const VectorXd&)>& score_func) {
    std::vector<double> scores;
    scores.reserve(X.cols());
    for (int col = 0; col < X.cols(); ++col) {
        double score = score_func(X.col(col), y);
        if (std::isnan(score) || std::isinf(score)) {
            score = 0.0;
        }
        scores.push_back(score);
    }
    return scores;
}

std::vector<double> scores_to_pvalues(const std::vector<double>& scores) {
    if (scores.empty()) {
        return {};
    }
    double min_score = *std::min_element(scores.begin(), scores.end());
    double max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<double> pvals(scores.size(), 1.0);
    if (max_score == min_score) {
        return pvals;
    }
    for (size_t i = 0; i < scores.size(); ++i) {
        double norm = (scores[i] - min_score) / (max_score - min_score);
        double p = 1.0 - norm;
        pvals[i] = std::min(1.0, std::max(0.0, p));
    }
    return pvals;
}

std::vector<int> select_top_k(const std::vector<double>& scores, int k) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    int keep = std::min(k, static_cast<int>(indices.size()));
    indices.resize(keep);
    std::sort(indices.begin(), indices.end());
    return indices;
}

std::vector<int> select_by_percentile(const std::vector<double>& scores, double percentile) {
    int n_features = static_cast<int>(scores.size());
    int k = std::max(1, static_cast<int>(std::ceil(n_features * percentile / 100.0)));
    return select_top_k(scores, k);
}

std::vector<int> select_by_threshold(const std::vector<double>& pvals, double threshold) {
    std::vector<int> selected;
    for (size_t i = 0; i < pvals.size(); ++i) {
        if (pvals[i] <= threshold) {
            selected.push_back(static_cast<int>(i));
        }
    }
    if (selected.empty() && !pvals.empty()) {
        int best = static_cast<int>(std::min_element(pvals.begin(), pvals.end()) - pvals.begin());
        selected.push_back(best);
    }
    return selected;
}

std::vector<int> select_fdr(const std::vector<double>& pvals, double alpha) {
    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(pvals.size());
    for (size_t i = 0; i < pvals.size(); ++i) {
        ranked.emplace_back(pvals[i], static_cast<int>(i));
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    double threshold = -1.0;
    for (size_t i = 0; i < ranked.size(); ++i) {
        double bound = (static_cast<double>(i + 1) / ranked.size()) * alpha;
        if (ranked[i].first <= bound) {
            threshold = ranked[i].first;
        }
    }

    if (threshold < 0.0) {
        return select_by_threshold(pvals, *std::min_element(pvals.begin(), pvals.end()));
    }
    return select_by_threshold(pvals, threshold);
}

MatrixXd select_columns(const MatrixXd& X, const std::vector<int>& columns) {
    MatrixXd result(X.rows(), columns.size());
    for (size_t i = 0; i < columns.size(); ++i) {
        result.col(i) = X.col(columns[i]);
    }
    return result;
}

std::vector<double> importances_from_coef(const VectorXd& coef) {
    std::vector<double> importances(coef.size());
    for (int i = 0; i < coef.size(); ++i) {
        importances[i] = std::abs(coef(i));
    }
    return importances;
}

std::vector<double> estimator_importances(const Estimator& estimator) {
    if (auto* model = dynamic_cast<const linear_model::LinearRegression*>(&estimator)) {
        return importances_from_coef(model->coef());
    }
    if (auto* model = dynamic_cast<const linear_model::Ridge*>(&estimator)) {
        return importances_from_coef(model->coef());
    }
    if (auto* model = dynamic_cast<const linear_model::Lasso*>(&estimator)) {
        return importances_from_coef(model->coef());
    }
    if (auto* model = dynamic_cast<const linear_model::ElasticNet*>(&estimator)) {
        return importances_from_coef(model->coef());
    }
    if (auto* model = dynamic_cast<const linear_model::LogisticRegression*>(&estimator)) {
        return importances_from_coef(model->coef());
    }
    if (auto* model = dynamic_cast<const svm::SVR*>(&estimator)) {
        return importances_from_coef(model->coef());
    }
    if (auto* model = dynamic_cast<const svm::LinearSVR*>(&estimator)) {
        return importances_from_coef(model->coef());
    }
    if (auto* model = dynamic_cast<const svm::NuSVR*>(&estimator)) {
        return importances_from_coef(model->coef());
    }

    throw std::runtime_error("Estimator does not expose coefficients for feature selection");
}
} // namespace

// VarianceThreshold implementation

VarianceThreshold::VarianceThreshold(double threshold)
    : threshold_(threshold), fitted_(false) {
}

Estimator& VarianceThreshold::fit(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }
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
    if (k_ > X.cols()) {
        throw std::invalid_argument("k cannot be greater than number of features");
    }

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

// SelectFpr implementation

SelectFpr::SelectFpr(std::function<double(const VectorXd&, const VectorXd&)> score_func, double alpha)
    : alpha_(alpha), score_func_(score_func), fitted_(false) {
    if (alpha_ <= 0.0 || alpha_ >= 1.0) {
        throw std::invalid_argument("alpha must be in (0, 1)");
    }
}

Estimator& SelectFpr::fit(const MatrixXd& X, const VectorXd& y) {
    scores_ = compute_scores(X, y, score_func_);
    auto pvals = scores_to_pvalues(scores_);
    selected_features_ = select_by_threshold(pvals, alpha_);
    fitted_ = true;
    return *this;
}

MatrixXd SelectFpr::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SelectFpr must be fitted before transform");
    }
    return select_columns(X, selected_features_);
}

MatrixXd SelectFpr::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for SelectFpr");
}

MatrixXd SelectFpr::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SelectFpr::get_params() const {
    Params params;
    params["alpha"] = std::to_string(alpha_);
    return params;
}

Estimator& SelectFpr::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    return *this;
}

std::vector<int> SelectFpr::get_support() const {
    return selected_features_;
}

// SelectFdr implementation

SelectFdr::SelectFdr(std::function<double(const VectorXd&, const VectorXd&)> score_func, double alpha)
    : alpha_(alpha), score_func_(score_func), fitted_(false) {
    if (alpha_ <= 0.0 || alpha_ >= 1.0) {
        throw std::invalid_argument("alpha must be in (0, 1)");
    }
}

Estimator& SelectFdr::fit(const MatrixXd& X, const VectorXd& y) {
    scores_ = compute_scores(X, y, score_func_);
    auto pvals = scores_to_pvalues(scores_);
    selected_features_ = select_fdr(pvals, alpha_);
    fitted_ = true;
    return *this;
}

MatrixXd SelectFdr::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SelectFdr must be fitted before transform");
    }
    return select_columns(X, selected_features_);
}

MatrixXd SelectFdr::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for SelectFdr");
}

MatrixXd SelectFdr::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SelectFdr::get_params() const {
    Params params;
    params["alpha"] = std::to_string(alpha_);
    return params;
}

Estimator& SelectFdr::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    return *this;
}

std::vector<int> SelectFdr::get_support() const {
    return selected_features_;
}

// SelectFwe implementation

SelectFwe::SelectFwe(std::function<double(const VectorXd&, const VectorXd&)> score_func, double alpha)
    : alpha_(alpha), score_func_(score_func), fitted_(false) {
    if (alpha_ <= 0.0 || alpha_ >= 1.0) {
        throw std::invalid_argument("alpha must be in (0, 1)");
    }
}

Estimator& SelectFwe::fit(const MatrixXd& X, const VectorXd& y) {
    scores_ = compute_scores(X, y, score_func_);
    auto pvals = scores_to_pvalues(scores_);
    double threshold = alpha_ / std::max(1, static_cast<int>(pvals.size()));
    selected_features_ = select_by_threshold(pvals, threshold);
    fitted_ = true;
    return *this;
}

MatrixXd SelectFwe::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SelectFwe must be fitted before transform");
    }
    return select_columns(X, selected_features_);
}

MatrixXd SelectFwe::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for SelectFwe");
}

MatrixXd SelectFwe::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SelectFwe::get_params() const {
    Params params;
    params["alpha"] = std::to_string(alpha_);
    return params;
}

Estimator& SelectFwe::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    return *this;
}

std::vector<int> SelectFwe::get_support() const {
    return selected_features_;
}

// GenericUnivariateSelect implementation

GenericUnivariateSelect::GenericUnivariateSelect(std::function<double(const VectorXd&, const VectorXd&)> score_func,
                                                 const std::string& mode, double param)
    : mode_(mode), param_(param), score_func_(score_func), fitted_(false) {
}

Estimator& GenericUnivariateSelect::fit(const MatrixXd& X, const VectorXd& y) {
    scores_ = compute_scores(X, y, score_func_);

    if (mode_ == "k_best") {
        int k = static_cast<int>(param_);
        if (k <= 0) {
            throw std::invalid_argument("k must be positive for k_best mode");
        }
        selected_features_ = select_top_k(scores_, k);
    } else if (mode_ == "percentile") {
        if (param_ < 0.0 || param_ > 100.0) {
            throw std::invalid_argument("percentile must be between 0 and 100");
        }
        selected_features_ = select_by_percentile(scores_, param_);
    } else if (mode_ == "fpr") {
        auto pvals = scores_to_pvalues(scores_);
        selected_features_ = select_by_threshold(pvals, param_);
    } else if (mode_ == "fdr") {
        auto pvals = scores_to_pvalues(scores_);
        selected_features_ = select_fdr(pvals, param_);
    } else if (mode_ == "fwe") {
        auto pvals = scores_to_pvalues(scores_);
        double threshold = param_ / std::max(1, static_cast<int>(pvals.size()));
        selected_features_ = select_by_threshold(pvals, threshold);
    } else {
        throw std::invalid_argument("Unsupported mode for GenericUnivariateSelect");
    }

    fitted_ = true;
    return *this;
}

MatrixXd GenericUnivariateSelect::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("GenericUnivariateSelect must be fitted before transform");
    }
    return select_columns(X, selected_features_);
}

MatrixXd GenericUnivariateSelect::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for GenericUnivariateSelect");
}

MatrixXd GenericUnivariateSelect::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params GenericUnivariateSelect::get_params() const {
    Params params;
    params["mode"] = mode_;
    params["param"] = std::to_string(param_);
    return params;
}

Estimator& GenericUnivariateSelect::set_params(const Params& params) {
    if (params.find("mode") != params.end()) {
        mode_ = params.at("mode");
    }
    if (params.find("param") != params.end()) {
        param_ = std::stod(params.at("param"));
    }
    return *this;
}

std::vector<int> GenericUnivariateSelect::get_support() const {
    return selected_features_;
}

// SelectFromModel implementation

SelectFromModel::SelectFromModel(Estimator& estimator, double threshold, int max_features)
    : estimator_(estimator), threshold_(threshold), max_features_(max_features), fitted_(false) {
}

Estimator& SelectFromModel::fit(const MatrixXd& X, const VectorXd& y) {
    estimator_.fit(X, y);
    importances_ = estimator_importances(estimator_);

    if (static_cast<int>(importances_.size()) != X.cols()) {
        throw std::runtime_error("Estimator importances size does not match number of features");
    }

    double threshold = threshold_;
    if (threshold <= 0.0) {
        double sum = std::accumulate(importances_.begin(), importances_.end(), 0.0);
        threshold = sum / importances_.size();
    }

    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(importances_.size());
    for (size_t i = 0; i < importances_.size(); ++i) {
        ranked.emplace_back(importances_[i], static_cast<int>(i));
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    selected_features_.clear();
    for (const auto& item : ranked) {
        if (item.first >= threshold) {
            selected_features_.push_back(item.second);
        }
    }

    if (selected_features_.empty() && !ranked.empty()) {
        selected_features_.push_back(ranked.front().second);
    }

    if (max_features_ > 0 && static_cast<int>(selected_features_.size()) > max_features_) {
        selected_features_.assign(selected_features_.begin(), selected_features_.begin() + max_features_);
    }

    std::sort(selected_features_.begin(), selected_features_.end());
    fitted_ = true;
    return *this;
}

MatrixXd SelectFromModel::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SelectFromModel must be fitted before transform");
    }
    return select_columns(X, selected_features_);
}

MatrixXd SelectFromModel::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for SelectFromModel");
}

MatrixXd SelectFromModel::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SelectFromModel::get_params() const {
    Params params;
    params["threshold"] = std::to_string(threshold_);
    params["max_features"] = std::to_string(max_features_);
    return params;
}

Estimator& SelectFromModel::set_params(const Params& params) {
    if (params.find("threshold") != params.end()) {
        threshold_ = std::stod(params.at("threshold"));
    }
    if (params.find("max_features") != params.end()) {
        max_features_ = std::stoi(params.at("max_features"));
    }
    return *this;
}

std::vector<int> SelectFromModel::get_support() const {
    return selected_features_;
}

// RFE implementation

RFE::RFE(Estimator& estimator, int n_features_to_select, int step)
    : estimator_(estimator), n_features_to_select_(n_features_to_select), step_(step), fitted_(false) {
    if (step_ < 1) {
        throw std::invalid_argument("step must be at least 1");
    }
}

Estimator& RFE::fit(const MatrixXd& X, const VectorXd& y) {
    int n_features = X.cols();
    int target = n_features_to_select_ > 0 ? n_features_to_select_ : std::max(1, n_features / 2);
    if (target > n_features) {
        throw std::invalid_argument("n_features_to_select cannot be greater than number of features");
    }

    std::vector<int> features(n_features);
    std::iota(features.begin(), features.end(), 0);

    while (static_cast<int>(features.size()) > target) {
        MatrixXd X_sub = select_columns(X, features);
        estimator_.fit(X_sub, y);

        std::vector<double> importances = estimator_importances(estimator_);
        if (importances.size() != features.size()) {
            throw std::runtime_error("Estimator importances size does not match number of features");
        }

        std::vector<int> order(features.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return importances[a] < importances[b];
        });

        int remove_count = std::min(step_, static_cast<int>(features.size()) - target);
        std::set<int> remove_indices(order.begin(), order.begin() + remove_count);
        std::vector<int> next_features;
        for (size_t i = 0; i < features.size(); ++i) {
            if (!remove_indices.count(static_cast<int>(i))) {
                next_features.push_back(features[i]);
            }
        }
        features.swap(next_features);
    }

    selected_features_ = features;
    fitted_ = true;
    return *this;
}

MatrixXd RFE::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RFE must be fitted before transform");
    }
    return select_columns(X, selected_features_);
}

MatrixXd RFE::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for RFE");
}

MatrixXd RFE::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params RFE::get_params() const {
    Params params;
    params["n_features_to_select"] = std::to_string(n_features_to_select_);
    params["step"] = std::to_string(step_);
    return params;
}

Estimator& RFE::set_params(const Params& params) {
    if (params.find("n_features_to_select") != params.end()) {
        n_features_to_select_ = std::stoi(params.at("n_features_to_select"));
    }
    if (params.find("step") != params.end()) {
        step_ = std::stoi(params.at("step"));
    }
    return *this;
}

std::vector<int> RFE::get_support() const {
    return selected_features_;
}

// RFECV implementation

RFECV::RFECV(Estimator& estimator, model_selection::BaseCrossValidator& cv,
             int step, const std::string& scoring, int min_features_to_select)
    : estimator_(estimator), cv_(cv), scoring_(scoring), step_(step),
      min_features_to_select_(min_features_to_select), fitted_(false) {
    if (step_ < 1) {
        throw std::invalid_argument("step must be at least 1");
    }
}

Estimator& RFECV::fit(const MatrixXd& X, const VectorXd& y) {
    int n_features = X.cols();
    int min_features = std::max(1, min_features_to_select_);
    if (min_features > n_features) {
        throw std::invalid_argument("min_features_to_select cannot be greater than number of features");
    }

    std::vector<int> features(n_features);
    std::iota(features.begin(), features.end(), 0);

    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<int> best_features = features;

    while (static_cast<int>(features.size()) >= min_features) {
        MatrixXd X_sub = select_columns(X, features);
        VectorXd scores = model_selection::cross_val_score(estimator_, X_sub, y, cv_, scoring_);
        double mean_score = scores.mean();

        if (mean_score > best_score) {
            best_score = mean_score;
            best_features = features;
        }

        if (static_cast<int>(features.size()) == min_features) {
            break;
        }

        estimator_.fit(X_sub, y);
        std::vector<double> importances = estimator_importances(estimator_);

        std::vector<int> order(features.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return importances[a] < importances[b];
        });

        int remove_count = std::min(step_, static_cast<int>(features.size()) - min_features);
        std::set<int> remove_indices(order.begin(), order.begin() + remove_count);
        std::vector<int> next_features;
        for (size_t i = 0; i < features.size(); ++i) {
            if (!remove_indices.count(static_cast<int>(i))) {
                next_features.push_back(features[i]);
            }
        }
        features.swap(next_features);
    }

    selected_features_ = best_features;
    fitted_ = true;
    return *this;
}

MatrixXd RFECV::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RFECV must be fitted before transform");
    }
    return select_columns(X, selected_features_);
}

MatrixXd RFECV::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for RFECV");
}

MatrixXd RFECV::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params RFECV::get_params() const {
    Params params;
    params["scoring"] = scoring_;
    params["step"] = std::to_string(step_);
    params["min_features_to_select"] = std::to_string(min_features_to_select_);
    return params;
}

Estimator& RFECV::set_params(const Params& params) {
    if (params.find("scoring") != params.end()) {
        scoring_ = params.at("scoring");
    }
    if (params.find("step") != params.end()) {
        step_ = std::stoi(params.at("step"));
    }
    if (params.find("min_features_to_select") != params.end()) {
        min_features_to_select_ = std::stoi(params.at("min_features_to_select"));
    }
    return *this;
}

std::vector<int> RFECV::get_support() const {
    return selected_features_;
}

// SequentialFeatureSelector implementation

SequentialFeatureSelector::SequentialFeatureSelector(Estimator& estimator, model_selection::BaseCrossValidator& cv,
                                                     int n_features_to_select, const std::string& direction,
                                                     const std::string& scoring)
    : estimator_(estimator), cv_(cv), scoring_(scoring), n_features_to_select_(n_features_to_select),
      direction_(direction), fitted_(false) {
}

Estimator& SequentialFeatureSelector::fit(const MatrixXd& X, const VectorXd& y) {
    int n_features = X.cols();
    int target = n_features_to_select_ > 0 ? n_features_to_select_ : std::max(1, n_features / 2);
    if (target > n_features) {
        throw std::invalid_argument("n_features_to_select cannot be greater than number of features");
    }

    std::vector<int> selected;
    std::vector<int> remaining(n_features);
    std::iota(remaining.begin(), remaining.end(), 0);

    if (direction_ == "forward") {
        while (static_cast<int>(selected.size()) < target) {
            double best_score = -std::numeric_limits<double>::infinity();
            int best_feature = -1;
            for (int feature : remaining) {
                std::vector<int> candidate = selected;
                candidate.push_back(feature);
                MatrixXd X_sub = select_columns(X, candidate);
                VectorXd scores = model_selection::cross_val_score(estimator_, X_sub, y, cv_, scoring_);
                double mean_score = scores.mean();
                if (mean_score > best_score) {
                    best_score = mean_score;
                    best_feature = feature;
                }
            }
            if (best_feature < 0) {
                break;
            }
            selected.push_back(best_feature);
            remaining.erase(std::remove(remaining.begin(), remaining.end(), best_feature), remaining.end());
        }
    } else if (direction_ == "backward") {
        selected = remaining;
        while (static_cast<int>(selected.size()) > target) {
            double best_score = -std::numeric_limits<double>::infinity();
            int worst_feature = -1;
            for (int feature : selected) {
                std::vector<int> candidate = selected;
                candidate.erase(std::remove(candidate.begin(), candidate.end(), feature), candidate.end());
                MatrixXd X_sub = select_columns(X, candidate);
                VectorXd scores = model_selection::cross_val_score(estimator_, X_sub, y, cv_, scoring_);
                double mean_score = scores.mean();
                if (mean_score > best_score) {
                    best_score = mean_score;
                    worst_feature = feature;
                }
            }
            if (worst_feature < 0) {
                break;
            }
            selected.erase(std::remove(selected.begin(), selected.end(), worst_feature), selected.end());
        }
    } else {
        throw std::invalid_argument("direction must be 'forward' or 'backward'");
    }

    selected_features_ = selected;
    std::sort(selected_features_.begin(), selected_features_.end());
    fitted_ = true;
    return *this;
}

MatrixXd SequentialFeatureSelector::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SequentialFeatureSelector must be fitted before transform");
    }
    return select_columns(X, selected_features_);
}

MatrixXd SequentialFeatureSelector::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("inverse_transform not supported for SequentialFeatureSelector");
}

MatrixXd SequentialFeatureSelector::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SequentialFeatureSelector::get_params() const {
    Params params;
    params["scoring"] = scoring_;
    params["n_features_to_select"] = std::to_string(n_features_to_select_);
    params["direction"] = direction_;
    return params;
}

Estimator& SequentialFeatureSelector::set_params(const Params& params) {
    if (params.find("scoring") != params.end()) {
        scoring_ = params.at("scoring");
    }
    if (params.find("n_features_to_select") != params.end()) {
        n_features_to_select_ = std::stoi(params.at("n_features_to_select"));
    }
    if (params.find("direction") != params.end()) {
        direction_ = params.at("direction");
    }
    return *this;
}

std::vector<int> SequentialFeatureSelector::get_support() const {
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
} // namespace ingenuityml

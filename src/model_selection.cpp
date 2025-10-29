#include "auroraml/model_selection.hpp"
#include "auroraml/base.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <numeric>
#include <map>
#include <set>

namespace auroraml {
namespace model_selection {

// train_test_split implementation
std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> train_test_split(
    const MatrixXd& X, const VectorXd& y, double test_size, double train_size,
    int random_state, bool shuffle, const VectorXd& stratify) {
    
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    int n_samples = X.rows();
    int n_test_samples = static_cast<int>(n_samples * test_size);
    int n_train_samples = n_samples - n_test_samples;
    
    if (train_size > 0) {
        n_train_samples = static_cast<int>(n_samples * train_size);
        n_test_samples = n_samples - n_train_samples;
    }
    
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle) {
        std::mt19937 rng(random_state >= 0 ? random_state : std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    std::vector<int> train_indices(indices.begin(), indices.begin() + n_train_samples);
    std::vector<int> test_indices(indices.begin() + n_train_samples, indices.end());
    
    // Create train and test sets
    MatrixXd X_train(n_train_samples, X.cols());
    MatrixXd X_test(n_test_samples, X.cols());
    VectorXd y_train(n_train_samples);
    VectorXd y_test(n_test_samples);
    
    for (int i = 0; i < n_train_samples; ++i) {
        X_train.row(i) = X.row(train_indices[i]);
        y_train(i) = y(train_indices[i]);
    }
    
    for (int i = 0; i < n_test_samples; ++i) {
        X_test.row(i) = X.row(test_indices[i]);
        y_test(i) = y(test_indices[i]);
    }
    
    return std::make_tuple(X_train, X_test, y_train, y_test);
}

// KFold implementation
KFold::KFold(int n_splits, bool shuffle, int random_state)
    : n_splits_(n_splits), shuffle_(shuffle), random_state_(random_state) {
    if (n_splits < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> KFold::split(const MatrixXd& X, const VectorXd& y) const {
    int n_samples = X.rows();
    if (n_splits_ > n_samples) {
        throw std::runtime_error("n_splits cannot be greater than the number of samples");
    }
    int fold_size = n_samples / n_splits_;
    
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle_) {
        std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    
    for (int i = 0; i < n_splits_; ++i) {
        int start = i * fold_size;
        int end = (i == n_splits_ - 1) ? n_samples : (i + 1) * fold_size;
        
        std::vector<int> test_indices(indices.begin() + start, indices.begin() + end);
        std::vector<int> train_indices;
        
        train_indices.insert(train_indices.end(), indices.begin(), indices.begin() + start);
        train_indices.insert(train_indices.end(), indices.begin() + end, indices.end());
        
        splits.emplace_back(train_indices, test_indices);
    }
    
    return splits;
}

// StratifiedKFold implementation
StratifiedKFold::StratifiedKFold(int n_splits, bool shuffle, int random_state)
    : n_splits_(n_splits), shuffle_(shuffle), random_state_(random_state) {
    if (n_splits < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> StratifiedKFold::split(
    const MatrixXd& X, const VectorXd& y) const {
    
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    if (n_splits_ > X.rows()) {
        throw std::runtime_error("n_splits cannot be greater than the number of samples");
    }
    
    // Group indices by class
    std::map<int, std::vector<int>> class_indices;
    for (int i = 0; i < y.size(); ++i) {
        int class_label = static_cast<int>(y(i));
        class_indices[class_label].push_back(i);
    }
    
    // Shuffle within each class if requested
    if (shuffle_) {
        std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
        for (auto& pair : class_indices) {
            std::shuffle(pair.second.begin(), pair.second.end(), rng);
        }
    }
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits(n_splits_);
    
    // Initialize splits
    for (int i = 0; i < n_splits_; ++i) {
        splits[i] = std::make_pair(std::vector<int>(), std::vector<int>());
    }
    
    // Distribute samples from each class across folds
    for (const auto& class_pair : class_indices) {
        const std::vector<int>& indices = class_pair.second;
        int n_class_samples = indices.size();
        int samples_per_fold = n_class_samples / n_splits_;
        int remainder = n_class_samples % n_splits_;
        
        int current_pos = 0;
        for (int fold = 0; fold < n_splits_; ++fold) {
            int fold_size = samples_per_fold + (fold < remainder ? 1 : 0);
            
            for (int i = 0; i < fold_size; ++i) {
                splits[fold].second.push_back(indices[current_pos + i]);
            }
            
            // Add remaining samples to training set
            for (int i = 0; i < current_pos; ++i) {
                splits[fold].first.push_back(indices[i]);
            }
            for (int i = current_pos + fold_size; i < n_class_samples; ++i) {
                splits[fold].first.push_back(indices[i]);
            }
            
            current_pos += fold_size;
        }
    }
    
    return splits;
}

// GroupKFold implementation
GroupKFold::GroupKFold(int n_splits) : n_splits_(n_splits) {
    if (n_splits < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> GroupKFold::split(
    const MatrixXd& X, const VectorXd& y, const VectorXd& groups) const {
    
    if (X.rows() == 0) {
        throw std::runtime_error("X cannot be empty");
    }
    if (X.rows() != y.size() || X.rows() != groups.size()) {
        throw std::invalid_argument("X, y, and groups must have the same number of samples");
    }
    
    // Group indices by group ID
    std::map<int, std::vector<int>> group_indices;
    for (int i = 0; i < groups.size(); ++i) {
        int group_id = static_cast<int>(groups(i));
        group_indices[group_id].push_back(i);
    }
    
    int n_groups = group_indices.size();
    if (n_splits_ > n_groups) {
        throw std::runtime_error("n_splits cannot be greater than the number of groups");
    }
    
    std::vector<int> group_ids;
    for (const auto& pair : group_indices) {
        group_ids.push_back(pair.first);
    }
    
    int groups_per_fold = n_groups / n_splits_;
    int remainder = n_groups % n_splits_;
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits(n_splits_);
    
    // Initialize splits
    for (int i = 0; i < n_splits_; ++i) {
        splits[i] = std::make_pair(std::vector<int>(), std::vector<int>());
    }
    
    // Distribute groups across folds
    int current_group = 0;
    for (int fold = 0; fold < n_splits_; ++fold) {
        int fold_groups = groups_per_fold + (fold < remainder ? 1 : 0);
        
        for (int g = 0; g < fold_groups; ++g) {
            int group_id = group_ids[current_group];
            const std::vector<int>& group_indices_vec = group_indices[group_id];
            
            splits[fold].second.insert(splits[fold].second.end(),
                                     group_indices_vec.begin(), group_indices_vec.end());
            current_group++;
        }
    }
    
    // Build training sets (all groups not in test set)
    for (int fold = 0; fold < n_splits_; ++fold) {
        std::set<int> test_groups;
        for (int idx : splits[fold].second) {
            int group_id = static_cast<int>(groups(idx));
            test_groups.insert(group_id);
        }
        
        for (int i = 0; i < groups.size(); ++i) {
            int group_id = static_cast<int>(groups(i));
            if (test_groups.find(group_id) == test_groups.end()) {
                splits[fold].first.push_back(i);
            }
        }
    }
    
    return splits;
}

// cross_val_score implementation
VectorXd cross_val_score(Estimator& estimator, const MatrixXd& X, const VectorXd& y,
                        const BaseCrossValidator& cv, const std::string& scoring) {
    
    auto splits = cv.split(X, y);
    int n_splits = splits.size();
    VectorXd scores(n_splits);
    
    for (int i = 0; i < n_splits; ++i) {
        const auto& train_indices = splits[i].first;
        const auto& test_indices = splits[i].second;
        
        // Create train and test sets
        MatrixXd X_train(train_indices.size(), X.cols());
        VectorXd y_train(train_indices.size());
        MatrixXd X_test(test_indices.size(), X.cols());
        VectorXd y_test(test_indices.size());
        
        for (size_t j = 0; j < train_indices.size(); ++j) {
            X_train.row(j) = X.row(train_indices[j]);
            y_train(j) = y(train_indices[j]);
        }
        
        for (size_t j = 0; j < test_indices.size(); ++j) {
            X_test.row(j) = X.row(test_indices[j]);
            y_test(j) = y(test_indices[j]);
        }
        
        // Train estimator
        estimator.fit(X_train, y_train);
        
        // Make predictions
        VectorXd y_pred;
        if (auto* predictor = dynamic_cast<const Predictor*>(&estimator)) {
            y_pred = predictor->predict(X_test);
        } else {
            throw std::runtime_error("Estimator must be a Predictor to make predictions");
        }
        
        // Calculate score
        if (scoring == "accuracy") {
            VectorXi y_true_int = y_test.cast<int>();
            VectorXi y_pred_int = y_pred.cast<int>();
            scores(i) = metrics::accuracy_score(y_true_int, y_pred_int);
        } else if (scoring == "precision") {
            VectorXi y_true_int = y_test.cast<int>();
            VectorXi y_pred_int = y_pred.cast<int>();
            scores(i) = metrics::precision_score(y_true_int, y_pred_int);
        } else if (scoring == "recall") {
            VectorXi y_true_int = y_test.cast<int>();
            VectorXi y_pred_int = y_pred.cast<int>();
            scores(i) = metrics::recall_score(y_true_int, y_pred_int);
        } else if (scoring == "f1") {
            VectorXi y_true_int = y_test.cast<int>();
            VectorXi y_pred_int = y_pred.cast<int>();
            scores(i) = metrics::f1_score(y_true_int, y_pred_int);
        } else if (scoring == "mse") {
            scores(i) = -metrics::mean_squared_error(y_test, y_pred);  // Negative for maximization
        } else if (scoring == "mae") {
            scores(i) = -metrics::mean_absolute_error(y_test, y_pred);  // Negative for maximization
        } else if (scoring == "r2") {
            scores(i) = metrics::r2_score(y_test, y_pred);
        } else {
            throw std::invalid_argument("Unsupported scoring metric: " + scoring);
        }
    }
    
    return scores;
}

// GridSearchCV implementation
GridSearchCV::GridSearchCV(Estimator& estimator, const std::vector<Params>& param_grid,
                          const BaseCrossValidator& cv, const std::string& scoring,
                          int n_jobs, bool verbose)
    : estimator_(estimator), param_grid_(param_grid), cv_(cv), scoring_(scoring),
      n_jobs_(n_jobs), verbose_(verbose) {
    if (param_grid.empty()) {
        throw std::invalid_argument("param_grid cannot be empty");
    }
    if (n_jobs < 1) {
        throw std::invalid_argument("n_jobs must be at least 1");
    }
}

Estimator& GridSearchCV::fit(const MatrixXd& X, const VectorXd& y) {
    double best_score = -std::numeric_limits<double>::infinity();
    Params best_params;
    
    for (const auto& params : param_grid_) {
        estimator_.set_params(params);
        VectorXd scores = cross_val_score(estimator_, X, y, cv_, scoring_);
        double mean_score = scores.mean();
        
        if (mean_score > best_score) {
            best_score = mean_score;
            best_params = params;
        }
    }
    
    estimator_.set_params(best_params);
    estimator_.fit(X, y);
    
    return estimator_;
}

VectorXd GridSearchCV::predict(const MatrixXd& X) const {
    if (auto* predictor = dynamic_cast<const Predictor*>(&estimator_)) {
        return predictor->predict(X);
    } else {
        throw std::runtime_error("Estimator must be a Predictor to make predictions");
    }
}

Params GridSearchCV::best_params() const {
    return estimator_.get_params();
}

double GridSearchCV::best_score() const {
    return 0.0;  // Would need to store this during fit
}

// RandomizedSearchCV implementation
RandomizedSearchCV::RandomizedSearchCV(Estimator& estimator, const std::vector<Params>& param_distributions,
                                     const BaseCrossValidator& cv, const std::string& scoring,
                                     int n_iter, int n_jobs, bool verbose)
    : estimator_(estimator), param_distributions_(param_distributions), cv_(cv), scoring_(scoring),
      n_iter_(n_iter), n_jobs_(n_jobs), verbose_(verbose) {
    if (param_distributions.empty()) {
        throw std::invalid_argument("param_distributions cannot be empty");
    }
    if (n_iter <= 0) {
        throw std::invalid_argument("n_iter must be positive");
    }
    if (n_jobs < 1) {
        throw std::invalid_argument("n_jobs must be at least 1");
    }
}

Estimator& RandomizedSearchCV::fit(const MatrixXd& X, const VectorXd& y) {
    double best_score = -std::numeric_limits<double>::infinity();
    Params best_params;
    
    std::mt19937 rng(std::random_device{}());
    
    for (int i = 0; i < n_iter_; ++i) {
        // Randomly select parameter set
        std::uniform_int_distribution<> dist(0, param_distributions_.size() - 1);
        Params params = param_distributions_[dist(rng)];
        
        estimator_.set_params(params);
        VectorXd scores = cross_val_score(estimator_, X, y, cv_, scoring_);
        double mean_score = scores.mean();
        
        if (mean_score > best_score) {
            best_score = mean_score;
            best_params = params;
        }
    }
    
    estimator_.set_params(best_params);
    estimator_.fit(X, y);
    
    return estimator_;
}

VectorXd RandomizedSearchCV::predict(const MatrixXd& X) const {
    if (auto* predictor = dynamic_cast<const Predictor*>(&estimator_)) {
        return predictor->predict(X);
    } else {
        throw std::runtime_error("Estimator must be a Predictor to make predictions");
    }
}

Params RandomizedSearchCV::best_params() const {
    return estimator_.get_params();
}

double RandomizedSearchCV::best_score() const {
    return 0.0;  // Would need to store this during fit
}

// Parameter management implementations for KFold
Params KFold::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)},
            {"shuffle", shuffle_ ? "1" : "0"},
            {"random_state", std::to_string(random_state_)}};
}

void KFold::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
    if (params.find("shuffle") != params.end()) {
        shuffle_ = (params.at("shuffle") == "1");
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
}

// Parameter management implementations for StratifiedKFold
Params StratifiedKFold::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)},
            {"shuffle", shuffle_ ? "1" : "0"},
            {"random_state", std::to_string(random_state_)}};
}

void StratifiedKFold::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
    if (params.find("shuffle") != params.end()) {
        shuffle_ = (params.at("shuffle") == "1");
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
}

// Parameter management implementations for GroupKFold
Params GroupKFold::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)}};
}

void GroupKFold::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
}

// Parameter management implementations for GridSearchCV
Params GridSearchCV::get_params() const {
    return {{"scoring", scoring_},
            {"n_jobs", std::to_string(n_jobs_)},
            {"verbose", verbose_ ? "1" : "0"}};
}

void GridSearchCV::set_params(const Params& params) {
    if (params.find("scoring") != params.end()) {
        scoring_ = params.at("scoring");
    }
    if (params.find("n_jobs") != params.end()) {
        n_jobs_ = std::stoi(params.at("n_jobs"));
    }
    if (params.find("verbose") != params.end()) {
        verbose_ = (params.at("verbose") == "1");
    }
}

int GridSearchCV::get_n_splits() const {
    return cv_.get_n_splits();
}

// Parameter management implementations for RandomizedSearchCV
Params RandomizedSearchCV::get_params() const {
    return {{"scoring", scoring_},
            {"n_iter", std::to_string(n_iter_)},
            {"n_jobs", std::to_string(n_jobs_)},
            {"verbose", verbose_ ? "1" : "0"}};
}

void RandomizedSearchCV::set_params(const Params& params) {
    if (params.find("scoring") != params.end()) {
        scoring_ = params.at("scoring");
    }
    if (params.find("n_iter") != params.end()) {
        n_iter_ = std::stoi(params.at("n_iter"));
    }
    if (params.find("n_jobs") != params.end()) {
        n_jobs_ = std::stoi(params.at("n_jobs"));
    }
    if (params.find("verbose") != params.end()) {
        verbose_ = (params.at("verbose") == "1");
    }
}

int RandomizedSearchCV::get_n_splits() const {
    return cv_.get_n_splits();
}

} // namespace model_selection
} // namespace cxml

#include "auroraml/ensemble_wrappers.hpp"
#include "auroraml/base.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <set>

namespace auroraml {
namespace ensemble {

// Helper function for bootstrap sampling
static std::vector<int> bootstrap_indices(int n, int max_samples, std::mt19937& rng) {
    int n_samples = (max_samples > 0 && max_samples < n) ? max_samples : n;
    std::uniform_int_distribution<int> uni(0, n - 1);
    std::vector<int> idx(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        idx[i] = uni(rng);
    }
    return idx;
}

// Helper function for feature sampling
static std::vector<int> sample_features(int n_features, int max_features, std::mt19937& rng) {
    if (max_features <= 0 || max_features >= n_features) {
        std::vector<int> all_features(n_features);
        std::iota(all_features.begin(), all_features.end(), 0);
        return all_features;
    }
    
    std::vector<int> all_features(n_features);
    std::iota(all_features.begin(), all_features.end(), 0);
    std::shuffle(all_features.begin(), all_features.end(), rng);
    all_features.resize(max_features);
    return all_features;
}

// BaggingClassifier implementation

BaggingClassifier::BaggingClassifier(
    std::shared_ptr<Classifier> base_estimator,
    int n_estimators,
    int max_samples,
    int max_features,
    int random_state
) : base_estimator_(base_estimator), n_estimators_(n_estimators),
    max_samples_(max_samples), max_features_(max_features),
    random_state_(random_state), fitted_(false) {
    if (!base_estimator_) {
        throw std::invalid_argument("Base estimator must not be null");
    }
}

Estimator& BaggingClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    int n_samples = X.rows();
    int n_features = X.cols();
    
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
    
    estimators_.clear();
    estimators_.reserve(n_estimators_);
    
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    
    for (int i = 0; i < n_estimators_; ++i) {
        // Bootstrap sample
        std::vector<int> sample_indices = bootstrap_indices(n_samples, max_samples_, rng);
        std::vector<int> feature_indices = sample_features(n_features, max_features_, rng);
        
        // Create bootstrap data
        MatrixXd X_bootstrap(sample_indices.size(), feature_indices.size());
        VectorXd y_bootstrap(sample_indices.size());
        
        for (size_t j = 0; j < sample_indices.size(); ++j) {
            for (size_t k = 0; k < feature_indices.size(); ++k) {
                X_bootstrap(j, k) = X(sample_indices[j], feature_indices[k]);
            }
            y_bootstrap(j) = y(sample_indices[j]);
        }
        
        // Clone and fit estimator
        auto estimator = base_estimator_;
        // Cast Classifier to Estimator to access fit
        Estimator* est = dynamic_cast<Estimator*>(estimator.get());
        if (est) {
            est->fit(X_bootstrap, y_bootstrap);
        } else {
            throw std::runtime_error("Base estimator must inherit from both Estimator and Classifier");
        }
        estimators_.push_back(estimator);
    }
    
    fitted_ = true;
    return *this;
}

VectorXi BaggingClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BaggingClassifier must be fitted before predict");
    }
    
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    // Voting
    for (int i = 0; i < X.rows(); ++i) {
        std::unordered_map<int, int> votes;
        for (const auto& estimator : estimators_) {
            VectorXi pred = estimator->predict_classes(X.row(i));
            votes[pred(0)]++;
        }
        
        // Find class with most votes
        int max_votes = 0;
        int predicted_class = classes_(0);
        for (const auto& [cls, count] : votes) {
            if (count > max_votes) {
                max_votes = count;
                predicted_class = cls;
            }
        }
        predictions(i) = predicted_class;
    }
    
    return predictions;
}

MatrixXd BaggingClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BaggingClassifier must be fitted before predict_proba");
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), classes_.size());
    
    for (const auto& estimator : estimators_) {
        MatrixXd est_proba = estimator->predict_proba(X);
        proba += est_proba;
    }
    
    proba /= estimators_.size();
    return proba;
}

VectorXd BaggingClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BaggingClassifier must be fitted before decision_function");
    }
    
    VectorXd decision = VectorXd::Zero(X.rows());
    
    for (const auto& estimator : estimators_) {
        decision += estimator->decision_function(X);
    }
    
    decision /= estimators_.size();
    return decision;
}

Params BaggingClassifier::get_params() const {
    Params params;
    params["n_estimators"] = std::to_string(n_estimators_);
    params["max_samples"] = std::to_string(max_samples_);
    params["max_features"] = std::to_string(max_features_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& BaggingClassifier::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    max_samples_ = utils::get_param_int(params, "max_samples", max_samples_);
    max_features_ = utils::get_param_int(params, "max_features", max_features_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// BaggingRegressor implementation

BaggingRegressor::BaggingRegressor(
    std::shared_ptr<Regressor> base_estimator,
    int n_estimators,
    int max_samples,
    int max_features,
    int random_state
) : base_estimator_(base_estimator), n_estimators_(n_estimators),
    max_samples_(max_samples), max_features_(max_features),
    random_state_(random_state), fitted_(false) {
    if (!base_estimator_) {
        throw std::invalid_argument("Base estimator must not be null");
    }
}

Estimator& BaggingRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    int n_samples = X.rows();
    int n_features = X.cols();
    
    estimators_.clear();
    estimators_.reserve(n_estimators_);
    
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    
    for (int i = 0; i < n_estimators_; ++i) {
        std::vector<int> sample_indices = bootstrap_indices(n_samples, max_samples_, rng);
        std::vector<int> feature_indices = sample_features(n_features, max_features_, rng);
        
        MatrixXd X_bootstrap(sample_indices.size(), feature_indices.size());
        VectorXd y_bootstrap(sample_indices.size());
        
        for (size_t j = 0; j < sample_indices.size(); ++j) {
            for (size_t k = 0; k < feature_indices.size(); ++k) {
                X_bootstrap(j, k) = X(sample_indices[j], feature_indices[k]);
            }
            y_bootstrap(j) = y(sample_indices[j]);
        }
        
        auto estimator = base_estimator_;
        // Cast Regressor to Estimator to access fit
        Estimator* est = dynamic_cast<Estimator*>(estimator.get());
        if (est) {
            est->fit(X_bootstrap, y_bootstrap);
        } else {
            throw std::runtime_error("Base estimator must inherit from both Estimator and Regressor");
        }
        estimators_.push_back(estimator);
    }
    
    fitted_ = true;
    return *this;
}

VectorXd BaggingRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BaggingRegressor must be fitted before predict");
    }
    
    VectorXd predictions = VectorXd::Zero(X.rows());
    
    for (const auto& estimator : estimators_) {
        predictions += estimator->predict(X);
    }
    
    predictions /= estimators_.size();
    return predictions;
}

Params BaggingRegressor::get_params() const {
    Params params;
    params["n_estimators"] = std::to_string(n_estimators_);
    params["max_samples"] = std::to_string(max_samples_);
    params["max_features"] = std::to_string(max_features_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& BaggingRegressor::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    max_samples_ = utils::get_param_int(params, "max_samples", max_samples_);
    max_features_ = utils::get_param_int(params, "max_features", max_features_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// VotingClassifier implementation

VotingClassifier::VotingClassifier(
    const std::vector<std::pair<std::string, std::shared_ptr<Classifier>>>& estimators,
    const std::string& voting
) : estimators_(estimators), voting_(voting), fitted_(false) {
    if (estimators_.empty()) {
        throw std::invalid_argument("VotingClassifier must have at least one estimator");
    }
}

Estimator& VotingClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
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
    
    // Fit all estimators
    for (auto& [name, estimator] : estimators_) {
        // Cast Classifier to Estimator to access fit
        Estimator* est = dynamic_cast<Estimator*>(estimator.get());
        if (est) {
            est->fit(X, y);
        } else {
            throw std::runtime_error("Estimator must inherit from both Estimator and Classifier");
        }
    }
    
    fitted_ = true;
    return *this;
}

VectorXi VotingClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("VotingClassifier must be fitted before predict");
    }
    
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    if (voting_ == "hard") {
        // Hard voting
        for (int i = 0; i < X.rows(); ++i) {
            std::unordered_map<int, int> votes;
            for (const auto& [name, estimator] : estimators_) {
                VectorXi pred = estimator->predict_classes(X.row(i));
                votes[pred(0)]++;
            }
            
            int max_votes = 0;
            int predicted_class = classes_(0);
            for (const auto& [cls, count] : votes) {
                if (count > max_votes) {
                    max_votes = count;
                    predicted_class = cls;
                }
            }
            predictions(i) = predicted_class;
        }
    } else {
        // Soft voting - use predict_proba
        MatrixXd proba = predict_proba(X);
        for (int i = 0; i < X.rows(); ++i) {
            int max_idx = 0;
            for (int j = 1; j < proba.cols(); ++j) {
                if (proba(i, j) > proba(i, max_idx)) {
                    max_idx = j;
                }
            }
            predictions(i) = classes_(max_idx);
        }
    }
    
    return predictions;
}

MatrixXd VotingClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("VotingClassifier must be fitted before predict_proba");
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), classes_.size());
    
    for (const auto& [name, estimator] : estimators_) {
        proba += estimator->predict_proba(X);
    }
    
    proba /= estimators_.size();
    return proba;
}

VectorXd VotingClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("VotingClassifier must be fitted before decision_function");
    }
    
    VectorXd decision = VectorXd::Zero(X.rows());
    
    for (const auto& [name, estimator] : estimators_) {
        decision += estimator->decision_function(X);
    }
    
    decision /= estimators_.size();
    return decision;
}

Params VotingClassifier::get_params() const {
    Params params;
    params["voting"] = voting_;
    params["n_estimators"] = std::to_string(estimators_.size());
    return params;
}

Estimator& VotingClassifier::set_params(const Params& params) {
    voting_ = utils::get_param_string(params, "voting", voting_);
    return *this;
}

// classes() is defined inline in the header

// VotingRegressor implementation

VotingRegressor::VotingRegressor(
    const std::vector<std::pair<std::string, std::shared_ptr<Regressor>>>& estimators
) : estimators_(estimators), fitted_(false) {
    if (estimators_.empty()) {
        throw std::invalid_argument("VotingRegressor must have at least one estimator");
    }
}

Estimator& VotingRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    for (auto& [name, estimator] : estimators_) {
        // Cast Regressor to Estimator to access fit
        Estimator* est = dynamic_cast<Estimator*>(estimator.get());
        if (est) {
            est->fit(X, y);
        } else {
            throw std::runtime_error("Estimator must inherit from both Estimator and Regressor");
        }
    }
    
    fitted_ = true;
    return *this;
}

VectorXd VotingRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("VotingRegressor must be fitted before predict");
    }
    
    VectorXd predictions = VectorXd::Zero(X.rows());
    
    for (const auto& [name, estimator] : estimators_) {
        predictions += estimator->predict(X);
    }
    
    predictions /= estimators_.size();
    return predictions;
}

Params VotingRegressor::get_params() const {
    Params params;
    params["n_estimators"] = std::to_string(estimators_.size());
    return params;
}

Estimator& VotingRegressor::set_params(const Params& params) {
    return *this;
}

// StackingClassifier implementation

StackingClassifier::StackingClassifier(
    const std::vector<std::pair<std::string, std::shared_ptr<Classifier>>>& base_estimators,
    std::shared_ptr<Classifier> meta_classifier
) : base_estimators_(base_estimators), meta_classifier_(meta_classifier), fitted_(false) {
    if (base_estimators_.empty()) {
        throw std::invalid_argument("StackingClassifier must have at least one base estimator");
    }
    if (!meta_classifier_) {
        throw std::invalid_argument("Meta classifier must not be null");
    }
}

Estimator& StackingClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    // Fit base estimators
    for (auto& [name, estimator] : base_estimators_) {
        // Cast Classifier to Estimator to access fit
        Estimator* est = dynamic_cast<Estimator*>(estimator.get());
        if (est) {
            est->fit(X, y);
        } else {
            throw std::runtime_error("Base estimator must inherit from both Estimator and Classifier");
        }
    }
    
    // Create meta features (predictions from base estimators)
    MatrixXd meta_features = MatrixXd::Zero(X.rows(), base_estimators_.size() * 2); // proba + decision
    int col_idx = 0;
    for (const auto& [name, estimator] : base_estimators_) {
        MatrixXd proba = estimator->predict_proba(X);
        VectorXd decision = estimator->decision_function(X);
        
        // Use first two columns of proba and decision
        for (int i = 0; i < X.rows(); ++i) {
            if (proba.cols() > 0) meta_features(i, col_idx) = proba(i, 0);
            if (proba.cols() > 1) meta_features(i, col_idx + 1) = proba(i, 1);
        }
        col_idx += 2;
    }
    
    // Fit meta classifier - cast Classifier to Estimator to access fit
    Estimator* meta_est = dynamic_cast<Estimator*>(meta_classifier_.get());
    if (meta_est) {
        meta_est->fit(meta_features, y);
    } else {
        throw std::runtime_error("Meta classifier must inherit from both Estimator and Classifier");
    }
    
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
    
    fitted_ = true;
    return *this;
}

VectorXi StackingClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("StackingClassifier must be fitted before predict");
    }
    
    // Get meta features
    MatrixXd meta_features = MatrixXd::Zero(X.rows(), base_estimators_.size() * 2);
    int col_idx = 0;
    for (const auto& [name, estimator] : base_estimators_) {
        MatrixXd proba = estimator->predict_proba(X);
        for (int i = 0; i < X.rows(); ++i) {
            if (proba.cols() > 0) meta_features(i, col_idx) = proba(i, 0);
            if (proba.cols() > 1) meta_features(i, col_idx + 1) = proba(i, 1);
        }
        col_idx += 2;
    }
    
    return meta_classifier_->predict_classes(meta_features);
}

MatrixXd StackingClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("StackingClassifier must be fitted before predict_proba");
    }
    
    MatrixXd meta_features = MatrixXd::Zero(X.rows(), base_estimators_.size() * 2);
    int col_idx = 0;
    for (const auto& [name, estimator] : base_estimators_) {
        MatrixXd proba = estimator->predict_proba(X);
        for (int i = 0; i < X.rows(); ++i) {
            if (proba.cols() > 0) meta_features(i, col_idx) = proba(i, 0);
            if (proba.cols() > 1) meta_features(i, col_idx + 1) = proba(i, 1);
        }
        col_idx += 2;
    }
    
    return meta_classifier_->predict_proba(meta_features);
}

VectorXd StackingClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("StackingClassifier must be fitted before decision_function");
    }
    
    MatrixXd meta_features = MatrixXd::Zero(X.rows(), base_estimators_.size() * 2);
    int col_idx = 0;
    for (const auto& [name, estimator] : base_estimators_) {
        MatrixXd proba = estimator->predict_proba(X);
        for (int i = 0; i < X.rows(); ++i) {
            if (proba.cols() > 0) meta_features(i, col_idx) = proba(i, 0);
            if (proba.cols() > 1) meta_features(i, col_idx + 1) = proba(i, 1);
        }
        col_idx += 2;
    }
    
    return meta_classifier_->decision_function(meta_features);
}

Params StackingClassifier::get_params() const {
    Params params;
    params["n_base_estimators"] = std::to_string(base_estimators_.size());
    return params;
}

Estimator& StackingClassifier::set_params(const Params& params) {
    return *this;
}

// classes() is defined inline in the header

// StackingRegressor implementation

StackingRegressor::StackingRegressor(
    const std::vector<std::pair<std::string, std::shared_ptr<Regressor>>>& base_estimators,
    std::shared_ptr<Regressor> meta_regressor
) : base_estimators_(base_estimators), meta_regressor_(meta_regressor), fitted_(false) {
    if (base_estimators_.empty()) {
        throw std::invalid_argument("StackingRegressor must have at least one base estimator");
    }
    if (!meta_regressor_) {
        throw std::invalid_argument("Meta regressor must not be null");
    }
}

Estimator& StackingRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    // Fit base estimators
    for (auto& [name, estimator] : base_estimators_) {
        // Cast Regressor to Estimator to access fit
        Estimator* est = dynamic_cast<Estimator*>(estimator.get());
        if (est) {
            est->fit(X, y);
        } else {
            throw std::runtime_error("Base estimator must inherit from both Estimator and Regressor");
        }
    }
    
    // Create meta features
    MatrixXd meta_features = MatrixXd::Zero(X.rows(), base_estimators_.size());
    int col_idx = 0;
    for (const auto& [name, estimator] : base_estimators_) {
        VectorXd pred = estimator->predict(X);
        meta_features.col(col_idx++) = pred;
    }
    
    // Fit meta regressor - cast Regressor to Estimator to access fit
    Estimator* meta_est = dynamic_cast<Estimator*>(meta_regressor_.get());
    if (meta_est) {
        meta_est->fit(meta_features, y);
    } else {
        throw std::runtime_error("Meta regressor must inherit from both Estimator and Regressor");
    }
    
    fitted_ = true;
    return *this;
}

VectorXd StackingRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("StackingRegressor must be fitted before predict");
    }
    
    // Get meta features
    MatrixXd meta_features = MatrixXd::Zero(X.rows(), base_estimators_.size());
    int col_idx = 0;
    for (const auto& [name, estimator] : base_estimators_) {
        VectorXd pred = estimator->predict(X);
        meta_features.col(col_idx++) = pred;
    }
    
    return meta_regressor_->predict(meta_features);
}

Params StackingRegressor::get_params() const {
    Params params;
    params["n_base_estimators"] = std::to_string(base_estimators_.size());
    return params;
}

Estimator& StackingRegressor::set_params(const Params& params) {
    return *this;
}

} // namespace ensemble
} // namespace auroraml


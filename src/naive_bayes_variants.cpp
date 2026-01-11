#include "ingenuityml/naive_bayes_variants.hpp"
#include "ingenuityml/base.hpp"
#include <set>
#include <cmath>
#include <algorithm>
#include <unordered_map>

namespace ingenuityml {
namespace naive_bayes {

// MultinomialNB implementation

MultinomialNB::MultinomialNB(double alpha, bool fit_prior)
    : alpha_(alpha), fit_prior_(fit_prior), fitted_(false), n_features_(0), n_classes_(0) {
}

Estimator& MultinomialNB::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    
    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    
    classes_.resize(unique_classes_set.size());
    n_classes_ = unique_classes_set.size();
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }
    
    // Count class frequencies and feature counts per class
    std::vector<int> class_counts(n_classes_, 0);
    std::vector<VectorXd> feature_counts(n_classes_, VectorXd::Zero(n_features_));
    
    for (int i = 0; i < y.size(); ++i) {
        int class_idx = -1;
        for (int c = 0; c < n_classes_; ++c) {
            if (classes_(c) == static_cast<int>(y(i))) {
                class_idx = c;
                break;
            }
        }
        
        if (class_idx >= 0) {
            class_counts[class_idx]++;
            for (int f = 0; f < n_features_; ++f) {
                feature_counts[class_idx](f) += X(i, f);
            }
        }
    }
    
    // Compute log priors
    class_log_prior_.clear();
    int total_samples = y.size();
    for (int c = 0; c < n_classes_; ++c) {
        if (fit_prior_) {
            class_log_prior_.push_back(std::log(class_counts[c] / static_cast<double>(total_samples)));
        } else {
            class_log_prior_.push_back(std::log(1.0 / n_classes_));
        }
    }
    
    // Compute log probabilities with smoothing
    feature_log_prob_.clear();
    for (int c = 0; c < n_classes_; ++c) {
        double total_counts = feature_counts[c].sum();
        double denominator = total_counts + alpha_ * n_features_;
        
        VectorXd log_prob = VectorXd::Zero(n_features_);
        for (int f = 0; f < n_features_; ++f) {
            double numerator = feature_counts[c](f) + alpha_;
            log_prob(f) = std::log(numerator) - std::log(denominator);
        }
        feature_log_prob_.push_back(log_prob);
    }
    
    fitted_ = true;
    return *this;
}

VectorXi MultinomialNB::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultinomialNB must be fitted before predict");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        int max_idx = 0;
        for (int j = 1; j < proba.cols(); ++j) {
            if (proba(i, j) > proba(i, max_idx)) {
                max_idx = j;
            }
        }
        predictions(i) = classes_(max_idx);
    }
    
    return predictions;
}

MatrixXd MultinomialNB::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultinomialNB must be fitted before predict_proba");
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), n_classes_);
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd log_proba = VectorXd::Zero(n_classes_);
        
        for (int c = 0; c < n_classes_; ++c) {
            log_proba(c) = class_log_prior_[c];
            for (int f = 0; f < n_features_; ++f) {
                log_proba(c) += X(i, f) * feature_log_prob_[c](f);
            }
        }
        
        // Convert to probabilities using softmax
        double max_log = log_proba.maxCoeff();
        VectorXd exp_log = (log_proba.array() - max_log).exp();
        double sum_exp = exp_log.sum();
        
        if (sum_exp > 0) {
            proba.row(i) = exp_log.transpose() / sum_exp;
        } else {
            proba.row(i).fill(1.0 / n_classes_);
        }
    }
    
    return proba;
}

VectorXd MultinomialNB::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultinomialNB must be fitted before decision_function");
    }
    
    VectorXd decision = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd log_proba = VectorXd::Zero(n_classes_);
        
        for (int c = 0; c < n_classes_; ++c) {
            log_proba(c) = class_log_prior_[c];
            for (int f = 0; f < n_features_; ++f) {
                log_proba(c) += X(i, f) * feature_log_prob_[c](f);
            }
        }
        
        decision(i) = log_proba.maxCoeff();
    }
    
    return decision;
}

Params MultinomialNB::get_params() const {
    Params params;
    params["alpha"] = std::to_string(alpha_);
    params["fit_prior"] = fit_prior_ ? "true" : "false";
    return params;
}

Estimator& MultinomialNB::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_prior_ = utils::get_param_bool(params, "fit_prior", fit_prior_);
    return *this;
}

// BernoulliNB implementation

BernoulliNB::BernoulliNB(double alpha, double binarize, bool fit_prior)
    : alpha_(alpha), binarize_(binarize), fit_prior_(fit_prior), fitted_(false), n_features_(0), n_classes_(0) {
}

Estimator& BernoulliNB::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    
    // Binarize features if needed
    MatrixXd X_binary = X;
    if (binarize_ != 0.0) {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                X_binary(i, j) = (X(i, j) > binarize_) ? 1.0 : 0.0;
            }
        }
    } else {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                X_binary(i, j) = (X(i, j) > 0.0) ? 1.0 : 0.0;
            }
        }
    }
    
    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    
    classes_.resize(unique_classes_set.size());
    n_classes_ = unique_classes_set.size();
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }
    
    // Count class frequencies and feature counts per class
    std::vector<int> class_counts(n_classes_, 0);
    std::vector<VectorXd> feature_counts(n_classes_, VectorXd::Zero(n_features_));
    std::vector<VectorXd> feature_counts_neg(n_classes_, VectorXd::Zero(n_features_));
    
    for (int i = 0; i < y.size(); ++i) {
        int class_idx = -1;
        for (int c = 0; c < n_classes_; ++c) {
            if (classes_(c) == static_cast<int>(y(i))) {
                class_idx = c;
                break;
            }
        }
        
        if (class_idx >= 0) {
            class_counts[class_idx]++;
            for (int f = 0; f < n_features_; ++f) {
                if (X_binary(i, f) > 0.5) {
                    feature_counts[class_idx](f)++;
                } else {
                    feature_counts_neg[class_idx](f)++;
                }
            }
        }
    }
    
    // Compute log priors
    class_log_prior_.clear();
    int total_samples = y.size();
    for (int c = 0; c < n_classes_; ++c) {
        if (fit_prior_) {
            class_log_prior_.push_back(std::log(class_counts[c] / static_cast<double>(total_samples)));
        } else {
            class_log_prior_.push_back(std::log(1.0 / n_classes_));
        }
    }
    
    // Compute log probabilities
    feature_log_prob_.clear();
    feature_log_prob_neg_.clear();
    for (int c = 0; c < n_classes_; ++c) {
        VectorXd log_prob = VectorXd::Zero(n_features_);
        VectorXd log_prob_neg = VectorXd::Zero(n_features_);
        
        for (int f = 0; f < n_features_; ++f) {
            double pos_count = feature_counts[c](f) + alpha_;
            double neg_count = feature_counts_neg[c](f) + alpha_;
            double total = pos_count + neg_count;
            
            log_prob(f) = std::log(pos_count) - std::log(total);
            log_prob_neg(f) = std::log(neg_count) - std::log(total);
        }
        
        feature_log_prob_.push_back(log_prob);
        feature_log_prob_neg_.push_back(log_prob_neg);
    }
    
    fitted_ = true;
    return *this;
}

VectorXi BernoulliNB::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BernoulliNB must be fitted before predict");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        int max_idx = 0;
        for (int j = 1; j < proba.cols(); ++j) {
            if (proba(i, j) > proba(i, max_idx)) {
                max_idx = j;
            }
        }
        predictions(i) = classes_(max_idx);
    }
    
    return predictions;
}

MatrixXd BernoulliNB::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BernoulliNB must be fitted before predict_proba");
    }
    
    // Binarize features
    MatrixXd X_binary = X;
    if (binarize_ != 0.0) {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                X_binary(i, j) = (X(i, j) > binarize_) ? 1.0 : 0.0;
            }
        }
    } else {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                X_binary(i, j) = (X(i, j) > 0.0) ? 1.0 : 0.0;
            }
        }
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), n_classes_);
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd log_proba = VectorXd::Zero(n_classes_);
        
        for (int c = 0; c < n_classes_; ++c) {
            log_proba(c) = class_log_prior_[c];
            for (int f = 0; f < n_features_; ++f) {
                if (X_binary(i, f) > 0.5) {
                    log_proba(c) += feature_log_prob_[c](f);
                } else {
                    log_proba(c) += feature_log_prob_neg_[c](f);
                }
            }
        }
        
        // Convert to probabilities using softmax
        double max_log = log_proba.maxCoeff();
        VectorXd exp_log = (log_proba.array() - max_log).exp();
        double sum_exp = exp_log.sum();
        
        if (sum_exp > 0) {
            proba.row(i) = exp_log.transpose() / sum_exp;
        } else {
            proba.row(i).fill(1.0 / n_classes_);
        }
    }
    
    return proba;
}

VectorXd BernoulliNB::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BernoulliNB must be fitted before decision_function");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXd decision = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        decision(i) = proba.row(i).maxCoeff();
    }
    
    return decision;
}

Params BernoulliNB::get_params() const {
    Params params;
    params["alpha"] = std::to_string(alpha_);
    params["binarize"] = std::to_string(binarize_);
    params["fit_prior"] = fit_prior_ ? "true" : "false";
    return params;
}

Estimator& BernoulliNB::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    binarize_ = utils::get_param_double(params, "binarize", binarize_);
    fit_prior_ = utils::get_param_bool(params, "fit_prior", fit_prior_);
    return *this;
}

// ComplementNB implementation

ComplementNB::ComplementNB(double alpha, bool fit_prior)
    : alpha_(alpha), fit_prior_(fit_prior), fitted_(false), n_features_(0), n_classes_(0) {
}

Estimator& ComplementNB::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    n_features_ = X.cols();
    
    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    
    classes_.resize(unique_classes_set.size());
    n_classes_ = unique_classes_set.size();
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }
    
    // Count class frequencies and feature counts per class
    std::vector<int> class_counts(n_classes_, 0);
    std::vector<VectorXd> feature_counts(n_classes_, VectorXd::Zero(n_features_));
    VectorXd total_feature_counts = VectorXd::Zero(n_features_);
    
    for (int i = 0; i < y.size(); ++i) {
        int class_idx = -1;
        for (int c = 0; c < n_classes_; ++c) {
            if (classes_(c) == static_cast<int>(y(i))) {
                class_idx = c;
                break;
            }
        }
        
        if (class_idx >= 0) {
            class_counts[class_idx]++;
            for (int f = 0; f < n_features_; ++f) {
                feature_counts[class_idx](f) += X(i, f);
                total_feature_counts(f) += X(i, f);
            }
        }
    }
    
    // Compute log priors (complement approach)
    class_log_prior_.clear();
    int total_samples = y.size();
    for (int c = 0; c < n_classes_; ++c) {
        if (fit_prior_) {
            // Use complement of class frequency
            double complement_freq = 1.0 - (class_counts[c] / static_cast<double>(total_samples));
            class_log_prior_.push_back(std::log(complement_freq));
        } else {
            class_log_prior_.push_back(std::log(1.0 / n_classes_));
        }
    }
    
    // Compute log probabilities using complement approach
    feature_log_prob_.clear();
    for (int c = 0; c < n_classes_; ++c) {
        VectorXd log_prob = VectorXd::Zero(n_features_);
        
        for (int f = 0; f < n_features_; ++f) {
            // Complement: use complement of class-specific counts
            double complement_count = total_feature_counts(f) - feature_counts[c](f) + alpha_;
            double total_complement = total_feature_counts.sum() - feature_counts[c].sum() + alpha_ * n_features_;
            
            log_prob(f) = std::log(complement_count) - std::log(total_complement);
        }
        
        feature_log_prob_.push_back(log_prob);
    }
    
    fitted_ = true;
    return *this;
}

VectorXi ComplementNB::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ComplementNB must be fitted before predict");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        int max_idx = 0;
        for (int j = 1; j < proba.cols(); ++j) {
            if (proba(i, j) > proba(i, max_idx)) {
                max_idx = j;
            }
        }
        predictions(i) = classes_(max_idx);
    }
    
    return predictions;
}

MatrixXd ComplementNB::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ComplementNB must be fitted before predict_proba");
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), n_classes_);
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd log_proba = VectorXd::Zero(n_classes_);
        
        for (int c = 0; c < n_classes_; ++c) {
            log_proba(c) = class_log_prior_[c];
            for (int f = 0; f < n_features_; ++f) {
                log_proba(c) += X(i, f) * feature_log_prob_[c](f);
            }
        }
        
        // Convert to probabilities using softmax
        double max_log = log_proba.maxCoeff();
        VectorXd exp_log = (log_proba.array() - max_log).exp();
        double sum_exp = exp_log.sum();
        
        if (sum_exp > 0) {
            proba.row(i) = exp_log.transpose() / sum_exp;
        } else {
            proba.row(i).fill(1.0 / n_classes_);
        }
    }
    
    return proba;
}

VectorXd ComplementNB::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ComplementNB must be fitted before decision_function");
    }
    
    VectorXd decision = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd log_proba = VectorXd::Zero(n_classes_);
        
        for (int c = 0; c < n_classes_; ++c) {
            log_proba(c) = class_log_prior_[c];
            for (int f = 0; f < n_features_; ++f) {
                log_proba(c) += X(i, f) * feature_log_prob_[c](f);
            }
        }
        
        decision(i) = log_proba.maxCoeff();
    }
    
    return decision;
}

Params ComplementNB::get_params() const {
    Params params;
    params["alpha"] = std::to_string(alpha_);
    params["fit_prior"] = fit_prior_ ? "true" : "false";
    return params;
}

Estimator& ComplementNB::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_prior_ = utils::get_param_bool(params, "fit_prior", fit_prior_);
    return *this;
}

// CategoricalNB implementation

CategoricalNB::CategoricalNB(double alpha, bool fit_prior)
    : alpha_(alpha), fit_prior_(fit_prior), fitted_(false), n_features_(0), n_classes_(0) {
}

Estimator& CategoricalNB::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);

    n_features_ = X.cols();

    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }

    classes_.resize(unique_classes_set.size());
    n_classes_ = unique_classes_set.size();
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }

    std::unordered_map<int, int> label_to_index;
    for (int c = 0; c < n_classes_; ++c) {
        label_to_index[classes_(c)] = c;
    }

    // Determine number of categories per feature and validate inputs
    n_categories_.assign(n_features_, 0);
    for (int f = 0; f < n_features_; ++f) {
        int max_cat = -1;
        for (int i = 0; i < X.rows(); ++i) {
            double value = X(i, f);
            double rounded = std::round(value);
            if (std::abs(value - rounded) > 1e-6) {
                throw std::invalid_argument("CategoricalNB requires integer-encoded categorical features");
            }
            int cat = static_cast<int>(rounded);
            if (cat < 0) {
                throw std::invalid_argument("CategoricalNB requires non-negative categorical features");
            }
            if (cat > max_cat) {
                max_cat = cat;
            }
        }
        n_categories_[f] = std::max(1, max_cat + 1);
    }

    std::vector<int> class_counts(n_classes_, 0);
    std::vector<MatrixXd> feature_counts;
    feature_counts.reserve(n_features_);
    for (int f = 0; f < n_features_; ++f) {
        feature_counts.push_back(MatrixXd::Zero(n_classes_, n_categories_[f]));
    }

    for (int i = 0; i < y.size(); ++i) {
        int label = static_cast<int>(y(i));
        int class_idx = label_to_index[label];
        class_counts[class_idx]++;

        for (int f = 0; f < n_features_; ++f) {
            int cat = static_cast<int>(std::round(X(i, f)));
            if (cat < 0 || cat >= n_categories_[f]) {
                throw std::invalid_argument("CategoricalNB feature value out of range");
            }
            feature_counts[f](class_idx, cat) += 1.0;
        }
    }

    // Compute log priors
    class_log_prior_.clear();
    int total_samples = y.size();
    for (int c = 0; c < n_classes_; ++c) {
        if (fit_prior_) {
            class_log_prior_.push_back(std::log(class_counts[c] / static_cast<double>(total_samples)));
        } else {
            class_log_prior_.push_back(std::log(1.0 / n_classes_));
        }
    }

    // Compute log probabilities with smoothing
    feature_log_prob_.clear();
    feature_log_prob_.reserve(n_features_);
    for (int f = 0; f < n_features_; ++f) {
        MatrixXd log_prob = MatrixXd::Zero(n_classes_, n_categories_[f]);
        for (int c = 0; c < n_classes_; ++c) {
            double denom = feature_counts[f].row(c).sum() + alpha_ * n_categories_[f];
            for (int cat = 0; cat < n_categories_[f]; ++cat) {
                double numerator = feature_counts[f](c, cat) + alpha_;
                log_prob(c, cat) = std::log(numerator) - std::log(denom);
            }
        }
        feature_log_prob_.push_back(log_prob);
    }

    fitted_ = true;
    return *this;
}

VectorXi CategoricalNB::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("CategoricalNB must be fitted before predict");
    }

    MatrixXd proba = predict_proba(X);
    VectorXi predictions = VectorXi::Zero(X.rows());

    for (int i = 0; i < X.rows(); ++i) {
        int max_idx = 0;
        for (int j = 1; j < proba.cols(); ++j) {
            if (proba(i, j) > proba(i, max_idx)) {
                max_idx = j;
            }
        }
        predictions(i) = classes_(max_idx);
    }

    return predictions;
}

MatrixXd CategoricalNB::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("CategoricalNB must be fitted before predict_proba");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd proba = MatrixXd::Zero(X.rows(), n_classes_);

    for (int i = 0; i < X.rows(); ++i) {
        VectorXd log_proba = VectorXd::Zero(n_classes_);
        for (int c = 0; c < n_classes_; ++c) {
            log_proba(c) = class_log_prior_[c];
            for (int f = 0; f < n_features_; ++f) {
                double value = X(i, f);
                double rounded = std::round(value);
                if (std::abs(value - rounded) > 1e-6) {
                    throw std::invalid_argument("CategoricalNB requires integer-encoded categorical features");
                }
                int cat = static_cast<int>(rounded);
                if (cat < 0 || cat >= n_categories_[f]) {
                    throw std::invalid_argument("CategoricalNB feature value out of range");
                }
                log_proba(c) += feature_log_prob_[f](c, cat);
            }
        }

        double max_log = log_proba.maxCoeff();
        VectorXd exp_log = (log_proba.array() - max_log).exp();
        double sum_exp = exp_log.sum();

        if (sum_exp > 0) {
            proba.row(i) = exp_log.transpose() / sum_exp;
        } else {
            proba.row(i).fill(1.0 / n_classes_);
        }
    }

    return proba;
}

VectorXd CategoricalNB::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("CategoricalNB must be fitted before decision_function");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    VectorXd decision = VectorXd::Zero(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd log_proba = VectorXd::Zero(n_classes_);
        for (int c = 0; c < n_classes_; ++c) {
            log_proba(c) = class_log_prior_[c];
            for (int f = 0; f < n_features_; ++f) {
                double value = X(i, f);
                double rounded = std::round(value);
                if (std::abs(value - rounded) > 1e-6) {
                    throw std::invalid_argument("CategoricalNB requires integer-encoded categorical features");
                }
                int cat = static_cast<int>(rounded);
                if (cat < 0 || cat >= n_categories_[f]) {
                    throw std::invalid_argument("CategoricalNB feature value out of range");
                }
                log_proba(c) += feature_log_prob_[f](c, cat);
            }
        }
        decision(i) = log_proba.maxCoeff();
    }

    return decision;
}

Params CategoricalNB::get_params() const {
    Params params;
    params["alpha"] = std::to_string(alpha_);
    params["fit_prior"] = fit_prior_ ? "true" : "false";
    return params;
}

Estimator& CategoricalNB::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_prior_ = utils::get_param_bool(params, "fit_prior", fit_prior_);
    return *this;
}

} // namespace naive_bayes
} // namespace ingenuityml

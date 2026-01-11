#include "ingenuityml/extra_trees.hpp"
#include "ingenuityml/base.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <set>

namespace ingenuityml {
namespace ensemble {

static std::vector<int> bootstrap_indices(int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> uni(0, n - 1);
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) {
        idx[i] = uni(rng);
    }
    return idx;
}

Estimator& ExtraTreesClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    n_features_ = X.cols();

    std::set<int> unique_classes;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes.insert(static_cast<int>(y(i)));
    }
    classes_.resize(unique_classes.size());
    int class_idx = 0;
    for (int cls : unique_classes) {
        classes_(class_idx++) = cls;
    }

    trees_.clear();
    trees_.reserve(n_estimators_);
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());

    const int n_samples = X.rows();
    std::vector<int> all_indices;
    if (!bootstrap_) {
        all_indices.resize(n_samples);
        std::iota(all_indices.begin(), all_indices.end(), 0);
    }

    for (int t = 0; t < n_estimators_; ++t) {
        tree::ExtraTreeClassifier tree(max_depth_, min_samples_split_, min_samples_leaf_,
                                       max_features_, static_cast<int>(rng()));
        if (bootstrap_) {
            std::vector<int> idx = bootstrap_indices(n_samples, rng);
            MatrixXd Xb(idx.size(), X.cols());
            VectorXd yb(idx.size());
            for (size_t i = 0; i < idx.size(); ++i) {
                Xb.row(i) = X.row(idx[i]);
                yb(i) = y(idx[i]);
            }
            tree.fit(Xb, yb);
        } else {
            tree.fit(X, y);
        }
        trees_.push_back(std::move(tree));
    }

    fitted_ = true;
    return *this;
}

VectorXi ExtraTreesClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ExtraTreesClassifier not fitted");
    }
    if (X.cols() != n_features_) {
        std::string msg = "X must have the same number of features as training data. Expected: " +
                         std::to_string(n_features_) + ", got: " + std::to_string(X.cols());
        throw std::runtime_error(msg);
    }
    VectorXi pred(X.rows());

    #pragma omp parallel for if(X.rows() > 16)
    for (int i = 0; i < X.rows(); ++i) {
        VectorXi counts = VectorXi::Zero(classes_.size());
        for (const auto& tree : trees_) {
            MatrixXd single_row = X.row(i);
            int cls = tree.predict_classes(single_row)(0);
            for (int c = 0; c < classes_.size(); ++c) {
                if (classes_(c) == cls) {
                    counts(c)++;
                    break;
                }
            }
        }
        Eigen::Index max_idx = 0;
        counts.maxCoeff(&max_idx);
        pred(i) = classes_(max_idx);
    }
    return pred;
}

MatrixXd ExtraTreesClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ExtraTreesClassifier not fitted");
    }
    if (X.cols() != n_features_) {
        std::string msg = "X must have the same number of features as training data. Expected: " +
                         std::to_string(n_features_) + ", got: " + std::to_string(X.cols());
        throw std::runtime_error(msg);
    }

    MatrixXd proba = MatrixXd::Zero(X.rows(), classes_.size());
    if (trees_.empty()) {
        return proba;
    }

    for (int i = 0; i < X.rows(); ++i) {
        VectorXd counts = VectorXd::Zero(classes_.size());
        for (const auto& tree : trees_) {
            MatrixXd single_row = X.row(i);
            int cls = tree.predict_classes(single_row)(0);
            for (int c = 0; c < classes_.size(); ++c) {
                if (classes_(c) == cls) {
                    counts(c) += 1.0;
                    break;
                }
            }
        }
        proba.row(i) = (counts / static_cast<double>(trees_.size())).transpose();
    }
    return proba;
}

VectorXd ExtraTreesClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ExtraTreesClassifier not fitted");
    }
    VectorXi cls = predict_classes(X);
    return cls.cast<double>();
}

Params ExtraTreesClassifier::get_params() const {
    Params params;
    params["n_estimators"] = std::to_string(n_estimators_);
    params["max_depth"] = std::to_string(max_depth_);
    params["min_samples_split"] = std::to_string(min_samples_split_);
    params["min_samples_leaf"] = std::to_string(min_samples_leaf_);
    params["max_features"] = std::to_string(max_features_);
    params["bootstrap"] = bootstrap_ ? "true" : "false";
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& ExtraTreesClassifier::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    min_samples_split_ = utils::get_param_int(params, "min_samples_split", min_samples_split_);
    min_samples_leaf_ = utils::get_param_int(params, "min_samples_leaf", min_samples_leaf_);
    max_features_ = utils::get_param_int(params, "max_features", max_features_);
    bootstrap_ = utils::get_param_bool(params, "bootstrap", bootstrap_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

Estimator& ExtraTreesRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    n_features_ = X.cols();

    trees_.clear();
    trees_.reserve(n_estimators_);
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());

    const int n_samples = X.rows();
    std::vector<int> all_indices;
    if (!bootstrap_) {
        all_indices.resize(n_samples);
        std::iota(all_indices.begin(), all_indices.end(), 0);
    }

    for (int t = 0; t < n_estimators_; ++t) {
        tree::ExtraTreeRegressor tree(max_depth_, min_samples_split_, min_samples_leaf_,
                                      max_features_, static_cast<int>(rng()));
        if (bootstrap_) {
            std::vector<int> idx = bootstrap_indices(n_samples, rng);
            MatrixXd Xb(idx.size(), X.cols());
            VectorXd yb(idx.size());
            for (size_t i = 0; i < idx.size(); ++i) {
                Xb.row(i) = X.row(idx[i]);
                yb(i) = y(idx[i]);
            }
            tree.fit(Xb, yb);
        } else {
            tree.fit(X, y);
        }
        trees_.push_back(std::move(tree));
    }

    fitted_ = true;
    return *this;
}

VectorXd ExtraTreesRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ExtraTreesRegressor not fitted");
    }
    if (X.cols() != n_features_) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd y = VectorXd::Zero(X.rows());
    for (const auto& tree : trees_) {
        y += tree.predict(X);
    }
    if (!trees_.empty()) {
        y /= static_cast<double>(trees_.size());
    }
    return y;
}

Params ExtraTreesRegressor::get_params() const {
    Params params;
    params["n_estimators"] = std::to_string(n_estimators_);
    params["max_depth"] = std::to_string(max_depth_);
    params["min_samples_split"] = std::to_string(min_samples_split_);
    params["min_samples_leaf"] = std::to_string(min_samples_leaf_);
    params["max_features"] = std::to_string(max_features_);
    params["bootstrap"] = bootstrap_ ? "true" : "false";
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& ExtraTreesRegressor::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    min_samples_split_ = utils::get_param_int(params, "min_samples_split", min_samples_split_);
    min_samples_leaf_ = utils::get_param_int(params, "min_samples_leaf", min_samples_leaf_);
    max_features_ = utils::get_param_int(params, "max_features", max_features_);
    bootstrap_ = utils::get_param_bool(params, "bootstrap", bootstrap_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

} // namespace ensemble
} // namespace ingenuityml

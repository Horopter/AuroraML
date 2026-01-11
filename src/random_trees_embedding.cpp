#include "ingenuityml/random_trees_embedding.hpp"
#include "ingenuityml/base.hpp"
#include <random>

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

Estimator& RandomTreesEmbedding::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;

    n_features_ = X.cols();
    trees_.clear();
    trees_.reserve(n_estimators_);

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const int n_samples = X.rows();
    for (int t = 0; t < n_estimators_; ++t) {
        tree::ExtraTreeRegressor tree(max_depth_, min_samples_split_, min_samples_leaf_,
                                      max_features_, static_cast<int>(rng()));
        if (bootstrap_) {
            std::vector<int> idx = bootstrap_indices(n_samples, rng);
            MatrixXd Xb(idx.size(), X.cols());
            VectorXd yb(idx.size());
            for (size_t i = 0; i < idx.size(); ++i) {
                Xb.row(i) = X.row(idx[i]);
                yb(i) = dist(rng);
            }
            tree.fit(Xb, yb);
        } else {
            VectorXd y_random(n_samples);
            for (int i = 0; i < n_samples; ++i) {
                y_random(i) = dist(rng);
            }
            tree.fit(X, y_random);
        }
        trees_.push_back(std::move(tree));
    }

    fitted_ = true;
    return *this;
}

MatrixXd RandomTreesEmbedding::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RandomTreesEmbedding must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd embedding = MatrixXd::Zero(X.rows(), static_cast<int>(trees_.size()));
    for (size_t t = 0; t < trees_.size(); ++t) {
        embedding.col(static_cast<int>(t)) = trees_[t].predict(X);
    }
    return embedding;
}

MatrixXd RandomTreesEmbedding::inverse_transform(const MatrixXd& X) const {
    return X;
}

MatrixXd RandomTreesEmbedding::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params RandomTreesEmbedding::get_params() const {
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

Estimator& RandomTreesEmbedding::set_params(const Params& params) {
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

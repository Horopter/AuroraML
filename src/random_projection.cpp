#include "ingenuityml/random_projection.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace ingenuityml {
namespace random_projection {

GaussianRandomProjection::GaussianRandomProjection(int n_components, int random_state)
    : n_components_(n_components),
      random_state_(random_state),
      fitted_(false),
      n_features_(0) {}

Estimator& GaussianRandomProjection::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;

    n_features_ = static_cast<int>(X.cols());
    int n_samples = static_cast<int>(X.rows());
    if (n_components_ <= 0) {
        n_components_ = std::min(n_samples, n_features_);
    }
    if (n_components_ <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }

    components_.resize(n_features_, n_components_);
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    double scale = 1.0 / std::sqrt(static_cast<double>(n_components_));
    std::normal_distribution<double> normal(0.0, scale);

    for (int i = 0; i < n_features_; ++i) {
        for (int j = 0; j < n_components_; ++j) {
            components_(i, j) = normal(rng);
        }
    }

    fitted_ = true;
    return *this;
}

MatrixXd GaussianRandomProjection::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("GaussianRandomProjection must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    return X * components_;
}

MatrixXd GaussianRandomProjection::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("GaussianRandomProjection must be fitted before inverse_transform");
    }
    return X * components_.transpose();
}

MatrixXd GaussianRandomProjection::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params GaussianRandomProjection::get_params() const {
    Params params;
    params["n_components"] = std::to_string(n_components_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& GaussianRandomProjection::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

SparseRandomProjection::SparseRandomProjection(int n_components, double density, int random_state)
    : n_components_(n_components),
      density_(density),
      random_state_(random_state),
      fitted_(false),
      n_features_(0) {}

Estimator& SparseRandomProjection::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;

    n_features_ = static_cast<int>(X.cols());
    int n_samples = static_cast<int>(X.rows());
    if (n_components_ <= 0) {
        n_components_ = std::min(n_samples, n_features_);
    }
    if (n_components_ <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }

    double density = density_;
    if (density <= 0.0) {
        density = 1.0 / std::sqrt(std::max(1, n_features_));
    }
    if (density <= 0.0 || density > 1.0) {
        throw std::invalid_argument("density must be in (0, 1]");
    }
    density_ = density;

    components_.setZero(n_features_, n_components_);
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double scale = std::sqrt(1.0 / density_) / std::sqrt(static_cast<double>(n_components_));

    for (int i = 0; i < n_features_; ++i) {
        for (int j = 0; j < n_components_; ++j) {
            double u = uni(rng);
            if (u < density_ * 0.5) {
                components_(i, j) = scale;
            } else if (u < density_) {
                components_(i, j) = -scale;
            }
        }
    }

    fitted_ = true;
    return *this;
}

MatrixXd SparseRandomProjection::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SparseRandomProjection must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    return X * components_;
}

MatrixXd SparseRandomProjection::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SparseRandomProjection must be fitted before inverse_transform");
    }
    return X * components_.transpose();
}

MatrixXd SparseRandomProjection::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SparseRandomProjection::get_params() const {
    Params params;
    params["n_components"] = std::to_string(n_components_);
    params["density"] = std::to_string(density_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& SparseRandomProjection::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    density_ = utils::get_param_double(params, "density", density_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

} // namespace random_projection
} // namespace ingenuityml

#include "ingenuityml/gaussian_process.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace ingenuityml {
namespace gaussian_process {

namespace {

constexpr double kEps = 1e-12;

double sigmoid(double x) {
    if (x >= 0.0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    double z = std::exp(x);
    return z / (1.0 + z);
}

} // namespace

GaussianProcessRegressor::GaussianProcessRegressor(double length_scale, double alpha, bool normalize_y)
    : length_scale_(length_scale),
      alpha_(alpha),
      normalize_y_(normalize_y),
      fitted_(false),
      y_mean_(0.0) {}

MatrixXd GaussianProcessRegressor::compute_kernel(const MatrixXd& A, const MatrixXd& B) const {
    if (length_scale_ <= 0.0) {
        throw std::invalid_argument("length_scale must be positive");
    }
    MatrixXd K(A.rows(), B.rows());
    double denom = 2.0 * length_scale_ * length_scale_;
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < B.rows(); ++j) {
            double dist = (A.row(i) - B.row(j)).squaredNorm();
            K(i, j) = std::exp(-dist / denom);
        }
    }
    return K;
}

Estimator& GaussianProcessRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (alpha_ < 0.0) {
        throw std::invalid_argument("alpha must be non-negative");
    }

    X_train_ = X;
    y_train_ = y;

    VectorXd y_centered = y;
    y_mean_ = 0.0;
    if (normalize_y_) {
        y_mean_ = y.mean();
        y_centered = y.array() - y_mean_;
    }

    MatrixXd K = compute_kernel(X_train_, X_train_);
    K.diagonal().array() += alpha_ + kEps;

    Eigen::LDLT<MatrixXd> ldlt(K);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve GaussianProcessRegressor system");
    }
    alpha_vec_ = ldlt.solve(y_centered);

    fitted_ = true;
    return *this;
}

VectorXd GaussianProcessRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("GaussianProcessRegressor must be fitted before predict");
    }
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd K_star = compute_kernel(X, X_train_);
    VectorXd y_pred = K_star * alpha_vec_;
    if (normalize_y_) {
        y_pred.array() += y_mean_;
    }
    return y_pred;
}

Params GaussianProcessRegressor::get_params() const {
    Params params;
    params["length_scale"] = std::to_string(length_scale_);
    params["alpha"] = std::to_string(alpha_);
    params["normalize_y"] = normalize_y_ ? "true" : "false";
    return params;
}

Estimator& GaussianProcessRegressor::set_params(const Params& params) {
    length_scale_ = utils::get_param_double(params, "length_scale", length_scale_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    normalize_y_ = utils::get_param_bool(params, "normalize_y", normalize_y_);
    return *this;
}

GaussianProcessClassifier::GaussianProcessClassifier(double length_scale, double alpha)
    : length_scale_(length_scale),
      alpha_(alpha),
      fitted_(false) {}

Estimator& GaussianProcessClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);

    VectorXi y_int = y.cast<int>();
    std::set<int> unique_classes;
    for (int i = 0; i < y_int.size(); ++i) {
        unique_classes.insert(y_int(i));
    }

    if (unique_classes.empty()) {
        throw std::invalid_argument("No classes found in y");
    }

    classes_.resize(static_cast<int>(unique_classes.size()));
    int idx = 0;
    for (int cls : unique_classes) {
        classes_(idx++) = cls;
    }

    regressors_.clear();
    regressors_.reserve(classes_.size());

    for (int c = 0; c < classes_.size(); ++c) {
        VectorXd y_binary(X.rows());
        for (int i = 0; i < y_int.size(); ++i) {
            y_binary(i) = (y_int(i) == classes_(c)) ? 1.0 : 0.0;
        }
        GaussianProcessRegressor reg(length_scale_, alpha_, false);
        reg.fit(X, y_binary);
        regressors_.push_back(reg);
    }

    fitted_ = true;
    return *this;
}

MatrixXd GaussianProcessClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("GaussianProcessClassifier must be fitted before predict_proba");
    }
    if (regressors_.empty()) {
        throw std::runtime_error("GaussianProcessClassifier has no trained regressors");
    }

    MatrixXd proba(X.rows(), classes_.size());
    for (int c = 0; c < classes_.size(); ++c) {
        VectorXd scores = regressors_[c].predict(X);
        for (int i = 0; i < scores.size(); ++i) {
            proba(i, c) = sigmoid(scores(i));
        }
    }

    for (int i = 0; i < proba.rows(); ++i) {
        double row_sum = proba.row(i).sum();
        if (row_sum > 0.0) {
            proba.row(i) /= row_sum;
        } else {
            proba.row(i).setConstant(1.0 / static_cast<double>(classes_.size()));
        }
    }

    return proba;
}

VectorXi GaussianProcessClassifier::predict_classes(const MatrixXd& X) const {
    MatrixXd proba = predict_proba(X);
    VectorXi preds(X.rows());
    for (int i = 0; i < proba.rows(); ++i) {
        Eigen::Index max_idx = 0;
        proba.row(i).maxCoeff(&max_idx);
        preds(i) = classes_(static_cast<int>(max_idx));
    }
    return preds;
}

VectorXd GaussianProcessClassifier::decision_function(const MatrixXd& X) const {
    MatrixXd proba = predict_proba(X);
    VectorXd decision(X.rows());
    for (int i = 0; i < proba.rows(); ++i) {
        decision(i) = proba.row(i).maxCoeff();
    }
    return decision;
}

Params GaussianProcessClassifier::get_params() const {
    Params params;
    params["length_scale"] = std::to_string(length_scale_);
    params["alpha"] = std::to_string(alpha_);
    return params;
}

Estimator& GaussianProcessClassifier::set_params(const Params& params) {
    length_scale_ = utils::get_param_double(params, "length_scale", length_scale_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    return *this;
}

} // namespace gaussian_process
} // namespace ingenuityml

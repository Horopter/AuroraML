#include "ingenuityml/cross_decomposition.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ingenuityml {
namespace cross_decomposition {

namespace {

constexpr double kEps = 1e-12;

void validate_xy(const MatrixXd& X, const MatrixXd& Y) {
    validation::check_X(X);
    if (X.rows() != Y.rows()) {
        throw std::invalid_argument("X and Y must have the same number of samples");
    }
    if (Y.cols() == 0) {
        throw std::invalid_argument("Y must have at least one target");
    }
}

int resolve_components(int n_components, int n_samples, int n_features, int n_targets) {
    int max_rank = std::min(n_features, n_targets);
    int sample_limit = std::max(1, n_samples - 1);
    int limit = std::min(max_rank, sample_limit);
    if (limit < 1) {
        throw std::invalid_argument("Not enough samples or features to compute components");
    }
    if (n_components <= 0) {
        return limit;
    }
    return std::min(n_components, limit);
}

void center_scale(const MatrixXd& X, bool scale, VectorXd& mean, VectorXd& std, MatrixXd& Xs) {
    mean = X.colwise().mean();
    MatrixXd Xc = X.rowwise() - mean.transpose();
    if (!scale) {
        std = VectorXd::Ones(X.cols());
        Xs = Xc;
        return;
    }
    int denom = std::max(1, static_cast<int>(X.rows()) - 1);
    VectorXd var = (Xc.array().square().colwise().sum() / static_cast<double>(denom)).matrix();
    std = var.array().sqrt();
    for (int i = 0; i < std.size(); ++i) {
        if (std(i) <= kEps) {
            std(i) = 1.0;
        }
    }
    Xs = Xc.array().rowwise() / std.transpose().array();
}

MatrixXd safe_inverse(const MatrixXd& M) {
    Eigen::CompleteOrthogonalDecomposition<MatrixXd> cod(M);
    return cod.pseudoInverse();
}

MatrixXd invsqrt_sym(const MatrixXd& A) {
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(A);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Failed to compute eigen decomposition");
    }
    VectorXd evals = es.eigenvalues();
    MatrixXd evecs = es.eigenvectors();
    VectorXd invsqrt = evals;
    for (int i = 0; i < evals.size(); ++i) {
        double v = std::max(evals(i), kEps);
        invsqrt(i) = 1.0 / std::sqrt(v);
    }
    return evecs * invsqrt.asDiagonal() * evecs.transpose();
}

} // namespace

PLSCanonical::PLSCanonical(int n_components, bool scale, int max_iter, double tol)
    : n_components_(n_components),
      scale_(scale),
      max_iter_(max_iter),
      tol_(tol),
      fitted_(false),
      n_features_(0),
      n_targets_(0) {}

Estimator& PLSCanonical::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd Y(X.rows(), 1);
    Y.col(0) = y;
    return fit(X, Y);
}

Estimator& PLSCanonical::fit(const MatrixXd& X, const MatrixXd& Y) {
    validate_xy(X, Y);
    n_features_ = static_cast<int>(X.cols());
    n_targets_ = static_cast<int>(Y.cols());

    MatrixXd Xs;
    MatrixXd Ys;
    center_scale(X, scale_, x_mean_, x_std_, Xs);
    center_scale(Y, scale_, y_mean_, y_std_, Ys);

    n_components_ = resolve_components(n_components_, static_cast<int>(X.rows()), n_features_, n_targets_);

    MatrixXd cross_cov = Xs.transpose() * Ys;
    Eigen::JacobiSVD<MatrixXd> svd(cross_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    int max_comp = std::min(static_cast<int>(U.cols()), static_cast<int>(V.cols()));
    int n_comp = std::min(n_components_, max_comp);
    n_components_ = n_comp;

    x_weights_ = U.leftCols(n_comp);
    y_weights_ = V.leftCols(n_comp);
    x_scores_ = Xs * x_weights_;
    y_scores_ = Ys * y_weights_;

    MatrixXd t_cov = x_scores_.transpose() * x_scores_;
    MatrixXd u_cov = y_scores_.transpose() * y_scores_;
    x_loadings_ = (Xs.transpose() * x_scores_) * safe_inverse(t_cov);
    y_loadings_ = (Ys.transpose() * y_scores_) * safe_inverse(u_cov);
    x_rotations_ = x_weights_ * safe_inverse(x_loadings_.transpose() * x_weights_);
    y_rotations_ = y_weights_ * safe_inverse(y_loadings_.transpose() * y_weights_);

    fitted_ = true;
    return *this;
}

MatrixXd PLSCanonical::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PLSCanonical must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd Xs = X.rowwise() - x_mean_.transpose();
    if (scale_) {
        Xs = Xs.array().rowwise() / x_std_.transpose().array();
    }
    if (x_rotations_.size() > 0) {
        return Xs * x_rotations_;
    }
    return Xs * x_weights_;
}

MatrixXd PLSCanonical::transform_y(const MatrixXd& Y) const {
    if (!fitted_) {
        throw std::runtime_error("PLSCanonical must be fitted before transform_y");
    }
    if (Y.cols() != n_targets_) {
        throw std::invalid_argument("Y must have the same number of targets as training data");
    }
    MatrixXd Ys = Y.rowwise() - y_mean_.transpose();
    if (scale_) {
        Ys = Ys.array().rowwise() / y_std_.transpose().array();
    }
    if (y_rotations_.size() > 0) {
        return Ys * y_rotations_;
    }
    return Ys * y_weights_;
}

MatrixXd PLSCanonical::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("PLSCanonical does not support inverse_transform");
}

MatrixXd PLSCanonical::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

MatrixXd PLSCanonical::fit_transform(const MatrixXd& X, const MatrixXd& Y) {
    fit(X, Y);
    return transform(X);
}

Params PLSCanonical::get_params() const {
    Params params;
    params["n_components"] = std::to_string(n_components_);
    params["scale"] = scale_ ? "true" : "false";
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    return params;
}

Estimator& PLSCanonical::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    scale_ = utils::get_param_bool(params, "scale", scale_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

PLSRegression::PLSRegression(int n_components, bool scale, int max_iter, double tol)
    : n_components_(n_components),
      scale_(scale),
      max_iter_(max_iter),
      tol_(tol),
      fitted_(false),
      n_features_(0),
      n_targets_(0) {}

Estimator& PLSRegression::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd Y(X.rows(), 1);
    Y.col(0) = y;
    return fit(X, Y);
}

Estimator& PLSRegression::fit(const MatrixXd& X, const MatrixXd& Y) {
    validate_xy(X, Y);
    n_features_ = static_cast<int>(X.cols());
    n_targets_ = static_cast<int>(Y.cols());

    MatrixXd Xs;
    MatrixXd Ys;
    center_scale(X, scale_, x_mean_, x_std_, Xs);
    center_scale(Y, scale_, y_mean_, y_std_, Ys);

    n_components_ = resolve_components(n_components_, static_cast<int>(X.rows()), n_features_, n_targets_);

    MatrixXd cross_cov = Xs.transpose() * Ys;
    Eigen::JacobiSVD<MatrixXd> svd(cross_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    int max_comp = std::min(static_cast<int>(U.cols()), static_cast<int>(V.cols()));
    int n_comp = std::min(n_components_, max_comp);
    n_components_ = n_comp;

    x_weights_ = U.leftCols(n_comp);
    y_weights_ = V.leftCols(n_comp);
    x_scores_ = Xs * x_weights_;
    y_scores_ = Ys * y_weights_;

    MatrixXd t_cov = x_scores_.transpose() * x_scores_;
    MatrixXd u_cov = y_scores_.transpose() * y_scores_;
    x_loadings_ = (Xs.transpose() * x_scores_) * safe_inverse(t_cov);
    y_loadings_ = (Ys.transpose() * y_scores_) * safe_inverse(u_cov);
    x_rotations_ = x_weights_ * safe_inverse(x_loadings_.transpose() * x_weights_);

    MatrixXd coef_scaled = x_rotations_ * y_loadings_.transpose();
    coef_ = coef_scaled;
    if (scale_) {
        for (int j = 0; j < n_targets_; ++j) {
            coef_.col(j) *= y_std_(j);
        }
        for (int i = 0; i < n_features_; ++i) {
            coef_.row(i) /= x_std_(i);
        }
    }
    intercept_ = y_mean_ - (x_mean_.transpose() * coef_).transpose();

    fitted_ = true;
    return *this;
}

VectorXd PLSRegression::predict(const MatrixXd& X) const {
    MatrixXd Y_pred = predict_multi(X);
    if (Y_pred.cols() != 1) {
        throw std::runtime_error("PLSRegression has multiple targets; use predict_multi");
    }
    return Y_pred.col(0);
}

MatrixXd PLSRegression::predict_multi(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PLSRegression must be fitted before predict");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd Y_pred = X * coef_;
    Y_pred.rowwise() += intercept_.transpose();
    return Y_pred;
}

MatrixXd PLSRegression::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PLSRegression must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd Xs = X.rowwise() - x_mean_.transpose();
    if (scale_) {
        Xs = Xs.array().rowwise() / x_std_.transpose().array();
    }
    if (x_rotations_.size() > 0) {
        return Xs * x_rotations_;
    }
    return Xs * x_weights_;
}

MatrixXd PLSRegression::transform_y(const MatrixXd& Y) const {
    if (!fitted_) {
        throw std::runtime_error("PLSRegression must be fitted before transform_y");
    }
    if (Y.cols() != n_targets_) {
        throw std::invalid_argument("Y must have the same number of targets as training data");
    }
    MatrixXd Ys = Y.rowwise() - y_mean_.transpose();
    if (scale_) {
        Ys = Ys.array().rowwise() / y_std_.transpose().array();
    }
    return Ys * y_weights_;
}

MatrixXd PLSRegression::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("PLSRegression does not support inverse_transform");
}

MatrixXd PLSRegression::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

MatrixXd PLSRegression::fit_transform(const MatrixXd& X, const MatrixXd& Y) {
    fit(X, Y);
    return transform(X);
}

Params PLSRegression::get_params() const {
    Params params;
    params["n_components"] = std::to_string(n_components_);
    params["scale"] = scale_ ? "true" : "false";
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    return params;
}

Estimator& PLSRegression::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    scale_ = utils::get_param_bool(params, "scale", scale_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

CCA::CCA(int n_components, bool scale, int max_iter, double tol)
    : n_components_(n_components),
      scale_(scale),
      max_iter_(max_iter),
      tol_(tol),
      fitted_(false),
      n_features_(0),
      n_targets_(0) {}

Estimator& CCA::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd Y(X.rows(), 1);
    Y.col(0) = y;
    return fit(X, Y);
}

Estimator& CCA::fit(const MatrixXd& X, const MatrixXd& Y) {
    validate_xy(X, Y);
    n_features_ = static_cast<int>(X.cols());
    n_targets_ = static_cast<int>(Y.cols());

    MatrixXd Xs;
    MatrixXd Ys;
    center_scale(X, scale_, x_mean_, x_std_, Xs);
    center_scale(Y, scale_, y_mean_, y_std_, Ys);

    n_components_ = resolve_components(n_components_, static_cast<int>(X.rows()), n_features_, n_targets_);

    int denom = std::max(1, static_cast<int>(Xs.rows()) - 1);
    MatrixXd Sxx = (Xs.transpose() * Xs) / static_cast<double>(denom);
    MatrixXd Syy = (Ys.transpose() * Ys) / static_cast<double>(denom);
    MatrixXd Sxy = (Xs.transpose() * Ys) / static_cast<double>(denom);

    MatrixXd Sxx_reg = Sxx + kEps * MatrixXd::Identity(n_features_, n_features_);
    MatrixXd Syy_reg = Syy + kEps * MatrixXd::Identity(n_targets_, n_targets_);
    MatrixXd invsqrt_x = invsqrt_sym(Sxx_reg);
    MatrixXd invsqrt_y = invsqrt_sym(Syy_reg);
    MatrixXd M = invsqrt_x * Sxy * invsqrt_y;

    Eigen::JacobiSVD<MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    int max_comp = std::min(static_cast<int>(U.cols()), static_cast<int>(V.cols()));
    int n_comp = std::min(n_components_, max_comp);
    n_components_ = n_comp;

    x_weights_ = invsqrt_x * U.leftCols(n_comp);
    y_weights_ = invsqrt_y * V.leftCols(n_comp);
    correlations_ = svd.singularValues().head(n_comp);

    x_scores_ = Xs * x_weights_;
    y_scores_ = Ys * y_weights_;

    fitted_ = true;
    return *this;
}

MatrixXd CCA::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("CCA must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd Xs = X.rowwise() - x_mean_.transpose();
    if (scale_) {
        Xs = Xs.array().rowwise() / x_std_.transpose().array();
    }
    return Xs * x_weights_;
}

MatrixXd CCA::transform_y(const MatrixXd& Y) const {
    if (!fitted_) {
        throw std::runtime_error("CCA must be fitted before transform_y");
    }
    if (Y.cols() != n_targets_) {
        throw std::invalid_argument("Y must have the same number of targets as training data");
    }
    MatrixXd Ys = Y.rowwise() - y_mean_.transpose();
    if (scale_) {
        Ys = Ys.array().rowwise() / y_std_.transpose().array();
    }
    return Ys * y_weights_;
}

MatrixXd CCA::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("CCA does not support inverse_transform");
}

MatrixXd CCA::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

MatrixXd CCA::fit_transform(const MatrixXd& X, const MatrixXd& Y) {
    fit(X, Y);
    return transform(X);
}

Params CCA::get_params() const {
    Params params;
    params["n_components"] = std::to_string(n_components_);
    params["scale"] = scale_ ? "true" : "false";
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    return params;
}

Estimator& CCA::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    scale_ = utils::get_param_bool(params, "scale", scale_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

PLSSVD::PLSSVD(int n_components, bool scale)
    : n_components_(n_components),
      scale_(scale),
      fitted_(false),
      n_features_(0),
      n_targets_(0) {}

Estimator& PLSSVD::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd Y(X.rows(), 1);
    Y.col(0) = y;
    return fit(X, Y);
}

Estimator& PLSSVD::fit(const MatrixXd& X, const MatrixXd& Y) {
    validate_xy(X, Y);
    n_features_ = static_cast<int>(X.cols());
    n_targets_ = static_cast<int>(Y.cols());

    MatrixXd Xs;
    MatrixXd Ys;
    center_scale(X, scale_, x_mean_, x_std_, Xs);
    center_scale(Y, scale_, y_mean_, y_std_, Ys);

    n_components_ = resolve_components(n_components_, static_cast<int>(X.rows()), n_features_, n_targets_);

    MatrixXd cross_cov = Xs.transpose() * Ys;
    Eigen::JacobiSVD<MatrixXd> svd(cross_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    int max_comp = std::min(static_cast<int>(U.cols()), static_cast<int>(V.cols()));
    int n_comp = std::min(n_components_, max_comp);
    n_components_ = n_comp;

    x_weights_ = U.leftCols(n_comp);
    y_weights_ = V.leftCols(n_comp);
    x_scores_ = Xs * x_weights_;
    y_scores_ = Ys * y_weights_;

    fitted_ = true;
    return *this;
}

MatrixXd PLSSVD::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PLSSVD must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd Xs = X.rowwise() - x_mean_.transpose();
    if (scale_) {
        Xs = Xs.array().rowwise() / x_std_.transpose().array();
    }
    return Xs * x_weights_;
}

MatrixXd PLSSVD::transform_y(const MatrixXd& Y) const {
    if (!fitted_) {
        throw std::runtime_error("PLSSVD must be fitted before transform_y");
    }
    if (Y.cols() != n_targets_) {
        throw std::invalid_argument("Y must have the same number of targets as training data");
    }
    MatrixXd Ys = Y.rowwise() - y_mean_.transpose();
    if (scale_) {
        Ys = Ys.array().rowwise() / y_std_.transpose().array();
    }
    return Ys * y_weights_;
}

MatrixXd PLSSVD::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("PLSSVD does not support inverse_transform");
}

MatrixXd PLSSVD::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

MatrixXd PLSSVD::fit_transform(const MatrixXd& X, const MatrixXd& Y) {
    fit(X, Y);
    return transform(X);
}

Params PLSSVD::get_params() const {
    Params params;
    params["n_components"] = std::to_string(n_components_);
    params["scale"] = scale_ ? "true" : "false";
    return params;
}

Estimator& PLSSVD::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    scale_ = utils::get_param_bool(params, "scale", scale_);
    return *this;
}

} // namespace cross_decomposition
} // namespace ingenuityml

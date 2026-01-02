#include "auroraml/linear_model.hpp"
#include "auroraml/model_selection.hpp"
#include "auroraml/metrics.hpp"
#include "auroraml/base.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

namespace auroraml {
namespace linear_model {

namespace {

double sigmoid(double z) {
    z = std::max(-500.0, std::min(500.0, z));
    return 1.0 / (1.0 + std::exp(-z));
}

double soft_threshold(double value, double threshold) {
    if (value > threshold) return value - threshold;
    if (value < -threshold) return value + threshold;
    return 0.0;
}

void center_data(const MatrixXd& X, const VectorXd& y, bool fit_intercept,
                 MatrixXd& X_centered, VectorXd& y_centered,
                 VectorXd& X_mean, double& y_mean) {
    X_centered = X;
    y_centered = y;
    X_mean = VectorXd::Zero(X.cols());
    y_mean = 0.0;
    if (fit_intercept) {
        X_mean = X.colwise().mean();
        y_mean = y.mean();
        X_centered.rowwise() -= X_mean.transpose();
        y_centered.array() -= y_mean;
    }
}

void fit_ridge_closed_form(const MatrixXd& X, const VectorXd& y, double alpha, bool fit_intercept,
                           VectorXd& coef, double& intercept) {
    MatrixXd X_work;
    VectorXd y_work;
    VectorXd X_mean;
    double y_mean = 0.0;
    center_data(X, y, fit_intercept, X_work, y_work, X_mean, y_mean);

    MatrixXd XtX = X_work.transpose() * X_work;
    MatrixXd reg = alpha * MatrixXd::Identity(X.cols(), X.cols());
    VectorXd Xty = X_work.transpose() * y_work;
    coef = (XtX + reg).ldlt().solve(Xty);
    intercept = fit_intercept ? (y_mean - X_mean.dot(coef)) : 0.0;
}

void fit_lasso_coordinate_descent(const MatrixXd& X, const VectorXd& y,
                                  double alpha, double l1_ratio, bool fit_intercept,
                                  int max_iter, double tol,
                                  VectorXd& coef, double& intercept) {
    MatrixXd X_work;
    VectorXd y_work;
    VectorXd X_mean;
    double y_mean = 0.0;
    center_data(X, y, fit_intercept, X_work, y_work, X_mean, y_mean);

    int n_samples = X_work.rows();
    int n_features = X_work.cols();
    coef = VectorXd::Zero(n_features);

    VectorXd col_norms(n_features);
    for (int j = 0; j < n_features; ++j) {
        col_norms(j) = X_work.col(j).squaredNorm() / static_cast<double>(n_samples);
    }

    double l1 = alpha * l1_ratio;
    double l2 = alpha * (1.0 - l1_ratio);

    for (int iter = 0; iter < max_iter; ++iter) {
        VectorXd coef_old = coef;
        VectorXd y_pred = X_work * coef;

        for (int j = 0; j < n_features; ++j) {
            double tmp = coef(j);
            double rho = X_work.col(j).dot(y_work - y_pred + X_work.col(j) * tmp) / static_cast<double>(n_samples);
            double denom = col_norms(j) + l2;
            coef(j) = soft_threshold(rho, l1) / denom;
            y_pred += X_work.col(j) * (coef(j) - tmp);
        }

        if ((coef - coef_old).norm() < tol) {
            break;
        }
    }

    intercept = fit_intercept ? (y_mean - X_mean.dot(coef)) : 0.0;
}

void fit_omp(const MatrixXd& X, const VectorXd& y, int n_nonzero_coefs,
             int max_iter, bool fit_intercept, double tol,
             VectorXd& coef, double& intercept) {
    MatrixXd X_work;
    VectorXd y_work;
    VectorXd X_mean;
    double y_mean = 0.0;
    center_data(X, y, fit_intercept, X_work, y_work, X_mean, y_mean);

    int n_features = X_work.cols();
    int max_features = std::min(n_features, max_iter);
    int target_nonzero = n_nonzero_coefs <= 0 ? max_features : std::min(n_nonzero_coefs, max_features);

    coef = VectorXd::Zero(n_features);
    VectorXd residual = y_work;

    std::vector<int> active;
    active.reserve(target_nonzero);
    std::vector<bool> selected(n_features, false);

    for (int iter = 0; iter < target_nonzero; ++iter) {
        VectorXd corr = X_work.transpose() * residual;
        int best_idx = -1;
        double best_val = -1.0;
        for (int j = 0; j < n_features; ++j) {
            if (selected[j]) {
                continue;
            }
            double val = std::abs(corr(j));
            if (val > best_val) {
                best_val = val;
                best_idx = j;
            }
        }
        if (best_idx < 0) {
            break;
        }
        selected[best_idx] = true;
        active.push_back(best_idx);

        MatrixXd X_active(X_work.rows(), static_cast<int>(active.size()));
        for (size_t k = 0; k < active.size(); ++k) {
            X_active.col(static_cast<int>(k)) = X_work.col(active[k]);
        }
        VectorXd coef_active = X_active.colPivHouseholderQr().solve(y_work);
        coef.setZero();
        for (size_t k = 0; k < active.size(); ++k) {
            coef(active[k]) = coef_active(static_cast<int>(k));
        }

        residual = y_work - X_work * coef;
        if (residual.norm() < tol) {
            break;
        }
    }

    intercept = fit_intercept ? (y_mean - X_mean.dot(coef)) : 0.0;
}

std::vector<int> unique_classes(const VectorXd& y) {
    std::set<int> unique;
    for (int i = 0; i < y.size(); ++i) {
        unique.insert(static_cast<int>(y(i)));
    }
    return std::vector<int>(unique.begin(), unique.end());
}

std::string join_double_list(const std::vector<double>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ",";
        oss << values[i];
    }
    return oss.str();
}

std::vector<double> parse_double_list(const std::string& value, const std::vector<double>& fallback) {
    if (value.empty()) {
        return fallback;
    }
    std::vector<double> result;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            result.push_back(std::stod(token));
        } catch (const std::exception&) {
        }
    }
    return result.empty() ? fallback : result;
}

double compute_learning_rate(const std::string& schedule, double eta0, double power_t, int t) {
    if (schedule == "constant") {
        return eta0;
    }
    if (schedule == "invscaling") {
        return eta0 / std::pow(static_cast<double>(t + 1), power_t);
    }
    if (schedule == "adaptive") {
        return eta0;
    }
    throw std::invalid_argument("Unsupported learning_rate: " + schedule);
}

} // namespace

Lars::Lars(int n_nonzero_coefs, bool fit_intercept, int max_iter, double eps)
    : coef_(), intercept_(0.0), fitted_(false), n_nonzero_coefs_(n_nonzero_coefs),
      max_iter_(max_iter), fit_intercept_(fit_intercept), eps_(eps) {}

Estimator& Lars::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
    fit_omp(X, y, n_nonzero_coefs_, max_iter_, fit_intercept_, eps_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd Lars::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Lars must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params Lars::get_params() const {
    return {
        {"n_nonzero_coefs", std::to_string(n_nonzero_coefs_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"eps", std::to_string(eps_)}
    };
}

Estimator& Lars::set_params(const Params& params) {
    n_nonzero_coefs_ = utils::get_param_int(params, "n_nonzero_coefs", n_nonzero_coefs_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    eps_ = utils::get_param_double(params, "eps", eps_);
    return *this;
}

bool Lars::is_fitted() const {
    return fitted_;
}

LarsCV::LarsCV(int cv_folds, bool fit_intercept, int max_iter, double eps)
    : coef_(), intercept_(0.0), fitted_(false), cv_folds_(cv_folds), max_iter_(max_iter),
      best_n_nonzero_coefs_(0), fit_intercept_(fit_intercept), eps_(eps) {}

Estimator& LarsCV::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (cv_folds_ <= 1) {
        throw std::invalid_argument("cv_folds must be at least 2");
    }
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }

    int max_nonzero = std::min(static_cast<int>(X.cols()), max_iter_);
    model_selection::KFold kfold(cv_folds_, true, 42);
    auto splits = kfold.split(X, y);

    double best_mse = std::numeric_limits<double>::infinity();
    int best_k = 1;

    for (int k = 1; k <= max_nonzero; ++k) {
        double mse_sum = 0.0;
        for (const auto& split : splits) {
            const auto& train_idx = split.first;
            const auto& test_idx = split.second;

            MatrixXd X_train(train_idx.size(), X.cols());
            VectorXd y_train(train_idx.size());
            MatrixXd X_test(test_idx.size(), X.cols());
            VectorXd y_test(test_idx.size());

            for (size_t i = 0; i < train_idx.size(); ++i) {
                X_train.row(i) = X.row(train_idx[i]);
                y_train(i) = y(train_idx[i]);
            }
            for (size_t i = 0; i < test_idx.size(); ++i) {
                X_test.row(i) = X.row(test_idx[i]);
                y_test(i) = y(test_idx[i]);
            }

            VectorXd coef_fold;
            double intercept_fold = 0.0;
            fit_omp(X_train, y_train, k, max_iter_, fit_intercept_, eps_, coef_fold, intercept_fold);
            VectorXd preds = X_test * coef_fold;
            preds.array() += intercept_fold;
            mse_sum += metrics::mean_squared_error(y_test, preds);
        }
        double mse = mse_sum / static_cast<double>(splits.size());
        if (mse < best_mse) {
            best_mse = mse;
            best_k = k;
        }
    }

    best_n_nonzero_coefs_ = best_k;
    fit_omp(X, y, best_n_nonzero_coefs_, max_iter_, fit_intercept_, eps_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd LarsCV::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LarsCV must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params LarsCV::get_params() const {
    return {
        {"cv", std::to_string(cv_folds_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"eps", std::to_string(eps_)}
    };
}

Estimator& LarsCV::set_params(const Params& params) {
    cv_folds_ = utils::get_param_int(params, "cv", cv_folds_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    eps_ = utils::get_param_double(params, "eps", eps_);
    return *this;
}

bool LarsCV::is_fitted() const {
    return fitted_;
}

LassoLars::LassoLars(double alpha, bool fit_intercept, int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), alpha_(alpha),
      fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

Estimator& LassoLars::fit(const MatrixXd& X, const VectorXd& y) {
    if (alpha_ <= 0.0) {
        throw std::invalid_argument("alpha must be positive");
    }
    validation::check_X_y(X, y);
    fit_lasso_coordinate_descent(X, y, alpha_, 1.0, fit_intercept_, max_iter_, tol_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd LassoLars::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LassoLars must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params LassoLars::get_params() const {
    return {
        {"alpha", std::to_string(alpha_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& LassoLars::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool LassoLars::is_fitted() const {
    return fitted_;
}

LassoLarsCV::LassoLarsCV(const std::vector<double>& alphas, int cv_folds,
                         bool fit_intercept, int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), alphas_(alphas), best_alpha_(0.0),
      cv_folds_(cv_folds), fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

Estimator& LassoLarsCV::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (alphas_.empty()) {
        throw std::invalid_argument("alphas must not be empty");
    }
    if (cv_folds_ <= 1) {
        throw std::invalid_argument("cv_folds must be at least 2");
    }

    model_selection::KFold kfold(cv_folds_, true, 42);
    auto splits = kfold.split(X, y);

    double best_mse = std::numeric_limits<double>::infinity();
    double best_alpha = alphas_.front();

    for (double alpha : alphas_) {
        if (alpha <= 0.0) {
            continue;
        }
        double mse_sum = 0.0;
        for (const auto& split : splits) {
            const auto& train_idx = split.first;
            const auto& test_idx = split.second;

            MatrixXd X_train(train_idx.size(), X.cols());
            VectorXd y_train(train_idx.size());
            MatrixXd X_test(test_idx.size(), X.cols());
            VectorXd y_test(test_idx.size());

            for (size_t i = 0; i < train_idx.size(); ++i) {
                X_train.row(i) = X.row(train_idx[i]);
                y_train(i) = y(train_idx[i]);
            }
            for (size_t i = 0; i < test_idx.size(); ++i) {
                X_test.row(i) = X.row(test_idx[i]);
                y_test(i) = y(test_idx[i]);
            }

            VectorXd coef_fold;
            double intercept_fold = 0.0;
            fit_lasso_coordinate_descent(X_train, y_train, alpha, 1.0, fit_intercept_,
                                         max_iter_, tol_, coef_fold, intercept_fold);
            VectorXd preds = X_test * coef_fold;
            preds.array() += intercept_fold;
            mse_sum += metrics::mean_squared_error(y_test, preds);
        }
        double mse = mse_sum / static_cast<double>(splits.size());
        if (mse < best_mse) {
            best_mse = mse;
            best_alpha = alpha;
        }
    }

    best_alpha_ = best_alpha;
    fit_lasso_coordinate_descent(X, y, best_alpha_, 1.0, fit_intercept_, max_iter_, tol_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd LassoLarsCV::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LassoLarsCV must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params LassoLarsCV::get_params() const {
    return {
        {"alphas", join_double_list(alphas_)},
        {"cv", std::to_string(cv_folds_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& LassoLarsCV::set_params(const Params& params) {
    alphas_ = parse_double_list(utils::get_param_string(params, "alphas", join_double_list(alphas_)), alphas_);
    cv_folds_ = utils::get_param_int(params, "cv", cv_folds_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool LassoLarsCV::is_fitted() const {
    return fitted_;
}

LassoLarsIC::LassoLarsIC(const std::vector<double>& alphas, const std::string& criterion,
                         bool fit_intercept, int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), alphas_(alphas), criterion_(criterion),
      best_alpha_(0.0), fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

Estimator& LassoLarsIC::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (alphas_.empty()) {
        throw std::invalid_argument("alphas must not be empty");
    }
    if (criterion_ != "aic" && criterion_ != "bic") {
        throw std::invalid_argument("criterion must be 'aic' or 'bic'");
    }

    int n_samples = X.rows();
    double best_ic = std::numeric_limits<double>::infinity();
    double best_alpha = alphas_.front();
    VectorXd best_coef;
    double best_intercept = 0.0;

    for (double alpha : alphas_) {
        if (alpha <= 0.0) {
            continue;
        }
        VectorXd coef_tmp;
        double intercept_tmp = 0.0;
        fit_lasso_coordinate_descent(X, y, alpha, 1.0, fit_intercept_, max_iter_, tol_, coef_tmp, intercept_tmp);
        VectorXd preds = X * coef_tmp;
        preds.array() += intercept_tmp;
        double rss = (y - preds).squaredNorm();
        int k = 1;
        for (int i = 0; i < coef_tmp.size(); ++i) {
            if (coef_tmp(i) != 0.0) {
                ++k;
            }
        }
        double ic = 0.0;
        double sigma2 = rss / static_cast<double>(n_samples);
        if (sigma2 <= 0.0) {
            sigma2 = 1e-12;
        }
        if (criterion_ == "aic") {
            ic = n_samples * std::log(sigma2) + 2.0 * k;
        } else {
            ic = n_samples * std::log(sigma2) + std::log(static_cast<double>(n_samples)) * k;
        }
        if (ic < best_ic) {
            best_ic = ic;
            best_alpha = alpha;
            best_coef = coef_tmp;
            best_intercept = intercept_tmp;
        }
    }

    best_alpha_ = best_alpha;
    coef_ = best_coef;
    intercept_ = best_intercept;
    fitted_ = true;
    return *this;
}

VectorXd LassoLarsIC::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LassoLarsIC must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params LassoLarsIC::get_params() const {
    return {
        {"alphas", join_double_list(alphas_)},
        {"criterion", criterion_},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& LassoLarsIC::set_params(const Params& params) {
    alphas_ = parse_double_list(utils::get_param_string(params, "alphas", join_double_list(alphas_)), alphas_);
    criterion_ = utils::get_param_string(params, "criterion", criterion_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool LassoLarsIC::is_fitted() const {
    return fitted_;
}

OrthogonalMatchingPursuit::OrthogonalMatchingPursuit(int n_nonzero_coefs, bool fit_intercept,
                                                     int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), n_nonzero_coefs_(n_nonzero_coefs),
      fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

Estimator& OrthogonalMatchingPursuit::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
    fit_omp(X, y, n_nonzero_coefs_, max_iter_, fit_intercept_, tol_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd OrthogonalMatchingPursuit::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OrthogonalMatchingPursuit must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params OrthogonalMatchingPursuit::get_params() const {
    return {
        {"n_nonzero_coefs", std::to_string(n_nonzero_coefs_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& OrthogonalMatchingPursuit::set_params(const Params& params) {
    n_nonzero_coefs_ = utils::get_param_int(params, "n_nonzero_coefs", n_nonzero_coefs_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool OrthogonalMatchingPursuit::is_fitted() const {
    return fitted_;
}

OrthogonalMatchingPursuitCV::OrthogonalMatchingPursuitCV(int cv_folds, bool fit_intercept,
                                                         int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), cv_folds_(cv_folds), max_iter_(max_iter),
      best_n_nonzero_coefs_(0), fit_intercept_(fit_intercept), tol_(tol) {}

Estimator& OrthogonalMatchingPursuitCV::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (cv_folds_ <= 1) {
        throw std::invalid_argument("cv_folds must be at least 2");
    }
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }

    int max_nonzero = std::min(static_cast<int>(X.cols()), max_iter_);
    model_selection::KFold kfold(cv_folds_, true, 42);
    auto splits = kfold.split(X, y);

    double best_mse = std::numeric_limits<double>::infinity();
    int best_k = 1;

    for (int k = 1; k <= max_nonzero; ++k) {
        double mse_sum = 0.0;
        for (const auto& split : splits) {
            const auto& train_idx = split.first;
            const auto& test_idx = split.second;

            MatrixXd X_train(train_idx.size(), X.cols());
            VectorXd y_train(train_idx.size());
            MatrixXd X_test(test_idx.size(), X.cols());
            VectorXd y_test(test_idx.size());

            for (size_t i = 0; i < train_idx.size(); ++i) {
                X_train.row(i) = X.row(train_idx[i]);
                y_train(i) = y(train_idx[i]);
            }
            for (size_t i = 0; i < test_idx.size(); ++i) {
                X_test.row(i) = X.row(test_idx[i]);
                y_test(i) = y(test_idx[i]);
            }

            VectorXd coef_fold;
            double intercept_fold = 0.0;
            fit_omp(X_train, y_train, k, max_iter_, fit_intercept_, tol_, coef_fold, intercept_fold);
            VectorXd preds = X_test * coef_fold;
            preds.array() += intercept_fold;
            mse_sum += metrics::mean_squared_error(y_test, preds);
        }
        double mse = mse_sum / static_cast<double>(splits.size());
        if (mse < best_mse) {
            best_mse = mse;
            best_k = k;
        }
    }

    best_n_nonzero_coefs_ = best_k;
    fit_omp(X, y, best_n_nonzero_coefs_, max_iter_, fit_intercept_, tol_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd OrthogonalMatchingPursuitCV::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OrthogonalMatchingPursuitCV must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params OrthogonalMatchingPursuitCV::get_params() const {
    return {
        {"cv", std::to_string(cv_folds_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& OrthogonalMatchingPursuitCV::set_params(const Params& params) {
    cv_folds_ = utils::get_param_int(params, "cv", cv_folds_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool OrthogonalMatchingPursuitCV::is_fitted() const {
    return fitted_;
}

RANSACRegressor::RANSACRegressor(int max_trials, int min_samples,
                                 double residual_threshold, int random_state,
                                 bool fit_intercept)
    : coef_(), intercept_(0.0), fitted_(false), max_trials_(max_trials),
      min_samples_(min_samples), residual_threshold_(residual_threshold),
      random_state_(random_state), fit_intercept_(fit_intercept) {}

Estimator& RANSACRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (max_trials_ <= 0) {
        throw std::invalid_argument("max_trials must be positive");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    int min_samples = min_samples_ <= 0 ? std::min(n_samples, n_features + 1) : min_samples_;
    if (min_samples <= 0 || min_samples > n_samples) {
        throw std::invalid_argument("min_samples must be between 1 and n_samples");
    }

    double threshold = residual_threshold_;
    if (threshold <= 0.0) {
        VectorXd centered = y.array() - y.mean();
        VectorXd abs_dev = centered.array().abs();
        std::vector<double> values(abs_dev.data(), abs_dev.data() + abs_dev.size());
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        double med = values[values.size() / 2];
        threshold = std::max(1e-3, 1.4826 * med);
    }

    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    int best_inliers = -1;
    VectorXd best_coef;
    double best_intercept = 0.0;

    for (int trial = 0; trial < max_trials_; ++trial) {
        std::shuffle(indices.begin(), indices.end(), rng);
        MatrixXd X_subset(min_samples, n_features);
        VectorXd y_subset(min_samples);
        for (int i = 0; i < min_samples; ++i) {
            X_subset.row(i) = X.row(indices[i]);
            y_subset(i) = y(indices[i]);
        }

        VectorXd coef_tmp;
        double intercept_tmp = 0.0;
        fit_ridge_closed_form(X_subset, y_subset, 0.0, fit_intercept_, coef_tmp, intercept_tmp);

        VectorXd preds = X * coef_tmp;
        preds.array() += intercept_tmp;
        VectorXd residuals = (y - preds).array().abs();
        int inliers = 0;
        for (int i = 0; i < residuals.size(); ++i) {
            if (residuals(i) <= threshold) {
                ++inliers;
            }
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_coef = coef_tmp;
            best_intercept = intercept_tmp;
        }
    }

    if (best_inliers <= 0) {
        fit_ridge_closed_form(X, y, 0.0, fit_intercept_, coef_, intercept_);
    } else {
        coef_ = best_coef;
        intercept_ = best_intercept;
    }

    fitted_ = true;
    return *this;
}

VectorXd RANSACRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RANSACRegressor must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params RANSACRegressor::get_params() const {
    return {
        {"max_trials", std::to_string(max_trials_)},
        {"min_samples", std::to_string(min_samples_)},
        {"residual_threshold", std::to_string(residual_threshold_)},
        {"random_state", std::to_string(random_state_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"}
    };
}

Estimator& RANSACRegressor::set_params(const Params& params) {
    max_trials_ = utils::get_param_int(params, "max_trials", max_trials_);
    min_samples_ = utils::get_param_int(params, "min_samples", min_samples_);
    residual_threshold_ = utils::get_param_double(params, "residual_threshold", residual_threshold_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    return *this;
}

bool RANSACRegressor::is_fitted() const {
    return fitted_;
}

TheilSenRegressor::TheilSenRegressor(int n_subsamples, int random_state, bool fit_intercept)
    : coef_(), intercept_(0.0), fitted_(false), n_subsamples_(n_subsamples),
      random_state_(random_state), fit_intercept_(fit_intercept) {}

Estimator& TheilSenRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);

    int n_samples = X.rows();
    int n_features = X.cols();
    int subset_size = std::min(n_samples, n_features + 1);
    int n_subsamples = n_subsamples_ <= 0 ? 100 : n_subsamples_;

    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    MatrixXd coef_samples(n_subsamples, n_features);
    VectorXd intercept_samples(n_subsamples);

    for (int s = 0; s < n_subsamples; ++s) {
        std::shuffle(indices.begin(), indices.end(), rng);
        MatrixXd X_subset(subset_size, n_features);
        VectorXd y_subset(subset_size);
        for (int i = 0; i < subset_size; ++i) {
            X_subset.row(i) = X.row(indices[i]);
            y_subset(i) = y(indices[i]);
        }
        VectorXd coef_tmp;
        double intercept_tmp = 0.0;
        fit_ridge_closed_form(X_subset, y_subset, 0.0, fit_intercept_, coef_tmp, intercept_tmp);
        coef_samples.row(s) = coef_tmp.transpose();
        intercept_samples(s) = intercept_tmp;
    }

    coef_ = VectorXd::Zero(n_features);
    for (int j = 0; j < n_features; ++j) {
        std::vector<double> values(coef_samples.rows());
        for (int i = 0; i < coef_samples.rows(); ++i) {
            values[i] = coef_samples(i, j);
        }
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        coef_(j) = values[values.size() / 2];
    }
    std::vector<double> intercept_vals(intercept_samples.data(), intercept_samples.data() + intercept_samples.size());
    std::nth_element(intercept_vals.begin(), intercept_vals.begin() + intercept_vals.size() / 2, intercept_vals.end());
    intercept_ = intercept_vals[intercept_vals.size() / 2];

    fitted_ = true;
    return *this;
}

VectorXd TheilSenRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("TheilSenRegressor must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params TheilSenRegressor::get_params() const {
    return {
        {"n_subsamples", std::to_string(n_subsamples_)},
        {"random_state", std::to_string(random_state_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"}
    };
}

Estimator& TheilSenRegressor::set_params(const Params& params) {
    n_subsamples_ = utils::get_param_int(params, "n_subsamples", n_subsamples_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    return *this;
}

bool TheilSenRegressor::is_fitted() const {
    return fitted_;
}

SGDRegressor::SGDRegressor(const std::string& loss, const std::string& penalty, double alpha,
                           double l1_ratio, bool fit_intercept, int max_iter, double tol,
                           const std::string& learning_rate, double eta0, double power_t,
                           bool shuffle, int random_state, double epsilon)
    : coef_(), intercept_(0.0), fitted_(false), loss_(loss), penalty_(penalty),
      alpha_(alpha), l1_ratio_(l1_ratio), fit_intercept_(fit_intercept),
      max_iter_(max_iter), tol_(tol), learning_rate_(learning_rate), eta0_(eta0),
      power_t_(power_t), shuffle_(shuffle), random_state_(random_state), epsilon_(epsilon) {}

Estimator& SGDRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (alpha_ < 0.0) {
        throw std::invalid_argument("alpha must be non-negative");
    }
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (eta0_ <= 0.0) {
        throw std::invalid_argument("eta0 must be positive");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    coef_ = VectorXd::Zero(n_features);
    intercept_ = 0.0;

    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    double prev_loss = std::numeric_limits<double>::infinity();
    int step = 0;

    for (int epoch = 0; epoch < max_iter_; ++epoch) {
        if (shuffle_) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }

        double total_loss = 0.0;
        for (int idx : indices) {
            double pred = X.row(idx).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
            double error = pred - y(idx);
            double grad = 0.0;

            if (loss_ == "squared_loss") {
                grad = error;
                total_loss += 0.5 * error * error;
            } else if (loss_ == "huber") {
                double abs_err = std::abs(error);
                if (abs_err <= epsilon_) {
                    grad = error;
                    total_loss += 0.5 * error * error;
                } else {
                    grad = epsilon_ * (error > 0.0 ? 1.0 : -1.0);
                    total_loss += epsilon_ * (abs_err - 0.5 * epsilon_);
                }
            } else if (loss_ == "epsilon_insensitive") {
                double abs_err = std::abs(error);
                if (abs_err <= epsilon_) {
                    grad = 0.0;
                } else {
                    grad = (error > 0.0) ? 1.0 : -1.0;
                    total_loss += abs_err - epsilon_;
                }
            } else {
                throw std::invalid_argument("Unsupported loss: " + loss_);
            }

            double eta = compute_learning_rate(learning_rate_, eta0_, power_t_, step++);
            if (grad != 0.0) {
                coef_ -= eta * grad * X.row(idx).transpose();
                if (fit_intercept_) {
                    intercept_ -= eta * grad;
                }
            }

            if (penalty_ == "l2") {
                coef_ -= eta * alpha_ * coef_;
            } else if (penalty_ == "l1") {
                for (int j = 0; j < coef_.size(); ++j) {
                    coef_(j) = soft_threshold(coef_(j), eta * alpha_);
                }
            } else if (penalty_ == "elasticnet") {
                coef_ -= eta * alpha_ * (1.0 - l1_ratio_) * coef_;
                for (int j = 0; j < coef_.size(); ++j) {
                    coef_(j) = soft_threshold(coef_(j), eta * alpha_ * l1_ratio_);
                }
            } else if (penalty_ != "none") {
                throw std::invalid_argument("Unsupported penalty: " + penalty_);
            }
        }

        if (std::abs(prev_loss - total_loss) < tol_) {
            break;
        }
        prev_loss = total_loss;
    }

    fitted_ = true;
    return *this;
}

VectorXd SGDRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SGDRegressor must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params SGDRegressor::get_params() const {
    return {
        {"loss", loss_},
        {"penalty", penalty_},
        {"alpha", std::to_string(alpha_)},
        {"l1_ratio", std::to_string(l1_ratio_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"learning_rate", learning_rate_},
        {"eta0", std::to_string(eta0_)},
        {"power_t", std::to_string(power_t_)},
        {"shuffle", shuffle_ ? "true" : "false"},
        {"random_state", std::to_string(random_state_)},
        {"epsilon", std::to_string(epsilon_)}
    };
}

Estimator& SGDRegressor::set_params(const Params& params) {
    loss_ = utils::get_param_string(params, "loss", loss_);
    penalty_ = utils::get_param_string(params, "penalty", penalty_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    l1_ratio_ = utils::get_param_double(params, "l1_ratio", l1_ratio_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    learning_rate_ = utils::get_param_string(params, "learning_rate", learning_rate_);
    eta0_ = utils::get_param_double(params, "eta0", eta0_);
    power_t_ = utils::get_param_double(params, "power_t", power_t_);
    shuffle_ = utils::get_param_bool(params, "shuffle", shuffle_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    epsilon_ = utils::get_param_double(params, "epsilon", epsilon_);
    return *this;
}

bool SGDRegressor::is_fitted() const {
    return fitted_;
}

SGDClassifier::SGDClassifier(const std::string& loss, const std::string& penalty, double alpha,
                             double l1_ratio, bool fit_intercept, int max_iter, double tol,
                             const std::string& learning_rate, double eta0, double power_t,
                             bool shuffle, int random_state)
    : coef_(), intercept_(0.0), fitted_(false), loss_(loss), penalty_(penalty),
      alpha_(alpha), l1_ratio_(l1_ratio), fit_intercept_(fit_intercept),
      max_iter_(max_iter), tol_(tol), learning_rate_(learning_rate), eta0_(eta0),
      power_t_(power_t), shuffle_(shuffle), random_state_(random_state),
      classes_(), n_classes_(0) {}

Estimator& SGDClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (alpha_ < 0.0) {
        throw std::invalid_argument("alpha must be non-negative");
    }
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (eta0_ <= 0.0) {
        throw std::invalid_argument("eta0 must be positive");
    }

    classes_ = unique_classes(y);
    n_classes_ = static_cast<int>(classes_.size());
    if (n_classes_ != 2) {
        throw std::invalid_argument("SGDClassifier currently supports binary classification only");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    coef_ = VectorXd::Zero(n_features);
    intercept_ = 0.0;

    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    double prev_loss = std::numeric_limits<double>::infinity();
    int step = 0;

    for (int epoch = 0; epoch < max_iter_; ++epoch) {
        if (shuffle_) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        double total_loss = 0.0;

        for (int idx : indices) {
            double score = X.row(idx).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
            double eta = compute_learning_rate(learning_rate_, eta0_, power_t_, step++);

            double grad_scale = 0.0;
            if (loss_ == "hinge") {
                double y_sign = (static_cast<int>(y(idx)) == classes_[1]) ? 1.0 : -1.0;
                double margin = y_sign * score;
                if (margin < 1.0) {
                    grad_scale = -y_sign;
                    total_loss += 1.0 - margin;
                }
            } else if (loss_ == "log") {
                double y_bin = (static_cast<int>(y(idx)) == classes_[1]) ? 1.0 : 0.0;
                double pred = sigmoid(score);
                grad_scale = (pred - y_bin);
                total_loss += -(y_bin * std::log(std::max(pred, 1e-12)) +
                                (1.0 - y_bin) * std::log(std::max(1.0 - pred, 1e-12)));
            } else if (loss_ == "perceptron") {
                double y_sign = (static_cast<int>(y(idx)) == classes_[1]) ? 1.0 : -1.0;
                double margin = y_sign * score;
                if (margin <= 0.0) {
                    grad_scale = -y_sign;
                    total_loss += -margin;
                }
            } else {
                throw std::invalid_argument("Unsupported loss: " + loss_);
            }

            if (grad_scale != 0.0) {
                coef_ -= eta * grad_scale * X.row(idx).transpose();
                if (fit_intercept_) {
                    intercept_ -= eta * grad_scale;
                }
            }

            if (penalty_ == "l2") {
                coef_ -= eta * alpha_ * coef_;
            } else if (penalty_ == "l1") {
                for (int j = 0; j < coef_.size(); ++j) {
                    coef_(j) = soft_threshold(coef_(j), eta * alpha_);
                }
            } else if (penalty_ == "elasticnet") {
                coef_ -= eta * alpha_ * (1.0 - l1_ratio_) * coef_;
                for (int j = 0; j < coef_.size(); ++j) {
                    coef_(j) = soft_threshold(coef_(j), eta * alpha_ * l1_ratio_);
                }
            } else if (penalty_ != "none") {
                throw std::invalid_argument("Unsupported penalty: " + penalty_);
            }
        }

        if (std::abs(prev_loss - total_loss) < tol_) {
            break;
        }
        prev_loss = total_loss;
    }

    fitted_ = true;
    return *this;
}

VectorXi SGDClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SGDClassifier must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXi preds(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        preds(i) = (score >= 0.0) ? classes_[1] : classes_[0];
    }
    return preds;
}

MatrixXd SGDClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SGDClassifier must be fitted before predict_proba");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    MatrixXd proba(X.rows(), 2);
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        double p1 = sigmoid(score);
        proba(i, 1) = p1;
        proba(i, 0) = 1.0 - p1;
    }
    return proba;
}

VectorXd SGDClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SGDClassifier must be fitted before decision_function");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd scores(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        scores(i) = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
    }
    return scores;
}

Params SGDClassifier::get_params() const {
    return {
        {"loss", loss_},
        {"penalty", penalty_},
        {"alpha", std::to_string(alpha_)},
        {"l1_ratio", std::to_string(l1_ratio_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"learning_rate", learning_rate_},
        {"eta0", std::to_string(eta0_)},
        {"power_t", std::to_string(power_t_)},
        {"shuffle", shuffle_ ? "true" : "false"},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& SGDClassifier::set_params(const Params& params) {
    loss_ = utils::get_param_string(params, "loss", loss_);
    penalty_ = utils::get_param_string(params, "penalty", penalty_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    l1_ratio_ = utils::get_param_double(params, "l1_ratio", l1_ratio_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    learning_rate_ = utils::get_param_string(params, "learning_rate", learning_rate_);
    eta0_ = utils::get_param_double(params, "eta0", eta0_);
    power_t_ = utils::get_param_double(params, "power_t", power_t_);
    shuffle_ = utils::get_param_bool(params, "shuffle", shuffle_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

bool SGDClassifier::is_fitted() const {
    return fitted_;
}

PassiveAggressiveRegressor::PassiveAggressiveRegressor(double C, double epsilon, bool fit_intercept,
                                                       int max_iter, bool shuffle, int random_state,
                                                       const std::string& loss)
    : coef_(), intercept_(0.0), fitted_(false), C_(C), epsilon_(epsilon),
      fit_intercept_(fit_intercept), max_iter_(max_iter), shuffle_(shuffle),
      random_state_(random_state), loss_(loss) {}

Estimator& PassiveAggressiveRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (C_ <= 0.0) {
        throw std::invalid_argument("C must be positive");
    }
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    coef_ = VectorXd::Zero(n_features);
    intercept_ = 0.0;

    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < max_iter_; ++epoch) {
        if (shuffle_) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        for (int idx : indices) {
            double pred = X.row(idx).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
            double diff = y(idx) - pred;
            double loss = std::max(0.0, std::abs(diff) - epsilon_);
            if (loss <= 0.0) {
                continue;
            }
            double norm_sq = X.row(idx).squaredNorm();
            double tau = 0.0;
            if (loss_ == "epsilon_insensitive") {
                tau = std::min(C_, loss / (norm_sq + 1e-12));
            } else if (loss_ == "squared_epsilon_insensitive") {
                tau = loss / (norm_sq + 1.0 / (2.0 * C_));
            } else {
                throw std::invalid_argument("Unsupported loss: " + loss_);
            }
            double update = tau * (diff > 0.0 ? 1.0 : -1.0);
            coef_ += update * X.row(idx).transpose();
            if (fit_intercept_) {
                intercept_ += update;
            }
        }
    }

    fitted_ = true;
    return *this;
}

VectorXd PassiveAggressiveRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PassiveAggressiveRegressor must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params PassiveAggressiveRegressor::get_params() const {
    return {
        {"C", std::to_string(C_)},
        {"epsilon", std::to_string(epsilon_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"shuffle", shuffle_ ? "true" : "false"},
        {"random_state", std::to_string(random_state_)},
        {"loss", loss_}
    };
}

Estimator& PassiveAggressiveRegressor::set_params(const Params& params) {
    C_ = utils::get_param_double(params, "C", C_);
    epsilon_ = utils::get_param_double(params, "epsilon", epsilon_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    shuffle_ = utils::get_param_bool(params, "shuffle", shuffle_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    loss_ = utils::get_param_string(params, "loss", loss_);
    return *this;
}

bool PassiveAggressiveRegressor::is_fitted() const {
    return fitted_;
}

PassiveAggressiveClassifier::PassiveAggressiveClassifier(double C, bool fit_intercept,
                                                         int max_iter, bool shuffle, int random_state)
    : coef_(), intercept_(0.0), fitted_(false), C_(C), fit_intercept_(fit_intercept),
      max_iter_(max_iter), shuffle_(shuffle), random_state_(random_state),
      classes_(), n_classes_(0) {}

Estimator& PassiveAggressiveClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (C_ <= 0.0) {
        throw std::invalid_argument("C must be positive");
    }
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }

    classes_ = unique_classes(y);
    n_classes_ = static_cast<int>(classes_.size());
    if (n_classes_ != 2) {
        throw std::invalid_argument("PassiveAggressiveClassifier supports binary classification only");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    coef_ = VectorXd::Zero(n_features);
    intercept_ = 0.0;

    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < max_iter_; ++epoch) {
        if (shuffle_) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        for (int idx : indices) {
            double y_sign = (static_cast<int>(y(idx)) == classes_[1]) ? 1.0 : -1.0;
            double score = X.row(idx).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
            double loss = std::max(0.0, 1.0 - y_sign * score);
            if (loss <= 0.0) {
                continue;
            }
            double norm_sq = X.row(idx).squaredNorm();
            double tau = std::min(C_, loss / (norm_sq + 1e-12));
            coef_ += tau * y_sign * X.row(idx).transpose();
            if (fit_intercept_) {
                intercept_ += tau * y_sign;
            }
        }
    }

    fitted_ = true;
    return *this;
}

VectorXi PassiveAggressiveClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PassiveAggressiveClassifier must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXi preds(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        preds(i) = (score >= 0.0) ? classes_[1] : classes_[0];
    }
    return preds;
}

MatrixXd PassiveAggressiveClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PassiveAggressiveClassifier must be fitted before predict_proba");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    MatrixXd proba(X.rows(), 2);
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        double p1 = sigmoid(score);
        proba(i, 1) = p1;
        proba(i, 0) = 1.0 - p1;
    }
    return proba;
}

VectorXd PassiveAggressiveClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PassiveAggressiveClassifier must be fitted before decision_function");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd scores(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        scores(i) = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
    }
    return scores;
}

Params PassiveAggressiveClassifier::get_params() const {
    return {
        {"C", std::to_string(C_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"shuffle", shuffle_ ? "true" : "false"},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& PassiveAggressiveClassifier::set_params(const Params& params) {
    C_ = utils::get_param_double(params, "C", C_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    shuffle_ = utils::get_param_bool(params, "shuffle", shuffle_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

bool PassiveAggressiveClassifier::is_fitted() const {
    return fitted_;
}

Perceptron::Perceptron(bool fit_intercept, int max_iter, double tol,
                       bool shuffle, int random_state)
    : coef_(), intercept_(0.0), fitted_(false), fit_intercept_(fit_intercept),
      max_iter_(max_iter), tol_(tol), shuffle_(shuffle),
      random_state_(random_state), classes_(), n_classes_(0) {}

Estimator& Perceptron::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }

    classes_ = unique_classes(y);
    n_classes_ = static_cast<int>(classes_.size());
    if (n_classes_ != 2) {
        throw std::invalid_argument("Perceptron supports binary classification only");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    coef_ = VectorXd::Zero(n_features);
    intercept_ = 0.0;

    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < max_iter_; ++epoch) {
        if (shuffle_) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        int errors = 0;
        for (int idx : indices) {
            double y_sign = (static_cast<int>(y(idx)) == classes_[1]) ? 1.0 : -1.0;
            double score = X.row(idx).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
            if (y_sign * score <= 0.0) {
                coef_ += y_sign * X.row(idx).transpose();
                if (fit_intercept_) {
                    intercept_ += y_sign;
                }
                ++errors;
            }
        }
        if (errors == 0 || errors < tol_) {
            break;
        }
    }

    fitted_ = true;
    return *this;
}

VectorXi Perceptron::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Perceptron must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXi preds(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        preds(i) = (score >= 0.0) ? classes_[1] : classes_[0];
    }
    return preds;
}

MatrixXd Perceptron::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Perceptron must be fitted before predict_proba");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    MatrixXd proba(X.rows(), 2);
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        double p1 = sigmoid(score);
        proba(i, 1) = p1;
        proba(i, 0) = 1.0 - p1;
    }
    return proba;
}

VectorXd Perceptron::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Perceptron must be fitted before decision_function");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd scores(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        scores(i) = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
    }
    return scores;
}

Params Perceptron::get_params() const {
    return {
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"shuffle", shuffle_ ? "true" : "false"},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& Perceptron::set_params(const Params& params) {
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    shuffle_ = utils::get_param_bool(params, "shuffle", shuffle_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

bool Perceptron::is_fitted() const {
    return fitted_;
}

LogisticRegressionCV::LogisticRegressionCV(const std::vector<double>& Cs, int cv_folds,
                                           const std::string& scoring, bool fit_intercept,
                                           int max_iter, double tol, int random_state)
    : coef_(), intercept_(0.0), fitted_(false), Cs_(Cs), cv_folds_(cv_folds),
      scoring_(scoring), fit_intercept_(fit_intercept), max_iter_(max_iter),
      tol_(tol), random_state_(random_state), best_C_(0.0), classes_(), n_classes_(0) {}

Estimator& LogisticRegressionCV::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (Cs_.empty()) {
        throw std::invalid_argument("Cs must not be empty");
    }
    if (cv_folds_ <= 1) {
        throw std::invalid_argument("cv_folds must be at least 2");
    }

    classes_ = unique_classes(y);
    n_classes_ = static_cast<int>(classes_.size());
    if (n_classes_ != 2) {
        throw std::invalid_argument("LogisticRegressionCV supports binary classification only");
    }

    model_selection::KFold kfold(cv_folds_, true, random_state_ == -1 ? 42 : random_state_);
    auto splits = kfold.split(X, y);

    double best_score = -std::numeric_limits<double>::infinity();
    double best_C = Cs_.front();

    for (double C : Cs_) {
        if (C <= 0.0) {
            continue;
        }
        double score_sum = 0.0;
        for (const auto& split : splits) {
            const auto& train_idx = split.first;
            const auto& test_idx = split.second;

            MatrixXd X_train(train_idx.size(), X.cols());
            VectorXd y_train(train_idx.size());
            MatrixXd X_test(test_idx.size(), X.cols());
            VectorXd y_test(test_idx.size());

            for (size_t i = 0; i < train_idx.size(); ++i) {
                X_train.row(i) = X.row(train_idx[i]);
                y_train(i) = y(train_idx[i]);
            }
            for (size_t i = 0; i < test_idx.size(); ++i) {
                X_test.row(i) = X.row(test_idx[i]);
                y_test(i) = y(test_idx[i]);
            }

            LogisticRegression lr(C, fit_intercept_, max_iter_, tol_, random_state_);
            lr.fit(X_train, y_train);
            VectorXi preds = lr.predict_classes(X_test);
            VectorXi y_true = y_test.cast<int>();
            score_sum += metrics::accuracy_score(y_true, preds);
        }
        double score = score_sum / static_cast<double>(splits.size());
        if (score > best_score) {
            best_score = score;
            best_C = C;
        }
    }

    best_C_ = best_C;
    LogisticRegression final_lr(best_C_, fit_intercept_, max_iter_, tol_, random_state_);
    final_lr.fit(X, y);
    coef_ = final_lr.coef();
    intercept_ = final_lr.intercept();
    fitted_ = true;
    return *this;
}

VectorXi LogisticRegressionCV::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LogisticRegressionCV must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXi preds(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        preds(i) = (score >= 0.0) ? classes_[1] : classes_[0];
    }
    return preds;
}

MatrixXd LogisticRegressionCV::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LogisticRegressionCV must be fitted before predict_proba");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    MatrixXd proba(X.rows(), 2);
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        double p1 = sigmoid(score);
        proba(i, 1) = p1;
        proba(i, 0) = 1.0 - p1;
    }
    return proba;
}

VectorXd LogisticRegressionCV::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LogisticRegressionCV must be fitted before decision_function");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd scores(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        scores(i) = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
    }
    return scores;
}

Params LogisticRegressionCV::get_params() const {
    return {
        {"Cs", join_double_list(Cs_)},
        {"cv", std::to_string(cv_folds_)},
        {"scoring", scoring_},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& LogisticRegressionCV::set_params(const Params& params) {
    Cs_ = parse_double_list(utils::get_param_string(params, "Cs", join_double_list(Cs_)), Cs_);
    cv_folds_ = utils::get_param_int(params, "cv", cv_folds_);
    scoring_ = utils::get_param_string(params, "scoring", scoring_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

bool LogisticRegressionCV::is_fitted() const {
    return fitted_;
}

RidgeClassifier::RidgeClassifier(double alpha, bool fit_intercept)
    : coef_(), intercept_(0.0), fitted_(false), alpha_(alpha),
      fit_intercept_(fit_intercept), classes_(), n_classes_(0) {}

Estimator& RidgeClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (alpha_ <= 0.0) {
        throw std::invalid_argument("alpha must be positive");
    }

    classes_ = unique_classes(y);
    n_classes_ = static_cast<int>(classes_.size());
    if (n_classes_ != 2) {
        throw std::invalid_argument("RidgeClassifier supports binary classification only");
    }

    VectorXd y_signed = VectorXd::Zero(y.size());
    for (int i = 0; i < y.size(); ++i) {
        y_signed(i) = (static_cast<int>(y(i)) == classes_[1]) ? 1.0 : -1.0;
    }
    fit_ridge_closed_form(X, y_signed, alpha_, fit_intercept_, coef_, intercept_);

    fitted_ = true;
    return *this;
}

VectorXi RidgeClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RidgeClassifier must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXi preds(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        preds(i) = (score >= 0.0) ? classes_[1] : classes_[0];
    }
    return preds;
}

MatrixXd RidgeClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RidgeClassifier must be fitted before predict_proba");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    MatrixXd proba(X.rows(), 2);
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        double p1 = sigmoid(score);
        proba(i, 1) = p1;
        proba(i, 0) = 1.0 - p1;
    }
    return proba;
}

VectorXd RidgeClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RidgeClassifier must be fitted before decision_function");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd scores(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        scores(i) = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
    }
    return scores;
}

Params RidgeClassifier::get_params() const {
    return {
        {"alpha", std::to_string(alpha_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"}
    };
}

Estimator& RidgeClassifier::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    return *this;
}

bool RidgeClassifier::is_fitted() const {
    return fitted_;
}

RidgeClassifierCV::RidgeClassifierCV(const std::vector<double>& alphas, int cv_folds,
                                     const std::string& scoring, bool fit_intercept)
    : coef_(), intercept_(0.0), fitted_(false), alphas_(alphas), best_alpha_(0.0),
      cv_folds_(cv_folds), scoring_(scoring), fit_intercept_(fit_intercept),
      classes_(), n_classes_(0) {}

Estimator& RidgeClassifierCV::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (alphas_.empty()) {
        throw std::invalid_argument("alphas must not be empty");
    }
    if (cv_folds_ <= 1) {
        throw std::invalid_argument("cv_folds must be at least 2");
    }

    classes_ = unique_classes(y);
    n_classes_ = static_cast<int>(classes_.size());
    if (n_classes_ != 2) {
        throw std::invalid_argument("RidgeClassifierCV supports binary classification only");
    }

    model_selection::KFold kfold(cv_folds_, true, 42);
    auto splits = kfold.split(X, y);

    double best_score = -std::numeric_limits<double>::infinity();
    double best_alpha = alphas_.front();

    for (double alpha : alphas_) {
        if (alpha <= 0.0) {
            continue;
        }
        double score_sum = 0.0;
        for (const auto& split : splits) {
            const auto& train_idx = split.first;
            const auto& test_idx = split.second;

            MatrixXd X_train(train_idx.size(), X.cols());
            VectorXd y_train(train_idx.size());
            MatrixXd X_test(test_idx.size(), X.cols());
            VectorXd y_test(test_idx.size());

            for (size_t i = 0; i < train_idx.size(); ++i) {
                X_train.row(i) = X.row(train_idx[i]);
                y_train(i) = y(train_idx[i]);
            }
            for (size_t i = 0; i < test_idx.size(); ++i) {
                X_test.row(i) = X.row(test_idx[i]);
                y_test(i) = y(test_idx[i]);
            }

            RidgeClassifier rc(alpha, fit_intercept_);
            rc.fit(X_train, y_train);
            VectorXi preds = rc.predict_classes(X_test);
            VectorXi y_true = y_test.cast<int>();
            score_sum += metrics::accuracy_score(y_true, preds);
        }
        double score = score_sum / static_cast<double>(splits.size());
        if (score > best_score) {
            best_score = score;
            best_alpha = alpha;
        }
    }

    best_alpha_ = best_alpha;
    RidgeClassifier final_rc(best_alpha_, fit_intercept_);
    final_rc.fit(X, y);
    coef_ = final_rc.coef();
    intercept_ = final_rc.intercept();
    fitted_ = true;
    return *this;
}

VectorXi RidgeClassifierCV::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RidgeClassifierCV must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXi preds(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        preds(i) = (score >= 0.0) ? classes_[1] : classes_[0];
    }
    return preds;
}

MatrixXd RidgeClassifierCV::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RidgeClassifierCV must be fitted before predict_proba");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    MatrixXd proba(X.rows(), 2);
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
        double p1 = sigmoid(score);
        proba(i, 1) = p1;
        proba(i, 0) = 1.0 - p1;
    }
    return proba;
}

VectorXd RidgeClassifierCV::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RidgeClassifierCV must be fitted before decision_function");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd scores(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        scores(i) = X.row(i).dot(coef_) + (fit_intercept_ ? intercept_ : 0.0);
    }
    return scores;
}

Params RidgeClassifierCV::get_params() const {
    return {
        {"alphas", join_double_list(alphas_)},
        {"cv", std::to_string(cv_folds_)},
        {"scoring", scoring_},
        {"fit_intercept", fit_intercept_ ? "true" : "false"}
    };
}

Estimator& RidgeClassifierCV::set_params(const Params& params) {
    alphas_ = parse_double_list(utils::get_param_string(params, "alphas", join_double_list(alphas_)), alphas_);
    cv_folds_ = utils::get_param_int(params, "cv", cv_folds_);
    scoring_ = utils::get_param_string(params, "scoring", scoring_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    return *this;
}

bool RidgeClassifierCV::is_fitted() const {
    return fitted_;
}

QuantileRegressor::QuantileRegressor(double quantile, double alpha, bool fit_intercept,
                                     int max_iter, double tol, double learning_rate)
    : coef_(), intercept_(0.0), fitted_(false), quantile_(quantile), alpha_(alpha),
      fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol),
      learning_rate_(learning_rate) {}

Estimator& QuantileRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (quantile_ <= 0.0 || quantile_ >= 1.0) {
        throw std::invalid_argument("quantile must be between 0 and 1");
    }
    if (learning_rate_ <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    coef_ = VectorXd::Zero(n_features);
    intercept_ = 0.0;

    for (int iter = 0; iter < max_iter_; ++iter) {
        VectorXd preds = X * coef_;
        preds.array() += (fit_intercept_ ? intercept_ : 0.0);
        VectorXd residual = y - preds;

        VectorXd grad = VectorXd::Zero(n_features);
        double grad_intercept = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            double g = 0.0;
            if (residual(i) > 0.0) {
                g = -quantile_;
            } else if (residual(i) < 0.0) {
                g = (1.0 - quantile_);
            }
            grad += g * X.row(i).transpose();
            grad_intercept += g;
        }
        grad /= static_cast<double>(n_samples);
        grad_intercept /= static_cast<double>(n_samples);
        grad += alpha_ * coef_;

        coef_ -= learning_rate_ * grad;
        if (fit_intercept_) {
            intercept_ -= learning_rate_ * grad_intercept;
        }

        if (grad.norm() < tol_) {
            break;
        }
    }

    fitted_ = true;
    return *this;
}

VectorXd QuantileRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("QuantileRegressor must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params QuantileRegressor::get_params() const {
    return {
        {"quantile", std::to_string(quantile_)},
        {"alpha", std::to_string(alpha_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"learning_rate", std::to_string(learning_rate_)}
    };
}

Estimator& QuantileRegressor::set_params(const Params& params) {
    quantile_ = utils::get_param_double(params, "quantile", quantile_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    learning_rate_ = utils::get_param_double(params, "learning_rate", learning_rate_);
    return *this;
}

bool QuantileRegressor::is_fitted() const {
    return fitted_;
}

PoissonRegressor::PoissonRegressor(double alpha, bool fit_intercept, int max_iter,
                                   double tol, double learning_rate)
    : coef_(), intercept_(0.0), fitted_(false), alpha_(alpha),
      fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol),
      learning_rate_(learning_rate) {}

Estimator& PoissonRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if ((y.array() < 0.0).any()) {
        throw std::invalid_argument("y must be non-negative for PoissonRegressor");
    }
    if (learning_rate_ <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    coef_ = VectorXd::Zero(n_features);
    intercept_ = 0.0;

    for (int iter = 0; iter < max_iter_; ++iter) {
        VectorXd eta = X * coef_;
        eta.array() += (fit_intercept_ ? intercept_ : 0.0);
        VectorXd mu = eta.array().exp();
        VectorXd grad = X.transpose() * (mu - y) / static_cast<double>(n_samples);
        grad += alpha_ * coef_;
        double grad_intercept = (mu - y).mean();

        coef_ -= learning_rate_ * grad;
        if (fit_intercept_) {
            intercept_ -= learning_rate_ * grad_intercept;
        }

        if (grad.norm() < tol_) {
            break;
        }
    }

    fitted_ = true;
    return *this;
}

VectorXd PoissonRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PoissonRegressor must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd eta = X * coef_;
    eta.array() += intercept_;
    return eta.array().exp();
}

Params PoissonRegressor::get_params() const {
    return {
        {"alpha", std::to_string(alpha_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"learning_rate", std::to_string(learning_rate_)}
    };
}

Estimator& PoissonRegressor::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    learning_rate_ = utils::get_param_double(params, "learning_rate", learning_rate_);
    return *this;
}

bool PoissonRegressor::is_fitted() const {
    return fitted_;
}

GammaRegressor::GammaRegressor(double alpha, bool fit_intercept, int max_iter,
                               double tol, double learning_rate)
    : coef_(), intercept_(0.0), fitted_(false), alpha_(alpha),
      fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol),
      learning_rate_(learning_rate) {}

Estimator& GammaRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if ((y.array() <= 0.0).any()) {
        throw std::invalid_argument("y must be positive for GammaRegressor");
    }
    if (learning_rate_ <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    coef_ = VectorXd::Zero(n_features);
    intercept_ = 0.0;

    for (int iter = 0; iter < max_iter_; ++iter) {
        VectorXd eta = X * coef_;
        eta.array() += (fit_intercept_ ? intercept_ : 0.0);
        VectorXd mu = eta.array().exp();
        VectorXd grad_vec = VectorXd::Ones(n_samples) - y.cwiseQuotient(mu);
        VectorXd grad = X.transpose() * grad_vec / static_cast<double>(n_samples);
        grad += alpha_ * coef_;
        double grad_intercept = grad_vec.mean();

        coef_ -= learning_rate_ * grad;
        if (fit_intercept_) {
            intercept_ -= learning_rate_ * grad_intercept;
        }

        if (grad.norm() < tol_) {
            break;
        }
    }

    fitted_ = true;
    return *this;
}

VectorXd GammaRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("GammaRegressor must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd eta = X * coef_;
    eta.array() += intercept_;
    return eta.array().exp();
}

Params GammaRegressor::get_params() const {
    return {
        {"alpha", std::to_string(alpha_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"learning_rate", std::to_string(learning_rate_)}
    };
}

Estimator& GammaRegressor::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    learning_rate_ = utils::get_param_double(params, "learning_rate", learning_rate_);
    return *this;
}

bool GammaRegressor::is_fitted() const {
    return fitted_;
}

TweedieRegressor::TweedieRegressor(double power, double alpha, bool fit_intercept,
                                   int max_iter, double tol, double learning_rate)
    : coef_(), intercept_(0.0), fitted_(false), power_(power), alpha_(alpha),
      fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol),
      learning_rate_(learning_rate) {}

Estimator& TweedieRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (power_ < 0.0 || power_ > 2.0) {
        throw std::invalid_argument("power must be between 0 and 2");
    }
    if ((y.array() < 0.0).any()) {
        throw std::invalid_argument("y must be non-negative for TweedieRegressor");
    }
    if (learning_rate_ <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }

    int n_samples = X.rows();
    int n_features = X.cols();
    coef_ = VectorXd::Zero(n_features);
    intercept_ = 0.0;

    for (int iter = 0; iter < max_iter_; ++iter) {
        VectorXd eta = X * coef_;
        eta.array() += (fit_intercept_ ? intercept_ : 0.0);
        VectorXd mu = eta.array().exp();

        VectorXd grad_vec = mu.array().pow(2.0 - power_) - y.array() * mu.array().pow(1.0 - power_);
        VectorXd grad = X.transpose() * grad_vec / static_cast<double>(n_samples);
        grad += alpha_ * coef_;
        double grad_intercept = grad_vec.mean();

        coef_ -= learning_rate_ * grad;
        if (fit_intercept_) {
            intercept_ -= learning_rate_ * grad_intercept;
        }

        if (grad.norm() < tol_) {
            break;
        }
    }

    fitted_ = true;
    return *this;
}

VectorXd TweedieRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("TweedieRegressor must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd eta = X * coef_;
    eta.array() += intercept_;
    return eta.array().exp();
}

Params TweedieRegressor::get_params() const {
    return {
        {"power", std::to_string(power_)},
        {"alpha", std::to_string(alpha_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"learning_rate", std::to_string(learning_rate_)}
    };
}

Estimator& TweedieRegressor::set_params(const Params& params) {
    power_ = utils::get_param_double(params, "power", power_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    learning_rate_ = utils::get_param_double(params, "learning_rate", learning_rate_);
    return *this;
}

bool TweedieRegressor::is_fitted() const {
    return fitted_;
}

MultiTaskLasso::MultiTaskLasso(double alpha, bool fit_intercept, int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), alpha_(alpha),
      fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

Estimator& MultiTaskLasso::fit(const MatrixXd& X, const VectorXd& y) {
    if (alpha_ <= 0.0) {
        throw std::invalid_argument("alpha must be positive");
    }
    validation::check_X_y(X, y);
    fit_lasso_coordinate_descent(X, y, alpha_, 1.0, fit_intercept_, max_iter_, tol_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd MultiTaskLasso::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultiTaskLasso must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params MultiTaskLasso::get_params() const {
    return {
        {"alpha", std::to_string(alpha_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& MultiTaskLasso::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool MultiTaskLasso::is_fitted() const {
    return fitted_;
}

MultiTaskLassoCV::MultiTaskLassoCV(const std::vector<double>& alphas, int cv_folds,
                                   bool fit_intercept, int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), alphas_(alphas), best_alpha_(0.0),
      cv_folds_(cv_folds), fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

Estimator& MultiTaskLassoCV::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (alphas_.empty()) {
        throw std::invalid_argument("alphas must not be empty");
    }
    if (cv_folds_ <= 1) {
        throw std::invalid_argument("cv_folds must be at least 2");
    }

    model_selection::KFold kfold(cv_folds_, true, 42);
    auto splits = kfold.split(X, y);

    double best_mse = std::numeric_limits<double>::infinity();
    double best_alpha = alphas_.front();

    for (double alpha : alphas_) {
        if (alpha <= 0.0) {
            continue;
        }
        double mse_sum = 0.0;
        for (const auto& split : splits) {
            const auto& train_idx = split.first;
            const auto& test_idx = split.second;

            MatrixXd X_train(train_idx.size(), X.cols());
            VectorXd y_train(train_idx.size());
            MatrixXd X_test(test_idx.size(), X.cols());
            VectorXd y_test(test_idx.size());

            for (size_t i = 0; i < train_idx.size(); ++i) {
                X_train.row(i) = X.row(train_idx[i]);
                y_train(i) = y(train_idx[i]);
            }
            for (size_t i = 0; i < test_idx.size(); ++i) {
                X_test.row(i) = X.row(test_idx[i]);
                y_test(i) = y(test_idx[i]);
            }

            VectorXd coef_fold;
            double intercept_fold = 0.0;
            fit_lasso_coordinate_descent(X_train, y_train, alpha, 1.0, fit_intercept_,
                                         max_iter_, tol_, coef_fold, intercept_fold);
            VectorXd preds = X_test * coef_fold;
            preds.array() += intercept_fold;
            mse_sum += metrics::mean_squared_error(y_test, preds);
        }
        double mse = mse_sum / static_cast<double>(splits.size());
        if (mse < best_mse) {
            best_mse = mse;
            best_alpha = alpha;
        }
    }

    best_alpha_ = best_alpha;
    fit_lasso_coordinate_descent(X, y, best_alpha_, 1.0, fit_intercept_, max_iter_, tol_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd MultiTaskLassoCV::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultiTaskLassoCV must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params MultiTaskLassoCV::get_params() const {
    return {
        {"alphas", join_double_list(alphas_)},
        {"cv", std::to_string(cv_folds_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& MultiTaskLassoCV::set_params(const Params& params) {
    alphas_ = parse_double_list(utils::get_param_string(params, "alphas", join_double_list(alphas_)), alphas_);
    cv_folds_ = utils::get_param_int(params, "cv", cv_folds_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool MultiTaskLassoCV::is_fitted() const {
    return fitted_;
}

MultiTaskElasticNet::MultiTaskElasticNet(double alpha, double l1_ratio, bool fit_intercept,
                                         int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), alpha_(alpha), l1_ratio_(l1_ratio),
      fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

Estimator& MultiTaskElasticNet::fit(const MatrixXd& X, const VectorXd& y) {
    if (alpha_ <= 0.0) {
        throw std::invalid_argument("alpha must be positive");
    }
    if (l1_ratio_ < 0.0 || l1_ratio_ > 1.0) {
        throw std::invalid_argument("l1_ratio must be between 0 and 1");
    }
    validation::check_X_y(X, y);
    fit_lasso_coordinate_descent(X, y, alpha_, l1_ratio_, fit_intercept_, max_iter_, tol_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd MultiTaskElasticNet::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultiTaskElasticNet must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params MultiTaskElasticNet::get_params() const {
    return {
        {"alpha", std::to_string(alpha_)},
        {"l1_ratio", std::to_string(l1_ratio_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& MultiTaskElasticNet::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    l1_ratio_ = utils::get_param_double(params, "l1_ratio", l1_ratio_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool MultiTaskElasticNet::is_fitted() const {
    return fitted_;
}

MultiTaskElasticNetCV::MultiTaskElasticNetCV(const std::vector<double>& alphas,
                                             const std::vector<double>& l1_ratios,
                                             int cv_folds, bool fit_intercept,
                                             int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), alphas_(alphas), l1_ratios_(l1_ratios),
      best_alpha_(0.0), best_l1_ratio_(0.0), cv_folds_(cv_folds),
      fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

Estimator& MultiTaskElasticNetCV::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (alphas_.empty() || l1_ratios_.empty()) {
        throw std::invalid_argument("alphas and l1_ratios must not be empty");
    }
    if (cv_folds_ <= 1) {
        throw std::invalid_argument("cv_folds must be at least 2");
    }

    model_selection::KFold kfold(cv_folds_, true, 42);
    auto splits = kfold.split(X, y);

    double best_mse = std::numeric_limits<double>::infinity();
    double best_alpha = alphas_.front();
    double best_l1_ratio = l1_ratios_.front();

    for (double alpha : alphas_) {
        if (alpha <= 0.0) {
            continue;
        }
        for (double l1_ratio : l1_ratios_) {
            if (l1_ratio < 0.0 || l1_ratio > 1.0) {
                continue;
            }
            double mse_sum = 0.0;
            for (const auto& split : splits) {
                const auto& train_idx = split.first;
                const auto& test_idx = split.second;

                MatrixXd X_train(train_idx.size(), X.cols());
                VectorXd y_train(train_idx.size());
                MatrixXd X_test(test_idx.size(), X.cols());
                VectorXd y_test(test_idx.size());

                for (size_t i = 0; i < train_idx.size(); ++i) {
                    X_train.row(i) = X.row(train_idx[i]);
                    y_train(i) = y(train_idx[i]);
                }
                for (size_t i = 0; i < test_idx.size(); ++i) {
                    X_test.row(i) = X.row(test_idx[i]);
                    y_test(i) = y(test_idx[i]);
                }

                VectorXd coef_fold;
                double intercept_fold = 0.0;
                fit_lasso_coordinate_descent(X_train, y_train, alpha, l1_ratio, fit_intercept_,
                                             max_iter_, tol_, coef_fold, intercept_fold);
                VectorXd preds = X_test * coef_fold;
                preds.array() += intercept_fold;
                mse_sum += metrics::mean_squared_error(y_test, preds);
            }
            double mse = mse_sum / static_cast<double>(splits.size());
            if (mse < best_mse) {
                best_mse = mse;
                best_alpha = alpha;
                best_l1_ratio = l1_ratio;
            }
        }
    }

    best_alpha_ = best_alpha;
    best_l1_ratio_ = best_l1_ratio;
    fit_lasso_coordinate_descent(X, y, best_alpha_, best_l1_ratio_, fit_intercept_,
                                 max_iter_, tol_, coef_, intercept_);
    fitted_ = true;
    return *this;
}

VectorXd MultiTaskElasticNetCV::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultiTaskElasticNetCV must be fitted before predict");
    }
    if (X.cols() != coef_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    VectorXd preds = X * coef_;
    preds.array() += intercept_;
    return preds;
}

Params MultiTaskElasticNetCV::get_params() const {
    return {
        {"alphas", join_double_list(alphas_)},
        {"l1_ratios", join_double_list(l1_ratios_)},
        {"cv", std::to_string(cv_folds_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& MultiTaskElasticNetCV::set_params(const Params& params) {
    alphas_ = parse_double_list(utils::get_param_string(params, "alphas", join_double_list(alphas_)), alphas_);
    l1_ratios_ = parse_double_list(utils::get_param_string(params, "l1_ratios", join_double_list(l1_ratios_)), l1_ratios_);
    cv_folds_ = utils::get_param_int(params, "cv", cv_folds_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool MultiTaskElasticNetCV::is_fitted() const {
    return fitted_;
}

} // namespace linear_model
} // namespace auroraml

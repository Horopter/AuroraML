#include "auroraml/linear_model.hpp"
#include "auroraml/base.hpp"
#include <Eigen/Dense>
#include <fstream>

namespace auroraml {
namespace linear_model {

// LinearRegression implementation
LinearRegression::LinearRegression(bool fit_intercept, bool copy_X, int n_jobs)
    : coef_(), intercept_(0.0), fitted_(false), fit_intercept_(fit_intercept), 
      copy_X_(copy_X), n_jobs_(n_jobs) {}

Estimator& LinearRegression::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    MatrixXd X_work = copy_X_ ? X : X;
    VectorXd y_work = y;
    
    if (fit_intercept_) {
        // Add intercept column
        MatrixXd X_with_intercept(X.rows(), X.cols() + 1);
        X_with_intercept.col(0) = VectorXd::Ones(X.rows());
        X_with_intercept.rightCols(X.cols()) = X_work;
        
        // Solve normal equations: (X'X)^-1 X'y
        MatrixXd XtX = X_with_intercept.transpose() * X_with_intercept;
        VectorXd Xty = X_with_intercept.transpose() * y_work;
        
        VectorXd solution = XtX.ldlt().solve(Xty);
        
        intercept_ = solution(0);
        coef_ = solution.tail(X.cols());
    } else {
        // No intercept
        MatrixXd XtX = X_work.transpose() * X_work;
        VectorXd Xty = X_work.transpose() * y_work;
        
        coef_ = XtX.ldlt().solve(Xty);
        intercept_ = 0.0;
    }
    
    fitted_ = true;
    return *this;
}

VectorXd LinearRegression::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LinearRegression must be fitted before predict");
    }
    
    if (X.cols() != coef_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXd predictions = X * coef_;
    if (fit_intercept_) {
        predictions.array() += intercept_;
    }
    
    return predictions;
}

Params LinearRegression::get_params() const {
    return {
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"copy_X", copy_X_ ? "true" : "false"},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& LinearRegression::set_params(const Params& params) {
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    copy_X_ = utils::get_param_bool(params, "copy_X", copy_X_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool LinearRegression::is_fitted() const {
    return fitted_;
}

void LinearRegression::save_to_file(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open file for saving LinearRegression");
    int sz = static_cast<int>(coef_.size());
    ofs.write(reinterpret_cast<const char*>(&sz), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(coef_.data()), sizeof(double) * sz);
    ofs.write(reinterpret_cast<const char*>(&intercept_), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&fit_intercept_), sizeof(bool));
}

void LinearRegression::load_from_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open file for loading LinearRegression");
    int sz = 0; ifs.read(reinterpret_cast<char*>(&sz), sizeof(int));
    coef_.resize(sz);
    ifs.read(reinterpret_cast<char*>(coef_.data()), sizeof(double) * sz);
    ifs.read(reinterpret_cast<char*>(&intercept_), sizeof(double));
    ifs.read(reinterpret_cast<char*>(&fit_intercept_), sizeof(bool));
    fitted_ = true;
}

// Ridge implementation
Ridge::Ridge(double alpha, bool fit_intercept, bool copy_X, int n_jobs)
    : coef_(), intercept_(0.0), fitted_(false), alpha_(alpha), 
      fit_intercept_(fit_intercept), copy_X_(copy_X), n_jobs_(n_jobs) {}

Estimator& Ridge::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    MatrixXd X_work = copy_X_ ? X : X;
    VectorXd y_work = y;
    
    if (fit_intercept_) {
        // Center the data
        VectorXd X_mean = X_work.colwise().mean();
        double y_mean = y_work.mean();
        
        X_work.rowwise() -= X_mean.transpose();
        y_work.array() -= y_mean;
        
        // Solve ridge regression: (X'X + αI)^-1 X'y
        MatrixXd XtX = X_work.transpose() * X_work;
        MatrixXd regularization = alpha_ * MatrixXd::Identity(X.cols(), X.cols());
        VectorXd Xty = X_work.transpose() * y_work;
        
        coef_ = (XtX + regularization).ldlt().solve(Xty);
        intercept_ = y_mean - X_mean.dot(coef_);
    } else {
        // No intercept
        MatrixXd XtX = X_work.transpose() * X_work;
        MatrixXd regularization = alpha_ * MatrixXd::Identity(X.cols(), X.cols());
        VectorXd Xty = X_work.transpose() * y_work;
        
        coef_ = (XtX + regularization).ldlt().solve(Xty);
        intercept_ = 0.0;
    }
    
    fitted_ = true;
    return *this;
}

VectorXd Ridge::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Ridge must be fitted before predict");
    }
    
    if (X.cols() != coef_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXd predictions = X * coef_;
    if (fit_intercept_) {
        predictions.array() += intercept_;
    }
    
    return predictions;
}

Params Ridge::get_params() const {
    return {
        {"alpha", std::to_string(alpha_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"copy_X", copy_X_ ? "true" : "false"},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& Ridge::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    copy_X_ = utils::get_param_bool(params, "copy_X", copy_X_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool Ridge::is_fitted() const {
    return fitted_;
}

void Ridge::save_to_file(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open file for saving Ridge");
    int sz = static_cast<int>(coef_.size());
    ofs.write(reinterpret_cast<const char*>(&sz), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(coef_.data()), sizeof(double) * sz);
    ofs.write(reinterpret_cast<const char*>(&intercept_), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&fit_intercept_), sizeof(bool));
    ofs.write(reinterpret_cast<const char*>(&alpha_), sizeof(double));
}

void Ridge::load_from_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open file for loading Ridge");
    int sz = 0; ifs.read(reinterpret_cast<char*>(&sz), sizeof(int));
    coef_.resize(sz);
    ifs.read(reinterpret_cast<char*>(coef_.data()), sizeof(double) * sz);
    ifs.read(reinterpret_cast<char*>(&intercept_), sizeof(double));
    ifs.read(reinterpret_cast<char*>(&fit_intercept_), sizeof(bool));
    ifs.read(reinterpret_cast<char*>(&alpha_), sizeof(double));
    fitted_ = true;
}

// Lasso implementation (simplified - just framework)
Lasso::Lasso(double alpha, bool fit_intercept, bool copy_X, int n_jobs)
    : coef_(), intercept_(0.0), fitted_(false), alpha_(alpha), 
      fit_intercept_(fit_intercept), copy_X_(copy_X), n_jobs_(n_jobs) {}

Estimator& Lasso::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    // Simplified implementation - just use Ridge for now
    // In a full implementation, this would use coordinate descent
    MatrixXd X_work = copy_X_ ? X : X;
    VectorXd y_work = y;
    
    if (fit_intercept_) {
        VectorXd X_mean = X_work.colwise().mean();
        double y_mean = y_work.mean();
        
        X_work.rowwise() -= X_mean.transpose();
        y_work.array() -= y_mean;
        
        MatrixXd XtX = X_work.transpose() * X_work;
        MatrixXd regularization = alpha_ * MatrixXd::Identity(X.cols(), X.cols());
        VectorXd Xty = X_work.transpose() * y_work;
        
        coef_ = (XtX + regularization).ldlt().solve(Xty);
        intercept_ = y_mean - X_mean.dot(coef_);
    } else {
        MatrixXd XtX = X_work.transpose() * X_work;
        MatrixXd regularization = alpha_ * MatrixXd::Identity(X.cols(), X.cols());
        VectorXd Xty = X_work.transpose() * y_work;
        
        coef_ = (XtX + regularization).ldlt().solve(Xty);
        intercept_ = 0.0;
    }
    
    fitted_ = true;
    return *this;
}

VectorXd Lasso::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Lasso must be fitted before predict");
    }
    
    if (X.cols() != coef_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXd predictions = X * coef_;
    if (fit_intercept_) {
        predictions.array() += intercept_;
    }
    
    return predictions;
}

Params Lasso::get_params() const {
    return {
        {"alpha", std::to_string(alpha_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"copy_X", copy_X_ ? "true" : "false"},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& Lasso::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    copy_X_ = utils::get_param_bool(params, "copy_X", copy_X_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool Lasso::is_fitted() const {
    return fitted_;
}

} // namespace linear_model
} // namespace cxml
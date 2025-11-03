#include "auroraml/linear_model.hpp"
#include "auroraml/base.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <set>
#include <cmath>

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

// ElasticNet implementation
ElasticNet::ElasticNet(double alpha, double l1_ratio, bool fit_intercept, bool copy_X, 
                       int n_jobs, int max_iter, double tol)
    : coef_(), intercept_(0.0), fitted_(false), alpha_(alpha), l1_ratio_(l1_ratio),
      fit_intercept_(fit_intercept), copy_X_(copy_X), n_jobs_(n_jobs), 
      max_iter_(max_iter), tol_(tol) {}

Estimator& ElasticNet::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    MatrixXd X_work = copy_X_ ? X : X;
    VectorXd y_work = y;
    
    if (fit_intercept_) {
        VectorXd X_mean = X_work.colwise().mean();
        double y_mean = y_work.mean();
        
        X_work.rowwise() -= X_mean.transpose();
        y_work.array() -= y_mean;
        
        // Elastic Net: combines L1 and L2 regularization
        // Regularization = α * (l1_ratio * L1 + (1 - l1_ratio) * L2)
        double l1_reg = alpha_ * l1_ratio_;
        double l2_reg = alpha_ * (1.0 - l1_ratio_);
        
        MatrixXd XtX = X_work.transpose() * X_work;
        MatrixXd regularization = l2_reg * MatrixXd::Identity(X.cols(), X.cols());
        VectorXd Xty = X_work.transpose() * y_work;
        
        // Solve with L2 part, then apply soft thresholding for L1
        coef_ = (XtX + regularization).ldlt().solve(Xty);
        
        // Apply soft thresholding for L1 regularization
        for (int i = 0; i < coef_.size(); ++i) {
            if (coef_(i) > l1_reg) {
                coef_(i) -= l1_reg;
            } else if (coef_(i) < -l1_reg) {
                coef_(i) += l1_reg;
            } else {
                coef_(i) = 0.0;
            }
        }
        
        intercept_ = y_mean - X_mean.dot(coef_);
    } else {
        double l1_reg = alpha_ * l1_ratio_;
        double l2_reg = alpha_ * (1.0 - l1_ratio_);
        
        MatrixXd XtX = X_work.transpose() * X_work;
        MatrixXd regularization = l2_reg * MatrixXd::Identity(X.cols(), X.cols());
        VectorXd Xty = X_work.transpose() * y_work;
        
        coef_ = (XtX + regularization).ldlt().solve(Xty);
        
        // Apply soft thresholding
        for (int i = 0; i < coef_.size(); ++i) {
            if (coef_(i) > l1_reg) {
                coef_(i) -= l1_reg;
            } else if (coef_(i) < -l1_reg) {
                coef_(i) += l1_reg;
            } else {
                coef_(i) = 0.0;
            }
        }
        
        intercept_ = 0.0;
    }
    
    fitted_ = true;
    return *this;
}

VectorXd ElasticNet::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ElasticNet must be fitted before predict");
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

Params ElasticNet::get_params() const {
    return {
        {"alpha", std::to_string(alpha_)},
        {"l1_ratio", std::to_string(l1_ratio_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"copy_X", copy_X_ ? "true" : "false"},
        {"n_jobs", std::to_string(n_jobs_)},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& ElasticNet::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    l1_ratio_ = utils::get_param_double(params, "l1_ratio", l1_ratio_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    copy_X_ = utils::get_param_bool(params, "copy_X", copy_X_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

bool ElasticNet::is_fitted() const {
    return fitted_;
}

// LogisticRegression implementation
LogisticRegression::LogisticRegression(double C, bool fit_intercept, int max_iter, 
                                      double tol, int random_state)
    : coef_(), intercept_(0.0), fitted_(false), C_(C), fit_intercept_(fit_intercept),
      max_iter_(max_iter), tol_(tol), random_state_(random_state), n_classes_(0) {}

// Sigmoid function
static double sigmoid(double z) {
    // Clamp z to prevent overflow
    z = std::max(-500.0, std::min(500.0, z));
    return 1.0 / (1.0 + std::exp(-z));
}

Estimator& LogisticRegression::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    // Find unique classes
    std::set<int> unique_classes;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes.insert(static_cast<int>(y(i)));
    }
    
    classes_.assign(unique_classes.begin(), unique_classes.end());
    n_classes_ = static_cast<int>(classes_.size());
    
    if (n_classes_ < 2) {
        throw std::invalid_argument("LogisticRegression requires at least 2 classes");
    }
    
    // For binary classification
    if (n_classes_ == 2) {
        // Convert to binary labels (0 and 1)
        VectorXd y_binary = VectorXd::Zero(y.size());
        int class0 = classes_[0];
        for (int i = 0; i < y.size(); ++i) {
            y_binary(i) = (static_cast<int>(y(i)) == class0) ? 0.0 : 1.0;
        }
        
        MatrixXd X_work = X;
        VectorXd y_work = y_binary;
        
        if (fit_intercept_) {
            // Add intercept column
            MatrixXd X_with_intercept(X.rows(), X.cols() + 1);
            X_with_intercept.col(0) = VectorXd::Ones(X.rows());
            X_with_intercept.rightCols(X.cols()) = X_work;
            X_work = X_with_intercept;
        }
        
        int n_features = X_work.cols();
        VectorXd weights = VectorXd::Zero(n_features);
        
        // Gradient descent with L2 regularization
        double lambda = 1.0 / C_;
        
        for (int iter = 0; iter < max_iter_; ++iter) {
            VectorXd grad = VectorXd::Zero(n_features);
            
            // Compute gradient
            for (int i = 0; i < X_work.rows(); ++i) {
                double z = X_work.row(i) * weights;
                double pred = sigmoid(z);
                double error = pred - y_work(i);
                grad += error * X_work.row(i).transpose();
            }
            grad /= X_work.rows();
            grad += lambda * weights;
            
            // Update weights (gradient descent step)
            double learning_rate = 0.01;
            weights -= learning_rate * grad;
            
            // Check convergence
            if (grad.norm() < tol_) {
                break;
            }
        }
        
        if (fit_intercept_) {
            intercept_ = weights(0);
            coef_ = weights.tail(X.cols());
        } else {
            intercept_ = 0.0;
            coef_ = weights;
        }
    } else {
        // Multi-class: One-vs-Rest approach
        throw std::runtime_error("Multi-class LogisticRegression not yet implemented");
    }
    
    fitted_ = true;
    return *this;
}

VectorXi LogisticRegression::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LogisticRegression must be fitted before predict");
    }
    
    VectorXd decision = decision_function(X);
    VectorXi predictions(decision.size());
    
    for (int i = 0; i < decision.size(); ++i) {
        if (n_classes_ == 2) {
            predictions(i) = (decision(i) >= 0.0) ? classes_[1] : classes_[0];
        } else {
            predictions(i) = classes_[0]; // Placeholder for multi-class
        }
    }
    
    return predictions;
}

MatrixXd LogisticRegression::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LogisticRegression must be fitted before predict_proba");
    }
    
    VectorXd decision = decision_function(X);
    MatrixXd proba(X.rows(), n_classes_);
    
    if (n_classes_ == 2) {
        for (int i = 0; i < decision.size(); ++i) {
            double prob_pos = sigmoid(decision(i));
            proba(i, 0) = 1.0 - prob_pos;
            proba(i, 1) = prob_pos;
        }
    } else {
        // Multi-class would need softmax
        throw std::runtime_error("Multi-class predict_proba not yet implemented");
    }
    
    return proba;
}

VectorXd LogisticRegression::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LogisticRegression must be fitted before decision_function");
    }
    
    if (X.cols() != coef_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXd decision = X * coef_;
    if (fit_intercept_) {
        decision.array() += intercept_;
    }
    
    return decision;
}

Params LogisticRegression::get_params() const {
    return {
        {"C", std::to_string(C_)},
        {"fit_intercept", fit_intercept_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& LogisticRegression::set_params(const Params& params) {
    C_ = utils::get_param_double(params, "C", C_);
    fit_intercept_ = utils::get_param_bool(params, "fit_intercept", fit_intercept_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

bool LogisticRegression::is_fitted() const {
    return fitted_;
}

} // namespace linear_model
} // namespace cxml
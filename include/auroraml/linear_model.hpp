#pragma once

#include "base.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace linear_model {

/**
 * Linear Regression (Ordinary Least Squares)
 */
class LinearRegression : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    bool fit_intercept_;
    bool copy_X_;
    int n_jobs_;

public:
    LinearRegression(bool fit_intercept = true, bool copy_X = true, int n_jobs = 1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }

    // Persistence
    void save_to_file(const std::string& path) const;
    void load_from_file(const std::string& path);
};

/**
 * Ridge Regression (L2 Regularization)
 */
class Ridge : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_;
    bool fit_intercept_;
    bool copy_X_;
    int n_jobs_;

public:
    Ridge(double alpha = 1.0, bool fit_intercept = true, bool copy_X = true, int n_jobs = 1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }

    // Persistence
    void save_to_file(const std::string& path) const;
    void load_from_file(const std::string& path);
};

/**
 * Lasso Regression (L1 Regularization)
 */
class Lasso : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_;
    bool fit_intercept_;
    bool copy_X_;
    int n_jobs_;

public:
    Lasso(double alpha = 1.0, bool fit_intercept = true, bool copy_X = true, int n_jobs = 1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

} // namespace linear_model
} // namespace cxml
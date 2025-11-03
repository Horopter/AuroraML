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

/**
 * Elastic Net (L1 + L2 Regularization)
 */
class ElasticNet : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_;
    double l1_ratio_;
    bool fit_intercept_;
    bool copy_X_;
    int n_jobs_;
    int max_iter_;
    double tol_;

public:
    ElasticNet(double alpha = 1.0, double l1_ratio = 0.5, bool fit_intercept = true, 
               bool copy_X = true, int n_jobs = 1, int max_iter = 1000, double tol = 1e-4);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * Logistic Regression
 */
class LogisticRegression : public Estimator, public Classifier {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double C_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    int random_state_;
    std::vector<int> classes_;
    int n_classes_;

public:
    LogisticRegression(double C = 1.0, bool fit_intercept = true, int max_iter = 100, 
                      double tol = 1e-4, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
    std::vector<int> classes() const { return classes_; }
    int n_classes() const { return n_classes_; }
};

} // namespace linear_model
} // namespace cxml
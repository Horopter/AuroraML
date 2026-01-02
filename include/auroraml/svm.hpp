#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>

namespace auroraml {
namespace svm {

// Linear SVM classifier (hinge loss) optimized via SGD
class LinearSVC : public Estimator, public Classifier {
private:
    VectorXd w_;
    double b_ = 0.0;
    bool fitted_ = false;

    // params
    double C_;          // regularization strength (1/lambda)
    int max_iter_;
    double lr_;         // learning rate
    int random_state_;

public:
    LinearSVC(double C = 1.0, int max_iter = 1000, double lr = 0.01, int random_state = -1)
        : C_(C), max_iter_(max_iter), lr_(lr), random_state_(random_state) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;

    // Classifier API
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;

    // Estimator API
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    // Model persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
};

/**
 * Support Vector Regression (SVR)
 * Uses epsilon-insensitive loss function
 */
class SVR : public Estimator, public Regressor {
private:
    VectorXd w_;
    double b_ = 0.0;
    bool fitted_ = false;
    
    // params
    double C_;          // regularization strength
    double epsilon_;    // epsilon-tube width
    int max_iter_;
    double lr_;         // learning rate
    int random_state_;

public:
    SVR(double C = 1.0, double epsilon = 0.1, int max_iter = 1000, 
        double lr = 0.01, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    VectorXd coef() const { return w_; }
    double intercept() const { return b_; }
};

/**
 * Linear Support Vector Regression (LinearSVR)
 * Linear kernel only, optimized for speed
 */
class LinearSVR : public Estimator, public Regressor {
private:
    VectorXd w_;
    double b_ = 0.0;
    bool fitted_ = false;
    
    // params
    double C_;          // regularization strength
    double epsilon_;    // epsilon-tube width
    int max_iter_;
    double lr_;         // learning rate
    int random_state_;

public:
    LinearSVR(double C = 1.0, double epsilon = 0.1, int max_iter = 1000, 
              double lr = 0.01, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    VectorXd coef() const { return w_; }
    double intercept() const { return b_; }
};

/**
 * Nu-Support Vector Classifier (NuSVC)
 * Alternative parameterization using nu parameter
 */
class NuSVC : public Estimator, public Classifier {
private:
    VectorXd w_;
    double b_ = 0.0;
    bool fitted_ = false;
    std::vector<int> classes_;
    
    // params
    double nu_;         // nu parameter (0 < nu <= 1)
    int max_iter_;
    double lr_;         // learning rate
    int random_state_;

public:
    NuSVC(double nu = 0.5, int max_iter = 1000, double lr = 0.01, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    
    // Classifier API
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    const std::vector<int>& classes() const { return classes_; }
};

/**
 * Nu-Support Vector Regression (NuSVR)
 * Alternative parameterization using nu parameter
 */
class NuSVR : public Estimator, public Regressor {
private:
    VectorXd w_;
    double b_ = 0.0;
    bool fitted_ = false;
    
    // params
    double nu_;         // nu parameter (0 < nu <= 1)
    double C_;          // regularization strength
    int max_iter_;
    double lr_;         // learning rate
    int random_state_;

public:
    NuSVR(double nu = 0.5, double C = 1.0, int max_iter = 1000, 
          double lr = 0.01, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    VectorXd coef() const { return w_; }
    double intercept() const { return b_; }
};

/**
 * One-Class Support Vector Machine (OneClassSVM)
 * Outlier detection using SVM with no labels
 */
class OneClassSVM : public Estimator {
private:
    VectorXd w_;
    double rho_ = 0.0;  // decision threshold
    bool fitted_ = false;
    MatrixXd support_vectors_;  // for RBF kernel
    VectorXd center_;  // for linear kernel
    
    // params
    double nu_;         // nu parameter (0 < nu <= 1)
    double gamma_;      // RBF kernel parameter
    int max_iter_;
    double lr_;         // learning rate
    int random_state_;
    
    // Support for different kernels
    enum class Kernel { LINEAR, RBF };
    Kernel kernel_;
    
    // Kernel functions
    double linear_kernel(const VectorXd& x1, const VectorXd& x2) const;
    double rbf_kernel(const VectorXd& x1, const VectorXd& x2) const;
    double kernel_function(const VectorXd& x1, const VectorXd& x2) const;

public:
    OneClassSVM(double nu = 0.5, double gamma = 1.0, const std::string& kernel = "rbf",
                int max_iter = 1000, double lr = 0.01, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    
    // Anomaly detection methods
    VectorXi predict(const MatrixXd& X) const;  // +1 for inliers, -1 for outliers
    VectorXd decision_function(const MatrixXd& X) const;  // distance to separating hyperplane
    VectorXd score_samples(const MatrixXd& X) const;      // raw anomaly scores
    
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    double get_threshold() const { return rho_; }
};

} // namespace svm
} // namespace auroraml



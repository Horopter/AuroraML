#pragma once

#include "base.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
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

/**
 * RidgeCV - Ridge Regression with Cross-Validation
 */
class RidgeCV : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::vector<double> alphas_;
    double best_alpha_;
    bool fit_intercept_;
    int cv_folds_;

public:
    RidgeCV(const std::vector<double>& alphas = {0.1, 1.0, 10.0}, 
            bool fit_intercept = true, int cv_folds = 5);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
    double best_alpha() const { return best_alpha_; }
};

/**
 * LassoCV - Lasso Regression with Cross-Validation
 */
class LassoCV : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::vector<double> alphas_;
    double best_alpha_;
    bool fit_intercept_;
    int cv_folds_;
    int max_iter_;
    double tol_;

public:
    LassoCV(const std::vector<double>& alphas = {0.1, 1.0, 10.0}, 
            bool fit_intercept = true, int cv_folds = 5, 
            int max_iter = 1000, double tol = 1e-4);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
    double best_alpha() const { return best_alpha_; }
};

/**
 * ElasticNetCV - ElasticNet Regression with Cross-Validation
 */
class ElasticNetCV : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::vector<double> alphas_;
    std::vector<double> l1_ratios_;
    double best_alpha_;
    double best_l1_ratio_;
    bool fit_intercept_;
    int cv_folds_;
    int max_iter_;
    double tol_;

public:
    ElasticNetCV(const std::vector<double>& alphas = {0.1, 1.0, 10.0}, 
                 const std::vector<double>& l1_ratios = {0.1, 0.5, 0.9},
                 bool fit_intercept = true, int cv_folds = 5, 
                 int max_iter = 1000, double tol = 1e-4);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
    double best_alpha() const { return best_alpha_; }
    double best_l1_ratio() const { return best_l1_ratio_; }
};

/**
 * ARDRegression - Automatic Relevance Determination Regression
 */
class ARDRegression : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_1_;
    double alpha_2_;
    double lambda_1_;
    double lambda_2_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    ARDRegression(double alpha_1 = 1e-6, double alpha_2 = 1e-6,
                  double lambda_1 = 1e-6, double lambda_2 = 1e-6,
                  bool fit_intercept = true, int max_iter = 300, double tol = 1e-3);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * BayesianRidge - Bayesian Ridge Regression
 */
class BayesianRidge : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_1_;
    double alpha_2_;
    double lambda_1_;
    double lambda_2_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    BayesianRidge(double alpha_1 = 1e-6, double alpha_2 = 1e-6,
                  double lambda_1 = 1e-6, double lambda_2 = 1e-6,
                  bool fit_intercept = true, int max_iter = 300, double tol = 1e-3);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * HuberRegressor - Huber Regression (robust to outliers)
 */
class HuberRegressor : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double epsilon_;
    double alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    HuberRegressor(double epsilon = 1.35, double alpha = 0.0001,
                   bool fit_intercept = true, int max_iter = 100, double tol = 1e-5);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * Least Angle Regression (Lars)
 */
class Lars : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    int n_nonzero_coefs_;
    int max_iter_;
    bool fit_intercept_;
    double eps_;

public:
    Lars(int n_nonzero_coefs = 0, bool fit_intercept = true, int max_iter = 500, double eps = 1e-3);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * LarsCV - Lars Regression with Cross-Validation
 */
class LarsCV : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    int cv_folds_;
    int max_iter_;
    int best_n_nonzero_coefs_;
    bool fit_intercept_;
    double eps_;

public:
    LarsCV(int cv_folds = 5, bool fit_intercept = true, int max_iter = 500, double eps = 1e-3);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    int best_n_nonzero_coefs() const { return best_n_nonzero_coefs_; }
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * LassoLars - Lasso Regression using a Lars-style solver
 */
class LassoLars : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    LassoLars(double alpha = 1.0, bool fit_intercept = true, int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * LassoLarsCV - LassoLars Regression with Cross-Validation
 */
class LassoLarsCV : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::vector<double> alphas_;
    double best_alpha_;
    int cv_folds_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    LassoLarsCV(const std::vector<double>& alphas = {0.1, 1.0, 10.0}, int cv_folds = 5,
                bool fit_intercept = true, int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    double best_alpha() const { return best_alpha_; }
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * LassoLarsIC - LassoLars with information-criterion model selection
 */
class LassoLarsIC : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::vector<double> alphas_;
    std::string criterion_;
    double best_alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    LassoLarsIC(const std::vector<double>& alphas = {0.1, 1.0, 10.0}, const std::string& criterion = "aic",
                bool fit_intercept = true, int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    double best_alpha() const { return best_alpha_; }
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * Orthogonal Matching Pursuit
 */
class OrthogonalMatchingPursuit : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    int n_nonzero_coefs_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    OrthogonalMatchingPursuit(int n_nonzero_coefs = 0, bool fit_intercept = true,
                              int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * Orthogonal Matching Pursuit with Cross-Validation
 */
class OrthogonalMatchingPursuitCV : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    int cv_folds_;
    int max_iter_;
    int best_n_nonzero_coefs_;
    bool fit_intercept_;
    double tol_;

public:
    OrthogonalMatchingPursuitCV(int cv_folds = 5, bool fit_intercept = true,
                                int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    int best_n_nonzero_coefs() const { return best_n_nonzero_coefs_; }
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * RANSACRegressor - Robust regression with RANSAC
 */
class RANSACRegressor : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    int max_trials_;
    int min_samples_;
    double residual_threshold_;
    int random_state_;
    bool fit_intercept_;

public:
    RANSACRegressor(int max_trials = 100, int min_samples = -1,
                    double residual_threshold = -1.0, int random_state = -1,
                    bool fit_intercept = true);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * TheilSenRegressor - Robust regression using median of least squares
 */
class TheilSenRegressor : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    int n_subsamples_;
    int random_state_;
    bool fit_intercept_;

public:
    TheilSenRegressor(int n_subsamples = 100, int random_state = -1,
                      bool fit_intercept = true);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * SGDRegressor - Linear regression with SGD
 */
class SGDRegressor : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::string loss_;
    std::string penalty_;
    double alpha_;
    double l1_ratio_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    std::string learning_rate_;
    double eta0_;
    double power_t_;
    bool shuffle_;
    int random_state_;
    double epsilon_;

public:
    SGDRegressor(const std::string& loss = "squared_loss", const std::string& penalty = "l2",
                 double alpha = 0.0001, double l1_ratio = 0.15, bool fit_intercept = true,
                 int max_iter = 1000, double tol = 1e-3, const std::string& learning_rate = "invscaling",
                 double eta0 = 0.01, double power_t = 0.5, bool shuffle = true,
                 int random_state = -1, double epsilon = 0.1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * SGDClassifier - Linear classifier with SGD
 */
class SGDClassifier : public Estimator, public Classifier {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::string loss_;
    std::string penalty_;
    double alpha_;
    double l1_ratio_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    std::string learning_rate_;
    double eta0_;
    double power_t_;
    bool shuffle_;
    int random_state_;
    std::vector<int> classes_;
    int n_classes_;

public:
    SGDClassifier(const std::string& loss = "hinge", const std::string& penalty = "l2",
                  double alpha = 0.0001, double l1_ratio = 0.15, bool fit_intercept = true,
                  int max_iter = 1000, double tol = 1e-3, const std::string& learning_rate = "invscaling",
                  double eta0 = 0.01, double power_t = 0.5, bool shuffle = true,
                  int random_state = -1);

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

/**
 * PassiveAggressiveRegressor
 */
class PassiveAggressiveRegressor : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double C_;
    double epsilon_;
    bool fit_intercept_;
    int max_iter_;
    bool shuffle_;
    int random_state_;
    std::string loss_;

public:
    PassiveAggressiveRegressor(double C = 1.0, double epsilon = 0.1, bool fit_intercept = true,
                               int max_iter = 1000, bool shuffle = true, int random_state = -1,
                               const std::string& loss = "epsilon_insensitive");

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * PassiveAggressiveClassifier
 */
class PassiveAggressiveClassifier : public Estimator, public Classifier {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double C_;
    bool fit_intercept_;
    int max_iter_;
    bool shuffle_;
    int random_state_;
    std::vector<int> classes_;
    int n_classes_;

public:
    PassiveAggressiveClassifier(double C = 1.0, bool fit_intercept = true,
                                int max_iter = 1000, bool shuffle = true, int random_state = -1);

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

/**
 * Perceptron
 */
class Perceptron : public Estimator, public Classifier {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    bool shuffle_;
    int random_state_;
    std::vector<int> classes_;
    int n_classes_;

public:
    Perceptron(bool fit_intercept = true, int max_iter = 1000, double tol = 1e-3,
               bool shuffle = true, int random_state = -1);

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

/**
 * LogisticRegressionCV - Logistic Regression with Cross-Validation
 */
class LogisticRegressionCV : public Estimator, public Classifier {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::vector<double> Cs_;
    int cv_folds_;
    std::string scoring_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    int random_state_;
    double best_C_;
    std::vector<int> classes_;
    int n_classes_;

public:
    LogisticRegressionCV(const std::vector<double>& Cs = {0.1, 1.0, 10.0}, int cv_folds = 5,
                         const std::string& scoring = "accuracy", bool fit_intercept = true,
                         int max_iter = 100, double tol = 1e-4, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    double best_C() const { return best_C_; }
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
    std::vector<int> classes() const { return classes_; }
    int n_classes() const { return n_classes_; }
};

/**
 * RidgeClassifier
 */
class RidgeClassifier : public Estimator, public Classifier {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_;
    bool fit_intercept_;
    std::vector<int> classes_;
    int n_classes_;

public:
    RidgeClassifier(double alpha = 1.0, bool fit_intercept = true);

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

/**
 * RidgeClassifierCV - RidgeClassifier with Cross-Validation
 */
class RidgeClassifierCV : public Estimator, public Classifier {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::vector<double> alphas_;
    double best_alpha_;
    int cv_folds_;
    std::string scoring_;
    bool fit_intercept_;
    std::vector<int> classes_;
    int n_classes_;

public:
    RidgeClassifierCV(const std::vector<double>& alphas = {0.1, 1.0, 10.0}, int cv_folds = 5,
                      const std::string& scoring = "accuracy", bool fit_intercept = true);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    double best_alpha() const { return best_alpha_; }
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
    std::vector<int> classes() const { return classes_; }
    int n_classes() const { return n_classes_; }
};

/**
 * QuantileRegressor
 */
class QuantileRegressor : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double quantile_;
    double alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    double learning_rate_;

public:
    QuantileRegressor(double quantile = 0.5, double alpha = 0.0, bool fit_intercept = true,
                      int max_iter = 1000, double tol = 1e-4, double learning_rate = 0.01);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * PoissonRegressor
 */
class PoissonRegressor : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    double learning_rate_;

public:
    PoissonRegressor(double alpha = 0.0, bool fit_intercept = true,
                     int max_iter = 1000, double tol = 1e-4, double learning_rate = 0.01);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * GammaRegressor
 */
class GammaRegressor : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    double learning_rate_;

public:
    GammaRegressor(double alpha = 0.0, bool fit_intercept = true,
                   int max_iter = 1000, double tol = 1e-4, double learning_rate = 0.01);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * TweedieRegressor
 */
class TweedieRegressor : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double power_;
    double alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    double learning_rate_;

public:
    TweedieRegressor(double power = 1.5, double alpha = 0.0, bool fit_intercept = true,
                     int max_iter = 1000, double tol = 1e-4, double learning_rate = 0.01);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * MultiTaskLasso
 */
class MultiTaskLasso : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    MultiTaskLasso(double alpha = 1.0, bool fit_intercept = true, int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * MultiTaskLassoCV
 */
class MultiTaskLassoCV : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::vector<double> alphas_;
    double best_alpha_;
    int cv_folds_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    MultiTaskLassoCV(const std::vector<double>& alphas = {0.1, 1.0, 10.0}, int cv_folds = 5,
                     bool fit_intercept = true, int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    double best_alpha() const { return best_alpha_; }
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * MultiTaskElasticNet
 */
class MultiTaskElasticNet : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    double alpha_;
    double l1_ratio_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    MultiTaskElasticNet(double alpha = 1.0, double l1_ratio = 0.5, bool fit_intercept = true,
                        int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

/**
 * MultiTaskElasticNetCV
 */
class MultiTaskElasticNetCV : public Estimator, public Regressor {
private:
    VectorXd coef_;
    double intercept_;
    bool fitted_;
    std::vector<double> alphas_;
    std::vector<double> l1_ratios_;
    double best_alpha_;
    double best_l1_ratio_;
    int cv_folds_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;

public:
    MultiTaskElasticNetCV(const std::vector<double>& alphas = {0.1, 1.0, 10.0},
                          const std::vector<double>& l1_ratios = {0.1, 0.5, 0.9},
                          int cv_folds = 5, bool fit_intercept = true,
                          int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

    double best_alpha() const { return best_alpha_; }
    double best_l1_ratio() const { return best_l1_ratio_; }
    VectorXd coef() const { return coef_; }
    double intercept() const { return intercept_; }
};

} // namespace linear_model
} // namespace ingenuityml

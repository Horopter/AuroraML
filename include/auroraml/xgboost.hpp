#pragma once

#include "base.hpp"
#include "tree.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <string>
#include <memory>

namespace auroraml {
namespace ensemble {

/**
 * XGBoost Classifier
 * 
 * Extreme Gradient Boosting for classification.
 * Implements regularized gradient boosting with tree pruning.
 */
class XGBClassifier : public Estimator, public Classifier {
private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    double gamma_;
    double reg_alpha_;
    double reg_lambda_;
    int min_child_weight_;
    double subsample_;
    double colsample_bytree_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    
    std::vector<tree::DecisionTreeRegressor> estimators_;
    std::vector<int> classes_;
    int n_classes_;
    VectorXd base_score_;

public:
    XGBClassifier(int n_estimators = 100, double learning_rate = 0.1,
                 int max_depth = 6, double gamma = 0.0, double reg_alpha = 0.0,
                 double reg_lambda = 1.0, int min_child_weight = 1,
                 double subsample = 1.0, double colsample_bytree = 1.0,
                 int random_state = -1)
        : n_estimators_(n_estimators), learning_rate_(learning_rate),
          max_depth_(max_depth), gamma_(gamma), reg_alpha_(reg_alpha),
          reg_lambda_(reg_lambda), min_child_weight_(min_child_weight),
          subsample_(subsample), colsample_bytree_(colsample_bytree),
          random_state_(random_state), n_features_(0), n_classes_(0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    std::vector<int> classes() const { return classes_; }
};

/**
 * XGBoost Regressor
 * 
 * Extreme Gradient Boosting for regression.
 * Implements regularized gradient boosting with tree pruning.
 */
class XGBRegressor : public Estimator, public Regressor {
private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    double gamma_;
    double reg_alpha_;
    double reg_lambda_;
    int min_child_weight_;
    double subsample_;
    double colsample_bytree_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    
    std::vector<tree::DecisionTreeRegressor> estimators_;
    double base_score_;

public:
    XGBRegressor(int n_estimators = 100, double learning_rate = 0.1,
                int max_depth = 6, double gamma = 0.0, double reg_alpha = 0.0,
                double reg_lambda = 1.0, int min_child_weight = 1,
                double subsample = 1.0, double colsample_bytree = 1.0,
                int random_state = -1)
        : n_estimators_(n_estimators), learning_rate_(learning_rate),
          max_depth_(max_depth), gamma_(gamma), reg_alpha_(reg_alpha),
          reg_lambda_(reg_lambda), min_child_weight_(min_child_weight),
          subsample_(subsample), colsample_bytree_(colsample_bytree),
          random_state_(random_state), n_features_(0), base_score_(0.0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

} // namespace ensemble
} // namespace cxml


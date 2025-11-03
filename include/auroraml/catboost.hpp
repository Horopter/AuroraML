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
 * CatBoost Classifier
 * 
 * Categorical Boosting for classification.
 * Optimized for handling categorical features with ordered boosting.
 */
class CatBoostClassifier : public Estimator, public Classifier {
private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    double l2_leaf_reg_;
    double border_count_;
    double bagging_temperature_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    
    std::vector<tree::DecisionTreeRegressor> estimators_;
    std::vector<int> classes_;
    int n_classes_;
    VectorXd base_score_;

public:
    CatBoostClassifier(int n_estimators = 100, double learning_rate = 0.03,
                      int max_depth = 6, double l2_leaf_reg = 3.0,
                      double border_count = 32.0, double bagging_temperature = 1.0,
                      int random_state = -1)
        : n_estimators_(n_estimators), learning_rate_(learning_rate),
          max_depth_(max_depth), l2_leaf_reg_(l2_leaf_reg),
          border_count_(border_count), bagging_temperature_(bagging_temperature),
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
 * CatBoost Regressor
 * 
 * Categorical Boosting for regression.
 * Optimized for handling categorical features with ordered boosting.
 */
class CatBoostRegressor : public Estimator, public Regressor {
private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    double l2_leaf_reg_;
    double border_count_;
    double bagging_temperature_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    
    std::vector<tree::DecisionTreeRegressor> estimators_;
    double base_score_;

public:
    CatBoostRegressor(int n_estimators = 100, double learning_rate = 0.03,
                     int max_depth = 6, double l2_leaf_reg = 3.0,
                     double border_count = 32.0, double bagging_temperature = 1.0,
                     int random_state = -1)
        : n_estimators_(n_estimators), learning_rate_(learning_rate),
          max_depth_(max_depth), l2_leaf_reg_(l2_leaf_reg),
          border_count_(border_count), bagging_temperature_(bagging_temperature),
          random_state_(random_state), n_features_(0), base_score_(0.0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

} // namespace ensemble
} // namespace cxml


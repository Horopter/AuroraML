#pragma once

#include "base.hpp"
#include "tree.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>

namespace auroraml {
namespace ensemble {

/**
 * AdaBoost Classifier
 * 
 * Adaptive Boosting algorithm for classification.
 * Uses decision stumps (depth-1 trees) as weak learners.
 */
class AdaBoostClassifier : public Estimator, public Classifier {
private:
    int n_estimators_;
    double learning_rate_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    
    std::vector<tree::DecisionTreeClassifier> estimators_;
    std::vector<double> estimator_weights_;
    std::vector<int> classes_;
    int n_classes_;

public:
    AdaBoostClassifier(int n_estimators = 50, double learning_rate = 1.0, int random_state = -1)
        : n_estimators_(n_estimators), learning_rate_(learning_rate), 
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
 * AdaBoost Regressor
 * 
 * Adaptive Boosting algorithm for regression.
 * Uses decision stumps (depth-1 trees) as weak learners.
 */
class AdaBoostRegressor : public Estimator, public Regressor {
private:
    int n_estimators_;
    double learning_rate_;
    std::string loss_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    
    std::vector<tree::DecisionTreeRegressor> estimators_;
    std::vector<double> estimator_weights_;

public:
    AdaBoostRegressor(int n_estimators = 50, double learning_rate = 1.0, 
                     const std::string& loss = "linear", int random_state = -1)
        : n_estimators_(n_estimators), learning_rate_(learning_rate), 
          loss_(loss), random_state_(random_state), n_features_(0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

} // namespace ensemble
} // namespace cxml


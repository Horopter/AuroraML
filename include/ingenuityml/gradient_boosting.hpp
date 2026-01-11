#pragma once

#include "base.hpp"
#include "tree.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>

namespace ingenuityml {
namespace ensemble {

/**
 * Gradient Boosting Classifier
 * 
 * Gradient boosting builds an additive model in a forward stage-wise fashion.
 * It optimizes an arbitrary differentiable loss function by fitting weak learners
 * to the negative gradients of the loss function.
 */
class GradientBoostingClassifier : public Estimator, public Classifier {
private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    double min_impurity_decrease_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    
    std::vector<tree::DecisionTreeRegressor> estimators_;
    VectorXd init_prediction_;
    std::vector<int> classes_;

public:
    GradientBoostingClassifier(int n_estimators = 100, double learning_rate = 0.1,
                              int max_depth = 3, int min_samples_split = 2,
                              int min_samples_leaf = 1, double min_impurity_decrease = 0.0,
                              int random_state = -1)
        : n_estimators_(n_estimators), learning_rate_(learning_rate), max_depth_(max_depth),
          min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf),
          min_impurity_decrease_(min_impurity_decrease), random_state_(random_state), n_features_(0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    // Model persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

    // Additional methods
    int n_estimators() const { return n_estimators_; }
    double learning_rate() const { return learning_rate_; }
    const std::vector<int>& classes() const { 
        if (!fitted_) throw std::runtime_error("GradientBoostingClassifier must be fitted before accessing classes.");
        return classes_; 
    }
};

/**
 * Gradient Boosting Regressor
 * 
 * Gradient boosting for regression tasks using squared error loss.
 */
class GradientBoostingRegressor : public Estimator, public Regressor {
private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    double min_impurity_decrease_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    
    std::vector<tree::DecisionTreeRegressor> estimators_;
    double init_prediction_;

public:
    GradientBoostingRegressor(int n_estimators = 100, double learning_rate = 0.1,
                              int max_depth = 3, int min_samples_split = 2,
                              int min_samples_leaf = 1, double min_impurity_decrease = 0.0,
                              int random_state = -1)
        : n_estimators_(n_estimators), learning_rate_(learning_rate), max_depth_(max_depth),
          min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf),
          min_impurity_decrease_(min_impurity_decrease), random_state_(random_state), n_features_(0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    // Model persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

    // Additional methods
    int n_estimators() const { return n_estimators_; }
    double learning_rate() const { return learning_rate_; }
};

/**
 * Hist Gradient Boosting Classifier
 *
 * Wrapper around GradientBoostingClassifier with a HistGradientBoosting-like interface.
 */
class HistGradientBoostingClassifier : public Estimator, public Classifier {
private:
    GradientBoostingClassifier impl_;

public:
    HistGradientBoostingClassifier(int max_iter = 100, double learning_rate = 0.1,
                                   int max_depth = 3, int min_samples_split = 2,
                                   int min_samples_leaf = 1, double min_impurity_decrease = 0.0,
                                   int random_state = -1)
        : impl_(max_iter, learning_rate, max_depth, min_samples_split,
                min_samples_leaf, min_impurity_decrease, random_state) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return impl_.is_fitted(); }

    void save(const std::string& filepath) const { impl_.save(filepath); }
    void load(const std::string& filepath) { impl_.load(filepath); }

    int max_iter() const { return impl_.n_estimators(); }
    double learning_rate() const { return impl_.learning_rate(); }
    const std::vector<int>& classes() const { return impl_.classes(); }
};

/**
 * Hist Gradient Boosting Regressor
 *
 * Wrapper around GradientBoostingRegressor with a HistGradientBoosting-like interface.
 */
class HistGradientBoostingRegressor : public Estimator, public Regressor {
private:
    GradientBoostingRegressor impl_;

public:
    HistGradientBoostingRegressor(int max_iter = 100, double learning_rate = 0.1,
                                  int max_depth = 3, int min_samples_split = 2,
                                  int min_samples_leaf = 1, double min_impurity_decrease = 0.0,
                                  int random_state = -1)
        : impl_(max_iter, learning_rate, max_depth, min_samples_split,
                min_samples_leaf, min_impurity_decrease, random_state) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return impl_.is_fitted(); }

    void save(const std::string& filepath) const { impl_.save(filepath); }
    void load(const std::string& filepath) { impl_.load(filepath); }

    int max_iter() const { return impl_.n_estimators(); }
    double learning_rate() const { return impl_.learning_rate(); }
};

} // namespace ensemble
} // namespace ingenuityml

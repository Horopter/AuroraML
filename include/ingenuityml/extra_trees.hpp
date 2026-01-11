#pragma once

#include "base.hpp"
#include "extratree.hpp"
#include <vector>

namespace ingenuityml {
namespace ensemble {

class ExtraTreesClassifier : public Estimator, public Classifier {
private:
    int n_estimators_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int max_features_;
    bool bootstrap_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    VectorXi classes_;
    std::vector<tree::ExtraTreeClassifier> trees_;

public:
    ExtraTreesClassifier(int n_estimators = 100, int max_depth = -1, int min_samples_split = 2,
                         int min_samples_leaf = 1, int max_features = -1,
                         bool bootstrap = false, int random_state = -1)
        : n_estimators_(n_estimators), max_depth_(max_depth),
          min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf),
          max_features_(max_features), bootstrap_(bootstrap),
          random_state_(random_state), n_features_(0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

class ExtraTreesRegressor : public Estimator, public Regressor {
private:
    int n_estimators_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int max_features_;
    bool bootstrap_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    std::vector<tree::ExtraTreeRegressor> trees_;

public:
    ExtraTreesRegressor(int n_estimators = 100, int max_depth = -1, int min_samples_split = 2,
                        int min_samples_leaf = 1, int max_features = -1,
                        bool bootstrap = false, int random_state = -1)
        : n_estimators_(n_estimators), max_depth_(max_depth),
          min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf),
          max_features_(max_features), bootstrap_(bootstrap),
          random_state_(random_state), n_features_(0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

} // namespace ensemble
} // namespace ingenuityml

#pragma once

#include "base.hpp"
#include "tree.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>

namespace auroraml {
namespace ensemble {

class RandomForestClassifier : public Estimator, public Classifier {
private:
    int n_estimators_;
    int max_depth_;
    int max_features_; // #features considered at split (sqrt for clf)
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    std::vector<tree::DecisionTreeClassifier> trees_;

public:
    RandomForestClassifier(int n_estimators = 100, int max_depth = -1, int max_features = -1, int random_state = -1)
        : n_estimators_(n_estimators), max_depth_(max_depth), max_features_(max_features), random_state_(random_state), n_features_(0) {}

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
};

class RandomForestRegressor : public Estimator, public Regressor {
private:
    int n_estimators_;
    int max_depth_;
    int max_features_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    std::vector<tree::DecisionTreeRegressor> trees_;

public:
    RandomForestRegressor(int n_estimators = 100, int max_depth = -1, int max_features = -1, int random_state = -1)
        : n_estimators_(n_estimators), max_depth_(max_depth), max_features_(max_features), random_state_(random_state), n_features_(0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    // Model persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
};

} // namespace ensemble
} // namespace cxml



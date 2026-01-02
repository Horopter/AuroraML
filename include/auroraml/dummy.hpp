#pragma once

#include "base.hpp"
#include <map>
#include <set>
#include <string>
#include <random>

namespace auroraml {
namespace ensemble {

class DummyClassifier : public Estimator, public Classifier {
private:
    std::string strategy_;
    bool fitted_;
    VectorXi classes_;
    int most_frequent_class_;
    std::map<int, int> class_counts_;
    int n_features_;

public:
    explicit DummyClassifier(const std::string& strategy = "most_frequent");

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

class DummyRegressor : public Estimator, public Regressor {
private:
    std::string strategy_;
    double quantile_;
    double constant_;
    double statistic_;
    bool fitted_;
    int n_features_;

public:
    DummyRegressor(const std::string& strategy = "mean", double quantile = 0.5, double constant = 0.0);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    double statistic() const { return statistic_; }
};

} // namespace ensemble
} // namespace auroraml

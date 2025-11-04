#pragma once

#include "base.hpp"
#include <memory>

namespace auroraml {
namespace discriminant_analysis {

/**
 * QuadraticDiscriminantAnalysis - Quadratic Discriminant Analysis
 * 
 * Similar to scikit-learn's QuadraticDiscriminantAnalysis, a classifier
 * with quadratic decision boundaries.
 */
class QuadraticDiscriminantAnalysis : public Estimator, public Classifier {
private:
    double regularization_;
    bool fitted_;
    int n_features_;
    int n_classes_;
    VectorXi classes_;
    std::vector<VectorXd> means_;
    std::vector<MatrixXd> covariances_;
    VectorXd priors_;

public:
    QuadraticDiscriminantAnalysis(double regularization = 0.0);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

} // namespace discriminant_analysis
} // namespace auroraml


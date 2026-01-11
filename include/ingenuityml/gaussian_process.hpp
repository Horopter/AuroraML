#pragma once

#include "base.hpp"
#include <vector>

namespace ingenuityml {
namespace gaussian_process {

class GaussianProcessRegressor : public Estimator, public Regressor {
private:
    double length_scale_;
    double alpha_;
    bool normalize_y_;
    bool fitted_;
    MatrixXd X_train_;
    VectorXd y_train_;
    double y_mean_;
    VectorXd alpha_vec_;

public:
    GaussianProcessRegressor(double length_scale = 1.0, double alpha = 1e-10, bool normalize_y = true);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    MatrixXd compute_kernel(const MatrixXd& A, const MatrixXd& B) const;
};

class GaussianProcessClassifier : public Estimator, public Classifier {
private:
    double length_scale_;
    double alpha_;
    bool fitted_;
    VectorXi classes_;
    std::vector<GaussianProcessRegressor> regressors_;

public:
    GaussianProcessClassifier(double length_scale = 1.0, double alpha = 1e-10);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

} // namespace gaussian_process
} // namespace ingenuityml

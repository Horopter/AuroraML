#pragma once

#include "base.hpp"

namespace ingenuityml {
namespace random_projection {

class GaussianRandomProjection : public Estimator, public Transformer {
private:
    int n_components_;
    int random_state_;
    bool fitted_;
    int n_features_;
    MatrixXd components_;

public:
    GaussianRandomProjection(int n_components = -1, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const { return components_; }
};

class SparseRandomProjection : public Estimator, public Transformer {
private:
    int n_components_;
    double density_;
    int random_state_;
    bool fitted_;
    int n_features_;
    MatrixXd components_;

public:
    SparseRandomProjection(int n_components = -1, double density = -1.0, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const { return components_; }
};

} // namespace random_projection
} // namespace ingenuityml

#pragma once

#include "base.hpp"

namespace ingenuityml {
namespace cross_decomposition {

class PLSCanonical : public Estimator, public Transformer {
private:
    int n_components_;
    bool scale_;
    int max_iter_;
    double tol_;
    bool fitted_;
    int n_features_;
    int n_targets_;
    VectorXd x_mean_;
    VectorXd y_mean_;
    VectorXd x_std_;
    VectorXd y_std_;
    MatrixXd x_weights_;
    MatrixXd y_weights_;
    MatrixXd x_loadings_;
    MatrixXd y_loadings_;
    MatrixXd x_scores_;
    MatrixXd y_scores_;
    MatrixXd x_rotations_;
    MatrixXd y_rotations_;

public:
    PLSCanonical(int n_components = 2, bool scale = true, int max_iter = 500, double tol = 1e-6);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    Estimator& fit(const MatrixXd& X, const MatrixXd& Y);
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd transform_y(const MatrixXd& Y) const;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd fit_transform(const MatrixXd& X, const MatrixXd& Y);
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& x_weights() const { return x_weights_; }
    const MatrixXd& y_weights() const { return y_weights_; }
    const MatrixXd& x_loadings() const { return x_loadings_; }
    const MatrixXd& y_loadings() const { return y_loadings_; }
    const MatrixXd& x_scores() const { return x_scores_; }
    const MatrixXd& y_scores() const { return y_scores_; }
};

class PLSRegression : public Estimator, public Regressor, public Transformer {
private:
    int n_components_;
    bool scale_;
    int max_iter_;
    double tol_;
    bool fitted_;
    int n_features_;
    int n_targets_;
    VectorXd x_mean_;
    VectorXd y_mean_;
    VectorXd x_std_;
    VectorXd y_std_;
    MatrixXd x_weights_;
    MatrixXd y_weights_;
    MatrixXd x_loadings_;
    MatrixXd y_loadings_;
    MatrixXd x_scores_;
    MatrixXd y_scores_;
    MatrixXd x_rotations_;
    MatrixXd coef_;
    VectorXd intercept_;

public:
    PLSRegression(int n_components = 2, bool scale = true, int max_iter = 500, double tol = 1e-6);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    Estimator& fit(const MatrixXd& X, const MatrixXd& Y);
    VectorXd predict(const MatrixXd& X) const override;
    MatrixXd predict_multi(const MatrixXd& X) const;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd transform_y(const MatrixXd& Y) const;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd fit_transform(const MatrixXd& X, const MatrixXd& Y);
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& x_weights() const { return x_weights_; }
    const MatrixXd& y_weights() const { return y_weights_; }
    const MatrixXd& x_loadings() const { return x_loadings_; }
    const MatrixXd& y_loadings() const { return y_loadings_; }
    const MatrixXd& x_scores() const { return x_scores_; }
    const MatrixXd& y_scores() const { return y_scores_; }
    const MatrixXd& coef() const { return coef_; }
    const VectorXd& intercept() const { return intercept_; }
};

class CCA : public Estimator, public Transformer {
private:
    int n_components_;
    bool scale_;
    int max_iter_;
    double tol_;
    bool fitted_;
    int n_features_;
    int n_targets_;
    VectorXd x_mean_;
    VectorXd y_mean_;
    VectorXd x_std_;
    VectorXd y_std_;
    MatrixXd x_weights_;
    MatrixXd y_weights_;
    MatrixXd x_scores_;
    MatrixXd y_scores_;
    VectorXd correlations_;

public:
    CCA(int n_components = 2, bool scale = true, int max_iter = 500, double tol = 1e-6);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    Estimator& fit(const MatrixXd& X, const MatrixXd& Y);
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd transform_y(const MatrixXd& Y) const;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd fit_transform(const MatrixXd& X, const MatrixXd& Y);
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& x_weights() const { return x_weights_; }
    const MatrixXd& y_weights() const { return y_weights_; }
    const MatrixXd& x_scores() const { return x_scores_; }
    const MatrixXd& y_scores() const { return y_scores_; }
    const VectorXd& correlations() const { return correlations_; }
};

class PLSSVD : public Estimator, public Transformer {
private:
    int n_components_;
    bool scale_;
    bool fitted_;
    int n_features_;
    int n_targets_;
    VectorXd x_mean_;
    VectorXd y_mean_;
    VectorXd x_std_;
    VectorXd y_std_;
    MatrixXd x_weights_;
    MatrixXd y_weights_;
    MatrixXd x_scores_;
    MatrixXd y_scores_;

public:
    PLSSVD(int n_components = 2, bool scale = true);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    Estimator& fit(const MatrixXd& X, const MatrixXd& Y);
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd transform_y(const MatrixXd& Y) const;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd fit_transform(const MatrixXd& X, const MatrixXd& Y);
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& x_weights() const { return x_weights_; }
    const MatrixXd& y_weights() const { return y_weights_; }
    const MatrixXd& x_scores() const { return x_scores_; }
    const MatrixXd& y_scores() const { return y_scores_; }
};

} // namespace cross_decomposition
} // namespace ingenuityml

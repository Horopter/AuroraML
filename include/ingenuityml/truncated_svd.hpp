#pragma once

#include "base.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace decomposition {

// TruncatedSVD for dense matrices using Eigen BDCSVD (thin V)
class TruncatedSVD : public Estimator, public Transformer {
private:
    int n_components_;
    bool fitted_ = false;
    MatrixXd components_; // right singular vectors (V)
    VectorXd singular_values_;
    VectorXd explained_variance_;

public:
    TruncatedSVD(int n_components) : n_components_(n_components) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override { return {{"n_components", std::to_string(n_components_)}}; }
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const { 
        if (!fitted_) throw std::runtime_error("TruncatedSVD must be fitted before accessing components.");
        return components_; 
    }
    const VectorXd& singular_values() const { 
        if (!fitted_) throw std::runtime_error("TruncatedSVD must be fitted before accessing singular values.");
        return singular_values_; 
    }
    const VectorXd& explained_variance() const { 
        if (!fitted_) throw std::runtime_error("TruncatedSVD must be fitted before accessing explained variance.");
        return explained_variance_; 
    }
};

} // namespace decomposition
} // namespace ingenuityml



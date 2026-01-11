#pragma once

#include "base.hpp"
#include <string>

namespace ingenuityml {
namespace density {

class KernelDensity : public Estimator {
private:
    double bandwidth_;
    std::string kernel_;
    bool fitted_;
    MatrixXd X_train_;

public:
    KernelDensity(double bandwidth = 1.0, const std::string& kernel = "gaussian");

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd score_samples(const MatrixXd& X) const;
    double score(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

} // namespace density
} // namespace ingenuityml

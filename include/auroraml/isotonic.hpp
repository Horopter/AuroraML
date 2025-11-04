#pragma once

#include "base.hpp"

namespace auroraml {
namespace isotonic {

/**
 * IsotonicRegression - Isotonic regression
 * 
 * Similar to scikit-learn's IsotonicRegression, fits a non-decreasing
 * function to the data.
 */
class IsotonicRegression : public Estimator, public Regressor {
private:
    bool increasing_;
    bool fitted_;
    VectorXd X_min_;
    VectorXd X_max_;
    VectorXd y_min_;
    VectorXd y_max_;
    std::vector<std::pair<double, double>> thresholds_;

public:
    IsotonicRegression(bool increasing = true);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    /**
     * Transform input
     */
    VectorXd transform(const VectorXd& X) const;
};

} // namespace isotonic
} // namespace auroraml


#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <algorithm>

namespace auroraml {
namespace preprocessing {

/**
 * MaxAbsScaler - Scale features by maximum absolute value
 * 
 * Similar to scikit-learn's MaxAbsScaler, scales features to [-1, 1]
 * by dividing by the maximum absolute value.
 */
class MaxAbsScaler : public Estimator, public Transformer {
private:
    VectorXd max_abs_;
    bool fitted_;

public:
    MaxAbsScaler();
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXd max_abs() const { return max_abs_; }
};

/**
 * Binarizer - Binarize features
 * 
 * Similar to scikit-learn's Binarizer, sets values above threshold to 1, below to 0.
 */
class Binarizer : public Estimator, public Transformer {
private:
    double threshold_;
    bool fitted_;

public:
    Binarizer(double threshold = 0.0);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

} // namespace preprocessing
} // namespace auroraml


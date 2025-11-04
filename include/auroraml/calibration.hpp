#pragma once

#include "base.hpp"
#include <memory>

namespace auroraml {
namespace calibration {

/**
 * CalibratedClassifierCV - Probability calibration using cross-validation
 * 
 * Similar to scikit-learn's CalibratedClassifierCV, calibrates classifier
 * probabilities using cross-validation.
 */
class CalibratedClassifierCV : public Estimator, public Classifier {
private:
    std::shared_ptr<Classifier> base_estimator_;
    std::string method_;
    int cv_;
    bool fitted_;
    std::vector<std::shared_ptr<Classifier>> calibrated_estimators_;
    VectorXi classes_;

public:
    /**
     * Constructor
     * @param base_estimator The base classifier to calibrate
     * @param method Calibration method ("sigmoid" or "isotonic")
     * @param cv Number of cross-validation folds
     */
    CalibratedClassifierCV(
        std::shared_ptr<Classifier> base_estimator,
        const std::string& method = "sigmoid",
        int cv = 3
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

} // namespace calibration
} // namespace auroraml


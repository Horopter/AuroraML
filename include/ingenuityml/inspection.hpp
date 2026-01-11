#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <memory>

namespace ingenuityml {
namespace inspection {

/**
 * PermutationImportance - Compute permutation importance
 * 
 * Computes feature importance by permuting each feature and measuring
 * the decrease in model performance.
 */
class PermutationImportance {
private:
    std::shared_ptr<Estimator> estimator_;
    std::string scoring_;
    int n_repeats_;
    int random_state_;
    std::vector<double> importances_;
    std::vector<std::vector<double>> importances_std_;

public:
    /**
     * Constructor
     * @param estimator The estimator to evaluate
     * @param scoring Scoring function name ("accuracy", "r2", etc.)
     * @param n_repeats Number of times to permute each feature
     * @param random_state Random seed
     */
    PermutationImportance(
        std::shared_ptr<Estimator> estimator,
        const std::string& scoring = "accuracy",
        int n_repeats = 5,
        int random_state = -1
    );
    
    /**
     * Fit and compute importance
     */
    void fit(const MatrixXd& X, const VectorXd& y);
    
    /**
     * Get feature importances
     */
    std::vector<double> feature_importances() const { return importances_; }
    
    /**
     * Get standard deviations of importances
     */
    std::vector<std::vector<double>> importances_std() const { return importances_std_; }
};

/**
 * PartialDependence - Compute partial dependence plots
 * 
 * Computes partial dependence of target variable on a set of features.
 */
class PartialDependence {
private:
    std::shared_ptr<Predictor> estimator_;
    std::vector<int> features_;
    MatrixXd grid_;
    VectorXd partial_dependence_;

public:
    /**
     * Constructor
     * @param estimator The predictor to evaluate
     * @param features Feature indices to compute partial dependence for
     */
    PartialDependence(
        std::shared_ptr<Predictor> estimator,
        const std::vector<int>& features
    );
    
    /**
     * Compute partial dependence
     */
    void compute(const MatrixXd& X);
    
    /**
     * Get grid values
     */
    MatrixXd grid() const { return grid_; }
    
    /**
     * Get partial dependence values
     */
    VectorXd partial_dependence() const { return partial_dependence_; }
};

} // namespace inspection
} // namespace ingenuityml


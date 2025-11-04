#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

namespace auroraml {
namespace feature_selection {

/**
 * VarianceThreshold - Remove features with low variance
 * 
 * Removes features whose variance is below a threshold.
 */
class VarianceThreshold : public Estimator, public Transformer {
private:
    double threshold_;
    std::vector<int> selected_features_;
    bool fitted_;

public:
    /**
     * Constructor
     * @param threshold Features with variance below this threshold will be removed
     */
    VarianceThreshold(double threshold = 0.0);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    /**
     * Get indices of selected features
     */
    std::vector<int> get_support() const;
};

/**
 * SelectKBest - Select K best features based on a scoring function
 * 
 * Selects the K best features according to a scoring function.
 */
class SelectKBest : public Estimator, public Transformer {
private:
    int k_;
    std::function<double(const VectorXd&, const VectorXd&)> score_func_;
    std::vector<int> selected_features_;
    std::vector<double> scores_;
    bool fitted_;

public:
    /**
     * Constructor
     * @param score_func Scoring function that takes (X_feature, y) and returns a score
     * @param k Number of top features to select
     */
    SelectKBest(std::function<double(const VectorXd&, const VectorXd&)> score_func, int k = 10);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    /**
     * Get indices of selected features
     */
    std::vector<int> get_support() const;
    
    /**
     * Get scores for all features
     */
    std::vector<double> scores() const { return scores_; }
};

/**
 * SelectPercentile - Select features based on percentile of scores
 * 
 * Selects features based on a percentile of the highest scores.
 */
class SelectPercentile : public Estimator, public Transformer {
private:
    int percentile_;
    std::function<double(const VectorXd&, const VectorXd&)> score_func_;
    std::vector<int> selected_features_;
    std::vector<double> scores_;
    bool fitted_;

public:
    /**
     * Constructor
     * @param score_func Scoring function that takes (X_feature, y) and returns a score
     * @param percentile Percentile of features to keep (0-100)
     */
    SelectPercentile(std::function<double(const VectorXd&, const VectorXd&)> score_func, int percentile = 10);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    /**
     * Get indices of selected features
     */
    std::vector<int> get_support() const;
    
    /**
     * Get scores for all features
     */
    std::vector<double> scores() const { return scores_; }
};

// Common scoring functions
namespace scores {
    /**
     * f_classif - F-value between label/feature for classification
     * Simplified version - returns correlation-based score
     */
    double f_classif(const VectorXd& X_feature, const VectorXd& y);
    
    /**
     * f_regression - F-value between label/feature for regression
     * Simplified version - returns R² score
     */
    double f_regression(const VectorXd& X_feature, const VectorXd& y);
    
    /**
     * mutual_info_classif - Mutual information for classification
     * Simplified version - returns correlation-based score
     */
    double mutual_info_classif(const VectorXd& X_feature, const VectorXd& y);
    
    /**
     * mutual_info_regression - Mutual information for regression
     * Simplified version - returns correlation-based score
     */
    double mutual_info_regression(const VectorXd& X_feature, const VectorXd& y);
    
    /**
     * chi2 - Chi-squared statistic
     * For classification tasks
     */
    double chi2(const VectorXd& X_feature, const VectorXi& y);
}

} // namespace feature_selection
} // namespace auroraml


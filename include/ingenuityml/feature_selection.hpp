#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

namespace ingenuityml {
namespace model_selection {
class BaseCrossValidator;
}

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

/**
 * SelectFpr - Select features based on false positive rate
 */
class SelectFpr : public Estimator, public Transformer {
private:
    double alpha_;
    std::function<double(const VectorXd&, const VectorXd&)> score_func_;
    std::vector<int> selected_features_;
    std::vector<double> scores_;
    bool fitted_;

public:
    SelectFpr(std::function<double(const VectorXd&, const VectorXd&)> score_func, double alpha = 0.05);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    std::vector<int> get_support() const;
    std::vector<double> scores() const { return scores_; }
};

/**
 * SelectFdr - Select features based on false discovery rate
 */
class SelectFdr : public Estimator, public Transformer {
private:
    double alpha_;
    std::function<double(const VectorXd&, const VectorXd&)> score_func_;
    std::vector<int> selected_features_;
    std::vector<double> scores_;
    bool fitted_;

public:
    SelectFdr(std::function<double(const VectorXd&, const VectorXd&)> score_func, double alpha = 0.05);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    std::vector<int> get_support() const;
    std::vector<double> scores() const { return scores_; }
};

/**
 * SelectFwe - Select features based on family-wise error rate
 */
class SelectFwe : public Estimator, public Transformer {
private:
    double alpha_;
    std::function<double(const VectorXd&, const VectorXd&)> score_func_;
    std::vector<int> selected_features_;
    std::vector<double> scores_;
    bool fitted_;

public:
    SelectFwe(std::function<double(const VectorXd&, const VectorXd&)> score_func, double alpha = 0.05);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    std::vector<int> get_support() const;
    std::vector<double> scores() const { return scores_; }
};

/**
 * GenericUnivariateSelect - Univariate feature selection with configurable mode
 */
class GenericUnivariateSelect : public Estimator, public Transformer {
private:
    std::string mode_;
    double param_;
    std::function<double(const VectorXd&, const VectorXd&)> score_func_;
    std::vector<int> selected_features_;
    std::vector<double> scores_;
    bool fitted_;

public:
    GenericUnivariateSelect(std::function<double(const VectorXd&, const VectorXd&)> score_func,
                            const std::string& mode = "percentile", double param = 10.0);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    std::vector<int> get_support() const;
    std::vector<double> scores() const { return scores_; }
};

/**
 * SelectFromModel - Select features based on model importances
 */
class SelectFromModel : public Estimator, public Transformer {
private:
    Estimator& estimator_;
    double threshold_;
    int max_features_;
    std::vector<int> selected_features_;
    std::vector<double> importances_;
    bool fitted_;

public:
    SelectFromModel(Estimator& estimator, double threshold = 0.0, int max_features = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    std::vector<int> get_support() const;
    std::vector<double> importances() const { return importances_; }
};

/**
 * RFE - Recursive feature elimination
 */
class RFE : public Estimator, public Transformer {
private:
    Estimator& estimator_;
    int n_features_to_select_;
    int step_;
    std::vector<int> selected_features_;
    bool fitted_;

public:
    RFE(Estimator& estimator, int n_features_to_select = -1, int step = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    std::vector<int> get_support() const;
};

/**
 * RFECV - Recursive feature elimination with cross-validation
 */
class RFECV : public Estimator, public Transformer {
private:
    Estimator& estimator_;
    model_selection::BaseCrossValidator& cv_;
    std::string scoring_;
    int step_;
    int min_features_to_select_;
    std::vector<int> selected_features_;
    bool fitted_;

public:
    RFECV(Estimator& estimator, model_selection::BaseCrossValidator& cv,
          int step = 1, const std::string& scoring = "accuracy", int min_features_to_select = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    std::vector<int> get_support() const;
};

/**
 * SequentialFeatureSelector - Forward/backward feature selection
 */
class SequentialFeatureSelector : public Estimator, public Transformer {
private:
    Estimator& estimator_;
    model_selection::BaseCrossValidator& cv_;
    std::string scoring_;
    int n_features_to_select_;
    std::string direction_;
    std::vector<int> selected_features_;
    bool fitted_;

public:
    SequentialFeatureSelector(Estimator& estimator, model_selection::BaseCrossValidator& cv,
                              int n_features_to_select = -1, const std::string& direction = "forward",
                              const std::string& scoring = "accuracy");

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    std::vector<int> get_support() const;
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
} // namespace ingenuityml

#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>

namespace auroraml {
namespace preprocessing {

/**
 * StandardScaler - Standardize features by removing the mean and scaling to unit variance
 */
class StandardScaler : public Estimator, public Transformer {
private:
    VectorXd mean_;
    VectorXd scale_;
    bool fitted_;
    bool with_mean_;
    bool with_std_;

public:
    StandardScaler(bool with_mean = true, bool with_std = true);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd mean() const { return mean_; }
    VectorXd scale() const { return scale_; }
};

/**
 * MinMaxScaler - Transform features by scaling each feature to a given range
 */
class MinMaxScaler : public Estimator, public Transformer {
private:
    VectorXd data_min_;
    VectorXd data_max_;
    VectorXd scale_;
    VectorXd min_;
    bool fitted_;
    double feature_range_min_;
    double feature_range_max_;

public:
    MinMaxScaler(double feature_range_min = 0.0, double feature_range_max = 1.0);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    VectorXd data_min() const { return data_min_; }
    VectorXd data_max() const { return data_max_; }
    VectorXd scale() const { return scale_; }
    VectorXd min() const { return min_; }
};

/**
 * LabelEncoder - Encode target labels with value between 0 and n_classes-1
 */
class LabelEncoder : public Estimator, public Transformer {
private:
    std::map<double, int> label_to_index_;
    std::map<int, double> index_to_label_;
    bool fitted_;
    int n_classes_;

public:
    LabelEncoder();
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi transform(const VectorXd& y) const;
    VectorXd inverse_transform(const VectorXi& y) const;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    int n_classes() const { return n_classes_; }
};

} // namespace preprocessing
} // namespace cxml

namespace auroraml {
namespace preprocessing {

/**
 * RobustScaler - Scale features using statistics that are robust to outliers.
 * Uses median and IQR (interquartile range).
 */
class RobustScaler : public Estimator, public Transformer {
private:
    VectorXd center_;   // medians
    VectorXd scale_;    // IQR
    bool with_centering_;
    bool with_scaling_;
    bool fitted_ = false;

public:
    RobustScaler(bool with_centering = true, bool with_scaling = true)
        : with_centering_(with_centering), with_scaling_(with_scaling) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    static double median(VectorXd v);
    static double quantile(VectorXd v, double q);
};

} // namespace preprocessing
} // namespace cxml

namespace auroraml {
namespace preprocessing {

/**
 * OneHotEncoder - Encode categorical features as a one-hot numeric array.
 * Assumes input is numeric but treated as categories per column.
 */
class OneHotEncoder : public Estimator, public Transformer {
private:
    // Per column, sorted unique categories
    std::vector<std::vector<double>> categories_;
    // Column start offsets in the transformed space
    std::vector<int> col_offsets_;
    int output_dim_ = 0;
    bool fitted_ = false;

public:
    OneHotEncoder() = default;

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override { return {}; }
    Estimator& set_params(const Params& params) override { (void)params; return *this; }
    bool is_fitted() const override { return fitted_; }

    const std::vector<std::vector<double>>& categories() const { return categories_; }
};

/**
 * OrdinalEncoder - Encode categorical features as an integer array
 */
class OrdinalEncoder : public Estimator, public Transformer {
private:
    std::vector<std::vector<double>> categories_;
    std::vector<std::map<double, int>> category_to_int_;
    bool fitted_;

public:
    OrdinalEncoder() = default;

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override { return {}; }
    Estimator& set_params(const Params& params) override { (void)params; return *this; }
    bool is_fitted() const override { return fitted_; }

    const std::vector<std::vector<double>>& categories() const { return categories_; }
};

/**
 * Normalizer - Normalize samples individually to unit norm
 */
class Normalizer : public Estimator, public Transformer {
private:
    std::string norm_;
    bool fitted_;

public:
    Normalizer(const std::string& norm = "l2");
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

/**
 * PolynomialFeatures - Generate polynomial and interaction features
 */
class PolynomialFeatures : public Estimator, public Transformer {
private:
    int degree_;
    bool interaction_only_;
    bool include_bias_;
    bool fitted_;
    int n_features_;
    int n_output_features_;

public:
    PolynomialFeatures(int degree = 2, bool interaction_only = false, bool include_bias = true);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    int n_input_features() const { return n_features_; }
    int n_output_features() const { return n_output_features_; }
};

/**
 * SimpleImputer - Imputation transformer for completing missing values
 */
class SimpleImputer : public Estimator, public Transformer {
private:
    std::string strategy_;
    double fill_value_;
    bool fitted_;
    VectorXd statistics_;

public:
    SimpleImputer(const std::string& strategy = "mean", double fill_value = 0.0);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    VectorXd statistics() const { return statistics_; }
};

} // namespace preprocessing
} // namespace cxml
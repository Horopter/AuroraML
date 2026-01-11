#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <map>

namespace ingenuityml {
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

/**
 * LabelBinarizer - Binarize labels for multiclass and binary targets.
 */
class LabelBinarizer : public Estimator, public Transformer {
private:
    int neg_label_;
    int pos_label_;
    bool fitted_;
    VectorXd classes_;
    std::map<double, int> class_to_index_;

public:
    LabelBinarizer(int neg_label = 0, int pos_label = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXd classes() const { return classes_; }
};

/**
 * MultiLabelBinarizer - Binarize multilabel outputs.
 */
class MultiLabelBinarizer : public Estimator, public Transformer {
private:
    bool fitted_;
    VectorXd classes_;
    std::map<double, int> class_to_index_;

public:
    MultiLabelBinarizer();

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXd classes() const { return classes_; }
};

/**
 * KBinsDiscretizer - Discretize continuous features into bins.
 */
class KBinsDiscretizer : public Estimator, public Transformer {
private:
    int n_bins_;
    std::string encode_;
    std::string strategy_;
    bool fitted_;
    int n_features_;
    int output_dim_;
    std::vector<VectorXd> bin_edges_;

public:
    KBinsDiscretizer(int n_bins = 5, const std::string& encode = "onehot",
                     const std::string& strategy = "uniform");

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    int n_bins() const { return n_bins_; }
    int output_dim() const { return output_dim_; }
};

/**
 * QuantileTransformer - Transform features using quantiles.
 */
class QuantileTransformer : public Estimator, public Transformer {
private:
    int n_quantiles_;
    std::string output_distribution_;
    bool fitted_;
    int n_features_;
    std::vector<VectorXd> quantiles_;

public:
    QuantileTransformer(int n_quantiles = 1000, const std::string& output_distribution = "uniform");

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

/**
 * PowerTransformer - Apply power transform (Yeo-Johnson or Box-Cox).
 */
class PowerTransformer : public Estimator, public Transformer {
private:
    std::string method_;
    bool standardize_;
    bool fitted_;
    VectorXd lambdas_;
    VectorXd mean_;
    VectorXd scale_;

public:
    PowerTransformer(const std::string& method = "yeo-johnson", bool standardize = true);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXd lambdas() const { return lambdas_; }
};

/**
 * FunctionTransformer - Apply a simple elementwise function.
 */
class FunctionTransformer : public Estimator, public Transformer {
private:
    std::string func_;
    std::string inverse_func_;
    bool validate_;
    bool fitted_;

public:
    FunctionTransformer(const std::string& func = "identity",
                        const std::string& inverse_func = "identity",
                        bool validate = true);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

/**
 * SplineTransformer - B-spline basis expansion.
 */
class SplineTransformer : public Estimator, public Transformer {
private:
    int n_knots_;
    int degree_;
    bool include_bias_;
    bool fitted_;
    int n_features_;
    int n_splines_;
    int output_dim_;
    std::vector<VectorXd> knot_vectors_;
    std::vector<VectorXd> greville_;

public:
    SplineTransformer(int n_knots = 5, int degree = 3, bool include_bias = true);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    int output_dim() const { return output_dim_; }
};

} // namespace preprocessing
} // namespace ingenuityml

#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace auroraml {
namespace compose {

/**
 * ColumnTransformer - Apply different transformers to different columns
 * 
 * Similar to scikit-learn's ColumnTransformer, this class applies different
 * transformers to different subsets of columns in the input data.
 */
class ColumnTransformer : public Estimator, public Transformer {
private:
    std::vector<std::tuple<std::string, std::shared_ptr<Transformer>, std::vector<int>>> transformers_;
    bool fitted_;
    std::vector<int> remaining_columns_; // Columns not specified in transformers
    bool drop_remaining_;

public:
    /**
     * Constructor
     * @param transformers Vector of (name, transformer, column_indices) tuples
     * @param remainder How to handle remaining columns: "drop", "passthrough", or a Transformer
     * @param sparse_threshold Not used in this implementation (always dense)
     */
    ColumnTransformer(
        const std::vector<std::tuple<std::string, std::shared_ptr<Transformer>, std::vector<int>>>& transformers,
        const std::string& remainder = "drop",
        double sparse_threshold = 0.3
    );
    
    /**
     * Fit all transformers
     * @param X Training data
     * @param y Target values (optional)
     */
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    
    /**
     * Transform data using the fitted transformers
     */
    MatrixXd transform(const MatrixXd& X) const override;
    
    /**
     * Inverse transform (not fully supported, returns transformed data)
     */
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    
    /**
     * Fit and transform in one step
     */
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    
    /**
     * Get parameters
     */
    Params get_params() const override;
    
    /**
     * Set parameters
     */
    Estimator& set_params(const Params& params) override;
    
    /**
     * Check if ColumnTransformer is fitted
     */
    bool is_fitted() const override { return fitted_; }
    
    /**
     * Get transformer by name
     */
    std::shared_ptr<Transformer> get_transformer(const std::string& name) const;
    
    /**
     * Get all transformer names
     */
    std::vector<std::string> get_transformer_names() const;

private:
    /**
     * Extract columns from X
     */
    MatrixXd extract_columns(const MatrixXd& X, const std::vector<int>& column_indices) const;
};

/**
 * TransformedTargetRegressor - Transform target before fitting regressor
 * 
 * Similar to scikit-learn's TransformedTargetRegressor, this class applies
 * a transformer to the target variable before fitting a regressor, and
 * inverse-transforms the predictions.
 */
class TransformedTargetRegressor : public Estimator, public Regressor {
private:
    std::shared_ptr<Regressor> regressor_;
    std::shared_ptr<Transformer> transformer_;
    bool fitted_;

public:
    /**
     * Constructor
     * @param regressor The regressor to use
     * @param transformer The transformer to apply to target (optional)
     */
    TransformedTargetRegressor(
        std::shared_ptr<Regressor> regressor,
        std::shared_ptr<Transformer> transformer = nullptr
    );
    
    /**
     * Fit the regressor with transformed target
     * @param X Training data
     * @param y Target values (will be transformed)
     */
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    
    /**
     * Predict and inverse-transform the predictions
     */
    VectorXd predict(const MatrixXd& X) const override;
    
    /**
     * Get parameters
     */
    Params get_params() const override;
    
    /**
     * Set parameters
     */
    Estimator& set_params(const Params& params) override;
    
    /**
     * Check if TransformedTargetRegressor is fitted
     */
    bool is_fitted() const override { return fitted_; }
    
    /**
     * Get the underlying regressor
     */
    std::shared_ptr<Regressor> regressor() const { return regressor_; }
    
    /**
     * Get the transformer
     */
    std::shared_ptr<Transformer> transformer() const { return transformer_; }
};

} // namespace compose
} // namespace auroraml

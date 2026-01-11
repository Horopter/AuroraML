#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <memory>

namespace ingenuityml {
namespace pipeline {

/**
 * Pipeline - Chain multiple estimators/transformers
 * 
 * Similar to scikit-learn's Pipeline, this class chains multiple
 * estimators/transformers together, where the output of each step
 * becomes the input to the next step.
 */
class Pipeline : public Estimator {
private:
    std::vector<std::pair<std::string, std::shared_ptr<Estimator>>> steps_;
    bool fitted_;

public:
    /**
     * Constructor
     * @param steps Vector of (name, estimator) pairs
     */
    Pipeline(const std::vector<std::pair<std::string, std::shared_ptr<Estimator>>>& steps);
    
    /**
     * Fit all steps in the pipeline
     * @param X Training data
     * @param y Target values (optional, only needed for final estimator)
     */
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    
    /**
     * Transform data through the pipeline
     * If the last step is a transformer, returns transformed data
     * If the last step is a predictor, returns predictions
     */
    MatrixXd transform(const MatrixXd& X) const;
    
    /**
     * Predict using the pipeline
     * The last step must be a Predictor
     */
    VectorXd predict(const MatrixXd& X) const;
    
    /**
     * Predict classes (for classifiers)
     */
    VectorXi predict_classes(const MatrixXd& X) const;
    
    /**
     * Predict probabilities (for classifiers)
     */
    MatrixXd predict_proba(const MatrixXd& X) const;
    
    /**
     * Fit and transform in one step
     */
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y);
    
    /**
     * Get parameters
     */
    Params get_params() const override;
    
    /**
     * Set parameters
     */
    Estimator& set_params(const Params& params) override;
    
    /**
     * Check if pipeline is fitted
     */
    bool is_fitted() const override { return fitted_; }
    
    /**
     * Get a specific step by name
     */
    std::shared_ptr<Estimator> get_step(const std::string& name) const;
    
    /**
     * Get all step names
     */
    std::vector<std::string> get_step_names() const;
};

/**
 * FeatureUnion - Combine multiple transformers
 * 
 * Similar to scikit-learn's FeatureUnion, this class combines
 * multiple transformers by concatenating their outputs horizontally.
 */
class FeatureUnion : public Estimator, public Transformer {
private:
    std::vector<std::pair<std::string, std::shared_ptr<Transformer>>> transformers_;
    bool fitted_;

public:
    /**
     * Constructor
     * @param transformers Vector of (name, transformer) pairs
     */
    FeatureUnion(const std::vector<std::pair<std::string, std::shared_ptr<Transformer>>>& transformers);
    
    /**
     * Fit all transformers
     * @param X Training data
     * @param y Target values (optional)
     */
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    
    /**
     * Transform data through all transformers and concatenate results
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
     * Check if FeatureUnion is fitted
     */
    bool is_fitted() const override { return fitted_; }
    
    /**
     * Get a specific transformer by name
     */
    std::shared_ptr<Transformer> get_transformer(const std::string& name) const;
    
    /**
     * Get all transformer names
     */
    std::vector<std::string> get_transformer_names() const;
};

} // namespace pipeline
} // namespace ingenuityml


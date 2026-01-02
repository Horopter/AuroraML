#pragma once

#include "base.hpp"
#include "neighbors.hpp"
#include <vector>
#include <string>
#include <cmath>

namespace auroraml {
namespace impute {

/**
 * SimpleImputer - Basic imputation strategies
 * Note: This already exists in preprocessing, but we'll add it here for completeness
 */

/**
 * KNNImputer - Impute missing values using k-nearest neighbors
 * 
 * Similar to scikit-learn's KNNImputer, uses k-nearest neighbors to impute
 * missing values based on similar samples.
 */
class KNNImputer : public Estimator, public Transformer {
private:
    int n_neighbors_;
    std::string metric_;
    bool fitted_;
    MatrixXd X_fitted_;
    std::vector<bool> missing_mask_;

public:
    /**
     * Constructor
     * @param n_neighbors Number of neighbors to use
     * @param metric Distance metric ("euclidean", "manhattan", "minkowski")
     */
    KNNImputer(int n_neighbors = 5, const std::string& metric = "euclidean");
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    /**
     * Find k nearest neighbors for a sample
     */
    std::vector<int> find_neighbors(const VectorXd& sample, const MatrixXd& X, int k) const;
    
    /**
     * Calculate distance between two samples (handling missing values)
     */
    double distance(const VectorXd& a, const VectorXd& b) const;
    
    /**
     * Impute missing values for a single sample
     */
    VectorXd impute_sample(const VectorXd& sample, const MatrixXd& X, const std::vector<int>& neighbors) const;
};

/**
 * IterativeImputer - Multivariate imputation using iterative regression
 * 
 * Similar to scikit-learn's IterativeImputer, uses iterative regression
 * to impute missing values.
 */
class IterativeImputer : public Estimator, public Transformer {
private:
    int max_iter_;
    double tol_;
    int random_state_;
    bool fitted_;
    std::vector<std::shared_ptr<Regressor>> imputation_models_;
    MatrixXd X_fitted_;

public:
    /**
     * Constructor
     * @param max_iter Maximum number of iterations
     * @param tol Tolerance for convergence
     * @param random_state Random seed
     */
    IterativeImputer(int max_iter = 10, double tol = 1e-3, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    /**
     * Initialize missing values with column means
     */
    MatrixXd initialize_missing(const MatrixXd& X) const;
    
    /**
     * Check convergence
     */
    bool check_convergence(const MatrixXd& X_old, const MatrixXd& X_new) const;
};

/**
 * MissingIndicator - Indicator for missing values
 *
 * Similar to scikit-learn's MissingIndicator, generates binary features
 * indicating the presence of missing values in each selected column.
 */
class MissingIndicator : public Estimator, public Transformer {
private:
    std::string features_;
    bool fitted_;
    std::vector<int> features_indices_;

public:
    explicit MissingIndicator(const std::string& features = "missing-only");

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const std::vector<int>& features() const { return features_indices_; }
};

} // namespace impute
} // namespace auroraml

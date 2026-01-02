#pragma once

#include "base.hpp"
#include "tree.hpp"
#include "neighbors.hpp"
#include "covariance.hpp"
#include <vector>
#include <random>

namespace auroraml {
namespace outlier_detection {

/**
 * IsolationForest - Isolation Forest for outlier detection
 * 
 * Similar to scikit-learn's IsolationForest, uses random trees to isolate outliers.
 */
class IsolationForest : public Estimator {
private:
    int n_estimators_;
    int max_samples_;
    double contamination_;
    int random_state_;
    bool fitted_;
    std::vector<std::unique_ptr<tree::TreeNode>> trees_;

public:
    IsolationForest(
        int n_estimators = 100,
        int max_samples = -1,
        double contamination = 0.1,
        int random_state = -1
    );
    IsolationForest(int n_estimators, double contamination, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict(const MatrixXd& X) const;
    VectorXd decision_function(const MatrixXd& X) const;
    VectorXi fit_predict(const MatrixXd& X);
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    std::unique_ptr<tree::TreeNode> build_isolation_tree(const MatrixXd& X, const std::vector<int>& samples, int depth, int max_depth, std::mt19937& rng);
    double path_length(const tree::TreeNode* node, const VectorXd& sample, int depth) const;
};

/**
 * LocalOutlierFactor - Local Outlier Factor for outlier detection
 * 
 * Similar to scikit-learn's LocalOutlierFactor, uses local density to detect outliers.
 */
class LocalOutlierFactor : public Estimator {
private:
    int n_neighbors_;
    std::string metric_;
    double contamination_;
    bool fitted_;
    MatrixXd X_fitted_;
    std::vector<double> lrd_scores_;

public:
    LocalOutlierFactor(
        int n_neighbors = 20,
        const std::string& metric = "euclidean",
        double contamination = 0.1
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict(const MatrixXd& X) const;
    VectorXd decision_function(const MatrixXd& X) const;
    VectorXi fit_predict(const MatrixXd& X);
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    double local_reachability_density(const MatrixXd& X, int sample_idx, int n_neighbors) const;
    double reachability_distance(const MatrixXd& X, int a, int b, int k) const;
};

using EllipticEnvelope = covariance::EllipticEnvelope;

} // namespace outlier_detection
} // namespace auroraml

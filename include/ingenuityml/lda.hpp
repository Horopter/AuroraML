#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <map>
#include <vector>

namespace ingenuityml {
namespace decomposition {

/**
 * Linear Discriminant Analysis (LDA)
 * 
 * LDA is a dimensionality reduction technique that finds linear combinations
 * of features that best separate classes. It maximizes the ratio of between-class
 * variance to within-class variance.
 */
class LDA : public Estimator, public Transformer {
private:
    int n_components_;
    bool fitted_ = false;
    MatrixXd components_;           // transformation matrix
    VectorXd explained_variance_;   // explained variance ratios
    VectorXd mean_;                // overall mean
    std::vector<VectorXd> class_means_;  // per-class means
    std::vector<int> class_labels_;      // unique class labels
    std::map<int, int> label_to_index_;  // mapping from label to index

public:
    LDA(int n_components = -1) : n_components_(n_components) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override { return {{"n_components", std::to_string(n_components_)}}; }
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    // Accessor methods
    const MatrixXd& components() const { 
        if (!fitted_) throw std::runtime_error("LDA must be fitted before accessing components.");
        return components_; 
    }
    const VectorXd& explained_variance() const { 
        if (!fitted_) throw std::runtime_error("LDA must be fitted before accessing explained variance.");
        return explained_variance_; 
    }
    VectorXd explained_variance_ratio() const {
        if (!fitted_) throw std::runtime_error("LDA must be fitted before accessing explained variance ratio.");
        if (explained_variance_.size() == 0) return VectorXd();
        double sum = explained_variance_.sum();
        if (sum <= 0) return VectorXd::Zero(explained_variance_.size());
        VectorXd r = explained_variance_ / sum;
        return r;
    }
    const VectorXd& mean() const { 
        if (!fitted_) throw std::runtime_error("LDA must be fitted before accessing mean.");
        return mean_; 
    }
    const std::vector<VectorXd>& class_means() const {
        if (!fitted_) throw std::runtime_error("LDA must be fitted before accessing class means.");
        return class_means_;
    }
    const std::vector<int>& classes() const {
        if (!fitted_) throw std::runtime_error("LDA must be fitted before accessing classes.");
        return class_labels_;
    }
};

} // namespace decomposition
} // namespace ingenuityml

#pragma once

#include "base.hpp"
#include "extratree.hpp"
#include <vector>

namespace ingenuityml {
namespace ensemble {

/**
 * RandomTreesEmbedding - Unsupervised random tree embedding
 *
 * Similar to scikit-learn's RandomTreesEmbedding, fits random trees
 * and uses their predictions as a feature embedding.
 */
class RandomTreesEmbedding : public Estimator, public Transformer {
private:
    int n_estimators_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int max_features_;
    bool bootstrap_;
    int random_state_;
    int n_features_;
    bool fitted_ = false;
    std::vector<tree::ExtraTreeRegressor> trees_;

public:
    RandomTreesEmbedding(int n_estimators = 100, int max_depth = -1,
                         int min_samples_split = 2, int min_samples_leaf = 1,
                         int max_features = -1, bool bootstrap = false,
                         int random_state = -1)
        : n_estimators_(n_estimators), max_depth_(max_depth),
          min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf),
          max_features_(max_features), bootstrap_(bootstrap),
          random_state_(random_state), n_features_(0) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

} // namespace ensemble
} // namespace ingenuityml

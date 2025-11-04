#pragma once

#include "base.hpp"
#include "tree.hpp"
#include <vector>
#include <random>

namespace auroraml {
namespace tree {

/**
 * ExtraTreeClassifier - Extremely Randomized Tree Classifier
 * 
 * Similar to scikit-learn's ExtraTreeClassifier, uses random splits
 * at each node instead of finding the optimal split.
 */
class ExtraTreeClassifier : public Estimator, public Classifier {
private:
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int max_features_;
    int random_state_;
    bool fitted_;
    int n_features_;
    int n_classes_;
    VectorXi classes_;
    std::unique_ptr<TreeNode> root_;

public:
    ExtraTreeClassifier(
        int max_depth = -1,
        int min_samples_split = 2,
        int min_samples_leaf = 1,
        int max_features = -1,
        int random_state = -1
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }

private:
    std::unique_ptr<TreeNode> build_tree_recursive(const MatrixXd& X, const VectorXi& y,
                                                  const std::vector<int>& samples, int depth,
                                                  std::mt19937& rng);
    std::pair<int, double> find_random_split(const MatrixXd& X, const VectorXi& y,
                                            const std::vector<int>& samples, std::mt19937& rng);
};

/**
 * ExtraTreeRegressor - Extremely Randomized Tree Regressor
 */
class ExtraTreeRegressor : public Estimator, public Regressor {
private:
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int max_features_;
    int random_state_;
    bool fitted_;
    int n_features_;
    std::unique_ptr<TreeNode> root_;

public:
    ExtraTreeRegressor(
        int max_depth = -1,
        int min_samples_split = 2,
        int min_samples_leaf = 1,
        int max_features = -1,
        int random_state = -1
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    std::unique_ptr<TreeNode> build_tree_recursive(const MatrixXd& X, const VectorXd& y,
                                                  const std::vector<int>& samples, int depth,
                                                  std::mt19937& rng);
    std::pair<int, double> find_random_split(const MatrixXd& X, const VectorXd& y,
                                            const std::vector<int>& samples, std::mt19937& rng);
};

} // namespace tree
} // namespace auroraml


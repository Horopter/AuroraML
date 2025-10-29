#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>
#include <algorithm>
#include <fstream>

namespace auroraml {
namespace tree {

/**
 * Decision Tree Node
 */
struct TreeNode {
    bool is_leaf;
    int feature_index;
    double threshold;
    double value;  // For regression or class probability
    int class_label;  // For classification
    std::map<int, int> class_counts;  // For classification probability calculation
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
    
    TreeNode() : is_leaf(false), feature_index(-1), threshold(0.0), 
                 value(0.0), class_label(-1) {}
};

/**
 * Decision Tree Classifier
 * 
 * Simplified CART implementation for classification
 */
class DecisionTreeClassifier : public Estimator, public Classifier {
private:
    std::unique_ptr<TreeNode> root_;
    bool fitted_;
    std::string criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    double min_impurity_decrease_;
    int n_classes_;
    int n_features_;
    std::vector<int> classes_;
    MatrixXd X_train_;

public:
    DecisionTreeClassifier(const std::string& criterion = "gini", int max_depth = -1,
                          int min_samples_split = 2, int min_samples_leaf = 1,
                          double min_impurity_decrease = 0.0);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    // Model persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    void build_tree(const MatrixXd& X, const VectorXi& y);
    std::unique_ptr<TreeNode> build_tree_recursive(const MatrixXd& X, const VectorXi& y,
                                                  const std::vector<int>& samples, int depth);
    std::pair<int, double> find_best_split(const MatrixXd& X, const VectorXi& y,
                                         const std::vector<int>& samples);
    double gini_impurity(const VectorXi& y, const std::vector<int>& samples);
    double entropy(const VectorXi& y, const std::vector<int>& samples);
    std::pair<std::vector<int>, std::vector<int>> split_samples(const MatrixXd& X,
                                                               const std::vector<int>& samples,
                                                               int feature, double threshold);
    std::unique_ptr<TreeNode> create_leaf(const VectorXi& y, const std::vector<int>& samples);
    int predict_single(const VectorXd& x) const;
    VectorXd predict_proba_single(const VectorXd& x) const;
};

/**
 * Decision Tree Regressor
 * 
 * Simplified CART implementation for regression
 */
class DecisionTreeRegressor : public Estimator, public Regressor {
private:
    std::unique_ptr<TreeNode> root_;
    bool fitted_;
    std::string criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    double min_impurity_decrease_;
    int n_features_;
    MatrixXd X_train_;

public:
    DecisionTreeRegressor(const std::string& criterion = "mse", int max_depth = -1,
                         int min_samples_split = 2, int min_samples_leaf = 1,
                         double min_impurity_decrease = 0.0);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    
    // Model persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    void build_tree(const MatrixXd& X, const VectorXd& y);
    std::unique_ptr<TreeNode> build_tree_recursive(const MatrixXd& X, const VectorXd& y,
                                                  const std::vector<int>& samples, int depth);
    std::pair<int, double> find_best_split(const MatrixXd& X, const VectorXd& y,
                                         const std::vector<int>& samples);
    double calculate_mse(const VectorXd& y, const std::vector<int>& samples);
    std::pair<std::vector<int>, std::vector<int>> split_samples(const MatrixXd& X,
                                                               const std::vector<int>& samples,
                                                               int feature, double threshold);
    std::unique_ptr<TreeNode> create_leaf(const VectorXd& y, const std::vector<int>& samples);
    double predict_single(const VectorXd& x) const;
};

} // namespace tree
} // namespace cxml

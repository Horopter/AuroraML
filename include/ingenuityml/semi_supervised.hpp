#pragma once

#include "base.hpp"
#include "neighbors.hpp"
#include <vector>

namespace ingenuityml {
namespace semi_supervised {

/**
 * LabelPropagation - Label Propagation for semi-supervised learning
 * 
 * Similar to scikit-learn's LabelPropagation, propagates labels to
 * unlabeled samples based on graph structure.
 */
class LabelPropagation : public Estimator, public Classifier {
private:
    double gamma_;
    int max_iter_;
    double tol_;
    std::string kernel_;
    bool fitted_;
    MatrixXd X_fitted_;
    VectorXi y_fitted_;
    VectorXi classes_;
    MatrixXd label_distributions_;

public:
    LabelPropagation(
        double gamma = 20.0,
        int max_iter = 30,
        double tol = 1e-3,
        const std::string& kernel = "rbf"
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
    MatrixXd build_affinity_matrix(const MatrixXd& X) const;
    void propagate_labels();
};

/**
 * LabelSpreading - Label Spreading for semi-supervised learning
 * 
 * Similar to scikit-learn's LabelSpreading, uses normalized graph Laplacian.
 */
class LabelSpreading : public Estimator, public Classifier {
private:
    double alpha_;
    double gamma_;
    int max_iter_;
    double tol_;
    std::string kernel_;
    bool fitted_;
    MatrixXd X_fitted_;
    VectorXi y_fitted_;
    VectorXi classes_;
    MatrixXd label_distributions_;

public:
    LabelSpreading(
        double alpha = 0.2,
        double gamma = 20.0,
        int max_iter = 30,
        double tol = 1e-3,
        const std::string& kernel = "rbf"
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
    MatrixXd build_affinity_matrix(const MatrixXd& X) const;
    void propagate_labels();
};

/**
 * SelfTrainingClassifier - Self-training with a KNN base classifier.
 */
class SelfTrainingClassifier : public Estimator, public Classifier {
private:
    int n_neighbors_;
    double threshold_;
    int max_iter_;
    bool fitted_;
    VectorXi classes_;
    neighbors::KNeighborsClassifier base_classifier_;
    MatrixXd X_fitted_;
    VectorXi y_fitted_;

public:
    SelfTrainingClassifier(int n_neighbors = 5, double threshold = 0.75, int max_iter = 10);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

} // namespace semi_supervised
} // namespace ingenuityml

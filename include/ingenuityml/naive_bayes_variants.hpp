#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace ingenuityml {
namespace naive_bayes {

/**
 * MultinomialNB - Multinomial Naive Bayes
 * 
 * Similar to scikit-learn's MultinomialNB, suitable for multinomial
 * distributed data (e.g., text classification with word counts).
 */
class MultinomialNB : public Estimator, public Classifier {
private:
    double alpha_;
    bool fit_prior_;
    bool fitted_;
    int n_features_;
    int n_classes_;
    VectorXi classes_;
    std::vector<double> class_log_prior_;
    std::vector<VectorXd> feature_log_prob_;

public:
    MultinomialNB(double alpha = 1.0, bool fit_prior = true);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

/**
 * BernoulliNB - Bernoulli Naive Bayes
 * 
 * Similar to scikit-learn's BernoulliNB, suitable for binary features.
 */
class BernoulliNB : public Estimator, public Classifier {
private:
    double alpha_;
    double binarize_;
    bool fit_prior_;
    bool fitted_;
    int n_features_;
    int n_classes_;
    VectorXi classes_;
    std::vector<double> class_log_prior_;
    std::vector<VectorXd> feature_log_prob_;
    std::vector<VectorXd> feature_log_prob_neg_;

public:
    BernoulliNB(double alpha = 1.0, double binarize = 0.0, bool fit_prior = true);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

/**
 * ComplementNB - Complement Naive Bayes
 * 
 * Similar to scikit-learn's ComplementNB, designed for imbalanced datasets.
 */
class ComplementNB : public Estimator, public Classifier {
private:
    double alpha_;
    bool fit_prior_;
    bool fitted_;
    int n_features_;
    int n_classes_;
    VectorXi classes_;
    std::vector<double> class_log_prior_;
    std::vector<VectorXd> feature_log_prob_;

public:
    ComplementNB(double alpha = 1.0, bool fit_prior = true);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

/**
 * CategoricalNB - Categorical Naive Bayes
 *
 * Similar to scikit-learn's CategoricalNB, suitable for categorical features
 * encoded as non-negative integers.
 */
class CategoricalNB : public Estimator, public Classifier {
private:
    double alpha_;
    bool fit_prior_;
    bool fitted_;
    int n_features_;
    int n_classes_;
    VectorXi classes_;
    std::vector<int> n_categories_;
    std::vector<double> class_log_prior_;
    std::vector<MatrixXd> feature_log_prob_;

public:
    CategoricalNB(double alpha = 1.0, bool fit_prior = true);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
    const std::vector<int>& n_categories() const { return n_categories_; }
};

} // namespace naive_bayes
} // namespace ingenuityml

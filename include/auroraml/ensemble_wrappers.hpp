#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace auroraml {
namespace ensemble {

/**
 * BaggingClassifier - Bagging ensemble for classification
 * 
 * Similar to scikit-learn's BaggingClassifier, creates an ensemble
 * by training multiple classifiers on bootstrap samples.
 */
class BaggingClassifier : public Estimator, public Classifier {
private:
    std::shared_ptr<Classifier> base_estimator_;
    int n_estimators_;
    int max_samples_;
    int max_features_;
    int random_state_;
    bool fitted_;
    std::vector<std::shared_ptr<Classifier>> estimators_;
    VectorXi classes_;

public:
    BaggingClassifier(
        std::shared_ptr<Classifier> base_estimator,
        int n_estimators = 10,
        int max_samples = -1,
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
};

/**
 * BaggingRegressor - Bagging ensemble for regression
 */
class BaggingRegressor : public Estimator, public Regressor {
private:
    std::shared_ptr<Regressor> base_estimator_;
    int n_estimators_;
    int max_samples_;
    int max_features_;
    int random_state_;
    bool fitted_;
    std::vector<std::shared_ptr<Regressor>> estimators_;

public:
    BaggingRegressor(
        std::shared_ptr<Regressor> base_estimator,
        int n_estimators = 10,
        int max_samples = -1,
        int max_features = -1,
        int random_state = -1
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

/**
 * VotingClassifier - Voting ensemble for classification
 * 
 * Similar to scikit-learn's VotingClassifier, combines predictions
 * from multiple classifiers using voting.
 */
class VotingClassifier : public Estimator, public Classifier {
private:
    std::vector<std::pair<std::string, std::shared_ptr<Classifier>>> estimators_;
    std::string voting_;
    bool fitted_;
    VectorXi classes_;

public:
    VotingClassifier(
        const std::vector<std::pair<std::string, std::shared_ptr<Classifier>>>& estimators,
        const std::string& voting = "hard"
    );
    
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
 * VotingRegressor - Voting ensemble for regression
 */
class VotingRegressor : public Estimator, public Regressor {
private:
    std::vector<std::pair<std::string, std::shared_ptr<Regressor>>> estimators_;
    bool fitted_;

public:
    VotingRegressor(
        const std::vector<std::pair<std::string, std::shared_ptr<Regressor>>>& estimators
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

/**
 * StackingClassifier - Stacking ensemble for classification
 * 
 * Similar to scikit-learn's StackingClassifier, uses a meta-classifier
 * to combine predictions from base classifiers.
 */
class StackingClassifier : public Estimator, public Classifier {
private:
    std::vector<std::pair<std::string, std::shared_ptr<Classifier>>> base_estimators_;
    std::shared_ptr<Classifier> meta_classifier_;
    bool fitted_;
    VectorXi classes_;

public:
    StackingClassifier(
        const std::vector<std::pair<std::string, std::shared_ptr<Classifier>>>& base_estimators,
        std::shared_ptr<Classifier> meta_classifier
    );
    
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
 * StackingRegressor - Stacking ensemble for regression
 */
class StackingRegressor : public Estimator, public Regressor {
private:
    std::vector<std::pair<std::string, std::shared_ptr<Regressor>>> base_estimators_;
    std::shared_ptr<Regressor> meta_regressor_;
    bool fitted_;

public:
    StackingRegressor(
        const std::vector<std::pair<std::string, std::shared_ptr<Regressor>>>& base_estimators,
        std::shared_ptr<Regressor> meta_regressor
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

} // namespace ensemble
} // namespace auroraml


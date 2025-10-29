#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <tuple>
#include <random>

namespace auroraml {
namespace model_selection {

/**
 * Split arrays or matrices into random train and test subsets
 */
std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> train_test_split(
    const MatrixXd& X, const VectorXd& y, double test_size = 0.25, 
    double train_size = -1, int random_state = -1, bool shuffle = true, 
    const VectorXd& stratify = VectorXd());

/**
 * Base class for cross-validators
 */
class BaseCrossValidator {
public:
    virtual ~BaseCrossValidator() = default;
    virtual std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const = 0;
    virtual int get_n_splits() const = 0;
};

/**
 * K-Fold cross-validator
 */
class KFold : public BaseCrossValidator {
private:
    int n_splits_;
    bool shuffle_;
    int random_state_;

public:
    KFold(int n_splits = 5, bool shuffle = false, int random_state = -1);
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_; }
    
    // Parameter management
    Params get_params() const;
    void set_params(const Params& params);
};

/**
 * Stratified K-Fold cross-validator
 */
class StratifiedKFold : public BaseCrossValidator {
private:
    int n_splits_;
    bool shuffle_;
    int random_state_;

public:
    StratifiedKFold(int n_splits = 5, bool shuffle = false, int random_state = -1);
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_; }
    
    // Parameter management
    Params get_params() const;
    void set_params(const Params& params);
};

/**
 * Group K-Fold cross-validator
 */
class GroupKFold {
private:
    int n_splits_;

public:
    GroupKFold(int n_splits = 5);
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y, const VectorXd& groups) const;
    int get_n_splits() const { return n_splits_; }
    
    // Parameter management
    Params get_params() const;
    void set_params(const Params& params);
};

/**
 * Cross-validation score
 */
VectorXd cross_val_score(Estimator& estimator, const MatrixXd& X, const VectorXd& y,
                        const BaseCrossValidator& cv, const std::string& scoring = "accuracy");

/**
 * Grid search cross-validation
 */
class GridSearchCV {
private:
    Estimator& estimator_;
    std::vector<Params> param_grid_;
    const BaseCrossValidator& cv_;
    std::string scoring_;
    int n_jobs_;
    bool verbose_;

public:
    GridSearchCV(Estimator& estimator, const std::vector<Params>& param_grid,
                const BaseCrossValidator& cv, const std::string& scoring = "accuracy",
                int n_jobs = 1, bool verbose = false);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y);
    VectorXd predict(const MatrixXd& X) const;
    Params best_params() const;
    double best_score() const;
    
    // Parameter management
    Params get_params() const;
    void set_params(const Params& params);
    int get_n_splits() const;
};

/**
 * Random search cross-validation
 */
class RandomizedSearchCV {
private:
    Estimator& estimator_;
    std::vector<Params> param_distributions_;
    const BaseCrossValidator& cv_;
    std::string scoring_;
    int n_iter_;
    int n_jobs_;
    bool verbose_;

public:
    RandomizedSearchCV(Estimator& estimator, const std::vector<Params>& param_distributions,
                      const BaseCrossValidator& cv, const std::string& scoring = "accuracy",
                      int n_iter = 10, int n_jobs = 1, bool verbose = false);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y);
    VectorXd predict(const MatrixXd& X) const;
    Params best_params() const;
    double best_score() const;
    
    // Parameter management
    Params get_params() const;
    void set_params(const Params& params);
    int get_n_splits() const;
};

} // namespace model_selection
} // namespace cxml
#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <map>
#include <memory>

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
 * ParameterGrid - Generate parameter combinations from a grid
 */
class ParameterGrid {
private:
    std::vector<Params> grid_;

public:
    ParameterGrid(const std::map<std::string, std::vector<std::string>>& param_grid);
    const std::vector<Params>& grid() const { return grid_; }
    size_t size() const { return grid_.size(); }
};

/**
 * ParameterSampler - Randomly sample parameter combinations
 */
class ParameterSampler {
private:
    std::vector<Params> samples_;

public:
    ParameterSampler(const std::map<std::string, std::vector<std::string>>& param_distributions,
                     int n_iter, int random_state = -1);
    const std::vector<Params>& samples() const { return samples_; }
    size_t size() const { return samples_.size(); }
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
 * Repeated K-Fold cross-validator
 */
class RepeatedKFold : public BaseCrossValidator {
private:
    int n_splits_;
    int n_repeats_;
    int random_state_;

public:
    RepeatedKFold(int n_splits = 5, int n_repeats = 10, int random_state = -1);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_ * n_repeats_; }

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
 * Repeated Stratified K-Fold cross-validator
 */
class RepeatedStratifiedKFold : public BaseCrossValidator {
private:
    int n_splits_;
    int n_repeats_;
    int random_state_;

public:
    RepeatedStratifiedKFold(int n_splits = 5, int n_repeats = 10, int random_state = -1);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_ * n_repeats_; }

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
 * ShuffleSplit cross-validator
 */
class ShuffleSplit : public BaseCrossValidator {
private:
    int n_splits_;
    double test_size_;
    double train_size_;
    int random_state_;

public:
    ShuffleSplit(int n_splits = 10, double test_size = 0.1, double train_size = -1.0, int random_state = -1);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_; }

    Params get_params() const;
    void set_params(const Params& params);
};

/**
 * StratifiedShuffleSplit cross-validator
 */
class StratifiedShuffleSplit : public BaseCrossValidator {
private:
    int n_splits_;
    double test_size_;
    double train_size_;
    int random_state_;

public:
    StratifiedShuffleSplit(int n_splits = 10, double test_size = 0.1, double train_size = -1.0, int random_state = -1);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_; }

    Params get_params() const;
    void set_params(const Params& params);
};

/**
 * GroupShuffleSplit cross-validator
 */
class GroupShuffleSplit {
private:
    int n_splits_;
    double test_size_;
    double train_size_;
    int random_state_;

public:
    GroupShuffleSplit(int n_splits = 5, double test_size = 0.2, double train_size = -1.0, int random_state = -1);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y, const VectorXd& groups) const;
    int get_n_splits() const { return n_splits_; }

    Params get_params() const;
    void set_params(const Params& params);
};

/**
 * PredefinedSplit cross-validator
 */
class PredefinedSplit : public BaseCrossValidator {
private:
    std::vector<int> test_fold_;
    int n_splits_;

public:
    PredefinedSplit(const std::vector<int>& test_fold);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_; }

    const std::vector<int>& test_fold() const { return test_fold_; }
};

/**
 * Leave-One-Out cross-validator
 */
class LeaveOneOut : public BaseCrossValidator {
private:
    mutable int n_splits_ = 0;

public:
    LeaveOneOut() = default;

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_; }
};

/**
 * Leave-P-Out cross-validator
 */
class LeavePOut : public BaseCrossValidator {
private:
    int p_;
    mutable int n_splits_ = 0;

public:
    LeavePOut(int p = 2);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_; }

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

/**
 * Time Series cross-validator
 * Provides train/test indices for time series data
 */
class TimeSeriesSplit : public BaseCrossValidator {
private:
    int n_splits_;
    int max_train_size_;
    int test_size_;
    int gap_;

public:
    TimeSeriesSplit(int n_splits = 5, int max_train_size = -1, int test_size = -1, int gap = 0);
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(const MatrixXd& X, const VectorXd& y = VectorXd()) const override;
    int get_n_splits() const override { return n_splits_; }
    
    // Parameter management
    Params get_params() const;
    void set_params(const Params& params);
};

/**
 * HalvingGridSearchCV - Successive Halving Grid Search Cross-Validation
 * 
 * Performs grid search with successive halving for efficient hyperparameter optimization.
 */
class HalvingGridSearchCV : public Estimator {
private:
    Estimator& estimator_;
    std::vector<Params> param_grid_;
    BaseCrossValidator& cv_;
    std::string scoring_;
    int factor_;
    int min_resources_;
    bool aggressive_elimination_;
    int n_jobs_;
    bool verbose_;
    Params best_params_;
    double best_score_;
    bool fitted_;

public:
    HalvingGridSearchCV(Estimator& estimator, const std::vector<Params>& param_grid,
                        BaseCrossValidator& cv, const std::string& scoring = "accuracy",
                        int factor = 3, int min_resources = 1, bool aggressive_elimination = false,
                        int n_jobs = 1, bool verbose = false);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const;
    Params best_params() const;
    double best_score() const;
    
    // Parameter management
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    int get_n_splits() const;

private:
    std::vector<std::pair<Params, double>> evaluate_candidates(
        const std::vector<Params>& candidates, const MatrixXd& X, const VectorXd& y, int n_resources);
};

/**
 * HalvingRandomSearchCV - Successive Halving Randomized Search
 */
class HalvingRandomSearchCV : public Estimator {
private:
    Estimator& estimator_;
    std::map<std::string, std::vector<std::string>> param_distributions_;
    BaseCrossValidator& cv_;
    std::string scoring_;
    int n_candidates_;
    int factor_;
    int min_resources_;
    bool aggressive_elimination_;
    int random_state_;
    int n_jobs_;
    bool verbose_;
    Params best_params_;
    double best_score_;
    bool fitted_;

public:
    HalvingRandomSearchCV(Estimator& estimator, const std::map<std::string, std::vector<std::string>>& param_distributions,
                          BaseCrossValidator& cv, const std::string& scoring = "accuracy",
                          int n_candidates = 10, int factor = 3, int min_resources = 1,
                          bool aggressive_elimination = false, int random_state = -1,
                          int n_jobs = 1, bool verbose = false);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const;
    Params best_params() const;
    double best_score() const;

    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    int get_n_splits() const;

private:
    std::vector<std::pair<Params, double>> evaluate_candidates(
        const std::vector<Params>& candidates, const MatrixXd& X, const VectorXd& y, int n_resources);
};

} // namespace model_selection
} // namespace cxml

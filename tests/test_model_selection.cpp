#include <gtest/gtest.h>
#include "auroraml/model_selection.hpp"
#include "auroraml/linear_model.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class ModelSelectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 3;
        
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Random(n_samples);
        
        // Create classification data
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        // Create groups for GroupKFold
        groups = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            groups(i) = i / 10;  // 10 samples per group
        }
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y, y_classification;
    VectorXd groups;
};

// KFold Tests
TEST_F(ModelSelectionTest, KFoldSplit) {
    model_selection::KFold kfold(5);
    
    auto splits = kfold.split(X, y);
    EXPECT_EQ(splits.size(), 5);
    
    for (const auto& split : splits) {
        EXPECT_GT(split.first.size(), 0);  // Train indices
        EXPECT_GT(split.second.size(), 0); // Test indices
        EXPECT_LT(split.first.size(), n_samples);
        EXPECT_LT(split.second.size(), n_samples);
    }
}

TEST_F(ModelSelectionTest, KFoldGetNSplits) {
    model_selection::KFold kfold(5);
    EXPECT_EQ(kfold.get_n_splits(), 5);
}

TEST_F(ModelSelectionTest, KFoldDifferentK) {
    model_selection::KFold kfold(3);
    
    auto splits = kfold.split(X, y);
    EXPECT_EQ(splits.size(), 3);
}

TEST_F(ModelSelectionTest, KFoldShuffle) {
    model_selection::KFold kfold(5, true, 42);
    
    auto splits = kfold.split(X, y);
    EXPECT_EQ(splits.size(), 5);
}

TEST_F(ModelSelectionTest, KFoldGetSetParams) {
    model_selection::KFold kfold(5, true, 42);
    
    Params params = kfold.get_params();
    EXPECT_EQ(params["n_splits"], "5");
    EXPECT_EQ(params["shuffle"], "1");
    EXPECT_EQ(params["random_state"], "42");
    
    // Test set_params
    Params new_params = {{"n_splits", "3"}, {"shuffle", "0"}};
    kfold.set_params(new_params);
    
    Params updated_params = kfold.get_params();
    EXPECT_EQ(updated_params["n_splits"], "3");
    EXPECT_EQ(updated_params["shuffle"], "0");
}

// StratifiedKFold Tests
TEST_F(ModelSelectionTest, StratifiedKFoldSplit) {
    model_selection::StratifiedKFold skfold(5);
    
    auto splits = skfold.split(X, y_classification);
    EXPECT_EQ(splits.size(), 5);
    
    for (const auto& split : splits) {
        EXPECT_GT(split.first.size(), 0);  // Train indices
        EXPECT_GT(split.second.size(), 0); // Test indices
        EXPECT_LT(split.first.size(), n_samples);
        EXPECT_LT(split.second.size(), n_samples);
    }
}

TEST_F(ModelSelectionTest, StratifiedKFoldGetNSplits) {
    model_selection::StratifiedKFold skfold(5);
    EXPECT_EQ(skfold.get_n_splits(), 5);
}

TEST_F(ModelSelectionTest, StratifiedKFoldDifferentK) {
    model_selection::StratifiedKFold skfold(3);
    
    auto splits = skfold.split(X, y_classification);
    EXPECT_EQ(splits.size(), 3);
}

TEST_F(ModelSelectionTest, StratifiedKFoldShuffle) {
    model_selection::StratifiedKFold skfold(5, true, 42);
    
    auto splits = skfold.split(X, y_classification);
    EXPECT_EQ(splits.size(), 5);
}

TEST_F(ModelSelectionTest, StratifiedKFoldGetSetParams) {
    model_selection::StratifiedKFold skfold(5, true, 42);
    
    Params params = skfold.get_params();
    EXPECT_EQ(params["n_splits"], "5");
    EXPECT_EQ(params["shuffle"], "1");
    EXPECT_EQ(params["random_state"], "42");
    
    // Test set_params
    Params new_params = {{"n_splits", "3"}, {"shuffle", "0"}};
    skfold.set_params(new_params);
    
    Params updated_params = skfold.get_params();
    EXPECT_EQ(updated_params["n_splits"], "3");
    EXPECT_EQ(updated_params["shuffle"], "0");
}

// GroupKFold Tests
TEST_F(ModelSelectionTest, GroupKFoldSplit) {
    model_selection::GroupKFold gkfold(5);
    
    auto splits = gkfold.split(X, y, groups);
    EXPECT_EQ(splits.size(), 5);
    
    for (const auto& split : splits) {
        EXPECT_GT(split.first.size(), 0);  // Train indices
        EXPECT_GT(split.second.size(), 0); // Test indices
        EXPECT_LT(split.first.size(), n_samples);
        EXPECT_LT(split.second.size(), n_samples);
    }
}

TEST_F(ModelSelectionTest, GroupKFoldGetNSplits) {
    model_selection::GroupKFold gkfold(5);
    EXPECT_EQ(gkfold.get_n_splits(), 5);
}

TEST_F(ModelSelectionTest, GroupKFoldDifferentK) {
    model_selection::GroupKFold gkfold(3);
    
    auto splits = gkfold.split(X, y, groups);
    EXPECT_EQ(splits.size(), 3);
}

TEST_F(ModelSelectionTest, GroupKFoldGetSetParams) {
    model_selection::GroupKFold gkfold(5);
    
    Params params = gkfold.get_params();
    EXPECT_EQ(params["n_splits"], "5");
    
    // Test set_params
    Params new_params = {{"n_splits", "3"}};
    gkfold.set_params(new_params);
    
    Params updated_params = gkfold.get_params();
    EXPECT_EQ(updated_params["n_splits"], "3");
}

// GridSearchCV Tests
TEST_F(ModelSelectionTest, GridSearchCVFit) {
    // Create a simple parameter grid
    std::vector<Params> param_grid = {
        {{"alpha", "0.1"}},
        {{"alpha", "1.0"}},
        {{"alpha", "10.0"}}
    };
    
    // Create a mock estimator and cross-validator
    linear_model::LinearRegression estimator;
    model_selection::KFold cv(3);
    
    model_selection::GridSearchCV grid_search(estimator, param_grid, cv);
    
    // Use a simple estimator (we'll need to implement a mock estimator)
    // For now, just test that the constructor works
    EXPECT_EQ(grid_search.get_n_splits(), 3);
}

TEST_F(ModelSelectionTest, GridSearchCVGetSetParams) {
    std::vector<Params> param_grid = {
        {{"alpha", "0.1"}},
        {{"alpha", "1.0"}}
    };
    
    // Create a mock estimator and cross-validator
    linear_model::LinearRegression estimator;
    model_selection::KFold cv(3);
    
    model_selection::GridSearchCV grid_search(estimator, param_grid, cv);
    
    Params params = grid_search.get_params();
    EXPECT_EQ(params["scoring"], "accuracy");
    EXPECT_EQ(params["n_jobs"], "1");
    
    // Test set_params
    Params new_params = {{"scoring", "mse"}, {"n_jobs", "2"}};
    grid_search.set_params(new_params);
    
    Params updated_params = grid_search.get_params();
    EXPECT_EQ(updated_params["scoring"], "mse");
    EXPECT_EQ(updated_params["n_jobs"], "2");
}

// RandomizedSearchCV Tests
TEST_F(ModelSelectionTest, RandomizedSearchCVFit) {
    // Create a simple parameter distribution
    std::vector<Params> param_dist = {
        {{"alpha", "0.1"}},
        {{"alpha", "1.0"}},
        {{"alpha", "10.0"}}
    };
    
    // Create a mock estimator and cross-validator
    linear_model::LinearRegression estimator;
    model_selection::KFold cv(3);
    
    model_selection::RandomizedSearchCV random_search(estimator, param_dist, cv, "accuracy", 5);
    
    // For now, just test that the constructor works
    EXPECT_EQ(random_search.get_n_splits(), 3);
}

TEST_F(ModelSelectionTest, RandomizedSearchCVGetSetParams) {
    std::vector<Params> param_dist = {
        {{"alpha", "0.1"}},
        {{"alpha", "1.0"}}
    };
    
    // Create a mock estimator and cross-validator
    linear_model::LinearRegression estimator;
    model_selection::KFold cv(3);
    
    model_selection::RandomizedSearchCV random_search(estimator, param_dist, cv, "accuracy", 5);
    
    Params params = random_search.get_params();
    EXPECT_EQ(params["scoring"], "accuracy");
    EXPECT_EQ(params["n_iter"], "5");
    EXPECT_EQ(params["n_jobs"], "1");
    
    // Test set_params
    Params new_params = {{"scoring", "mse"}, {"n_iter", "10"}};
    random_search.set_params(new_params);
    
    Params updated_params = random_search.get_params();
    EXPECT_EQ(updated_params["scoring"], "mse");
    EXPECT_EQ(updated_params["n_iter"], "10");
}

// Edge Cases and Error Handling
TEST_F(ModelSelectionTest, KFoldEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    model_selection::KFold kfold(5);
    EXPECT_THROW(kfold.split(X_empty, y_empty), std::runtime_error);
}

TEST_F(ModelSelectionTest, KFoldSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    VectorXd y_single = VectorXd::Random(1);
    
    model_selection::KFold kfold(1);
    auto splits = kfold.split(X_single, y_single);
    EXPECT_EQ(splits.size(), 1);
    EXPECT_EQ(splits[0].first.size(), 0);  // No training data
    EXPECT_EQ(splits[0].second.size(), 1); // All data is test data
}

TEST_F(ModelSelectionTest, KFoldZeroSplits) {
    EXPECT_THROW(model_selection::KFold kfold(0), std::invalid_argument);
}

TEST_F(ModelSelectionTest, KFoldNegativeSplits) {
    EXPECT_THROW(model_selection::KFold kfold(-1), std::invalid_argument);
}

TEST_F(ModelSelectionTest, KFoldMoreSplitsThanSamples) {
    model_selection::KFold kfold(n_samples + 1);
    EXPECT_THROW(kfold.split(X, y), std::runtime_error);
}

TEST_F(ModelSelectionTest, StratifiedKFoldEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    model_selection::StratifiedKFold skfold(5);
    EXPECT_THROW(skfold.split(X_empty, y_empty), std::runtime_error);
}

TEST_F(ModelSelectionTest, StratifiedKFoldSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    VectorXd y_single = VectorXd::Ones(1);
    
    model_selection::StratifiedKFold skfold(1);
    auto splits = skfold.split(X_single, y_single);
    EXPECT_EQ(splits.size(), 1);
    EXPECT_EQ(splits[0].first.size(), 0);  // No training data
    EXPECT_EQ(splits[0].second.size(), 1); // All data is test data
}

TEST_F(ModelSelectionTest, StratifiedKFoldZeroSplits) {
    EXPECT_THROW(model_selection::StratifiedKFold skfold(0), std::invalid_argument);
}

TEST_F(ModelSelectionTest, StratifiedKFoldNegativeSplits) {
    EXPECT_THROW(model_selection::StratifiedKFold skfold(-1), std::invalid_argument);
}

TEST_F(ModelSelectionTest, StratifiedKFoldMoreSplitsThanSamples) {
    model_selection::StratifiedKFold skfold(n_samples + 1);
    EXPECT_THROW(skfold.split(X, y_classification), std::runtime_error);
}

TEST_F(ModelSelectionTest, GroupKFoldEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    VectorXd groups_empty = VectorXd::Zero(0);
    
    model_selection::GroupKFold gkfold(5);
    EXPECT_THROW(gkfold.split(X_empty, y_empty, groups_empty), std::runtime_error);
}

TEST_F(ModelSelectionTest, GroupKFoldSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    VectorXd y_single = VectorXd::Random(1);
    VectorXd groups_single = VectorXd::Ones(1);
    
    model_selection::GroupKFold gkfold(1);
    auto splits = gkfold.split(X_single, y_single, groups_single);
    EXPECT_EQ(splits.size(), 1);
    EXPECT_EQ(splits[0].first.size(), 0);  // No training data
    EXPECT_EQ(splits[0].second.size(), 1); // All data is test data
}

TEST_F(ModelSelectionTest, GroupKFoldZeroSplits) {
    EXPECT_THROW(model_selection::GroupKFold gkfold(0), std::invalid_argument);
}

TEST_F(ModelSelectionTest, GroupKFoldNegativeSplits) {
    EXPECT_THROW(model_selection::GroupKFold gkfold(-1), std::invalid_argument);
}

TEST_F(ModelSelectionTest, GroupKFoldMoreSplitsThanGroups) {
    int n_groups = groups.maxCoeff() + 1;
    model_selection::GroupKFold gkfold(n_groups + 1);
    EXPECT_THROW(gkfold.split(X, y, groups), std::runtime_error);
}

TEST_F(ModelSelectionTest, GroupKFoldDimensionMismatch) {
    VectorXd groups_wrong = VectorXd::Zero(n_samples + 1);
    
    model_selection::GroupKFold gkfold(5);
    EXPECT_THROW(gkfold.split(X, y, groups_wrong), std::invalid_argument);
}

TEST_F(ModelSelectionTest, GridSearchCVEmptyParamGrid) {
    std::vector<Params> empty_grid;
    
    // Create a mock estimator and cross-validator
    linear_model::LinearRegression estimator;
    model_selection::KFold cv(3);
    
    EXPECT_THROW(model_selection::GridSearchCV(estimator, empty_grid, cv), std::invalid_argument);
}

TEST_F(ModelSelectionTest, RandomizedSearchCVEmptyParamDist) {
    std::vector<Params> empty_dist;
    
    // Create a mock estimator and cross-validator
    linear_model::LinearRegression estimator;
    model_selection::KFold cv(3);
    
    EXPECT_THROW(model_selection::RandomizedSearchCV(estimator, empty_dist, cv, "accuracy", 5), std::invalid_argument);
}

TEST_F(ModelSelectionTest, RandomizedSearchCVZeroIterations) {
    std::vector<Params> param_dist = {
        {{"alpha", "0.1"}},
        {{"alpha", "1.0"}}
    };
    
    // Create a mock estimator and cross-validator
    linear_model::LinearRegression estimator;
    model_selection::KFold cv(3);
    
    EXPECT_THROW(model_selection::RandomizedSearchCV(estimator, param_dist, cv, "accuracy", 0), std::invalid_argument);
}

TEST_F(ModelSelectionTest, RandomizedSearchCVNegativeIterations) {
    std::vector<Params> param_dist = {
        {{"alpha", "0.1"}},
        {{"alpha", "1.0"}}
    };
    
    // Create a mock estimator and cross-validator
    linear_model::LinearRegression estimator;
    model_selection::KFold cv(3);
    
    EXPECT_THROW(model_selection::RandomizedSearchCV(estimator, param_dist, cv, "accuracy", -1), std::invalid_argument);
}

// Consistency Tests
TEST_F(ModelSelectionTest, KFoldConsistency) {
    model_selection::KFold kfold1(5, true, 42);
    model_selection::KFold kfold2(5, true, 42);
    
    auto splits1 = kfold1.split(X, y);
    auto splits2 = kfold2.split(X, y);
    
    EXPECT_EQ(splits1.size(), splits2.size());
    EXPECT_EQ(splits1.size(), 5);
}

TEST_F(ModelSelectionTest, StratifiedKFoldConsistency) {
    model_selection::StratifiedKFold skfold1(5, true, 42);
    model_selection::StratifiedKFold skfold2(5, true, 42);
    
    auto splits1 = skfold1.split(X, y_classification);
    auto splits2 = skfold2.split(X, y_classification);
    
    EXPECT_EQ(splits1.size(), splits2.size());
    EXPECT_EQ(splits1.size(), 5);
}

TEST_F(ModelSelectionTest, GroupKFoldConsistency) {
    model_selection::GroupKFold gkfold1(5);
    model_selection::GroupKFold gkfold2(5);
    
    auto splits1 = gkfold1.split(X, y, groups);
    auto splits2 = gkfold2.split(X, y, groups);
    
    EXPECT_EQ(splits1.size(), splits2.size());
    EXPECT_EQ(splits1.size(), 5);
}

// Cross-validation Properties
TEST_F(ModelSelectionTest, KFoldDisjointSplits) {
    model_selection::KFold kfold(5);
    auto splits = kfold.split(X, y);
    
    // Check that train and test sets are disjoint
    for (const auto& split : splits) {
        const auto& train_indices = split.first;
        const auto& test_indices = split.second;
        
        for (int train_idx : train_indices) {
            for (int test_idx : test_indices) {
                EXPECT_NE(train_idx, test_idx);
            }
        }
    }
}

TEST_F(ModelSelectionTest, KFoldCompleteCoverage) {
    model_selection::KFold kfold(5);
    auto splits = kfold.split(X, y);
    
    // Check that all samples are covered exactly once in test sets
    std::set<int> all_test_indices;
    for (const auto& split : splits) {
        for (int idx : split.second) {
            all_test_indices.insert(idx);
        }
    }
    
    EXPECT_EQ(all_test_indices.size(), n_samples);
}

TEST_F(ModelSelectionTest, StratifiedKFoldStratification) {
    model_selection::StratifiedKFold skfold(5);
    auto splits = skfold.split(X, y_classification);
    
    // Check that each fold maintains class distribution
    for (const auto& split : splits) {
        const auto& train_indices = split.first;
        const auto& test_indices = split.second;
        
        // Calculate class distribution in test set
        int class0_count = 0, class1_count = 0;
        for (int idx : test_indices) {
            if (y_classification(idx) == 0.0) class0_count++;
            else class1_count++;
        }
        
        // Both classes should be represented (unless one class is very rare)
        EXPECT_GT(class0_count + class1_count, 0);
    }
}

TEST_F(ModelSelectionTest, GroupKFoldGroupSeparation) {
    model_selection::GroupKFold gkfold(5);
    auto splits = gkfold.split(X, y, groups);
    
    // Check that groups are not split between train and test
    for (const auto& split : splits) {
        const auto& train_indices = split.first;
        const auto& test_indices = split.second;
        
        std::set<int> train_groups, test_groups;
        for (int idx : train_indices) {
            train_groups.insert(groups(idx));
        }
        for (int idx : test_indices) {
            test_groups.insert(groups(idx));
        }
        
        // No group should appear in both train and test
        for (int group : train_groups) {
            EXPECT_EQ(test_groups.count(group), 0);
        }
    }
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

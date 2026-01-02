#include <gtest/gtest.h>
#include "auroraml/linear_model.hpp"
#include "auroraml/model_selection.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>

namespace auroraml {
namespace test {

class LogisticRegressionCVTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases - Using LogisticRegression with cross-validation manually
TEST_F(LogisticRegressionCVTest, BasicFit) {
    // Use LogisticRegression with KFold cross-validation
    model_selection::KFold kfold(5, true, 42);
    auto splits = kfold.split(X, y_classification);
    
    EXPECT_EQ(splits.size(), 5);
    
    // Fit on first fold
    std::vector<int> train_indices = splits[0].first;
    MatrixXd X_train = MatrixXd(train_indices.size(), n_features);
    VectorXd y_train = VectorXd(train_indices.size());
    
    for (size_t i = 0; i < train_indices.size(); ++i) {
        X_train.row(i) = X.row(train_indices[i]);
        y_train(i) = y_classification(train_indices[i]);
    }
    
    linear_model::LogisticRegression lr(1.0, true, 100, 1e-4, 42);
    lr.fit(X_train, y_train);
    
    EXPECT_TRUE(lr.is_fitted());
}

TEST_F(LogisticRegressionCVTest, CrossValidationScore) {
    linear_model::LogisticRegression lr(1.0, true, 100, 1e-4, 42);
    model_selection::KFold kfold(5, true, 42);
    
    VectorXd scores = model_selection::cross_val_score(
        lr, X, y_classification, kfold, "accuracy"
    );
    
    EXPECT_EQ(scores.size(), 5);
    for (int i = 0; i < scores.size(); ++i) {
        EXPECT_GE(scores(i), 0.0);
        EXPECT_LE(scores(i), 1.0);
    }
}

TEST_F(LogisticRegressionCVTest, GridSearchCV) {
    linear_model::LogisticRegression lr(1.0, true, 100, 1e-4, 42);
    model_selection::KFold kfold(5, true, 42);
    
    std::vector<Params> param_grid;
    Params p1;
    p1["C"] = "1.0";
    param_grid.push_back(p1);
    
    model_selection::GridSearchCV gs(lr, param_grid, kfold, "accuracy");
    gs.fit(X, y_classification);
    
    // GridSearchCV doesn't have is_fitted(), but we can check it worked by getting best params
    Params best = gs.best_params();
    EXPECT_FALSE(best.empty());
}

TEST_F(LogisticRegressionCVTest, GridSearchCVPredict) {
    linear_model::LogisticRegression lr(1.0, true, 100, 1e-4, 42);
    model_selection::KFold kfold(5, true, 42);
    
    std::vector<Params> param_grid;
    Params p1;
    p1["C"] = "1.0";
    param_grid.push_back(p1);
    
    model_selection::GridSearchCV gs(lr, param_grid, kfold, "accuracy");
    gs.fit(X, y_classification);
    
    VectorXd predictions = gs.predict(X_test);
    EXPECT_EQ(predictions.size(), X_test.rows());
}

TEST_F(LogisticRegressionCVTest, MultipleCValues) {
    linear_model::LogisticRegression lr(1.0, true, 100, 1e-4, 42);
    model_selection::KFold kfold(5, true, 42);
    
    // Test with multiple C values
    std::vector<double> C_values = {0.01, 0.1, 1.0, 10.0};
    std::vector<Params> param_grid;
    
    for (double C : C_values) {
        Params p;
        p["C"] = std::to_string(C);
        param_grid.push_back(p);
    }
    
    model_selection::GridSearchCV gs(lr, param_grid, kfold, "accuracy");
    gs.fit(X, y_classification);
    
    // GridSearchCV doesn't have is_fitted(), but we can check it worked by getting best params
    Params best = gs.best_params();
    EXPECT_FALSE(best.empty());
}

// Negative test cases
TEST_F(LogisticRegressionCVTest, GridSearchCVNotFitted) {
    linear_model::LogisticRegression lr(1.0);
    model_selection::KFold kfold(5);
    
    std::vector<Params> param_grid;
    Params p1;
    p1["C"] = "1.0";
    param_grid.push_back(p1);
    
    model_selection::GridSearchCV gs(lr, param_grid, kfold, "accuracy");
    EXPECT_THROW(gs.predict(X_test), std::runtime_error);
}

TEST_F(LogisticRegressionCVTest, EmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    linear_model::LogisticRegression lr(1.0);
    model_selection::KFold kfold(5);
    
    std::vector<Params> param_grid;
    Params p1;
    p1["C"] = "1.0";
    param_grid.push_back(p1);
    
    model_selection::GridSearchCV gs(lr, param_grid, kfold, "accuracy");
    // Empty data causes KFold to throw runtime_error about n_splits
    EXPECT_THROW(gs.fit(X_empty, y_empty), std::runtime_error);
}

TEST_F(LogisticRegressionCVTest, InvalidKFold) {
    linear_model::LogisticRegression lr(1.0);
    // Invalid n_splits throws exception during KFold construction
    EXPECT_THROW(model_selection::KFold kfold(0), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

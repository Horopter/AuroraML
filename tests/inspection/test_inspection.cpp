#include <gtest/gtest.h>
#include "ingenuityml/inspection.hpp"
#include "ingenuityml/tree.hpp"
#include "ingenuityml/linear_model.hpp"
#include <Eigen/Dense>
#include <memory>

namespace ingenuityml {
namespace test {

class InspectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        y_regression = VectorXd::Random(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y_classification, y_regression;
};

// Positive test cases
TEST_F(InspectionTest, PermutationImportanceClassification) {
    auto clf = std::make_shared<tree::DecisionTreeClassifier>("gini", 3, 2, 1, 42);
    clf->fit(X, y_classification);
    
    inspection::PermutationImportance perm_imp(clf, "accuracy", 3, 42);
    perm_imp.fit(X, y_classification);
    
    std::vector<double> importances = perm_imp.feature_importances();
    EXPECT_EQ(importances.size(), n_features);
    
    for (double imp : importances) {
        EXPECT_GE(imp, 0.0);
    }
}

TEST_F(InspectionTest, PermutationImportanceRegression) {
    auto reg = std::make_shared<linear_model::LinearRegression>();
    reg->fit(X, y_regression);
    
    inspection::PermutationImportance perm_imp(reg, "r2", 3, 42);
    perm_imp.fit(X, y_regression);
    
    std::vector<double> importances = perm_imp.feature_importances();
    EXPECT_EQ(importances.size(), n_features);
}

TEST_F(InspectionTest, PartialDependenceFit) {
    auto reg = std::make_shared<linear_model::LinearRegression>();
    reg->fit(X, y_regression);
    
    std::vector<int> features = {0, 1};
    inspection::PartialDependence pd(reg, features);
    pd.compute(X);
    
    MatrixXd grid = pd.grid();
    EXPECT_GT(grid.rows(), 0);
    EXPECT_EQ(grid.cols(), features.size());
}

// Negative test cases
TEST_F(InspectionTest, PermutationImportanceNotFitted) {
    auto clf = std::make_shared<tree::DecisionTreeClassifier>("gini", 3);
    inspection::PermutationImportance perm_imp(clf, "accuracy", 3);
    
    // Should work even if base estimator not fitted
    perm_imp.fit(X, y_classification);
    EXPECT_GT(perm_imp.feature_importances().size(), 0);
}

TEST_F(InspectionTest, PermutationImportanceEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    auto clf = std::make_shared<tree::DecisionTreeClassifier>("gini", 3);
    clf->fit(X, y_classification);
    
    inspection::PermutationImportance perm_imp(clf, "accuracy", 3);
    EXPECT_THROW(perm_imp.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(InspectionTest, PartialDependenceEmptyFeatures) {
    auto reg = std::make_shared<linear_model::LinearRegression>();
    reg->fit(X, y_regression);
    
    std::vector<int> features = {};
    inspection::PartialDependence pd(reg, features);
    
    EXPECT_THROW(pd.compute(X), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

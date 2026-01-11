#include <gtest/gtest.h>
#include "ingenuityml/truncated_svd.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class TruncatedSVDTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 10;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_dummy = VectorXd::Zero(n_samples);
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_dummy;
};

// Positive test cases
TEST_F(TruncatedSVDTest, TruncatedSVDFit) {
    decomposition::TruncatedSVD svd(3);
    svd.fit(X, y_dummy);
    
    EXPECT_TRUE(svd.is_fitted());
}

TEST_F(TruncatedSVDTest, TruncatedSVDTransform) {
    decomposition::TruncatedSVD svd(3);
    svd.fit(X, y_dummy);
    
    MatrixXd X_transformed = svd.transform(X_test);
    EXPECT_EQ(X_transformed.rows(), X_test.rows());
    EXPECT_EQ(X_transformed.cols(), 3);
    EXPECT_FALSE(X_transformed.array().isNaN().any());
}

TEST_F(TruncatedSVDTest, TruncatedSVDComponents) {
    decomposition::TruncatedSVD svd(3);
    svd.fit(X, y_dummy);
    
    MatrixXd components = svd.components();
    EXPECT_EQ(components.rows(), 3);
    EXPECT_EQ(components.cols(), n_features);
}

// Negative test cases
TEST_F(TruncatedSVDTest, TruncatedSVDNotFitted) {
    decomposition::TruncatedSVD svd(3);
    EXPECT_THROW(svd.transform(X_test), std::runtime_error);
}

TEST_F(TruncatedSVDTest, TruncatedSVDWrongFeatureCount) {
    decomposition::TruncatedSVD svd(3);
    svd.fit(X, y_dummy);
    
    MatrixXd X_wrong = MatrixXd::Random(20, n_features + 1);
    EXPECT_THROW(svd.transform(X_wrong), std::runtime_error);
}

TEST_F(TruncatedSVDTest, TruncatedSVDEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    decomposition::TruncatedSVD svd(3);
    EXPECT_THROW(svd.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(TruncatedSVDTest, TruncatedSVDZeroComponents) {
    decomposition::TruncatedSVD svd(0);
    EXPECT_THROW(svd.fit(X, y_dummy), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


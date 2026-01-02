#include <gtest/gtest.h>
#include "auroraml/impute.hpp"
#include <Eigen/Dense>
#include <limits>

namespace auroraml {
namespace test {

class ImputeTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 50;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_dummy = VectorXd::Zero(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y_dummy;
};

// Positive test cases
TEST_F(ImputeTest, KNNImputerFit) {
    impute::KNNImputer imputer(5);
    imputer.fit(X, y_dummy);
    
    EXPECT_TRUE(imputer.is_fitted());
}

TEST_F(ImputeTest, KNNImputerTransform) {
    impute::KNNImputer imputer(5);
    imputer.fit(X, y_dummy);
    
    MatrixXd X_transformed = imputer.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_EQ(X_transformed.cols(), X.cols());
}

TEST_F(ImputeTest, KNNImputerFitTransform) {
    impute::KNNImputer imputer(5);
    MatrixXd X_transformed = imputer.fit_transform(X, y_dummy);
    
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_EQ(X_transformed.cols(), X.cols());
    EXPECT_TRUE(imputer.is_fitted());
}

TEST_F(ImputeTest, MissingIndicatorFitTransform) {
    MatrixXd X_missing = X;
    X_missing(0, 1) = std::numeric_limits<double>::quiet_NaN();
    X_missing(1, 3) = std::numeric_limits<double>::quiet_NaN();

    impute::MissingIndicator indicator("missing-only");
    indicator.fit(X_missing, y_dummy);

    MatrixXd indicators = indicator.transform(X_missing);
    EXPECT_EQ(indicators.rows(), X_missing.rows());
    EXPECT_EQ(indicators.cols(), static_cast<int>(indicator.features().size()));
}

TEST_F(ImputeTest, MissingIndicatorAllFeatures) {
    impute::MissingIndicator indicator("all");
    indicator.fit(X, y_dummy);

    MatrixXd indicators = indicator.transform(X);
    EXPECT_EQ(indicators.rows(), X.rows());
    EXPECT_EQ(indicators.cols(), X.cols());
}

// Negative test cases
TEST_F(ImputeTest, KNNImputerNotFitted) {
    impute::KNNImputer imputer(5);
    EXPECT_THROW(imputer.transform(X), std::runtime_error);
}

TEST_F(ImputeTest, KNNImputerEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    impute::KNNImputer imputer(5);
    EXPECT_THROW(imputer.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(ImputeTest, KNNImputerNegativeNeighbors) {
    impute::KNNImputer imputer(-1);
    EXPECT_THROW(imputer.fit(X, y_dummy), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

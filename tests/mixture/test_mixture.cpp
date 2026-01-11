#include <gtest/gtest.h>
#include "ingenuityml/mixture.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class MixtureTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 2;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_dummy = VectorXd::Zero(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y_dummy;
};

// Positive test cases
TEST_F(MixtureTest, GaussianMixtureFit) {
    mixture::GaussianMixture gm(3, 42);
    gm.fit(X, y_dummy);
    
    EXPECT_TRUE(gm.is_fitted());
}

TEST_F(MixtureTest, GaussianMixturePredict) {
    mixture::GaussianMixture gm(3, 42);
    gm.fit(X, y_dummy);
    
    VectorXi labels = gm.predict(X);
    EXPECT_EQ(labels.size(), n_samples);
    
    for (int i = 0; i < labels.size(); ++i) {
        EXPECT_GE(labels(i), 0);
        EXPECT_LT(labels(i), 3);
    }
}

TEST_F(MixtureTest, GaussianMixturePredictProba) {
    mixture::GaussianMixture gm(3, 42);
    gm.fit(X, y_dummy);
    
    MatrixXd proba = gm.predict_proba(X);
    EXPECT_EQ(proba.rows(), n_samples);
    EXPECT_EQ(proba.cols(), 3);
    
    for (int i = 0; i < proba.rows(); ++i) {
        double sum = proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

// Negative test cases
TEST_F(MixtureTest, GaussianMixtureNotFitted) {
    mixture::GaussianMixture gm(3);
    EXPECT_THROW(gm.predict(X), std::runtime_error);
}

TEST_F(MixtureTest, GaussianMixtureEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    mixture::GaussianMixture gm(3);
    EXPECT_THROW(gm.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(MixtureTest, GaussianMixtureZeroComponents) {
    // Zero components throws exception during construction
    EXPECT_THROW(mixture::GaussianMixture gm(0), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

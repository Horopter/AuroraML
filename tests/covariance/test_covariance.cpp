#include <gtest/gtest.h>
#include "auroraml/covariance.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class CovarianceTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 120;
        n_features = 3;
        X = MatrixXd::Random(n_samples, n_features);
        X.row(0).array() += 5.0;
        X.row(1).array() -= 4.0;
        y_dummy = VectorXd::Zero(n_samples);
    }

    int n_samples;
    int n_features;
    MatrixXd X;
    VectorXd y_dummy;
};

TEST_F(CovarianceTest, EmpiricalCovarianceFit) {
    covariance::EmpiricalCovariance cov;
    cov.fit(X, y_dummy);
    EXPECT_TRUE(cov.is_fitted());
    EXPECT_EQ(cov.covariance().rows(), n_features);
    EXPECT_EQ(cov.covariance().cols(), n_features);

    VectorXd distances = cov.mahalanobis(X);
    EXPECT_EQ(distances.size(), n_samples);
}

TEST_F(CovarianceTest, ShrunkCovarianceFit) {
    covariance::ShrunkCovariance cov(0.2);
    cov.fit(X, y_dummy);
    EXPECT_TRUE(cov.is_fitted());
    EXPECT_NEAR(cov.shrinkage(), 0.2, 1e-9);
    EXPECT_EQ(cov.covariance().rows(), n_features);
}

TEST_F(CovarianceTest, LedoitWolfFit) {
    covariance::LedoitWolf cov;
    cov.fit(X, y_dummy);
    EXPECT_TRUE(cov.is_fitted());
    EXPECT_GE(cov.shrinkage(), 0.0);
    EXPECT_LE(cov.shrinkage(), 1.0);
}

TEST_F(CovarianceTest, OASFit) {
    covariance::OAS cov;
    cov.fit(X, y_dummy);
    EXPECT_TRUE(cov.is_fitted());
    EXPECT_GE(cov.shrinkage(), 0.0);
    EXPECT_LE(cov.shrinkage(), 1.0);
}

TEST_F(CovarianceTest, MinCovDetFit) {
    covariance::MinCovDet cov(0.75);
    cov.fit(X, y_dummy);
    EXPECT_TRUE(cov.is_fitted());
    EXPECT_EQ(cov.support().size(), n_samples);

    VectorXd distances = cov.mahalanobis(X);
    EXPECT_EQ(distances.size(), n_samples);
}

TEST_F(CovarianceTest, EllipticEnvelopeFit) {
    covariance::EllipticEnvelope envelope(0.1, 0.75, 50, 1e-3, 42);
    envelope.fit(X, y_dummy);
    EXPECT_TRUE(envelope.is_fitted());

    VectorXi labels = envelope.predict(X);
    EXPECT_EQ(labels.size(), n_samples);

    VectorXd scores = envelope.decision_function(X);
    EXPECT_EQ(scores.size(), n_samples);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

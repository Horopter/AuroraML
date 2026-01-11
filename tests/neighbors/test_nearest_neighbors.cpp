#include <gtest/gtest.h>
#include "ingenuityml/neighbors.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class NearestNeighborsTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 60;
        n_features = 3;
        X = MatrixXd::Random(n_samples, n_features);
        X_test = MatrixXd::Random(15, n_features);
        y_dummy = VectorXd::Zero(n_samples);
    }

    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_dummy;
};

TEST_F(NearestNeighborsTest, Fit) {
    neighbors::NearestNeighbors nn(3, 1.0);
    nn.fit(X, y_dummy);
    EXPECT_TRUE(nn.is_fitted());
}

TEST_F(NearestNeighborsTest, KNeighbors) {
    neighbors::NearestNeighbors nn(3, 1.0);
    nn.fit(X, y_dummy);

    auto result = nn.kneighbors(X_test);
    const MatrixXd& distances = result.first;
    const MatrixXi& indices = result.second;

    EXPECT_EQ(distances.rows(), X_test.rows());
    EXPECT_EQ(distances.cols(), 3);
    EXPECT_EQ(indices.rows(), X_test.rows());
    EXPECT_EQ(indices.cols(), 3);

    for (int i = 0; i < distances.rows(); ++i) {
        for (int j = 0; j < distances.cols(); ++j) {
            EXPECT_GE(distances(i, j), 0.0);
            EXPECT_LT(indices(i, j), n_samples);
        }
    }
}

TEST_F(NearestNeighborsTest, RadiusNeighbors) {
    neighbors::NearestNeighbors nn(3, 0.8);
    nn.fit(X, y_dummy);

    auto result = nn.radius_neighbors(X_test, 0.8);
    const auto& distances = result.first;
    const auto& indices = result.second;

    EXPECT_EQ(distances.size(), static_cast<size_t>(X_test.rows()));
    EXPECT_EQ(indices.size(), static_cast<size_t>(X_test.rows()));

    for (size_t i = 0; i < distances.size(); ++i) {
        EXPECT_EQ(distances[i].size(), indices[i].size());
        for (size_t j = 0; j < distances[i].size(); ++j) {
            EXPECT_GE(distances[i][j], 0.0);
            EXPECT_LT(indices[i][j], n_samples);
        }
    }
}

TEST_F(NearestNeighborsTest, NotFitted) {
    neighbors::NearestNeighbors nn(3, 1.0);
    EXPECT_THROW(nn.kneighbors(X_test), std::runtime_error);
    EXPECT_THROW(nn.radius_neighbors(X_test, 1.0), std::runtime_error);
}

TEST_F(NearestNeighborsTest, WrongFeatureCount) {
    neighbors::NearestNeighbors nn(3, 1.0);
    nn.fit(X, y_dummy);

    MatrixXd X_wrong = MatrixXd::Random(10, n_features + 1);
    EXPECT_THROW(nn.kneighbors(X_wrong), std::invalid_argument);
    EXPECT_THROW(nn.radius_neighbors(X_wrong, 1.0), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

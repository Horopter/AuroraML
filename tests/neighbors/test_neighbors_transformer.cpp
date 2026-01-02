#include <gtest/gtest.h>
#include "auroraml/neighbors.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class NeighborsTransformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 40;
        n_features = 3;
        X = MatrixXd::Random(n_samples, n_features);
        X_test = MatrixXd::Random(12, n_features);
        y_dummy = VectorXd::Zero(n_samples);
    }

    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_dummy;
};

TEST_F(NeighborsTransformerTest, KNeighborsTransformerDistance) {
    neighbors::KNeighborsTransformer transformer(4, "distance");
    transformer.fit(X, y_dummy);

    MatrixXd graph = transformer.transform(X_test);
    EXPECT_EQ(graph.rows(), X_test.rows());
    EXPECT_EQ(graph.cols(), X.rows());

    for (int i = 0; i < graph.rows(); ++i) {
        for (int j = 0; j < graph.cols(); ++j) {
            EXPECT_GE(graph(i, j), 0.0);
        }
    }
}

TEST_F(NeighborsTransformerTest, KNeighborsTransformerConnectivity) {
    neighbors::KNeighborsTransformer transformer(3, "connectivity");
    transformer.fit(X, y_dummy);

    MatrixXd graph = transformer.transform(X_test);
    EXPECT_EQ(graph.rows(), X_test.rows());
    EXPECT_EQ(graph.cols(), X.rows());

    for (int i = 0; i < graph.rows(); ++i) {
        int ones = 0;
        for (int j = 0; j < graph.cols(); ++j) {
            double v = graph(i, j);
            EXPECT_TRUE(v == 0.0 || v == 1.0);
            if (v == 1.0) {
                ones++;
            }
        }
        EXPECT_LE(ones, 3);
    }
}

TEST_F(NeighborsTransformerTest, RadiusNeighborsTransformerConnectivity) {
    neighbors::RadiusNeighborsTransformer transformer(0.8, "connectivity");
    transformer.fit(X, y_dummy);

    MatrixXd graph = transformer.transform(X_test);
    EXPECT_EQ(graph.rows(), X_test.rows());
    EXPECT_EQ(graph.cols(), X.rows());

    for (int i = 0; i < graph.rows(); ++i) {
        for (int j = 0; j < graph.cols(); ++j) {
            double v = graph(i, j);
            EXPECT_TRUE(v == 0.0 || v == 1.0);
        }
    }
}

TEST_F(NeighborsTransformerTest, NotFitted) {
    neighbors::KNeighborsTransformer knn(3, "distance");
    EXPECT_THROW(knn.transform(X_test), std::runtime_error);

    neighbors::RadiusNeighborsTransformer radius(1.0, "distance");
    EXPECT_THROW(radius.transform(X_test), std::runtime_error);
}

TEST_F(NeighborsTransformerTest, WrongFeatureCount) {
    neighbors::KNeighborsTransformer knn(3, "distance");
    knn.fit(X, y_dummy);

    MatrixXd X_wrong = MatrixXd::Random(10, n_features + 2);
    EXPECT_THROW(knn.transform(X_wrong), std::invalid_argument);

    neighbors::RadiusNeighborsTransformer radius(1.0, "distance");
    radius.fit(X, y_dummy);
    EXPECT_THROW(radius.transform(X_wrong), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

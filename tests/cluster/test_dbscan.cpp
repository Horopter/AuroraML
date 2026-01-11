#include <gtest/gtest.h>
#include "ingenuityml/dbscan.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class DBSCANTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data with well-separated clusters
        n_samples = 100;
        n_features = 2;
        
        X_clustered = MatrixXd::Zero(n_samples, n_features);
        for (int i = 0; i < n_samples/3; ++i) {
            X_clustered(i, 0) = 1.0 + 0.1 * (rand() % 100) / 100.0;
            X_clustered(i, 1) = 1.0 + 0.1 * (rand() % 100) / 100.0;
        }
        for (int i = n_samples/3; i < 2*n_samples/3; ++i) {
            X_clustered(i, 0) = -1.0 + 0.1 * (rand() % 100) / 100.0;
            X_clustered(i, 1) = 1.0 + 0.1 * (rand() % 100) / 100.0;
        }
        for (int i = 2*n_samples/3; i < n_samples; ++i) {
            X_clustered(i, 0) = 0.0 + 0.1 * (rand() % 100) / 100.0;
            X_clustered(i, 1) = -1.0 + 0.1 * (rand() % 100) / 100.0;
        }
        
        y_dummy = VectorXd::Zero(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X_clustered;
    VectorXd y_dummy;
};

// Positive test cases
TEST_F(DBSCANTest, DBSCANFitPredict) {
    cluster::DBSCAN dbscan(0.5, 5);
    dbscan.fit(X_clustered, y_dummy);
    VectorXi labels = dbscan.labels();
    
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid (>= -1, where -1 is noise)
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_GE(labels(i), -1);
    }
}

TEST_F(DBSCANTest, DBSCANLabels) {
    cluster::DBSCAN dbscan(0.5, 5);
    dbscan.fit(X_clustered, y_dummy);
    
    VectorXi labels = dbscan.labels();
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_GE(labels(i), -1);
    }
}

TEST_F(DBSCANTest, DBSCANDifferentEps) {
    std::vector<double> eps_values = {0.3, 0.5, 1.0};
    
    for (double eps : eps_values) {
        cluster::DBSCAN dbscan(eps, 5);
        dbscan.fit(X_clustered, y_dummy);
        VectorXi labels = dbscan.labels();
        EXPECT_EQ(labels.size(), n_samples);
    }
}

TEST_F(DBSCANTest, DBSCANDifferentMinSamples) {
    std::vector<int> min_samples_values = {3, 5, 10};
    
    for (int min_samples : min_samples_values) {
        cluster::DBSCAN dbscan(0.5, min_samples);
        dbscan.fit(X_clustered, y_dummy);
        VectorXi labels = dbscan.labels();
        EXPECT_EQ(labels.size(), n_samples);
    }
}

// Negative test cases
TEST_F(DBSCANTest, DBSCANEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    cluster::DBSCAN dbscan(0.5, 5);
    EXPECT_THROW(dbscan.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(DBSCANTest, DBSCANNegativeEps) {
    cluster::DBSCAN dbscan(-0.1, 5);
    EXPECT_THROW(dbscan.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(DBSCANTest, DBSCANZeroMinSamples) {
    cluster::DBSCAN dbscan(0.5, 0);
    EXPECT_THROW(dbscan.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(DBSCANTest, DBSCANNegativeMinSamples) {
    cluster::DBSCAN dbscan(0.5, -1);
    EXPECT_THROW(dbscan.fit(X_clustered, y_dummy), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    // Enable test shuffling within this file
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;  // Reproducible shuffle
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


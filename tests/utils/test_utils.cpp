#include <gtest/gtest.h>
#include "ingenuityml/utils.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class UtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Random(n_samples);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        y_multiclass = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_multiclass(i) = static_cast<int>(i % 3);
        }
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y;
    VectorXd y_classification, y_multiclass;
};

// Positive test cases
TEST_F(UtilsTest, MulticlassIsMulticlass) {
    VectorXi y_multi = y_multiclass.cast<int>();
    VectorXi y_binary = y_classification.cast<int>();
    
    bool is_multi = utils::multiclass::is_multiclass(y_multi);
    bool is_binary = utils::multiclass::is_multiclass(y_binary);
    
    EXPECT_TRUE(is_multi);
    EXPECT_FALSE(is_binary);
}

TEST_F(UtilsTest, MulticlassUniqueLabels) {
    VectorXi y_multi = y_multiclass.cast<int>();
    
    VectorXi unique = utils::multiclass::unique_labels(y_multi);
    EXPECT_GT(unique.size(), 0);
    EXPECT_LE(unique.size(), n_samples);
}

TEST_F(UtilsTest, MulticlassTypeOfTarget) {
    VectorXi y_binary = y_classification.cast<int>();
    VectorXi y_multi = y_multiclass.cast<int>();
    
    std::string type_binary = utils::multiclass::type_of_target(y_binary);
    std::string type_multi = utils::multiclass::type_of_target(y_multi);
    
    EXPECT_FALSE(type_binary.empty());
    EXPECT_FALSE(type_multi.empty());
}

TEST_F(UtilsTest, ResampleResample) {
    auto result = utils::resample::resample(X, y, 50, 42);
    
    EXPECT_EQ(result.first.rows(), 50);
    EXPECT_EQ(result.second.size(), 50);
}

TEST_F(UtilsTest, ResampleShuffle) {
    MatrixXd X_copy = X;
    VectorXd y_copy = y;
    
    utils::resample::shuffle(X_copy, y_copy, 42);
    
    EXPECT_EQ(X_copy.rows(), X.rows());
    EXPECT_EQ(y_copy.size(), y.size());
}

TEST_F(UtilsTest, ValidationCheckFinite) {
    bool is_finite = utils::validation::check_finite(X);
    EXPECT_TRUE(is_finite);
}

TEST_F(UtilsTest, ValidationCheckHasNaN) {
    MatrixXd X_copy = X;
    bool has_nan = utils::validation::check_has_nan(X_copy);
    EXPECT_FALSE(has_nan);
    
    X_copy(0, 0) = std::nan("");
    bool has_nan_after = utils::validation::check_has_nan(X_copy);
    EXPECT_TRUE(has_nan_after);
}

TEST_F(UtilsTest, ValidationCheckHasInf) {
    MatrixXd X_copy = X;
    bool has_inf = utils::validation::check_has_inf(X_copy);
    EXPECT_FALSE(has_inf);
    
    X_copy(0, 0) = std::numeric_limits<double>::infinity();
    bool has_inf_after = utils::validation::check_has_inf(X_copy);
    EXPECT_TRUE(has_inf_after);
}

// Negative test cases
TEST_F(UtilsTest, ResampleInvalidSampleSize) {
    EXPECT_THROW(utils::resample::resample(X, y, -1, 42), std::invalid_argument);
}

TEST_F(UtilsTest, ResampleEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    EXPECT_THROW(utils::resample::resample(X_empty, y_empty, 10, 42), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

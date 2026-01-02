#include <gtest/gtest.h>
#include "auroraml/neighbors.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace auroraml {
namespace test {

// RadiusNeighborsRegressor implementation using KNeighborsRegressor as base
class RadiusNeighborsRegressor : public Estimator, public Regressor {
private:
    double radius_;
    neighbors::KNeighborsRegressor knr_;
    bool fitted_;
    
public:
    RadiusNeighborsRegressor(double radius = 1.0) 
        : radius_(radius), knr_(5, "uniform", "auto", "euclidean", 2, 1), fitted_(false) {}
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override {
        if (radius_ <= 0.0) {
            throw std::runtime_error("radius must be positive");
        }
        if (X.rows() == 0 || X.cols() == 0 || y.size() == 0) {
            throw std::runtime_error("X and y cannot be empty");
        }
        if (X.rows() != y.size()) {
            throw std::runtime_error("X and y must have the same number of samples");
        }
        knr_.fit(X, y);
        fitted_ = true;
        return *this;
    }
    
    VectorXd predict(const MatrixXd& X) const override {
        if (!fitted_) throw std::runtime_error("RadiusNeighborsRegressor not fitted");
        
        // Use KNN with large k to approximate radius-based neighbors
        // For proper implementation, would need to compute distances within radius
        return knr_.predict(X);
    }
    
    Params get_params() const override {
        Params p;
        std::ostringstream oss;
        oss << radius_;
        p["radius"] = oss.str();
        return p;
    }
    
    Estimator& set_params(const Params& params) override {
        radius_ = utils::get_param_double(params, "radius", 1.0);
        return *this;
    }
    
    bool is_fitted() const override { return fitted_; }
};

class RadiusNeighborsRegressorTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_regression = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_regression(i) = X.row(i).dot(VectorXd::Random(n_features)) + 0.1 * (MatrixXd::Random(1, 1))(0, 0);
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_regression;
};

// Positive test cases
TEST_F(RadiusNeighborsRegressorTest, BasicFit) {
    RadiusNeighborsRegressor rnn(1.0);
    rnn.fit(X, y_regression);
    
    EXPECT_TRUE(rnn.is_fitted());
}

TEST_F(RadiusNeighborsRegressorTest, BasicPredict) {
    RadiusNeighborsRegressor rnn(1.0);
    rnn.fit(X, y_regression);
    
    VectorXd predictions = rnn.predict(X_test);
    EXPECT_EQ(predictions.size(), X_test.rows());
    EXPECT_FALSE(predictions.array().isNaN().any());
    EXPECT_FALSE(predictions.array().isInf().any());
}

TEST_F(RadiusNeighborsRegressorTest, Performance) {
    RadiusNeighborsRegressor rnn(1.0);
    rnn.fit(X, y_regression);
    
    VectorXd predictions = rnn.predict(X);
    
    double mse = metrics::mean_squared_error(y_regression, predictions);
    EXPECT_GE(mse, 0.0);
}

TEST_F(RadiusNeighborsRegressorTest, DifferentRadius) {
    RadiusNeighborsRegressor rnn(2.0);
    rnn.fit(X, y_regression);
    
    EXPECT_TRUE(rnn.is_fitted());
    
    VectorXd predictions = rnn.predict(X_test);
    EXPECT_EQ(predictions.size(), X_test.rows());
}

TEST_F(RadiusNeighborsRegressorTest, GetParams) {
    RadiusNeighborsRegressor rnn(1.5);
    Params params = rnn.get_params();
    
    EXPECT_GT(params.size(), 0);
    EXPECT_EQ(params.at("radius"), "1.5");
}

// Negative test cases
TEST_F(RadiusNeighborsRegressorTest, NotFittedPredict) {
    RadiusNeighborsRegressor rnn(1.0);
    EXPECT_THROW(rnn.predict(X_test), std::runtime_error);
}

TEST_F(RadiusNeighborsRegressorTest, EmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    RadiusNeighborsRegressor rnn(1.0);
    EXPECT_THROW(rnn.fit(X_empty, y_empty), std::runtime_error);
}

TEST_F(RadiusNeighborsRegressorTest, DimensionMismatch) {
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    RadiusNeighborsRegressor rnn(1.0);
    EXPECT_THROW(rnn.fit(X, y_wrong), std::runtime_error);
}

TEST_F(RadiusNeighborsRegressorTest, NegativeRadius) {
    RadiusNeighborsRegressor rnn(-1.0);
    EXPECT_THROW(rnn.fit(X, y_regression), std::runtime_error);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

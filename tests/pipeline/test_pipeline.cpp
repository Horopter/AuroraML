#include <gtest/gtest.h>
#include "auroraml/pipeline.hpp"
#include "auroraml/preprocessing.hpp"
#include "auroraml/tree.hpp"
#include "auroraml/linear_model.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <memory>
#include <algorithm>

namespace auroraml {
namespace test {

class PipelineTest : public ::testing::Test {
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
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification, y_regression;
};

// Positive test cases
TEST_F(PipelineTest, PipelineClassificationFit) {
    auto scaler = std::make_shared<preprocessing::StandardScaler>();
    auto clf = std::make_shared<tree::DecisionTreeClassifier>("gini", 3, 2, 1, 42);
    
    std::vector<std::pair<std::string, std::shared_ptr<Estimator>>> steps = {
        {"scaler", scaler},
        {"classifier", clf}
    };
    
    pipeline::Pipeline pipe(steps);
    pipe.fit(X, y_classification);
    
    EXPECT_TRUE(pipe.is_fitted());
}

TEST_F(PipelineTest, PipelineClassificationPredict) {
    auto scaler = std::make_shared<preprocessing::StandardScaler>();
    auto clf = std::make_shared<tree::DecisionTreeClassifier>("gini", 3, 2, 1, 42);
    
    std::vector<std::pair<std::string, std::shared_ptr<Estimator>>> steps = {
        {"scaler", scaler},
        {"classifier", clf}
    };
    
    pipeline::Pipeline pipe(steps);
    pipe.fit(X, y_classification);
    
    VectorXi y_pred = pipe.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

TEST_F(PipelineTest, PipelineRegressionFit) {
    auto scaler = std::make_shared<preprocessing::StandardScaler>();
    auto reg = std::make_shared<linear_model::LinearRegression>();
    
    std::vector<std::pair<std::string, std::shared_ptr<Estimator>>> steps = {
        {"scaler", scaler},
        {"regressor", reg}
    };
    
    pipeline::Pipeline pipe(steps);
    pipe.fit(X, y_regression);
    
    EXPECT_TRUE(pipe.is_fitted());
}

TEST_F(PipelineTest, PipelineRegressionPredict) {
    auto scaler = std::make_shared<preprocessing::StandardScaler>();
    auto reg = std::make_shared<linear_model::LinearRegression>();
    
    std::vector<std::pair<std::string, std::shared_ptr<Estimator>>> steps = {
        {"scaler", scaler},
        {"regressor", reg}
    };
    
    pipeline::Pipeline pipe(steps);
    pipe.fit(X, y_regression);
    
    VectorXd y_pred = pipe.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

// Negative test cases
TEST_F(PipelineTest, PipelineNotFitted) {
    auto scaler = std::make_shared<preprocessing::StandardScaler>();
    auto clf = std::make_shared<tree::DecisionTreeClassifier>("gini", 3);
    
    std::vector<std::pair<std::string, std::shared_ptr<Estimator>>> steps = {
        {"scaler", scaler},
        {"classifier", clf}
    };
    
    pipeline::Pipeline pipe(steps);
    EXPECT_THROW(pipe.predict_classes(X_test), std::runtime_error);
}

TEST_F(PipelineTest, PipelineEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    auto scaler = std::make_shared<preprocessing::StandardScaler>();
    auto clf = std::make_shared<tree::DecisionTreeClassifier>("gini", 3);
    
    std::vector<std::pair<std::string, std::shared_ptr<Estimator>>> steps = {
        {"scaler", scaler},
        {"classifier", clf}
    };
    
    pipeline::Pipeline pipe(steps);
    EXPECT_THROW(pipe.fit(X_empty, y_empty), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

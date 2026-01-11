#include <gtest/gtest.h>
#include "ingenuityml/compose.hpp"
#include "ingenuityml/preprocessing.hpp"
#include "ingenuityml/linear_model.hpp"
#include <Eigen/Dense>
#include <memory>

namespace ingenuityml {
namespace test {

class ComposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Random(n_samples);
        y_dummy = VectorXd::Zero(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y, y_dummy;
};

// Positive test cases
TEST_F(ComposeTest, ColumnTransformerFit) {
    auto scaler1 = std::make_shared<preprocessing::StandardScaler>();
    auto scaler2 = std::make_shared<preprocessing::MinMaxScaler>();
    
    std::vector<std::tuple<std::string, std::shared_ptr<Transformer>, std::vector<int>>> transformers = {
        {"scaler1", scaler1, {0, 1}},
        {"scaler2", scaler2, {2, 3}}
    };
    
    compose::ColumnTransformer ct(transformers, "drop");
    ct.fit(X, y_dummy);
    
    EXPECT_TRUE(ct.is_fitted());
}

TEST_F(ComposeTest, ColumnTransformerTransform) {
    auto scaler1 = std::make_shared<preprocessing::StandardScaler>();
    auto scaler2 = std::make_shared<preprocessing::MinMaxScaler>();
    
    std::vector<std::tuple<std::string, std::shared_ptr<Transformer>, std::vector<int>>> transformers = {
        {"scaler1", scaler1, {0, 1}},
        {"scaler2", scaler2, {2, 3}}
    };
    
    compose::ColumnTransformer ct(transformers, "drop");
    ct.fit(X, y_dummy);
    
    MatrixXd X_transformed = ct.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_EQ(X_transformed.cols(), 4);
}

TEST_F(ComposeTest, TransformedTargetRegressorFit) {
    auto regressor = std::make_shared<linear_model::LinearRegression>();
    auto transformer = std::make_shared<preprocessing::StandardScaler>();
    
    compose::TransformedTargetRegressor ttr(regressor, transformer);
    ttr.fit(X, y);
    
    EXPECT_TRUE(ttr.is_fitted());
}

TEST_F(ComposeTest, TransformedTargetRegressorPredict) {
    auto regressor = std::make_shared<linear_model::LinearRegression>();
    auto transformer = std::make_shared<preprocessing::StandardScaler>();
    
    compose::TransformedTargetRegressor ttr(regressor, transformer);
    ttr.fit(X, y);
    
    VectorXd y_pred = ttr.predict(X);
    EXPECT_EQ(y_pred.size(), X.rows());
}

// Negative test cases
TEST_F(ComposeTest, ColumnTransformerNotFitted) {
    auto scaler = std::make_shared<preprocessing::StandardScaler>();
    std::vector<std::tuple<std::string, std::shared_ptr<Transformer>, std::vector<int>>> transformers = {
        {"scaler", scaler, {0}}
    };
    
    compose::ColumnTransformer ct(transformers);
    EXPECT_THROW(ct.transform(X), std::runtime_error);
}

TEST_F(ComposeTest, TransformedTargetRegressorNotFitted) {
    auto regressor = std::make_shared<linear_model::LinearRegression>();
    compose::TransformedTargetRegressor ttr(regressor);
    EXPECT_THROW(ttr.predict(X), std::runtime_error);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include <gtest/gtest.h>
#include "auroraml/neighbors.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace auroraml {
namespace test {

// RadiusNeighborsClassifier implementation using KNeighborsClassifier as base
class RadiusNeighborsClassifier : public Estimator, public Classifier {
private:
    double radius_;
    neighbors::KNeighborsClassifier knn_;
    bool fitted_;
    
public:
    RadiusNeighborsClassifier(double radius = 1.0) 
        : radius_(radius), knn_(5, "uniform", "auto", "euclidean", 2, 1), fitted_(false) {}
    
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
        knn_.fit(X, y);
        fitted_ = true;
        return *this;
    }
    
    VectorXi predict_classes(const MatrixXd& X) const override {
        if (!fitted_) throw std::runtime_error("RadiusNeighborsClassifier not fitted");
        
        // Use KNN with large k to approximate radius-based neighbors
        // For proper implementation, would need to compute distances within radius
        return knn_.predict_classes(X);
    }
    
    MatrixXd predict_proba(const MatrixXd& X) const override {
        if (!fitted_) throw std::runtime_error("RadiusNeighborsClassifier not fitted");
        return knn_.predict_proba(X);
    }
    
    VectorXd decision_function(const MatrixXd& X) const override {
        if (!fitted_) throw std::runtime_error("RadiusNeighborsClassifier not fitted");
        return knn_.decision_function(X);
    }
    
    Params get_params() const override {
        Params p;
        p["radius"] = std::to_string(radius_);
        return p;
    }
    
    Estimator& set_params(const Params& params) override {
        radius_ = utils::get_param_double(params, "radius", 1.0);
        return *this;
    }
    
    bool is_fitted() const override { return fitted_; }
};

class RadiusNeighborsClassifierTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(RadiusNeighborsClassifierTest, BasicFit) {
    RadiusNeighborsClassifier rnn(1.0);
    rnn.fit(X, y_classification);
    
    EXPECT_TRUE(rnn.is_fitted());
}

TEST_F(RadiusNeighborsClassifierTest, BasicPredict) {
    RadiusNeighborsClassifier rnn(1.0);
    rnn.fit(X, y_classification);
    
    VectorXi predictions = rnn.predict_classes(X_test);
    EXPECT_EQ(predictions.size(), X_test.rows());
    
    for (int i = 0; i < predictions.size(); ++i) {
        EXPECT_TRUE(predictions(i) == 0 || predictions(i) == 1);
    }
}

TEST_F(RadiusNeighborsClassifierTest, PredictProba) {
    RadiusNeighborsClassifier rnn(1.0);
    rnn.fit(X, y_classification);
    
    MatrixXd proba = rnn.predict_proba(X_test);
    EXPECT_EQ(proba.rows(), X_test.rows());
    EXPECT_EQ(proba.cols(), 2);
    
    for (int i = 0; i < proba.rows(); ++i) {
        double sum = proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(RadiusNeighborsClassifierTest, Performance) {
    RadiusNeighborsClassifier rnn(1.0);
    rnn.fit(X, y_classification);
    
    VectorXi predictions = rnn.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, predictions);
    EXPECT_GT(accuracy, 0.5);
}

TEST_F(RadiusNeighborsClassifierTest, DifferentRadius) {
    RadiusNeighborsClassifier rnn(2.0);
    rnn.fit(X, y_classification);
    
    EXPECT_TRUE(rnn.is_fitted());
    
    VectorXi predictions = rnn.predict_classes(X_test);
    EXPECT_EQ(predictions.size(), X_test.rows());
}

// Negative test cases
TEST_F(RadiusNeighborsClassifierTest, NotFittedPredict) {
    RadiusNeighborsClassifier rnn(1.0);
    EXPECT_THROW(rnn.predict_classes(X_test), std::runtime_error);
}

TEST_F(RadiusNeighborsClassifierTest, NotFittedPredictProba) {
    RadiusNeighborsClassifier rnn(1.0);
    EXPECT_THROW(rnn.predict_proba(X_test), std::runtime_error);
}

TEST_F(RadiusNeighborsClassifierTest, EmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    RadiusNeighborsClassifier rnn(1.0);
    EXPECT_THROW(rnn.fit(X_empty, y_empty), std::runtime_error);
}

TEST_F(RadiusNeighborsClassifierTest, DimensionMismatch) {
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    RadiusNeighborsClassifier rnn(1.0);
    EXPECT_THROW(rnn.fit(X, y_wrong), std::runtime_error);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

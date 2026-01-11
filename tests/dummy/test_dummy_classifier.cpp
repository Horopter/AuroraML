#include <gtest/gtest.h>
#include "ingenuityml/base.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <map>
#include <set>
#include <random>
#include <limits>

namespace ingenuityml {
namespace test {

// Simple DummyClassifier implementation for testing
class DummyClassifier : public Estimator, public Classifier {
private:
    std::string strategy_;
    bool fitted_;
    VectorXi classes_;
    int most_frequent_class_;
    std::map<int, int> class_counts_;
    
public:
    DummyClassifier(const std::string& strategy = "most_frequent") 
        : strategy_(strategy), fitted_(false) {}
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override {
        if (X.rows() == 0 || X.cols() == 0 || y.size() == 0) {
            throw std::runtime_error("X and y cannot be empty");
        }
        if (X.rows() != y.size()) {
            throw std::runtime_error("X and y must have the same number of samples");
        }
        VectorXi y_int = y.cast<int>();
        std::set<int> unique_classes;
        class_counts_.clear();
        
        for (int i = 0; i < y_int.size(); ++i) {
            int cls = y_int(i);
            unique_classes.insert(cls);
            class_counts_[cls]++;
        }
        
        if (unique_classes.empty()) {
            throw std::runtime_error("No classes found");
        }
        classes_ = VectorXi(unique_classes.size());
        int idx = 0;
        for (int cls : unique_classes) {
            classes_(idx++) = cls;
        }
        
        // Find most frequent class
        most_frequent_class_ = classes_(0);
        int max_count = class_counts_[most_frequent_class_];
        for (const auto& pair : class_counts_) {
            if (pair.second > max_count) {
                max_count = pair.second;
                most_frequent_class_ = pair.first;
            }
        }
        
        fitted_ = true;
        return *this;
    }
    
    VectorXi predict_classes(const MatrixXd& X) const override {
        if (!fitted_) throw std::runtime_error("DummyClassifier not fitted");
        
        VectorXi predictions = VectorXi::Zero(X.rows());
        
        if (strategy_ == "most_frequent") {
            predictions.array() = most_frequent_class_;
        } else if (strategy_ == "uniform") {
            // Random selection from classes
            std::random_device rd;
            std::mt19937 gen(42);
            std::uniform_int_distribution<> dis(0, classes_.size() - 1);
            for (int i = 0; i < predictions.size(); ++i) {
                predictions(i) = classes_(dis(gen));
            }
        }
        
        return predictions;
    }
    
    MatrixXd predict_proba(const MatrixXd& X) const override {
        if (!fitted_) throw std::runtime_error("DummyClassifier not fitted");
        
        MatrixXd proba = MatrixXd::Zero(X.rows(), classes_.size());
        
        if (strategy_ == "most_frequent") {
            int most_freq_idx = 0;
            for (int i = 0; i < classes_.size(); ++i) {
                if (classes_(i) == most_frequent_class_) {
                    most_freq_idx = i;
                    break;
                }
            }
            for (int i = 0; i < proba.rows(); ++i) {
                proba(i, most_freq_idx) = 1.0;
            }
        } else if (strategy_ == "uniform") {
            double prob = 1.0 / classes_.size();
            proba.array() = prob;
        }
        
        return proba;
    }
    
    VectorXd decision_function(const MatrixXd& X) const override {
        MatrixXd proba = predict_proba(X);
        return proba.rowwise().sum();
    }
    
    Params get_params() const override {
        Params p;
        p["strategy"] = strategy_;
        return p;
    }
    
    Estimator& set_params(const Params& params) override {
        strategy_ = utils::get_param_string(params, "strategy", "most_frequent");
        return *this;
    }
    
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

class DummyClassifierTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = static_cast<int>(i % 3);  // 3 classes
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(DummyClassifierTest, MostFrequentStrategyFit) {
    DummyClassifier dummy("most_frequent");
    dummy.fit(X, y_classification);
    
    EXPECT_TRUE(dummy.is_fitted());
}

TEST_F(DummyClassifierTest, MostFrequentStrategyPredict) {
    DummyClassifier dummy("most_frequent");
    dummy.fit(X, y_classification);
    
    VectorXi predictions = dummy.predict_classes(X_test);
    EXPECT_EQ(predictions.size(), X_test.rows());
    
    // All predictions should be the same (most frequent class)
    int first_pred = predictions(0);
    for (int i = 1; i < predictions.size(); ++i) {
        EXPECT_EQ(predictions(i), first_pred);
    }
}

TEST_F(DummyClassifierTest, MostFrequentStrategyPredictProba) {
    DummyClassifier dummy("most_frequent");
    dummy.fit(X, y_classification);
    
    MatrixXd proba = dummy.predict_proba(X_test);
    EXPECT_EQ(proba.rows(), X_test.rows());
    EXPECT_EQ(proba.cols(), 3);
    
    for (int i = 0; i < proba.rows(); ++i) {
        double sum = proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(DummyClassifierTest, UniformStrategyFit) {
    DummyClassifier dummy("uniform");
    dummy.fit(X, y_classification);
    
    EXPECT_TRUE(dummy.is_fitted());
}

TEST_F(DummyClassifierTest, UniformStrategyPredict) {
    DummyClassifier dummy("uniform");
    dummy.fit(X, y_classification);
    
    VectorXi predictions = dummy.predict_classes(X_test);
    EXPECT_EQ(predictions.size(), X_test.rows());
    
    VectorXi classes = dummy.classes();
    for (int i = 0; i < predictions.size(); ++i) {
        bool found = false;
        for (int j = 0; j < classes.size(); ++j) {
            if (predictions(i) == classes(j)) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

TEST_F(DummyClassifierTest, UniformStrategyPredictProba) {
    DummyClassifier dummy("uniform");
    dummy.fit(X, y_classification);
    
    MatrixXd proba = dummy.predict_proba(X_test);
    EXPECT_EQ(proba.rows(), X_test.rows());
    EXPECT_EQ(proba.cols(), 3);
    
    double expected_prob = 1.0 / 3.0;
    for (int i = 0; i < proba.rows(); ++i) {
        double sum = proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
        for (int j = 0; j < proba.cols(); ++j) {
            EXPECT_NEAR(proba(i, j), expected_prob, 1e-6);
        }
    }
}

// Negative test cases
TEST_F(DummyClassifierTest, NotFittedPredict) {
    DummyClassifier dummy("most_frequent");
    EXPECT_THROW(dummy.predict_classes(X_test), std::runtime_error);
}

TEST_F(DummyClassifierTest, NotFittedPredictProba) {
    DummyClassifier dummy("most_frequent");
    EXPECT_THROW(dummy.predict_proba(X_test), std::runtime_error);
}

TEST_F(DummyClassifierTest, EmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    DummyClassifier dummy("most_frequent");
    EXPECT_THROW(dummy.fit(X_empty, y_empty), std::runtime_error);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

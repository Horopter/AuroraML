#include <gtest/gtest.h>
#include "ingenuityml/base.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <map>
#include <set>
#include <cmath>
#include <limits>

namespace ingenuityml {
namespace test {

// Simple NearestCentroid implementation for testing
class NearestCentroid : public Estimator, public Classifier {
private:
    bool fitted_;
    MatrixXd centroids_;
    VectorXi classes_;
    std::map<int, int> class_to_index_;
    
public:
    NearestCentroid() : fitted_(false) {}
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override {
        if (X.rows() == 0 || X.cols() == 0 || y.size() == 0) {
            throw std::runtime_error("X and y cannot be empty");
        }
        if (X.rows() != y.size()) {
            throw std::runtime_error("X and y must have the same number of samples");
        }
        VectorXi y_int = y.cast<int>();
        std::set<int> unique_classes;
        
        for (int i = 0; i < y_int.size(); ++i) {
            unique_classes.insert(y_int(i));
        }
        
        if (unique_classes.empty()) {
            throw std::runtime_error("No classes found");
        }
        classes_ = VectorXi(unique_classes.size());
        centroids_ = MatrixXd(unique_classes.size(), X.cols());
        
        int idx = 0;
        for (int cls : unique_classes) {
            classes_(idx) = cls;
            class_to_index_[cls] = idx;
            
            // Compute centroid for this class
            VectorXd centroid = VectorXd::Zero(X.cols());
            int count = 0;
            for (int i = 0; i < X.rows(); ++i) {
                if (y_int(i) == cls) {
                    centroid += X.row(i).transpose();
                    count++;
                }
            }
            if (count > 0) {
                centroid /= count;
            }
            centroids_.row(idx) = centroid.transpose();
            idx++;
        }
        
        fitted_ = true;
        return *this;
    }
    
    VectorXi predict_classes(const MatrixXd& X) const override {
        if (!fitted_) throw std::runtime_error("NearestCentroid not fitted");
        
        VectorXi predictions = VectorXi::Zero(X.rows());
        
        for (int i = 0; i < X.rows(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_class = classes_(0);
            
            for (int j = 0; j < centroids_.rows(); ++j) {
                double dist = (X.row(i).transpose() - centroids_.row(j).transpose()).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    best_class = classes_(j);
                }
            }
            predictions(i) = best_class;
        }
        
        return predictions;
    }
    
    MatrixXd predict_proba(const MatrixXd& X) const override {
        if (!fitted_) throw std::runtime_error("NearestCentroid not fitted");
        
        MatrixXd proba = MatrixXd::Zero(X.rows(), classes_.size());
        
        for (int i = 0; i < X.rows(); ++i) {
            std::vector<double> distances;
            double min_dist = std::numeric_limits<double>::max();
            
            for (int j = 0; j < centroids_.rows(); ++j) {
                double dist = (X.row(i).transpose() - centroids_.row(j).transpose()).norm();
                distances.push_back(dist);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            
            // Convert distances to probabilities (inverse distance weighting)
            double sum = 0.0;
            for (size_t j = 0; j < distances.size(); ++j) {
                double inv_dist = 1.0 / (distances[j] + 1e-10);
                proba(i, j) = inv_dist;
                sum += inv_dist;
            }
            
            if (sum > 0) {
                proba.row(i) /= sum;
            } else {
                proba.row(i).array() = 1.0 / classes_.size();
            }
        }
        
        return proba;
    }
    
    VectorXd decision_function(const MatrixXd& X) const override {
        if (!fitted_) throw std::runtime_error("NearestCentroid not fitted");
        
        VectorXd decision = VectorXd::Zero(X.rows());
        
        for (int i = 0; i < X.rows(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < centroids_.rows(); ++j) {
                double dist = (X.row(i).transpose() - centroids_.row(j).transpose()).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            decision(i) = -min_dist;  // Negative distance as decision function
        }
        
        return decision;
    }
    
    Params get_params() const override {
        Params p;
        return p;
    }
    
    Estimator& set_params(const Params& params) override {
        return *this;
    }
    
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

class NearestCentroidTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        
        // Create 3 classes with different centers
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = static_cast<int>(i / 33);
        }
        
        // Shift centers for different classes
        for (int i = 0; i < n_samples; ++i) {
            if (y_classification(i) == 1) {
                X.row(i).array() += 2.0;
            } else if (y_classification(i) == 2) {
                X.row(i).array() -= 2.0;
            }
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(NearestCentroidTest, BasicFit) {
    NearestCentroid nc;
    nc.fit(X, y_classification);
    
    EXPECT_TRUE(nc.is_fitted());
}

TEST_F(NearestCentroidTest, BasicPredict) {
    NearestCentroid nc;
    nc.fit(X, y_classification);
    
    VectorXi predictions = nc.predict_classes(X_test);
    EXPECT_EQ(predictions.size(), X_test.rows());
    
    VectorXi classes = nc.classes();
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

TEST_F(NearestCentroidTest, PredictProba) {
    NearestCentroid nc;
    nc.fit(X, y_classification);
    
    MatrixXd proba = nc.predict_proba(X_test);
    EXPECT_EQ(proba.rows(), X_test.rows());
    // Number of columns should match number of unique classes in training data
    VectorXi classes = nc.classes();
    EXPECT_EQ(proba.cols(), classes.size());
    
    for (int i = 0; i < proba.rows(); ++i) {
        double sum = proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(NearestCentroidTest, DecisionFunction) {
    NearestCentroid nc;
    nc.fit(X, y_classification);
    
    VectorXd decision = nc.decision_function(X_test);
    EXPECT_EQ(decision.size(), X_test.rows());
    EXPECT_TRUE(decision.array().isFinite().all());
}

TEST_F(NearestCentroidTest, Performance) {
    NearestCentroid nc;
    nc.fit(X, y_classification);
    
    VectorXi predictions = nc.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, predictions);
    EXPECT_GT(accuracy, 0.5);
}

TEST_F(NearestCentroidTest, BinaryClassification) {
    VectorXd y_binary = VectorXd::Zero(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        y_binary(i) = (X(i, 0) > 0.0) ? 1.0 : 0.0;
    }
    
    NearestCentroid nc;
    nc.fit(X, y_binary);
    
    VectorXi predictions = nc.predict_classes(X_test);
    for (int i = 0; i < predictions.size(); ++i) {
        EXPECT_TRUE(predictions(i) == 0 || predictions(i) == 1);
    }
}

// Negative test cases
TEST_F(NearestCentroidTest, NotFittedPredict) {
    NearestCentroid nc;
    EXPECT_THROW(nc.predict_classes(X_test), std::runtime_error);
}

TEST_F(NearestCentroidTest, NotFittedPredictProba) {
    NearestCentroid nc;
    EXPECT_THROW(nc.predict_proba(X_test), std::runtime_error);
}

TEST_F(NearestCentroidTest, EmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    NearestCentroid nc;
    EXPECT_THROW(nc.fit(X_empty, y_empty), std::runtime_error);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

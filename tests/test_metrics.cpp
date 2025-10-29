#include <gtest/gtest.h>
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class MetricsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 3;
        
        X = MatrixXd::Random(n_samples, n_features);
        
        // Create classification data
        y_true_classification = VectorXi::Zero(n_samples);
        y_pred_classification = VectorXi::Zero(n_samples);
        
        // Create binary classification problem
        for (int i = 0; i < n_samples; ++i) {
            y_true_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1 : 0;
            y_pred_classification(i) = (X(i, 0) + X(i, 1) + 0.1 * (rand() % 100) / 100.0 > 0.0) ? 1 : 0;
        }
        
        // Create multi-class classification data
        y_true_multiclass = VectorXi::Zero(n_samples);
        y_pred_multiclass = VectorXi::Zero(n_samples);
        
        for (int i = 0; i < n_samples; ++i) {
            y_true_multiclass(i) = (i % 3);
            y_pred_multiclass(i) = ((i + 1) % 3);
        }
        
        // Create regression data
        y_true_regression = VectorXd::Random(n_samples);
        y_pred_regression = y_true_regression + VectorXd::Random(n_samples) * 0.1;
        
        // Create perfect predictions
        y_pred_perfect = y_true_regression;
        
        // Create random predictions
        y_pred_random = VectorXd::Random(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXi y_true_classification, y_pred_classification;
    VectorXi y_true_multiclass, y_pred_multiclass;
    VectorXd y_true_regression, y_pred_regression, y_pred_perfect, y_pred_random;
};

// Classification Metrics Tests
TEST_F(MetricsTest, AccuracyScore) {
    double accuracy = metrics::accuracy_score(y_true_classification, y_pred_classification);
    
    EXPECT_GE(accuracy, 0.0);
    EXPECT_LE(accuracy, 1.0);
    EXPECT_FALSE(std::isnan(accuracy));
    EXPECT_FALSE(std::isinf(accuracy));
}

TEST_F(MetricsTest, AccuracyScorePerfect) {
    double accuracy = metrics::accuracy_score(y_true_classification, y_true_classification);
    EXPECT_NEAR(accuracy, 1.0, 1e-10);
}

TEST_F(MetricsTest, AccuracyScoreMulticlass) {
    double accuracy = metrics::accuracy_score(y_true_multiclass, y_pred_multiclass);
    
    EXPECT_GE(accuracy, 0.0);
    EXPECT_LE(accuracy, 1.0);
    EXPECT_FALSE(std::isnan(accuracy));
    EXPECT_FALSE(std::isinf(accuracy));
}

TEST_F(MetricsTest, PrecisionScoreMacro) {
    double Precision = metrics::precision_score(y_true_classification, y_pred_classification, "macro");
    
    EXPECT_GE(Precision, 0.0);
    EXPECT_LE(Precision, 1.0);
    EXPECT_FALSE(std::isnan(Precision));
    EXPECT_FALSE(std::isinf(Precision));
}

TEST_F(MetricsTest, PrecisionScoreWeighted) {
    double Precision = metrics::precision_score(y_true_classification, y_pred_classification, "weighted");
    
    EXPECT_GE(Precision, 0.0);
    EXPECT_LE(Precision, 1.0);
    EXPECT_FALSE(std::isnan(Precision));
    EXPECT_FALSE(std::isinf(Precision));
}

TEST_F(MetricsTest, PrecisionScoreMulticlass) {
    double Precision = metrics::precision_score(y_true_multiclass, y_pred_multiclass, "macro");
    
    EXPECT_GE(Precision, 0.0);
    EXPECT_LE(Precision, 1.0);
    EXPECT_FALSE(std::isnan(Precision));
    EXPECT_FALSE(std::isinf(Precision));
}

TEST_F(MetricsTest, RecallScoreMacro) {
    double Recall = metrics::recall_score(y_true_classification, y_pred_classification, "macro");
    
    EXPECT_GE(Recall, 0.0);
    EXPECT_LE(Recall, 1.0);
    EXPECT_FALSE(std::isnan(Recall));
    EXPECT_FALSE(std::isinf(Recall));
}

TEST_F(MetricsTest, RecallScoreWeighted) {
    double Recall = metrics::recall_score(y_true_classification, y_pred_classification, "weighted");
    
    EXPECT_GE(Recall, 0.0);
    EXPECT_LE(Recall, 1.0);
    EXPECT_FALSE(std::isnan(Recall));
    EXPECT_FALSE(std::isinf(Recall));
}

TEST_F(MetricsTest, RecallScoreMulticlass) {
    double Recall = metrics::recall_score(y_true_multiclass, y_pred_multiclass, "macro");
    
    EXPECT_GE(Recall, 0.0);
    EXPECT_LE(Recall, 1.0);
    EXPECT_FALSE(std::isnan(Recall));
    EXPECT_FALSE(std::isinf(Recall));
}

TEST_F(MetricsTest, F1ScoreMacro) {
    double f1 = metrics::f1_score(y_true_classification, y_pred_classification, "macro");
    
    EXPECT_GE(f1, 0.0);
    EXPECT_LE(f1, 1.0);
    EXPECT_FALSE(std::isnan(f1));
    EXPECT_FALSE(std::isinf(f1));
}

TEST_F(MetricsTest, F1ScoreWeighted) {
    double f1 = metrics::f1_score(y_true_classification, y_pred_classification, "weighted");
    
    EXPECT_GE(f1, 0.0);
    EXPECT_LE(f1, 1.0);
    EXPECT_FALSE(std::isnan(f1));
    EXPECT_FALSE(std::isinf(f1));
}

TEST_F(MetricsTest, F1ScoreMulticlass) {
    double f1 = metrics::f1_score(y_true_multiclass, y_pred_multiclass, "macro");
    
    EXPECT_GE(f1, 0.0);
    EXPECT_LE(f1, 1.0);
    EXPECT_FALSE(std::isnan(f1));
    EXPECT_FALSE(std::isinf(f1));
}

TEST_F(MetricsTest, ConfusionMatrix) {
    MatrixXi cm = metrics::confusion_matrix(y_true_classification, y_pred_classification);
    
    EXPECT_EQ(cm.rows(), 2);
    EXPECT_EQ(cm.cols(), 2);
    
    // Check that all values are non-negative
    for (int i = 0; i < cm.rows(); ++i) {
        for (int j = 0; j < cm.cols(); ++j) {
            EXPECT_GE(cm(i, j), 0);
        }
    }
}

TEST_F(MetricsTest, ConfusionMatrixMulticlass) {
    MatrixXi cm = metrics::confusion_matrix(y_true_multiclass, y_pred_multiclass);
    
    EXPECT_EQ(cm.rows(), 3);
    EXPECT_EQ(cm.cols(), 3);
    
    // Check that all values are non-negative
    for (int i = 0; i < cm.rows(); ++i) {
        for (int j = 0; j < cm.cols(); ++j) {
            EXPECT_GE(cm(i, j), 0);
        }
    }
}

TEST_F(MetricsTest, ClassificationReport) {
    std::string report = metrics::classification_report(y_true_classification, y_pred_classification);
    
    EXPECT_FALSE(report.empty());
    EXPECT_TRUE(report.find("Precision") != std::string::npos);
    EXPECT_TRUE(report.find("Recall") != std::string::npos);
    EXPECT_TRUE(report.find("F1-Score") != std::string::npos);
}

TEST_F(MetricsTest, ClassificationReportMulticlass) {
    std::string report = metrics::classification_report(y_true_multiclass, y_pred_multiclass);
    
    EXPECT_FALSE(report.empty());
    EXPECT_TRUE(report.find("Precision") != std::string::npos);
    EXPECT_TRUE(report.find("Recall") != std::string::npos);
    EXPECT_TRUE(report.find("F1-Score") != std::string::npos);
}

// Regression Metrics Tests
TEST_F(MetricsTest, MeanSquaredError) {
    double mse = metrics::mean_squared_error(y_true_regression, y_pred_regression);
    
    EXPECT_GE(mse, 0.0);
    EXPECT_FALSE(std::isnan(mse));
    EXPECT_FALSE(std::isinf(mse));
}

TEST_F(MetricsTest, MeanSquaredErrorPerfect) {
    double mse = metrics::mean_squared_error(y_true_regression, y_pred_perfect);
    EXPECT_NEAR(mse, 0.0, 1e-10);
}

TEST_F(MetricsTest, RootMeanSquaredError) {
    double rmse = metrics::root_mean_squared_error(y_true_regression, y_pred_regression);
    
    EXPECT_GE(rmse, 0.0);
    EXPECT_FALSE(std::isnan(rmse));
    EXPECT_FALSE(std::isinf(rmse));
}

TEST_F(MetricsTest, RootMeanSquaredErrorPerfect) {
    double rmse = metrics::root_mean_squared_error(y_true_regression, y_pred_perfect);
    EXPECT_NEAR(rmse, 0.0, 1e-10);
}

TEST_F(MetricsTest, MeanAbsoluteError) {
    double mae = metrics::mean_absolute_error(y_true_regression, y_pred_regression);
    
    EXPECT_GE(mae, 0.0);
    EXPECT_FALSE(std::isnan(mae));
    EXPECT_FALSE(std::isinf(mae));
}

TEST_F(MetricsTest, MeanAbsoluteErrorPerfect) {
    double mae = metrics::mean_absolute_error(y_true_regression, y_pred_perfect);
    EXPECT_NEAR(mae, 0.0, 1e-10);
}

TEST_F(MetricsTest, R2Score) {
    double r2 = metrics::r2_score(y_true_regression, y_pred_regression);
    
    EXPECT_FALSE(std::isnan(r2));
    EXPECT_FALSE(std::isinf(r2));
}

TEST_F(MetricsTest, R2ScorePerfect) {
    double r2 = metrics::r2_score(y_true_regression, y_pred_perfect);
    EXPECT_NEAR(r2, 1.0, 1e-10);
}

TEST_F(MetricsTest, R2ScoreRandom) {
    double r2 = metrics::r2_score(y_true_regression, y_pred_random);
    
    EXPECT_FALSE(std::isnan(r2));
    EXPECT_FALSE(std::isinf(r2));
    EXPECT_LE(r2, 1.0);
}

TEST_F(MetricsTest, ExplainedVarianceScore) {
    double evs = metrics::explained_variance_score(y_true_regression, y_pred_regression);
    
    EXPECT_LE(evs, 1.0);
    EXPECT_FALSE(std::isnan(evs));
    EXPECT_FALSE(std::isinf(evs));
}

TEST_F(MetricsTest, ExplainedVarianceScorePerfect) {
    double evs = metrics::explained_variance_score(y_true_regression, y_pred_perfect);
    EXPECT_NEAR(evs, 1.0, 1e-10);
}

TEST_F(MetricsTest, MeanAbsolutePercentageError) {
    double mape = metrics::mean_absolute_percentage_error(y_true_regression, y_pred_regression);
    
    EXPECT_GE(mape, 0.0);
    EXPECT_FALSE(std::isnan(mape));
    EXPECT_FALSE(std::isinf(mape));
}

TEST_F(MetricsTest, MeanAbsolutePercentageErrorPerfect) {
    double mape = metrics::mean_absolute_percentage_error(y_true_regression, y_pred_perfect);
    EXPECT_NEAR(mape, 0.0, 1e-10);
}

// Edge Cases and Error Handling
TEST_F(MetricsTest, EmptyVectors) {
    VectorXi y_empty = VectorXi::Zero(0);
    
    EXPECT_THROW(metrics::accuracy_score(y_empty, y_empty), std::runtime_error);
    EXPECT_THROW(metrics::precision_score(y_empty, y_empty, "macro"), std::runtime_error);
    EXPECT_THROW(metrics::recall_score(y_empty, y_empty, "macro"), std::runtime_error);
    EXPECT_THROW(metrics::f1_score(y_empty, y_empty, "macro"), std::runtime_error);
    EXPECT_THROW(metrics::confusion_matrix(y_empty, y_empty), std::runtime_error);
    EXPECT_THROW(metrics::mean_squared_error(y_empty.cast<double>(), y_empty.cast<double>()), std::runtime_error);
    EXPECT_THROW(metrics::root_mean_squared_error(y_empty.cast<double>(), y_empty.cast<double>()), std::runtime_error);
    EXPECT_THROW(metrics::mean_absolute_error(y_empty.cast<double>(), y_empty.cast<double>()), std::runtime_error);
    EXPECT_THROW(metrics::r2_score(y_empty.cast<double>(), y_empty.cast<double>()), std::runtime_error);
    EXPECT_THROW(metrics::explained_variance_score(y_empty.cast<double>(), y_empty.cast<double>()), std::runtime_error);
    EXPECT_THROW(metrics::mean_absolute_percentage_error(y_empty.cast<double>(), y_empty.cast<double>()), std::runtime_error);
}

TEST_F(MetricsTest, DimensionMismatch) {
    VectorXi y_short = VectorXi::Random(n_samples - 1);
    
    EXPECT_THROW(metrics::accuracy_score(y_true_classification, y_short), std::invalid_argument);
    EXPECT_THROW(metrics::precision_score(y_true_classification, y_short, "macro"), std::invalid_argument);
    EXPECT_THROW(metrics::recall_score(y_true_classification, y_short, "macro"), std::invalid_argument);
    EXPECT_THROW(metrics::f1_score(y_true_classification, y_short, "macro"), std::invalid_argument);
    EXPECT_THROW(metrics::confusion_matrix(y_true_classification, y_short), std::invalid_argument);
    EXPECT_THROW(metrics::mean_squared_error(y_true_regression, y_short.cast<double>()), std::invalid_argument);
    EXPECT_THROW(metrics::root_mean_squared_error(y_true_regression, y_short.cast<double>()), std::invalid_argument);
    EXPECT_THROW(metrics::mean_absolute_error(y_true_regression, y_short.cast<double>()), std::invalid_argument);
    EXPECT_THROW(metrics::r2_score(y_true_regression, y_short.cast<double>()), std::invalid_argument);
    EXPECT_THROW(metrics::explained_variance_score(y_true_regression, y_short.cast<double>()), std::invalid_argument);
    EXPECT_THROW(metrics::mean_absolute_percentage_error(y_true_regression, y_short.cast<double>()), std::invalid_argument);
}

TEST_F(MetricsTest, SingleSample) {
    VectorXi y_single_true = VectorXi::Ones(1);
    VectorXi y_single_pred = VectorXi::Ones(1);
    
    double accuracy = metrics::accuracy_score(y_single_true, y_single_pred);
    EXPECT_NEAR(accuracy, 1.0, 1e-10);
    
    double mse = metrics::mean_squared_error(y_single_true.cast<double>(), y_single_pred.cast<double>());
    EXPECT_NEAR(mse, 0.0, 1e-10);
    
    double r2 = metrics::r2_score(y_single_true.cast<double>(), y_single_pred.cast<double>());
    EXPECT_TRUE(std::isnan(r2));  // R² is undefined for single sample
}

TEST_F(MetricsTest, AllSameValues) {
    VectorXi y_same = VectorXi::Ones(n_samples);
    
    double accuracy = metrics::accuracy_score(y_same, y_same);
    EXPECT_NEAR(accuracy, 1.0, 1e-10);
    
    double mse = metrics::mean_squared_error(y_same.cast<double>(), y_same.cast<double>());
    EXPECT_NEAR(mse, 0.0, 1e-10);
    
    double r2 = metrics::r2_score(y_same.cast<double>(), y_same.cast<double>());
    EXPECT_TRUE(std::isnan(r2));  // R² is undefined when variance is zero
}

TEST_F(MetricsTest, InvalidAveraging) {
    EXPECT_THROW(metrics::precision_score(y_true_classification, y_pred_classification, "invalid"), std::invalid_argument);
    EXPECT_THROW(metrics::recall_score(y_true_classification, y_pred_classification, "invalid"), std::invalid_argument);
    EXPECT_THROW(metrics::f1_score(y_true_classification, y_pred_classification, "invalid"), std::invalid_argument);
}

TEST_F(MetricsTest, NegativeValues) {
    VectorXd y_negative = -VectorXd::Ones(n_samples);
    VectorXd y_positive = VectorXd::Ones(n_samples);
    
    double mse = metrics::mean_squared_error(y_negative, y_positive);
    EXPECT_GT(mse, 0.0);
    
    double mae = metrics::mean_absolute_error(y_negative, y_positive);
    EXPECT_GT(mae, 0.0);
}

TEST_F(MetricsTest, ZeroValues) {
    VectorXd y_zero = VectorXd::Zero(n_samples);
    VectorXd y_ones = VectorXd::Ones(n_samples);
    
    double mse = metrics::mean_squared_error(y_zero, y_ones);
    EXPECT_NEAR(mse, 1.0, 1e-10);
    
    double mae = metrics::mean_absolute_error(y_zero, y_ones);
    EXPECT_NEAR(mae, 1.0, 1e-10);
}

TEST_F(MetricsTest, LargeValues) {
    VectorXd y_large = VectorXd::Ones(n_samples) * 1e6;
    VectorXd y_small = VectorXd::Ones(n_samples) * 1e-6;
    
    double mse = metrics::mean_squared_error(y_large, y_small);
    EXPECT_GT(mse, 0.0);
    EXPECT_FALSE(std::isnan(mse));
    EXPECT_FALSE(std::isinf(mse));
}

TEST_F(MetricsTest, VerySmallValues) {
    VectorXd y_tiny = VectorXd::Ones(n_samples) * 1e-10;
    VectorXd y_zero = VectorXd::Zero(n_samples);
    
    double mse = metrics::mean_squared_error(y_tiny, y_zero);
    EXPECT_GT(mse, 0.0);
    EXPECT_FALSE(std::isnan(mse));
    EXPECT_FALSE(std::isinf(mse));
}

// Consistency Tests
TEST_F(MetricsTest, MetricsConsistency) {
    // Test that metrics are consistent with each other
    double accuracy = metrics::accuracy_score(y_true_classification, y_pred_classification);
    double Precision = metrics::precision_score(y_true_classification, y_pred_classification, "macro");
    double Recall = metrics::recall_score(y_true_classification, y_pred_classification, "macro");
    double f1 = metrics::f1_score(y_true_classification, y_pred_classification, "macro");
    
    // F1 should be harmonic mean of Precision and Recall
    double expected_f1 = 2.0 * Precision * Recall / (Precision + Recall);
    if (Precision + Recall > 0) {
        EXPECT_NEAR(f1, expected_f1, 1e-10);
    }
}

TEST_F(MetricsTest, RegressionMetricsConsistency) {
    // Test that regression metrics are consistent
    double mse = metrics::mean_squared_error(y_true_regression, y_pred_regression);
    double rmse = metrics::root_mean_squared_error(y_true_regression, y_pred_regression);
    
    // RMSE should be square root of MSE
    EXPECT_NEAR(rmse, std::sqrt(mse), 1e-10);
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

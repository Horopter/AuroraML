#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace auroraml {
namespace metrics {

// Classification Metrics
double accuracy_score(const VectorXi& y_true, const VectorXi& y_pred);
double precision_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average = "macro");
double recall_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average = "macro");
double f1_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average = "macro");
MatrixXi confusion_matrix(const VectorXi& y_true, const VectorXi& y_pred);
std::string classification_report(const VectorXi& y_true, const VectorXi& y_pred);

// Regression Metrics
double mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred);
double root_mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred);
double mean_absolute_error(const VectorXd& y_true, const VectorXd& y_pred);
double r2_score(const VectorXd& y_true, const VectorXd& y_pred);
double explained_variance_score(const VectorXd& y_true, const VectorXd& y_pred);
double mean_absolute_percentage_error(const VectorXd& y_true, const VectorXd& y_pred);

// Utility functions
std::vector<int> unique_values(const VectorXi& vec);

} // namespace metrics
} // namespace cxml

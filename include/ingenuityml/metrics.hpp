#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace ingenuityml {
namespace metrics {

// Classification Metrics
double accuracy_score(const VectorXi& y_true, const VectorXi& y_pred);
double balanced_accuracy_score(const VectorXi& y_true, const VectorXi& y_pred);
double top_k_accuracy_score(const VectorXi& y_true, const MatrixXd& y_score, int k = 5);
double roc_auc_score(const VectorXi& y_true, const VectorXd& y_score);
double roc_auc_score_multiclass(const VectorXi& y_true, const MatrixXd& y_score, const std::string& average = "macro");
double average_precision_score(const VectorXi& y_true, const VectorXd& y_score);
double log_loss(const VectorXi& y_true, const MatrixXd& y_pred);
double hinge_loss(const VectorXi& y_true, const VectorXd& pred_decision);
double cohen_kappa_score(const VectorXi& y_true, const VectorXi& y_pred);
double matthews_corrcoef(const VectorXi& y_true, const VectorXi& y_pred);
double hamming_loss(const VectorXi& y_true, const VectorXi& y_pred);
double jaccard_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average = "macro");
double zero_one_loss(const VectorXi& y_true, const VectorXi& y_pred);
double brier_score_loss(const VectorXi& y_true, const VectorXd& y_prob);
double precision_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average = "macro");
double recall_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average = "macro");
double f1_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average = "macro");
MatrixXi confusion_matrix(const VectorXi& y_true, const VectorXi& y_pred);
std::string classification_report(const VectorXi& y_true, const VectorXi& y_pred);

// Regression Metrics
double mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred);
double root_mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred);
double mean_absolute_error(const VectorXd& y_true, const VectorXd& y_pred);
double median_absolute_error(const VectorXd& y_true, const VectorXd& y_pred);
double max_error(const VectorXd& y_true, const VectorXd& y_pred);
double mean_poisson_deviance(const VectorXd& y_true, const VectorXd& y_pred);
double mean_gamma_deviance(const VectorXd& y_true, const VectorXd& y_pred);
double mean_tweedie_deviance(const VectorXd& y_true, const VectorXd& y_pred, double power = 0.0);
double d2_tweedie_score(const VectorXd& y_true, const VectorXd& y_pred, double power = 0.0);
double d2_pinball_score(const VectorXd& y_true, const VectorXd& y_pred, double alpha = 0.5);
double d2_absolute_error_score(const VectorXd& y_true, const VectorXd& y_pred);
double r2_score(const VectorXd& y_true, const VectorXd& y_pred);
double explained_variance_score(const VectorXd& y_true, const VectorXd& y_pred);
double mean_absolute_percentage_error(const VectorXd& y_true, const VectorXd& y_pred);

// Clustering Metrics
double silhouette_score(const MatrixXd& X, const VectorXi& labels);
VectorXd silhouette_samples(const MatrixXd& X, const VectorXi& labels);
double calinski_harabasz_score(const MatrixXd& X, const VectorXi& labels);
double davies_bouldin_score(const MatrixXd& X, const VectorXi& labels);

// Clustering Comparison Metrics
double adjusted_rand_score(const VectorXi& labels_true, const VectorXi& labels_pred);
double adjusted_mutual_info_score(const VectorXi& labels_true, const VectorXi& labels_pred);
double normalized_mutual_info_score(const VectorXi& labels_true, const VectorXi& labels_pred);
double homogeneity_score(const VectorXi& labels_true, const VectorXi& labels_pred);
double completeness_score(const VectorXi& labels_true, const VectorXi& labels_pred);
double v_measure_score(const VectorXi& labels_true, const VectorXi& labels_pred);
double fowlkes_mallows_score(const VectorXi& labels_true, const VectorXi& labels_pred);

// Utility functions
std::vector<int> unique_values(const VectorXi& vec);

} // namespace metrics
} // namespace ingenuityml

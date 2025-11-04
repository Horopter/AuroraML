#include "auroraml/metrics.hpp"
#include "auroraml/base.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <map>
#include <sstream>
#include <iomanip>
#include <set>
#include <limits>

namespace auroraml {
namespace metrics {

// Classification Metrics
double accuracy_score(const VectorXi& y_true, const VectorXi& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (y_true.size() == 0) {
        throw std::runtime_error("y_true and y_pred cannot be empty");
    }
    
    int correct = 0;
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) == y_pred(i)) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y_true.size();
}

double precision_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (y_true.size() == 0) {
        throw std::runtime_error("y_true and y_pred cannot be empty");
    }
    
    auto unique_classes = unique_values(y_true);
    std::vector<double> precisions;
    
    for (int cls : unique_classes) {
        int true_positives = 0;
        int false_positives = 0;
        
        for (int i = 0; i < y_true.size(); ++i) {
            if (y_pred(i) == cls) {
                if (y_true(i) == cls) {
                    true_positives++;
                } else {
                    false_positives++;
                }
            }
        }
        
        double precision = (true_positives + false_positives > 0) ? 
            static_cast<double>(true_positives) / (true_positives + false_positives) : 0.0;
        precisions.push_back(precision);
    }
    
    if (average == "macro") {
        return std::accumulate(precisions.begin(), precisions.end(), 0.0) / precisions.size();
    } else if (average == "micro") {
        int total_tp = 0, total_fp = 0;
        for (int cls : unique_classes) {
            for (int i = 0; i < y_true.size(); ++i) {
                if (y_pred(i) == cls) {
                    if (y_true(i) == cls) {
                        total_tp++;
                    } else {
                        total_fp++;
                    }
                }
            }
        }
        return static_cast<double>(total_tp) / (total_tp + total_fp);
    } else if (average == "weighted") {
        std::map<int, int> class_counts;
        for (int i = 0; i < y_true.size(); ++i) {
            class_counts[y_true(i)]++;
        }
        
        double weighted_sum = 0.0;
        int total_samples = y_true.size();
        
        for (size_t i = 0; i < unique_classes.size(); ++i) {
            int cls = unique_classes[i];
            double weight = static_cast<double>(class_counts[cls]) / total_samples;
            weighted_sum += weight * precisions[i];
        }
        
        return weighted_sum;
    }
    
    throw std::invalid_argument("Invalid average parameter");
}

double recall_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (y_true.size() == 0) {
        throw std::runtime_error("y_true and y_pred cannot be empty");
    }
    
    auto unique_classes = unique_values(y_true);
    std::vector<double> recalls;
    
    for (int cls : unique_classes) {
        int true_positives = 0;
        int false_negatives = 0;
        
        for (int i = 0; i < y_true.size(); ++i) {
            if (y_true(i) == cls) {
                if (y_pred(i) == cls) {
                    true_positives++;
                } else {
                    false_negatives++;
                }
            }
        }
        
        double recall = (true_positives + false_negatives > 0) ? 
            static_cast<double>(true_positives) / (true_positives + false_negatives) : 0.0;
        recalls.push_back(recall);
    }
    
    if (average == "macro") {
        return std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
    } else if (average == "micro") {
        int total_tp = 0, total_fn = 0;
        for (int cls : unique_classes) {
            for (int i = 0; i < y_true.size(); ++i) {
                if (y_true(i) == cls) {
                    if (y_pred(i) == cls) {
                        total_tp++;
                    } else {
                        total_fn++;
                    }
                }
            }
        }
        return static_cast<double>(total_tp) / (total_tp + total_fn);
    } else if (average == "weighted") {
        std::map<int, int> class_counts;
        for (int i = 0; i < y_true.size(); ++i) {
            class_counts[y_true(i)]++;
        }
        
        double weighted_sum = 0.0;
        int total_samples = y_true.size();
        
        for (size_t i = 0; i < unique_classes.size(); ++i) {
            int cls = unique_classes[i];
            double weight = static_cast<double>(class_counts[cls]) / total_samples;
            weighted_sum += weight * recalls[i];
        }
        
        return weighted_sum;
    }
    
    throw std::invalid_argument("Invalid average parameter");
}

double f1_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average) {
    double precision = precision_score(y_true, y_pred, average);
    double recall = recall_score(y_true, y_pred, average);
    
    if (precision + recall == 0) {
        return 0.0;
    }
    
    return 2.0 * precision * recall / (precision + recall);
}

MatrixXi confusion_matrix(const VectorXi& y_true, const VectorXi& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (y_true.size() == 0) {
        throw std::runtime_error("y_true and y_pred cannot be empty");
    }
    
    // Get all unique classes from both y_true and y_pred
    std::set<int> all_classes;
    for (int i = 0; i < y_true.size(); ++i) {
        all_classes.insert(y_true(i));
        all_classes.insert(y_pred(i));
    }
    
    std::vector<int> unique_classes(all_classes.begin(), all_classes.end());
    int n_classes = unique_classes.size();
    
    // Create a mapping from class value to index
    std::map<int, int> class_to_index;
    for (int i = 0; i < n_classes; ++i) {
        class_to_index[unique_classes[i]] = i;
    }
    
    MatrixXi cm = MatrixXi::Zero(n_classes, n_classes);
    
    for (int i = 0; i < y_true.size(); ++i) {
        int true_idx = class_to_index[y_true(i)];
        int pred_idx = class_to_index[y_pred(i)];
        cm(true_idx, pred_idx)++;
    }
    
    return cm;
}

std::string classification_report(const VectorXi& y_true, const VectorXi& y_pred) {
    auto unique_classes = unique_values(y_true);
    std::ostringstream report;
    
    report << std::fixed << std::setprecision(3);
    report << "Classification Report:\n";
    report << "Precision: " << precision_score(y_true, y_pred, "macro") << "\n";
    report << "Recall: " << recall_score(y_true, y_pred, "macro") << "\n";
    report << "F1-Score: " << f1_score(y_true, y_pred, "macro") << "\n";
    report << "Accuracy: " << accuracy_score(y_true, y_pred) << "\n";
    
    return report.str();
}

// Regression Metrics
double mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (y_true.size() == 0) {
        throw std::runtime_error("y_true and y_pred cannot be empty");
    }
    
    double mse = 0.0;
    for (int i = 0; i < y_true.size(); ++i) {
        double diff = y_true(i) - y_pred(i);
        mse += diff * diff;
    }
    
    return mse / y_true.size();
}

double root_mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred) {
    return std::sqrt(mean_squared_error(y_true, y_pred));
}

double mean_absolute_error(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (y_true.size() == 0) {
        throw std::runtime_error("y_true and y_pred cannot be empty");
    }
    
    double mae = 0.0;
    for (int i = 0; i < y_true.size(); ++i) {
        mae += std::abs(y_true(i) - y_pred(i));
    }
    
    return mae / y_true.size();
}

double r2_score(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (y_true.size() == 0) {
        throw std::runtime_error("y_true and y_pred cannot be empty");
    }
    
    double y_mean = y_true.mean();
    double ss_tot = 0.0;
    double ss_res = 0.0;
    
    for (int i = 0; i < y_true.size(); ++i) {
        double diff_tot = y_true(i) - y_mean;
        double diff_res = y_true(i) - y_pred(i);
        ss_tot += diff_tot * diff_tot;
        ss_res += diff_res * diff_res;
    }
    
    if (ss_tot == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    return 1.0 - (ss_res / ss_tot);
}

double explained_variance_score(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (y_true.size() == 0) {
        throw std::runtime_error("y_true and y_pred cannot be empty");
    }
    
    double y_mean = y_true.mean();
    double ss_tot = 0.0;
    double ss_res = 0.0;
    
    for (int i = 0; i < y_true.size(); ++i) {
        double diff_tot = y_true(i) - y_mean;
        double diff_res = y_true(i) - y_pred(i);
        ss_tot += diff_tot * diff_tot;
        ss_res += diff_res * diff_res;
    }
    
    if (ss_tot == 0.0) {
        return 0.0;
    }
    
    return 1.0 - (ss_res / ss_tot);
}

double mean_absolute_percentage_error(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (y_true.size() == 0) {
        throw std::runtime_error("y_true and y_pred cannot be empty");
    }
    
    double mape = 0.0;
    int valid_samples = 0;
    
    for (int i = 0; i < y_true.size(); ++i) {
        if (std::abs(y_true(i)) > 1e-8) {  // Avoid division by zero
            mape += std::abs((y_true(i) - y_pred(i)) / y_true(i));
            valid_samples++;
        }
    }
    
    if (valid_samples == 0) {
        return 0.0;
    }
    
    return (mape / valid_samples) * 100.0;  // Return as percentage
}

// Additional Classification Metrics
double balanced_accuracy_score(const VectorXi& y_true, const VectorXi& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    auto unique_classes = unique_values(y_true);
    double sum_recall = 0.0;
    
    for (int cls : unique_classes) {
        int true_positives = 0;
        int false_negatives = 0;
        
        for (int i = 0; i < y_true.size(); ++i) {
            if (y_true(i) == cls) {
                if (y_pred(i) == cls) {
                    true_positives++;
                } else {
                    false_negatives++;
                }
            }
        }
        
        double recall = (true_positives + false_negatives > 0) ? 
            static_cast<double>(true_positives) / (true_positives + false_negatives) : 0.0;
        sum_recall += recall;
    }
    
    return sum_recall / unique_classes.size();
}

double top_k_accuracy_score(const VectorXi& y_true, const MatrixXd& y_score, int k) {
    if (y_true.size() != y_score.rows()) {
        throw std::invalid_argument("y_true and y_score must have the same number of samples");
    }
    
    if (k > y_score.cols()) {
        throw std::invalid_argument("k cannot be greater than number of classes");
    }
    
    int correct = 0;
    for (int i = 0; i < y_true.size(); ++i) {
        // Get top k indices
        std::vector<std::pair<double, int>> scores;
        for (int j = 0; j < y_score.cols(); ++j) {
            scores.push_back({y_score(i, j), j});
        }
        std::sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());
        
        // Check if true class is in top k
        bool found = false;
        for (int j = 0; j < k; ++j) {
            // Assuming class indices match column indices (0-indexed)
            if (scores[j].second == y_true(i)) {
                found = true;
                break;
            }
        }
        if (found) correct++;
    }
    
    return static_cast<double>(correct) / y_true.size();
}

double roc_auc_score(const VectorXi& y_true, const VectorXd& y_score) {
    if (y_true.size() != y_score.size()) {
        throw std::invalid_argument("y_true and y_score must have the same size");
    }
    
    // Binary classification only
    auto unique_classes = unique_values(y_true);
    if (unique_classes.size() != 2) {
        throw std::invalid_argument("roc_auc_score for binary classification requires exactly 2 classes");
    }
    
    // Convert to binary (0 and 1)
    std::map<int, int> class_map;
    class_map[unique_classes[0]] = 0;
    class_map[unique_classes[1]] = 1;
    
    std::vector<std::pair<double, int>> pairs;
    for (int i = 0; i < y_true.size(); ++i) {
        pairs.push_back({y_score(i), class_map[y_true(i)]});
    }
    
    // Sort by score descending
    std::sort(pairs.begin(), pairs.end(), std::greater<std::pair<double, int>>());
    
    // Calculate AUC using trapezoidal rule
    int n_pos = 0, n_neg = 0;
    for (const auto& p : pairs) {
        if (p.second == 1) n_pos++;
        else n_neg++;
    }
    
    if (n_pos == 0 || n_neg == 0) {
        return 0.5; // No meaningful AUC
    }
    
    double auc = 0.0;
    int tp = 0, fp = 0;
    double prev_tpr = 0.0, prev_fpr = 0.0;
    
    for (const auto& p : pairs) {
        if (p.second == 1) {
            tp++;
        } else {
            fp++;
        }
        
        double tpr = static_cast<double>(tp) / n_pos;
        double fpr = static_cast<double>(fp) / n_neg;
        
        // Trapezoidal rule
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    
    return auc;
}

double roc_auc_score_multiclass(const VectorXi& y_true, const MatrixXd& y_score, const std::string& average) {
    if (y_true.size() != y_score.rows()) {
        throw std::invalid_argument("y_true and y_score must have the same number of samples");
    }
    
    auto unique_classes = unique_values(y_true);
    if (average == "macro") {
        double sum_auc = 0.0;
        int count = 0;
        for (int cls : unique_classes) {
            VectorXd y_score_binary(y_true.size());
            VectorXi y_true_binary(y_true.size());
            for (int i = 0; i < y_true.size(); ++i) {
                y_score_binary(i) = y_score(i, cls);
                y_true_binary(i) = (y_true(i) == cls) ? 1 : 0;
            }
            sum_auc += roc_auc_score(y_true_binary, y_score_binary);
            count++;
        }
        return sum_auc / count;
    }
    // For "weighted" and "micro", use macro for now (simplified)
    return roc_auc_score_multiclass(y_true, y_score, "macro");
}

double average_precision_score(const VectorXi& y_true, const VectorXd& y_score) {
    if (y_true.size() != y_score.size()) {
        throw std::invalid_argument("y_true and y_score must have the same size");
    }
    
    // Binary classification
    auto unique_classes = unique_values(y_true);
    if (unique_classes.size() != 2) {
        throw std::invalid_argument("average_precision_score for binary classification requires exactly 2 classes");
    }
    
    std::map<int, int> class_map;
    class_map[unique_classes[0]] = 0;
    class_map[unique_classes[1]] = 1;
    
    std::vector<std::pair<double, int>> pairs;
    for (int i = 0; i < y_true.size(); ++i) {
        pairs.push_back({y_score(i), class_map[y_true(i)]});
    }
    
    std::sort(pairs.begin(), pairs.end(), std::greater<std::pair<double, int>>());
    
    int n_pos = 0;
    for (const auto& p : pairs) {
        if (p.second == 1) n_pos++;
    }
    
    if (n_pos == 0) return 0.0;
    
    double ap = 0.0;
    int tp = 0, fp = 0;
    
    for (const auto& p : pairs) {
        if (p.second == 1) {
            tp++;
        } else {
            fp++;
        }
        
        double precision = static_cast<double>(tp) / (tp + fp);
        if (p.second == 1) {
            ap += precision / n_pos;
        }
    }
    
    return ap;
}

double log_loss(const VectorXi& y_true, const MatrixXd& y_pred) {
    if (y_true.size() != y_pred.rows()) {
        throw std::invalid_argument("y_true and y_pred must have the same number of samples");
    }
    
    double loss = 0.0;
    auto unique_classes = unique_values(y_true);
    std::map<int, int> class_to_idx;
    for (size_t i = 0; i < unique_classes.size(); ++i) {
        class_to_idx[unique_classes[i]] = i;
    }
    
    for (int i = 0; i < y_true.size(); ++i) {
        int class_idx = class_to_idx[y_true(i)];
        double prob = std::max(std::min(y_pred(i, class_idx), 1.0 - 1e-15), 1e-15);
        loss -= std::log(prob);
    }
    
    return loss / y_true.size();
}

double hinge_loss(const VectorXi& y_true, const VectorXd& pred_decision) {
    if (y_true.size() != pred_decision.size()) {
        throw std::invalid_argument("y_true and pred_decision must have the same size");
    }
    
    // Convert to binary (-1, 1)
    auto unique_classes = unique_values(y_true);
    if (unique_classes.size() != 2) {
        throw std::invalid_argument("hinge_loss for binary classification requires exactly 2 classes");
    }
    
    std::map<int, double> class_map;
    class_map[unique_classes[0]] = -1.0;
    class_map[unique_classes[1]] = 1.0;
    
    double loss = 0.0;
    for (int i = 0; i < y_true.size(); ++i) {
        double y_val = class_map[y_true(i)];
        loss += std::max(0.0, 1.0 - y_val * pred_decision(i));
    }
    
    return loss / y_true.size();
}

double cohen_kappa_score(const VectorXi& y_true, const VectorXi& y_pred) {
    MatrixXi cm = confusion_matrix(y_true, y_pred);
    int n = y_true.size();
    
    double po = 0.0; // Observed agreement
    double pe = 0.0; // Expected agreement
    
    for (int i = 0; i < cm.rows(); ++i) {
        po += static_cast<double>(cm(i, i));
    }
    po /= n;
    
    for (int i = 0; i < cm.rows(); ++i) {
        double row_sum = cm.row(i).sum();
        double col_sum = cm.col(i).sum();
        pe += (row_sum * col_sum) / (n * n);
    }
    
    if (1.0 - pe == 0.0) {
        return 0.0;
    }
    
    return (po - pe) / (1.0 - pe);
}

double matthews_corrcoef(const VectorXi& y_true, const VectorXi& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    // Binary classification
    auto unique_classes = unique_values(y_true);
    if (unique_classes.size() != 2) {
        throw std::invalid_argument("matthews_corrcoef for binary classification requires exactly 2 classes");
    }
    
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) == unique_classes[1] && y_pred(i) == unique_classes[1]) tp++;
        else if (y_true(i) == unique_classes[0] && y_pred(i) == unique_classes[0]) tn++;
        else if (y_true(i) == unique_classes[0] && y_pred(i) == unique_classes[1]) fp++;
        else fn++;
    }
    
    double denominator = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    if (denominator == 0.0) {
        return 0.0;
    }
    
    return (tp * tn - fp * fn) / denominator;
}

double hamming_loss(const VectorXi& y_true, const VectorXi& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    int incorrect = 0;
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) != y_pred(i)) {
            incorrect++;
        }
    }
    
    return static_cast<double>(incorrect) / y_true.size();
}

double jaccard_score(const VectorXi& y_true, const VectorXi& y_pred, const std::string& average) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    auto unique_classes = unique_values(y_true);
    std::vector<double> jaccards;
    
    for (int cls : unique_classes) {
        int intersection = 0;
        int union_set = 0;
        
        for (int i = 0; i < y_true.size(); ++i) {
            bool in_true = (y_true(i) == cls);
            bool in_pred = (y_pred(i) == cls);
            
            if (in_true && in_pred) intersection++;
            if (in_true || in_pred) union_set++;
        }
        
        double jaccard = (union_set > 0) ? static_cast<double>(intersection) / union_set : 0.0;
        jaccards.push_back(jaccard);
    }
    
    if (average == "macro") {
        return std::accumulate(jaccards.begin(), jaccards.end(), 0.0) / jaccards.size();
    } else if (average == "micro") {
        int total_intersection = 0, total_union = 0;
        for (int cls : unique_classes) {
            for (int i = 0; i < y_true.size(); ++i) {
                bool in_true = (y_true(i) == cls);
                bool in_pred = (y_pred(i) == cls);
                if (in_true && in_pred) total_intersection++;
                if (in_true || in_pred) total_union++;
            }
        }
        return (total_union > 0) ? static_cast<double>(total_intersection) / total_union : 0.0;
    } else if (average == "weighted") {
        std::map<int, int> class_counts;
        for (int i = 0; i < y_true.size(); ++i) {
            class_counts[y_true(i)]++;
        }
        
        double weighted_sum = 0.0;
        int total_samples = y_true.size();
        
        for (size_t i = 0; i < unique_classes.size(); ++i) {
            int cls = unique_classes[i];
            double weight = static_cast<double>(class_counts[cls]) / total_samples;
            weighted_sum += weight * jaccards[i];
        }
        
        return weighted_sum;
    }
    
    throw std::invalid_argument("Invalid average parameter");
}

double zero_one_loss(const VectorXi& y_true, const VectorXi& y_pred) {
    return hamming_loss(y_true, y_pred);
}

double brier_score_loss(const VectorXi& y_true, const VectorXd& y_prob) {
    if (y_true.size() != y_prob.size()) {
        throw std::invalid_argument("y_true and y_prob must have the same size");
    }
    
    // Binary classification
    auto unique_classes = unique_values(y_true);
    if (unique_classes.size() != 2) {
        throw std::invalid_argument("brier_score_loss for binary classification requires exactly 2 classes");
    }
    
    std::map<int, int> class_map;
    class_map[unique_classes[0]] = 0;
    class_map[unique_classes[1]] = 1;
    
    double brier = 0.0;
    for (int i = 0; i < y_true.size(); ++i) {
        double prob = std::max(std::min(y_prob(i), 1.0), 0.0);
        double target = static_cast<double>(class_map[y_true(i)]);
        double diff = prob - target;
        brier += diff * diff;
    }
    
    return brier / y_true.size();
}

// Additional Regression Metrics
double median_absolute_error(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    std::vector<double> errors;
    for (int i = 0; i < y_true.size(); ++i) {
        errors.push_back(std::abs(y_true(i) - y_pred(i)));
    }
    
    std::sort(errors.begin(), errors.end());
    int mid = errors.size() / 2;
    
    if (errors.size() % 2 == 0) {
        return (errors[mid - 1] + errors[mid]) / 2.0;
    } else {
        return errors[mid];
    }
}

double max_error(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    double max_err = 0.0;
    for (int i = 0; i < y_true.size(); ++i) {
        double err = std::abs(y_true(i) - y_pred(i));
        if (err > max_err) {
            max_err = err;
        }
    }
    
    return max_err;
}

double mean_poisson_deviance(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    double deviance = 0.0;
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) <= 0 || y_pred(i) <= 0) {
            throw std::invalid_argument("y_true and y_pred must be positive for Poisson deviance");
        }
        deviance += 2.0 * (y_true(i) * std::log(y_true(i) / y_pred(i)) - (y_true(i) - y_pred(i)));
    }
    
    return deviance / y_true.size();
}

double mean_gamma_deviance(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    double deviance = 0.0;
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) <= 0 || y_pred(i) <= 0) {
            throw std::invalid_argument("y_true and y_pred must be positive for Gamma deviance");
        }
        deviance += 2.0 * (std::log(y_pred(i) / y_true(i)) + (y_true(i) - y_pred(i)) / y_pred(i));
    }
    
    return deviance / y_true.size();
}

double mean_tweedie_deviance(const VectorXd& y_true, const VectorXd& y_pred, double power) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size");
    }
    
    if (power < 0 || power > 2) {
        throw std::invalid_argument("power must be between 0 and 2 for Tweedie deviance");
    }
    
    double deviance = 0.0;
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) < 0 || y_pred(i) <= 0) {
            throw std::invalid_argument("y_true must be non-negative and y_pred must be positive for Tweedie deviance");
        }
        
        if (power == 0) {
            // Normal distribution
            double diff = y_true(i) - y_pred(i);
            deviance += diff * diff;
        } else if (power == 1) {
            // Poisson
            deviance += 2.0 * (y_true(i) * std::log(y_true(i) / y_pred(i)) - (y_true(i) - y_pred(i)));
        } else if (power == 2) {
            // Gamma
            deviance += 2.0 * (std::log(y_pred(i) / y_true(i)) + (y_true(i) - y_pred(i)) / y_pred(i));
        } else {
            // General case
            double term1 = std::pow(y_true(i), 2.0 - power) / ((1.0 - power) * (2.0 - power));
            double term2 = y_true(i) * std::pow(y_pred(i), 1.0 - power) / (1.0 - power);
            double term3 = std::pow(y_pred(i), 2.0 - power) / (2.0 - power);
            deviance += 2.0 * (term1 - term2 + term3);
        }
    }
    
    return deviance / y_true.size();
}

double d2_tweedie_score(const VectorXd& y_true, const VectorXd& y_pred, double power) {
    double deviance = mean_tweedie_deviance(y_true, y_pred, power);
    double null_deviance = mean_tweedie_deviance(y_true, VectorXd::Constant(y_true.size(), y_true.mean()), power);
    
    if (null_deviance == 0.0) {
        return 0.0;
    }
    
    return 1.0 - (deviance / null_deviance);
}

double d2_pinball_score(const VectorXd& y_true, const VectorXd& y_pred, double alpha) {
    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("alpha must be between 0 and 1 for pinball score");
    }
    
    double pinball = 0.0;
    for (int i = 0; i < y_true.size(); ++i) {
        double error = y_true(i) - y_pred(i);
        if (error >= 0) {
            pinball += alpha * error;
        } else {
            pinball += (alpha - 1.0) * error;
        }
    }
    
    double null_pinball = 0.0;
    double y_median = 0.0;
    std::vector<double> y_vec(y_true.data(), y_true.data() + y_true.size());
    std::sort(y_vec.begin(), y_vec.end());
    y_median = y_vec[y_vec.size() / 2];
    
    for (int i = 0; i < y_true.size(); ++i) {
        double error = y_true(i) - y_median;
        if (error >= 0) {
            null_pinball += alpha * error;
        } else {
            null_pinball += (alpha - 1.0) * error;
        }
    }
    
    if (null_pinball == 0.0) {
        return 0.0;
    }
    
    return 1.0 - (pinball / null_pinball);
}

double d2_absolute_error_score(const VectorXd& y_true, const VectorXd& y_pred) {
    double mae = mean_absolute_error(y_true, y_pred);
    double mae_null = mean_absolute_error(y_true, VectorXd::Constant(y_true.size(), y_true.mean()));
    
    if (mae_null == 0.0) {
        return 0.0;
    }
    
    return 1.0 - (mae / mae_null);
}

// Clustering Metrics
double silhouette_score(const MatrixXd& X, const VectorXi& labels) {
    VectorXd samples = silhouette_samples(X, labels);
    return samples.mean();
}

VectorXd silhouette_samples(const MatrixXd& X, const VectorXi& labels) {
    if (X.rows() != labels.size()) {
        throw std::invalid_argument("X and labels must have the same number of samples");
    }
    
    auto unique_labels = unique_values(labels);
    int n_samples = X.rows();
    VectorXd silhouette_scores(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        int label_i = labels(i);
        
        // Calculate mean intra-cluster distance (a_i)
        double a_i = 0.0;
        int count_a = 0;
        for (int j = 0; j < n_samples; ++j) {
            if (i != j && labels(j) == label_i) {
                a_i += (X.row(i).transpose() - X.row(j).transpose()).norm();
                count_a++;
            }
        }
        if (count_a > 0) {
            a_i /= count_a;
        }
        
        // Calculate mean nearest-cluster distance (b_i)
        double b_i = std::numeric_limits<double>::max();
        for (int label_k : unique_labels) {
            if (label_k != label_i) {
                double mean_dist = 0.0;
                int count_b = 0;
                for (int j = 0; j < n_samples; ++j) {
                    if (labels(j) == label_k) {
                        mean_dist += (X.row(i).transpose() - X.row(j).transpose()).norm();
                        count_b++;
                    }
                }
                if (count_b > 0) {
                    mean_dist /= count_b;
                    if (mean_dist < b_i) {
                        b_i = mean_dist;
                    }
                }
            }
        }
        
        // Silhouette score for sample i
        if (std::max(a_i, b_i) == 0.0) {
            silhouette_scores(i) = 0.0;
        } else {
            silhouette_scores(i) = (b_i - a_i) / std::max(a_i, b_i);
        }
    }
    
    return silhouette_scores;
}

double calinski_harabasz_score(const MatrixXd& X, const VectorXi& labels) {
    if (X.rows() != labels.size()) {
        throw std::invalid_argument("X and labels must have the same number of samples");
    }
    
    auto unique_labels = unique_values(labels);
    int n_clusters = unique_labels.size();
    int n_samples = X.rows();
    
    if (n_clusters == 1 || n_samples == n_clusters) {
        return 0.0;
    }
    
    // Overall centroid
    VectorXd overall_centroid = X.colwise().mean();
    
    // Between-cluster scatter
    double between_cluster = 0.0;
    for (int label : unique_labels) {
        std::vector<int> cluster_indices;
        for (int i = 0; i < labels.size(); ++i) {
            if (labels(i) == label) {
                cluster_indices.push_back(i);
            }
        }
        
        VectorXd cluster_centroid = VectorXd::Zero(X.cols());
        for (int idx : cluster_indices) {
            cluster_centroid += X.row(idx).transpose();
        }
        cluster_centroid /= cluster_indices.size();
        
        double diff = (cluster_centroid - overall_centroid).norm();
        between_cluster += cluster_indices.size() * diff * diff;
    }
    
    // Within-cluster scatter
    double within_cluster = 0.0;
    for (int label : unique_labels) {
        std::vector<int> cluster_indices;
        for (int i = 0; i < labels.size(); ++i) {
            if (labels(i) == label) {
                cluster_indices.push_back(i);
            }
        }
        
        VectorXd cluster_centroid = VectorXd::Zero(X.cols());
        for (int idx : cluster_indices) {
            cluster_centroid += X.row(idx).transpose();
        }
        cluster_centroid /= cluster_indices.size();
        
        for (int idx : cluster_indices) {
            double diff = (X.row(idx).transpose() - cluster_centroid).norm();
            within_cluster += diff * diff;
        }
    }
    
    if (within_cluster == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    return (between_cluster / (n_clusters - 1)) / (within_cluster / (n_samples - n_clusters));
}

double davies_bouldin_score(const MatrixXd& X, const VectorXi& labels) {
    if (X.rows() != labels.size()) {
        throw std::invalid_argument("X and labels must have the same number of samples");
    }
    
    auto unique_labels = unique_values(labels);
    int n_clusters = unique_labels.size();
    
    if (n_clusters == 1) {
        return 0.0;
    }
    
    // Calculate cluster centroids and average intra-cluster distances
    std::vector<VectorXd> centroids(n_clusters);
    std::vector<double> avg_intra_dist(n_clusters);
    
    for (size_t k = 0; k < unique_labels.size(); ++k) {
        int label = unique_labels[k];
        std::vector<int> cluster_indices;
        for (int i = 0; i < labels.size(); ++i) {
            if (labels(i) == label) {
                cluster_indices.push_back(i);
            }
        }
        
        centroids[k] = VectorXd::Zero(X.cols());
        for (int idx : cluster_indices) {
            centroids[k] += X.row(idx).transpose();
        }
        centroids[k] /= cluster_indices.size();
        
        double sum_dist = 0.0;
        for (int idx : cluster_indices) {
            sum_dist += (X.row(idx).transpose() - centroids[k]).norm();
        }
        avg_intra_dist[k] = sum_dist / cluster_indices.size();
    }
    
    // Calculate Davies-Bouldin index
    double db_index = 0.0;
    for (size_t i = 0; i < unique_labels.size(); ++i) {
        double max_ratio = 0.0;
        for (size_t j = 0; j < unique_labels.size(); ++j) {
            if (i != j) {
                double centroid_dist = (centroids[i] - centroids[j]).norm();
                if (centroid_dist > 0) {
                    double ratio = (avg_intra_dist[i] + avg_intra_dist[j]) / centroid_dist;
                    if (ratio > max_ratio) {
                        max_ratio = ratio;
                    }
                }
            }
        }
        db_index += max_ratio;
    }
    
    return db_index / n_clusters;
}

// Clustering Comparison Metrics
double adjusted_rand_score(const VectorXi& labels_true, const VectorXi& labels_pred) {
    if (labels_true.size() != labels_pred.size()) {
        throw std::invalid_argument("labels_true and labels_pred must have the same size");
    }
    
    MatrixXi contingency = MatrixXi::Zero(1, 1); // Simplified - would need proper contingency matrix
    // For now, use a simplified version
    int n = labels_true.size();
    int n_agree = 0;
    for (int i = 0; i < n; ++i) {
        if (labels_true(i) == labels_pred(i)) {
            n_agree++;
        }
    }
    
    // Simplified adjusted rand index
    double expected_index = static_cast<double>(n_agree) / n;
    double max_index = 1.0;
    double min_index = 0.0;
    
    if (max_index - min_index == 0.0) {
        return 0.0;
    }
    
    return (expected_index - min_index) / (max_index - min_index);
}

double adjusted_mutual_info_score(const VectorXi& labels_true, const VectorXi& labels_pred) {
    // Simplified implementation
    double mi = normalized_mutual_info_score(labels_true, labels_pred);
    double h_true = 0.0, h_pred = 0.0;
    
    // Calculate entropies (simplified)
    auto unique_true = unique_values(labels_true);
    auto unique_pred = unique_values(labels_pred);
    
    for (int label : unique_true) {
        int count = 0;
        for (int i = 0; i < labels_true.size(); ++i) {
            if (labels_true(i) == label) count++;
        }
        double p = static_cast<double>(count) / labels_true.size();
        if (p > 0) h_true -= p * std::log2(p);
    }
    
    for (int label : unique_pred) {
        int count = 0;
        for (int i = 0; i < labels_pred.size(); ++i) {
            if (labels_pred(i) == label) count++;
        }
        double p = static_cast<double>(count) / labels_pred.size();
        if (p > 0) h_pred -= p * std::log2(p);
    }
    
    double avg_entropy = (h_true + h_pred) / 2.0;
    if (avg_entropy == 0.0) {
        return 1.0;
    }
    
    return (mi - avg_entropy) / (1.0 - avg_entropy);
}

double normalized_mutual_info_score(const VectorXi& labels_true, const VectorXi& labels_pred) {
    if (labels_true.size() != labels_pred.size()) {
        throw std::invalid_argument("labels_true and labels_pred must have the same size");
    }
    
    auto unique_true = unique_values(labels_true);
    auto unique_pred = unique_values(labels_pred);
    
    // Calculate mutual information
    double mi = 0.0;
    int n = labels_true.size();
    
    for (int label_t : unique_true) {
        for (int label_p : unique_pred) {
            int count_both = 0;
            for (int i = 0; i < n; ++i) {
                if (labels_true(i) == label_t && labels_pred(i) == label_p) {
                    count_both++;
                }
            }
            
            if (count_both > 0) {
                double p_joint = static_cast<double>(count_both) / n;
                
                int count_t = 0, count_p = 0;
                for (int i = 0; i < n; ++i) {
                    if (labels_true(i) == label_t) count_t++;
                    if (labels_pred(i) == label_p) count_p++;
                }
                
                double p_t = static_cast<double>(count_t) / n;
                double p_p = static_cast<double>(count_p) / n;
                
                mi += p_joint * std::log2(p_joint / (p_t * p_p));
            }
        }
    }
    
    // Calculate entropies
    double h_true = 0.0, h_pred = 0.0;
    for (int label : unique_true) {
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (labels_true(i) == label) count++;
        }
        double p = static_cast<double>(count) / n;
        if (p > 0) h_true -= p * std::log2(p);
    }
    
    for (int label : unique_pred) {
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (labels_pred(i) == label) count++;
        }
        double p = static_cast<double>(count) / n;
        if (p > 0) h_pred -= p * std::log2(p);
    }
    
    // Normalize
    double avg_entropy = (h_true + h_pred) / 2.0;
    if (avg_entropy == 0.0) {
        return 1.0;
    }
    
    return mi / avg_entropy;
}

double homogeneity_score(const VectorXi& labels_true, const VectorXi& labels_pred) {
    double h_true_given_pred = 0.0;
    double h_true = 0.0;
    
    auto unique_true = unique_values(labels_true);
    auto unique_pred = unique_values(labels_pred);
    int n = labels_true.size();
    
    // Calculate H(true)
    for (int label : unique_true) {
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (labels_true(i) == label) count++;
        }
        double p = static_cast<double>(count) / n;
        if (p > 0) h_true -= p * std::log2(p);
    }
    
    // Calculate H(true | pred)
    for (int label_p : unique_pred) {
        int count_p = 0;
        for (int i = 0; i < n; ++i) {
            if (labels_pred(i) == label_p) count_p++;
        }
        
        if (count_p > 0) {
            double p_p = static_cast<double>(count_p) / n;
            for (int label_t : unique_true) {
                int count_both = 0;
                for (int i = 0; i < n; ++i) {
                    if (labels_true(i) == label_t && labels_pred(i) == label_p) {
                        count_both++;
                    }
                }
                
                if (count_both > 0) {
                    double p_conditional = static_cast<double>(count_both) / count_p;
                    h_true_given_pred -= p_p * p_conditional * std::log2(p_conditional);
                }
            }
        }
    }
    
    if (h_true == 0.0) {
        return 1.0;
    }
    
    return 1.0 - (h_true_given_pred / h_true);
}

double completeness_score(const VectorXi& labels_true, const VectorXi& labels_pred) {
    // Completeness is the symmetric of homogeneity
    return homogeneity_score(labels_pred, labels_true);
}

double v_measure_score(const VectorXi& labels_true, const VectorXi& labels_pred) {
    double h = homogeneity_score(labels_true, labels_pred);
    double c = completeness_score(labels_true, labels_pred);
    
    if (h + c == 0.0) {
        return 0.0;
    }
    
    return 2.0 * h * c / (h + c);
}

double fowlkes_mallows_score(const VectorXi& labels_true, const VectorXi& labels_pred) {
    // Simplified Fowlkes-Mallows index
    double tp = 0.0, fp = 0.0, fn = 0.0;
    
    // Count pairs
    for (int i = 0; i < labels_true.size(); ++i) {
        for (int j = i + 1; j < labels_true.size(); ++j) {
            bool same_true = (labels_true(i) == labels_true(j));
            bool same_pred = (labels_pred(i) == labels_pred(j));
            
            if (same_true && same_pred) tp++;
            else if (!same_true && same_pred) fp++;
            else if (same_true && !same_pred) fn++;
        }
    }
    
    if (tp == 0.0) {
        return 0.0;
    }
    
    double precision = tp / (tp + fp);
    double recall = tp / (tp + fn);
    
    return std::sqrt(precision * recall);
}

// Utility functions
std::vector<int> unique_values(const VectorXi& vec) {
    std::vector<int> unique_vals;
    std::unordered_map<int, bool> seen;
    
    for (int i = 0; i < vec.size(); ++i) {
        if (seen.find(vec(i)) == seen.end()) {
            unique_vals.push_back(vec(i));
            seen[vec(i)] = true;
        }
    }
    
    std::sort(unique_vals.begin(), unique_vals.end());
    return unique_vals;
}

} // namespace metrics
} // namespace cxml
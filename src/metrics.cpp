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
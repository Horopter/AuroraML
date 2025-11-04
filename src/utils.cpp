#include "auroraml/utils.hpp"
#include "auroraml/base.hpp"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <random>

namespace auroraml {
namespace utils {

// Multiclass utilities

namespace multiclass {

bool is_multiclass(const VectorXi& y) {
    if (y.size() == 0) return false;
    
    std::unordered_set<int> unique_labels;
    for (int i = 0; i < y.size(); ++i) {
        unique_labels.insert(y(i));
    }
    
    return unique_labels.size() > 2;
}

VectorXi unique_labels(const VectorXi& y) {
    std::unordered_set<int> unique_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_set.insert(y(i));
    }
    
    std::vector<int> unique_vec(unique_set.begin(), unique_set.end());
    std::sort(unique_vec.begin(), unique_vec.end());
    
    VectorXi result(unique_vec.size());
    for (size_t i = 0; i < unique_vec.size(); ++i) {
        result(i) = unique_vec[i];
    }
    
    return result;
}

std::string type_of_target(const VectorXi& y) {
    if (y.size() == 0) {
        return "unknown";
    }
    
    std::unordered_set<int> unique_labels;
    int min_val = y(0), max_val = y(0);
    
    for (int i = 0; i < y.size(); ++i) {
        unique_labels.insert(y(i));
        if (y(i) < min_val) min_val = y(i);
        if (y(i) > max_val) max_val = y(i);
    }
    
    if (unique_labels.size() == 2 && min_val == 0 && max_val == 1) {
        return "binary";
    } else if (unique_labels.size() > 2) {
        return "multiclass";
    } else {
        return "continuous";
    }
}

} // namespace multiclass

// Resampling utilities

namespace resample {

std::pair<MatrixXd, VectorXd> resample(const MatrixXd& X, const VectorXd& y, int n_samples, int random_state) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    int n = X.rows();
    if (n_samples <= 0) {
        n_samples = n;
    }
    
    // Initialize random number generator
    std::mt19937 rng;
    if (random_state >= 0) {
        rng.seed(random_state);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    std::uniform_int_distribution<int> dist(0, n - 1);
    
    MatrixXd X_resampled(n_samples, X.cols());
    VectorXd y_resampled(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        int idx = dist(rng);
        X_resampled.row(i) = X.row(idx);
        y_resampled(i) = y(idx);
    }
    
    return {X_resampled, y_resampled};
}

void shuffle(MatrixXd& X, VectorXd& y, int random_state) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    int n = X.rows();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Initialize random number generator
    std::mt19937 rng;
    if (random_state >= 0) {
        rng.seed(random_state);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    std::shuffle(indices.begin(), indices.end(), rng);
    
    // Create shuffled copies
    MatrixXd X_shuffled(n, X.cols());
    VectorXd y_shuffled(n);
    
    for (int i = 0; i < n; ++i) {
        X_shuffled.row(i) = X.row(indices[i]);
        y_shuffled(i) = y(indices[i]);
    }
    
    X = X_shuffled;
    y = y_shuffled;
}

std::pair<std::pair<MatrixXd, VectorXd>, std::pair<MatrixXd, VectorXd>> 
train_test_split_stratified(const MatrixXd& X, const VectorXi& y, double test_size, int random_state) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    // Group indices by class
    std::unordered_map<int, std::vector<int>> class_indices;
    for (int i = 0; i < y.size(); ++i) {
        class_indices[y(i)].push_back(i);
    }
    
    // Shuffle indices for each class
    std::mt19937 rng;
    if (random_state >= 0) {
        rng.seed(random_state);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    for (auto& [label, indices] : class_indices) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    // Split each class proportionally
    std::vector<int> train_indices, test_indices;
    
    for (const auto& [label, indices] : class_indices) {
        int n_test = static_cast<int>(indices.size() * test_size);
        int n_train = indices.size() - n_test;
        
        for (int i = 0; i < n_train; ++i) {
            train_indices.push_back(indices[i]);
        }
        for (int i = n_train; i < static_cast<int>(indices.size()); ++i) {
            test_indices.push_back(indices[i]);
        }
    }
    
    // Create train/test splits
    MatrixXd X_train(train_indices.size(), X.cols());
    VectorXi y_train(train_indices.size());
    MatrixXd X_test(test_indices.size(), X.cols());
    VectorXi y_test(test_indices.size());
    
    for (size_t i = 0; i < train_indices.size(); ++i) {
        X_train.row(i) = X.row(train_indices[i]);
        y_train(i) = y(train_indices[i]);
    }
    
    for (size_t i = 0; i < test_indices.size(); ++i) {
        X_test.row(i) = X.row(test_indices[i]);
        y_test(i) = y(test_indices[i]);
    }
    
    return {{X_train, y_train.cast<double>()}, {X_test, y_test.cast<double>()}};
}

} // namespace resample

// Validation utilities

namespace validation {

bool check_finite(const MatrixXd& X) {
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            if (!std::isfinite(X(i, j))) {
                return false;
            }
        }
    }
    return true;
}

bool check_has_nan(const MatrixXd& X) {
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            if (std::isnan(X(i, j))) {
                return true;
            }
        }
    }
    return false;
}

bool check_has_inf(const MatrixXd& X) {
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            if (std::isinf(X(i, j))) {
                return true;
            }
        }
    }
    return false;
}

} // namespace validation

// Class weight utilities

namespace class_weight {

std::unordered_map<int, double> compute_class_weight(const std::string& mode, const VectorXi& y) {
    std::unordered_map<int, double> weights;
    
    if (mode == "balanced") {
        // Count class frequencies
        std::unordered_map<int, int> class_counts;
        for (int i = 0; i < y.size(); ++i) {
            class_counts[y(i)]++;
        }
        
        int n_classes = class_counts.size();
        double n_samples = y.size();
        
        for (const auto& [label, count] : class_counts) {
            weights[label] = n_samples / (n_classes * count);
        }
    } else {
        // Uniform weights
        std::unordered_set<int> unique_labels;
        for (int i = 0; i < y.size(); ++i) {
            unique_labels.insert(y(i));
        }
        
        double weight = 1.0 / unique_labels.size();
        for (int label : unique_labels) {
            weights[label] = weight;
        }
    }
    
    return weights;
}

std::unordered_map<int, double> compute_sample_weight(const VectorXi& y, const std::unordered_map<int, double>& class_weight) {
    std::unordered_map<int, double> sample_weights;
    
    for (int i = 0; i < y.size(); ++i) {
        int label = y(i);
        auto it = class_weight.find(label);
        if (it != class_weight.end()) {
            sample_weights[i] = it->second;
        } else {
            sample_weights[i] = 1.0;
        }
    }
    
    return sample_weights;
}

} // namespace class_weight

// Array utilities

namespace array {

bool issparse(const MatrixXd& X) {
    // Eigen matrices are always dense
    return false;
}

std::pair<int, int> shape(const MatrixXd& X) {
    return {X.rows(), X.cols()};
}

} // namespace array

// Index utilities

namespace index {

int safe_index(int idx, int size) {
    if (idx < 0) {
        idx = size + idx;
    }
    if (idx < 0 || idx >= size) {
        throw std::out_of_range("Index out of range");
    }
    return idx;
}

} // namespace index

} // namespace utils
} // namespace auroraml


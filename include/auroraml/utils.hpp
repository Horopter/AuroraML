#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <random>

namespace auroraml {
namespace utils {

/**
 * Multiclass utilities
 */
namespace multiclass {
    /**
     * Check if the problem is multiclass
     */
    bool is_multiclass(const VectorXi& y);
    
    /**
     * Get unique class labels
     */
    VectorXi unique_labels(const VectorXi& y);
    
    /**
     * Type of multiclass problem
     */
    std::string type_of_target(const VectorXi& y);
}

/**
 * Resampling utilities
 */
namespace resample {
    /**
     * Resample arrays with replacement
     */
    std::pair<MatrixXd, VectorXd> resample(const MatrixXd& X, const VectorXd& y, int n_samples = -1, int random_state = -1);
    
    /**
     * Shuffle arrays in unison
     */
    void shuffle(MatrixXd& X, VectorXd& y, int random_state = -1);
    
    /**
     * Stratified shuffle split
     */
    std::pair<std::pair<MatrixXd, VectorXd>, std::pair<MatrixXd, VectorXd>> 
    train_test_split_stratified(const MatrixXd& X, const VectorXi& y, double test_size = 0.25, int random_state = -1);
}

/**
 * Data validation utilities (extended)
 */
namespace validation {
    /**
     * Check if array is finite
     */
    bool check_finite(const MatrixXd& X);
    
    /**
     * Check if array has any NaN values
     */
    bool check_has_nan(const MatrixXd& X);
    
    /**
     * Check if array has any infinite values
     */
    bool check_has_inf(const MatrixXd& X);
}

/**
 * Class weight utilities
 */
namespace class_weight {
    /**
     * Compute class weights
     */
    std::unordered_map<int, double> compute_class_weight(const std::string& mode, const VectorXi& y);
    
    /**
     * Balance class weights
     */
    std::unordered_map<int, double> compute_sample_weight(const VectorXi& y, const std::unordered_map<int, double>& class_weight);
}

/**
 * Array utilities
 */
namespace array {
    /**
     * Check if array is sparse (not implemented, always returns false)
     */
    bool issparse(const MatrixXd& X);
    
    /**
     * Get array shape
     */
    std::pair<int, int> shape(const MatrixXd& X);
}

/**
 * Index utilities
 */
namespace index {
    /**
     * Safe index conversion
     */
    int safe_index(int idx, int size);
}

} // namespace utils
} // namespace auroraml


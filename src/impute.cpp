#include "auroraml/impute.hpp"
#include "auroraml/linear_model.hpp"
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace auroraml {
namespace impute {

// KNNImputer implementation

KNNImputer::KNNImputer(int n_neighbors, const std::string& metric)
    : n_neighbors_(n_neighbors), metric_(metric), fitted_(false) {
    if (n_neighbors <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
}

Estimator& KNNImputer::fit(const MatrixXd& X, const VectorXd& y) {
    X_fitted_ = X;
    
    // Create missing mask (NaN values)
    missing_mask_.clear();
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            if (std::isnan(X(i, j))) {
                missing_mask_.push_back(true);
            } else {
                missing_mask_.push_back(false);
            }
        }
    }
    
    fitted_ = true;
    return *this;
}

double KNNImputer::distance(const VectorXd& a, const VectorXd& b) const {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    
    double dist = 0.0;
    int valid_dims = 0;
    
    for (int i = 0; i < a.size(); ++i) {
        if (!std::isnan(a(i)) && !std::isnan(b(i))) {
            double diff = a(i) - b(i);
            if (metric_ == "euclidean" || metric_ == "minkowski") {
                dist += diff * diff;
            } else if (metric_ == "manhattan") {
                dist += std::abs(diff);
            }
            valid_dims++;
        }
    }
    
    if (valid_dims == 0) {
        return std::numeric_limits<double>::max();
    }
    
    if (metric_ == "euclidean" || metric_ == "minkowski") {
        return std::sqrt(dist);
    } else {
        return dist;
    }
}

std::vector<int> KNNImputer::find_neighbors(const VectorXd& sample, const MatrixXd& X, int k) const {
    std::vector<std::pair<double, int>> distances;
    
    for (int i = 0; i < X.rows(); ++i) {
        double dist = distance(sample, X.row(i));
        distances.push_back({dist, i});
    }
    
    // Sort by distance
    std::sort(distances.begin(), distances.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first < b.first;
              });
    
    std::vector<int> neighbors;
    int k_selected = std::min(k, static_cast<int>(distances.size()));
    for (int i = 0; i < k_selected; ++i) {
        neighbors.push_back(distances[i].second);
    }
    
    return neighbors;
}

VectorXd KNNImputer::impute_sample(const VectorXd& sample, const MatrixXd& X, const std::vector<int>& neighbors) const {
    VectorXd imputed = sample;
    
    for (int j = 0; j < sample.size(); ++j) {
        if (std::isnan(sample(j))) {
            // Calculate mean from neighbors
            double sum = 0.0;
            int count = 0;
            
            for (int neighbor_idx : neighbors) {
                double val = X(neighbor_idx, j);
                if (!std::isnan(val)) {
                    sum += val;
                    count++;
                }
            }
            
            if (count > 0) {
                imputed(j) = sum / count;
            } else {
                // If no valid neighbors, use column mean
                double col_sum = 0.0;
                int col_count = 0;
                for (int i = 0; i < X.rows(); ++i) {
                    if (!std::isnan(X(i, j))) {
                        col_sum += X(i, j);
                        col_count++;
                    }
                }
                imputed(j) = col_count > 0 ? col_sum / col_count : 0.0;
            }
        }
    }
    
    return imputed;
}

MatrixXd KNNImputer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KNNImputer must be fitted before transform");
    }
    
    MatrixXd X_imputed = X;
    
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd sample = X.row(i);
        bool has_missing = false;
        for (int j = 0; j < sample.size(); ++j) {
            if (std::isnan(sample(j))) {
                has_missing = true;
                break;
            }
        }
        
        if (has_missing) {
            std::vector<int> neighbors = find_neighbors(sample, X_fitted_, n_neighbors_);
            VectorXd imputed = impute_sample(sample, X_fitted_, neighbors);
            X_imputed.row(i) = imputed;
        }
    }
    
    return X_imputed;
}

MatrixXd KNNImputer::inverse_transform(const MatrixXd& X) const {
    // Inverse transform not meaningful for imputation
    return X;
}

MatrixXd KNNImputer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params KNNImputer::get_params() const {
    Params params;
    params["n_neighbors"] = std::to_string(n_neighbors_);
    params["metric"] = metric_;
    return params;
}

Estimator& KNNImputer::set_params(const Params& params) {
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", 5);
    metric_ = utils::get_param_string(params, "metric", "euclidean");
    return *this;
}

// IterativeImputer implementation

IterativeImputer::IterativeImputer(int max_iter, double tol, int random_state)
    : max_iter_(max_iter), tol_(tol), random_state_(random_state), fitted_(false) {
    if (max_iter <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
}

MatrixXd IterativeImputer::initialize_missing(const MatrixXd& X) const {
    MatrixXd X_init = X;
    
    // Initialize missing values with column means
    for (int j = 0; j < X.cols(); ++j) {
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < X.rows(); ++i) {
            if (!std::isnan(X(i, j))) {
                sum += X(i, j);
                count++;
            }
        }
        
        double mean = count > 0 ? sum / count : 0.0;
        
        for (int i = 0; i < X.rows(); ++i) {
            if (std::isnan(X_init(i, j))) {
                X_init(i, j) = mean;
            }
        }
    }
    
    return X_init;
}

bool IterativeImputer::check_convergence(const MatrixXd& X_old, const MatrixXd& X_new) const {
    double max_diff = 0.0;
    
    for (int i = 0; i < X_old.rows(); ++i) {
        for (int j = 0; j < X_old.cols(); ++j) {
            double diff = std::abs(X_old(i, j) - X_new(i, j));
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    return max_diff < tol_;
}

Estimator& IterativeImputer::fit(const MatrixXd& X, const VectorXd& y) {
    X_fitted_ = X;
    imputation_models_.clear();
    imputation_models_.resize(X.cols());
    
    // Initialize missing values
    MatrixXd X_current = initialize_missing(X);
    
    // Iterative imputation
    for (int iter = 0; iter < max_iter_; ++iter) {
        MatrixXd X_old = X_current;
        
        // For each feature with missing values, fit a model on other features
        for (int j = 0; j < X.cols(); ++j) {
            // Find rows with missing values in column j
            std::vector<int> missing_rows;
            std::vector<int> complete_rows;
            
            for (int i = 0; i < X.rows(); ++i) {
                if (std::isnan(X(i, j))) {
                    missing_rows.push_back(i);
                } else {
                    complete_rows.push_back(i);
                }
            }
            
            if (missing_rows.empty()) {
                continue; // No missing values in this column
            }
            
            // Build feature matrix (all columns except j)
            MatrixXd X_features(complete_rows.size(), X.cols() - 1);
            VectorXd y_target(complete_rows.size());
            
            int feature_col = 0;
            for (int col = 0; col < X.cols(); ++col) {
                if (col == j) continue;
                
                for (size_t row_idx = 0; row_idx < complete_rows.size(); ++row_idx) {
                    X_features(row_idx, feature_col) = X_current(complete_rows[row_idx], col);
                }
                feature_col++;
            }
            
            for (size_t row_idx = 0; row_idx < complete_rows.size(); ++row_idx) {
                y_target(row_idx) = X_current(complete_rows[row_idx], j);
            }
            
            // Fit a simple linear regression model
            auto regressor = std::make_shared<linear_model::LinearRegression>();
            regressor->fit(X_features, y_target);
            imputation_models_[j] = regressor;
            
            // Predict missing values
            if (!missing_rows.empty()) {
                MatrixXd X_missing(missing_rows.size(), X.cols() - 1);
                feature_col = 0;
                for (int col = 0; col < X.cols(); ++col) {
                    if (col == j) continue;
                    
                    for (size_t row_idx = 0; row_idx < missing_rows.size(); ++row_idx) {
                        X_missing(row_idx, feature_col) = X_current(missing_rows[row_idx], col);
                    }
                    feature_col++;
                }
                
                VectorXd predictions = regressor->predict(X_missing);
                for (size_t row_idx = 0; row_idx < missing_rows.size(); ++row_idx) {
                    X_current(missing_rows[row_idx], j) = predictions(row_idx);
                }
            }
        }
        
        // Check convergence
        if (check_convergence(X_old, X_current)) {
            break;
        }
    }
    
    X_fitted_ = X_current;
    fitted_ = true;
    return *this;
}

MatrixXd IterativeImputer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("IterativeImputer must be fitted before transform");
    }
    
    MatrixXd X_imputed = initialize_missing(X);
    
    // Apply imputation models
    for (int iter = 0; iter < max_iter_; ++iter) {
        for (int j = 0; j < X.cols(); ++j) {
            if (imputation_models_[j] == nullptr) {
                continue;
            }
            
            // Find rows with missing values in column j
            std::vector<int> missing_rows;
            for (int i = 0; i < X.rows(); ++i) {
                if (std::isnan(X(i, j))) {
                    missing_rows.push_back(i);
                }
            }
            
            if (missing_rows.empty()) {
                continue;
            }
            
            // Build feature matrix
            MatrixXd X_features(missing_rows.size(), X.cols() - 1);
            int feature_col = 0;
            for (int col = 0; col < X.cols(); ++col) {
                if (col == j) continue;
                
                for (size_t row_idx = 0; row_idx < missing_rows.size(); ++row_idx) {
                    X_features(row_idx, feature_col) = X_imputed(missing_rows[row_idx], col);
                }
                feature_col++;
            }
            
            // Predict
            VectorXd predictions = imputation_models_[j]->predict(X_features);
            for (size_t row_idx = 0; row_idx < missing_rows.size(); ++row_idx) {
                X_imputed(missing_rows[row_idx], j) = predictions(row_idx);
            }
        }
    }
    
    return X_imputed;
}

MatrixXd IterativeImputer::inverse_transform(const MatrixXd& X) const {
    return X;
}

MatrixXd IterativeImputer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params IterativeImputer::get_params() const {
    Params params;
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& IterativeImputer::set_params(const Params& params) {
    max_iter_ = utils::get_param_int(params, "max_iter", 10);
    tol_ = utils::get_param_double(params, "tol", 1e-3);
    random_state_ = utils::get_param_int(params, "random_state", -1);
    return *this;
}

} // namespace impute
} // namespace auroraml


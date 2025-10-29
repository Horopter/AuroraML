#include "auroraml/neighbors.hpp"
#include "auroraml/base.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <map>
#include <set>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace auroraml {
namespace neighbors {

// KNeighborsClassifier implementation
KNeighborsClassifier::KNeighborsClassifier(int n_neighbors, const std::string& weights,
                                         const std::string& algorithm, const std::string& metric,
                                         double p, int n_jobs)
    : X_train_(), y_train_(), fitted_(false), n_neighbors_(n_neighbors), 
      weights_(weights), algorithm_(algorithm), metric_(metric), p_(p), n_jobs_(n_jobs) {
    if (n_neighbors <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
}

Estimator& KNeighborsClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    X_train_ = X;
    y_train_ = y;
    fitted_ = true;
    return *this;
}

VectorXi KNeighborsClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KNeighborsClassifier must be fitted before predict");
    }
    
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd distances = compute_distances(X);
    std::vector<std::vector<int>> neighbor_indices = find_k_neighbors(distances);
    
    return predict_from_neighbors(neighbor_indices);
}

MatrixXd KNeighborsClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KNeighborsClassifier must be fitted before predict_proba");
    }
    
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd distances = compute_distances(X);
    std::vector<std::vector<int>> neighbor_indices = find_k_neighbors(distances);
    
    return predict_proba_from_neighbors(neighbor_indices);
}

VectorXd KNeighborsClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KNeighborsClassifier must be fitted before decision_function");
    }
    
    MatrixXd probabilities = predict_proba(X);
    VectorXd decision_values(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        // Simple decision function: probability of positive class
        decision_values(i) = probabilities(i, 1);
    }
    
    return decision_values;
}

Params KNeighborsClassifier::get_params() const {
    return {
        {"n_neighbors", std::to_string(n_neighbors_)},
        {"weights", weights_},
        {"algorithm", algorithm_},
        {"metric", metric_},
        {"p", std::to_string(p_)},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& KNeighborsClassifier::set_params(const Params& params) {
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", n_neighbors_);
    weights_ = utils::get_param_string(params, "weights", weights_);
    algorithm_ = utils::get_param_string(params, "algorithm", algorithm_);
    metric_ = utils::get_param_string(params, "metric", metric_);
    p_ = utils::get_param_double(params, "p", p_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool KNeighborsClassifier::is_fitted() const {
    return fitted_;
}

MatrixXd KNeighborsClassifier::compute_distances(const MatrixXd& X) const {
    int n_test = X.rows();
    int n_train = X_train_.rows();
    
    MatrixXd distances(n_test, n_train);
    
    #pragma omp parallel for if(n_test > 32)
    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_train; ++j) {
            if (metric_ == "minkowski") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    dist += std::pow(std::abs(X(i, k) - X_train_(j, k)), p_);
                }
                distances(i, j) = std::pow(dist, 1.0 / p_);
            } else if (metric_ == "euclidean") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    double diff = X(i, k) - X_train_(j, k);
                    dist += diff * diff;
                }
                distances(i, j) = std::sqrt(dist);
            } else if (metric_ == "manhattan") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    dist += std::abs(X(i, k) - X_train_(j, k));
                }
                distances(i, j) = dist;
            } else {
                throw std::invalid_argument("Unsupported metric: " + metric_);
            }
        }
    }
    
    return distances;
}

std::vector<std::vector<int>> KNeighborsClassifier::find_k_neighbors(const MatrixXd& distances) const {
    int n_test = distances.rows();
    std::vector<std::vector<int>> neighbor_indices(n_test);
    
    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        std::vector<std::pair<double, int>> dist_idx_pairs;
        for (int j = 0; j < distances.cols(); ++j) {
            dist_idx_pairs.emplace_back(distances(i, j), j);
        }
        
        std::sort(dist_idx_pairs.begin(), dist_idx_pairs.end());
        
        neighbor_indices[i].reserve(n_neighbors_);
        for (int k = 0; k < n_neighbors_; ++k) {
            neighbor_indices[i].push_back(dist_idx_pairs[k].second);
        }
    }
    
    return neighbor_indices;
}

VectorXi KNeighborsClassifier::predict_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const {
    int n_test = neighbor_indices.size();
    VectorXi predictions(n_test);
    
    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        std::map<int, int> class_counts;
        
        for (int neighbor_idx : neighbor_indices[i]) {
            int class_label = static_cast<int>(y_train_(neighbor_idx));
            class_counts[class_label]++;
        }
        
        // Find the class with maximum count
        int max_count = 0;
        int predicted_class = class_counts.begin()->first;
        
        for (const auto& pair : class_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                predicted_class = pair.first;
            }
        }
        
        predictions(i) = predicted_class;
    }
    
    return predictions;
}

MatrixXd KNeighborsClassifier::predict_proba_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const {
    int n_test = neighbor_indices.size();
    
    // Find unique classes
    std::set<int> unique_classes;
    for (int i = 0; i < y_train_.size(); ++i) {
        unique_classes.insert(static_cast<int>(y_train_(i)));
    }
    
    int n_classes = unique_classes.size();
    std::vector<int> class_labels(unique_classes.begin(), unique_classes.end());
    
    // Initialize matrix with zeros
    MatrixXd probabilities = MatrixXd::Zero(n_test, n_classes);
    
    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        std::map<int, int> class_counts;
        
        // Initialize counts for all classes
        for (int cls : class_labels) {
            class_counts[cls] = 0;
        }
        
        for (int neighbor_idx : neighbor_indices[i]) {
            int class_label = static_cast<int>(y_train_(neighbor_idx));
            class_counts[class_label]++;
        }
        
        // Calculate probabilities
        int total_neighbors = neighbor_indices[i].size();
        if (total_neighbors > 0) {
            for (int j = 0; j < n_classes; ++j) {
                int class_label = class_labels[j];
                probabilities(i, j) = static_cast<double>(class_counts[class_label]) / total_neighbors;
            }
        }
    }
    
    return probabilities;
}

void KNeighborsClassifier::save(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&n_neighbors_), sizeof(n_neighbors_));
    ofs.write(reinterpret_cast<const char*>(&p_), sizeof(p_));
    ofs.write(reinterpret_cast<const char*>(&n_jobs_), sizeof(n_jobs_));
    
    // Save string parameters
    size_t weights_len = weights_.length();
    ofs.write(reinterpret_cast<const char*>(&weights_len), sizeof(weights_len));
    ofs.write(weights_.c_str(), weights_len);
    
    size_t algorithm_len = algorithm_.length();
    ofs.write(reinterpret_cast<const char*>(&algorithm_len), sizeof(algorithm_len));
    ofs.write(algorithm_.c_str(), algorithm_len);
    
    size_t metric_len = metric_.length();
    ofs.write(reinterpret_cast<const char*>(&metric_len), sizeof(metric_len));
    ofs.write(metric_.c_str(), metric_len);
    
    // Save training data
    int rows = X_train_.rows();
    int cols = X_train_.cols();
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    if (rows > 0 && cols > 0) {
        ofs.write(reinterpret_cast<const char*>(X_train_.data()), rows * cols * sizeof(double));
    }
    
    int y_size = y_train_.size();
    ofs.write(reinterpret_cast<const char*>(&y_size), sizeof(y_size));
    if (y_size > 0) {
        ofs.write(reinterpret_cast<const char*>(y_train_.data()), y_size * sizeof(double));
    }
    
    ofs.close();
}

void KNeighborsClassifier::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&n_neighbors_), sizeof(n_neighbors_));
    ifs.read(reinterpret_cast<char*>(&p_), sizeof(p_));
    ifs.read(reinterpret_cast<char*>(&n_jobs_), sizeof(n_jobs_));
    
    // Load string parameters
    size_t weights_len;
    ifs.read(reinterpret_cast<char*>(&weights_len), sizeof(weights_len));
    weights_.resize(weights_len);
    ifs.read(&weights_[0], weights_len);
    
    size_t algorithm_len;
    ifs.read(reinterpret_cast<char*>(&algorithm_len), sizeof(algorithm_len));
    algorithm_.resize(algorithm_len);
    ifs.read(&algorithm_[0], algorithm_len);
    
    size_t metric_len;
    ifs.read(reinterpret_cast<char*>(&metric_len), sizeof(metric_len));
    metric_.resize(metric_len);
    ifs.read(&metric_[0], metric_len);
    
    // Load training data
    int rows, cols;
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (rows > 0 && cols > 0) {
        X_train_.resize(rows, cols);
        ifs.read(reinterpret_cast<char*>(X_train_.data()), rows * cols * sizeof(double));
    }
    
    int y_size;
    ifs.read(reinterpret_cast<char*>(&y_size), sizeof(y_size));
    if (y_size > 0) {
        y_train_.resize(y_size);
        ifs.read(reinterpret_cast<char*>(y_train_.data()), y_size * sizeof(double));
    }
    
    ifs.close();
}

// KNeighborsRegressor implementation
KNeighborsRegressor::KNeighborsRegressor(int n_neighbors, const std::string& weights,
                                       const std::string& algorithm, const std::string& metric,
                                       double p, int n_jobs)
    : X_train_(), y_train_(), fitted_(false), n_neighbors_(n_neighbors), 
      weights_(weights), algorithm_(algorithm), metric_(metric), p_(p), n_jobs_(n_jobs) {
    if (n_neighbors <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
}

Estimator& KNeighborsRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    X_train_ = X;
    y_train_ = y;
    fitted_ = true;
    return *this;
}

VectorXd KNeighborsRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KNeighborsRegressor must be fitted before predict");
    }
    
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd distances = compute_distances(X);
    std::vector<std::vector<int>> neighbor_indices = find_k_neighbors(distances);
    
    return predict_from_neighbors(neighbor_indices);
}

Params KNeighborsRegressor::get_params() const {
    return {
        {"n_neighbors", std::to_string(n_neighbors_)},
        {"weights", weights_},
        {"algorithm", algorithm_},
        {"metric", metric_},
        {"p", std::to_string(p_)},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& KNeighborsRegressor::set_params(const Params& params) {
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", n_neighbors_);
    weights_ = utils::get_param_string(params, "weights", weights_);
    algorithm_ = utils::get_param_string(params, "algorithm", algorithm_);
    metric_ = utils::get_param_string(params, "metric", metric_);
    p_ = utils::get_param_double(params, "p", p_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool KNeighborsRegressor::is_fitted() const {
    return fitted_;
}

MatrixXd KNeighborsRegressor::compute_distances(const MatrixXd& X) const {
    int n_test = X.rows();
    int n_train = X_train_.rows();
    
    MatrixXd distances(n_test, n_train);
    
    #pragma omp parallel for if(n_test > 32)
    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_train; ++j) {
            if (metric_ == "minkowski") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    dist += std::pow(std::abs(X(i, k) - X_train_(j, k)), p_);
                }
                distances(i, j) = std::pow(dist, 1.0 / p_);
            } else if (metric_ == "euclidean") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    double diff = X(i, k) - X_train_(j, k);
                    dist += diff * diff;
                }
                distances(i, j) = std::sqrt(dist);
            } else if (metric_ == "manhattan") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    dist += std::abs(X(i, k) - X_train_(j, k));
                }
                distances(i, j) = dist;
            } else {
                throw std::invalid_argument("Unsupported metric: " + metric_);
            }
        }
    }
    
    return distances;
}

std::vector<std::vector<int>> KNeighborsRegressor::find_k_neighbors(const MatrixXd& distances) const {
    int n_test = distances.rows();
    std::vector<std::vector<int>> neighbor_indices(n_test);
    
    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        std::vector<std::pair<double, int>> dist_idx_pairs;
        for (int j = 0; j < distances.cols(); ++j) {
            dist_idx_pairs.emplace_back(distances(i, j), j);
        }
        
        std::sort(dist_idx_pairs.begin(), dist_idx_pairs.end());
        
        neighbor_indices[i].reserve(n_neighbors_);
        for (int k = 0; k < n_neighbors_; ++k) {
            neighbor_indices[i].push_back(dist_idx_pairs[k].second);
        }
    }
    
    return neighbor_indices;
}

VectorXd KNeighborsRegressor::predict_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const {
    int n_test = neighbor_indices.size();
    VectorXd predictions(n_test);
    
    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        double sum = 0.0;
        for (int neighbor_idx : neighbor_indices[i]) {
            sum += y_train_(neighbor_idx);
        }
        predictions(i) = sum / neighbor_indices[i].size();
    }
    
    return predictions;
}

void KNeighborsRegressor::save(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&n_neighbors_), sizeof(n_neighbors_));
    ofs.write(reinterpret_cast<const char*>(&p_), sizeof(p_));
    ofs.write(reinterpret_cast<const char*>(&n_jobs_), sizeof(n_jobs_));
    
    // Save string parameters
    size_t weights_len = weights_.length();
    ofs.write(reinterpret_cast<const char*>(&weights_len), sizeof(weights_len));
    ofs.write(weights_.c_str(), weights_len);
    
    size_t algorithm_len = algorithm_.length();
    ofs.write(reinterpret_cast<const char*>(&algorithm_len), sizeof(algorithm_len));
    ofs.write(algorithm_.c_str(), algorithm_len);
    
    size_t metric_len = metric_.length();
    ofs.write(reinterpret_cast<const char*>(&metric_len), sizeof(metric_len));
    ofs.write(metric_.c_str(), metric_len);
    
    // Save training data
    int rows = X_train_.rows();
    int cols = X_train_.cols();
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    if (rows > 0 && cols > 0) {
        ofs.write(reinterpret_cast<const char*>(X_train_.data()), rows * cols * sizeof(double));
    }
    
    int y_size = y_train_.size();
    ofs.write(reinterpret_cast<const char*>(&y_size), sizeof(y_size));
    if (y_size > 0) {
        ofs.write(reinterpret_cast<const char*>(y_train_.data()), y_size * sizeof(double));
    }
    
    ofs.close();
}

void KNeighborsRegressor::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&n_neighbors_), sizeof(n_neighbors_));
    ifs.read(reinterpret_cast<char*>(&p_), sizeof(p_));
    ifs.read(reinterpret_cast<char*>(&n_jobs_), sizeof(n_jobs_));
    
    // Load string parameters
    size_t weights_len;
    ifs.read(reinterpret_cast<char*>(&weights_len), sizeof(weights_len));
    weights_.resize(weights_len);
    ifs.read(&weights_[0], weights_len);
    
    size_t algorithm_len;
    ifs.read(reinterpret_cast<char*>(&algorithm_len), sizeof(algorithm_len));
    algorithm_.resize(algorithm_len);
    ifs.read(&algorithm_[0], algorithm_len);
    
    size_t metric_len;
    ifs.read(reinterpret_cast<char*>(&metric_len), sizeof(metric_len));
    metric_.resize(metric_len);
    ifs.read(&metric_[0], metric_len);
    
    // Load training data
    int rows, cols;
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (rows > 0 && cols > 0) {
        X_train_.resize(rows, cols);
        ifs.read(reinterpret_cast<char*>(X_train_.data()), rows * cols * sizeof(double));
    }
    
    int y_size;
    ifs.read(reinterpret_cast<char*>(&y_size), sizeof(y_size));
    if (y_size > 0) {
        y_train_.resize(y_size);
        ifs.read(reinterpret_cast<char*>(y_train_.data()), y_size * sizeof(double));
    }
    
    ifs.close();
}

} // namespace neighbors
} // namespace cxml

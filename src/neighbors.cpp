#include "ingenuityml/neighbors.hpp"
#include "ingenuityml/base.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <map>
#include <set>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ingenuityml {
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
      weights_(weights), algorithm_(algorithm), metric_(metric), p_(p), n_jobs_(n_jobs) {}

Estimator& KNeighborsRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    if (n_neighbors_ <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
    validation::check_X_y(X, y);
    if (n_neighbors_ > X.rows()) {
        throw std::invalid_argument("n_neighbors cannot be greater than number of samples");
    }
    
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
        throw std::runtime_error("X must have the same number of features as training data");
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

// RadiusNeighborsClassifier implementation
RadiusNeighborsClassifier::RadiusNeighborsClassifier(double radius, const std::string& weights,
                                                   const std::string& algorithm, const std::string& metric,
                                                   double p, int n_jobs)
    : X_train_(), y_train_(), fitted_(false), radius_(radius),
      weights_(weights), algorithm_(algorithm), metric_(metric), p_(p), n_jobs_(n_jobs) {
    if (radius <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
}

Estimator& RadiusNeighborsClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (radius_ <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
    X_train_ = X;
    y_train_ = y;
    fitted_ = true;
    return *this;
}

VectorXi RadiusNeighborsClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RadiusNeighborsClassifier must be fitted before predict");
    }
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd distances = compute_distances(X);
    std::vector<std::vector<int>> neighbor_indices = find_radius_neighbors(distances);
    return predict_from_neighbors(neighbor_indices);
}

MatrixXd RadiusNeighborsClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RadiusNeighborsClassifier must be fitted before predict_proba");
    }
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd distances = compute_distances(X);
    std::vector<std::vector<int>> neighbor_indices = find_radius_neighbors(distances);
    return predict_proba_from_neighbors(neighbor_indices);
}

VectorXd RadiusNeighborsClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RadiusNeighborsClassifier must be fitted before decision_function");
    }
    MatrixXd probabilities = predict_proba(X);
    VectorXd decision_values(X.rows());
    for (int i = 0; i < probabilities.rows(); ++i) {
        decision_values(i) = probabilities.row(i).maxCoeff();
    }
    return decision_values;
}

Params RadiusNeighborsClassifier::get_params() const {
    return {
        {"radius", std::to_string(radius_)},
        {"weights", weights_},
        {"algorithm", algorithm_},
        {"metric", metric_},
        {"p", std::to_string(p_)},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& RadiusNeighborsClassifier::set_params(const Params& params) {
    radius_ = utils::get_param_double(params, "radius", radius_);
    weights_ = utils::get_param_string(params, "weights", weights_);
    algorithm_ = utils::get_param_string(params, "algorithm", algorithm_);
    metric_ = utils::get_param_string(params, "metric", metric_);
    p_ = utils::get_param_double(params, "p", p_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool RadiusNeighborsClassifier::is_fitted() const {
    return fitted_;
}

MatrixXd RadiusNeighborsClassifier::compute_distances(const MatrixXd& X) const {
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

std::vector<std::vector<int>> RadiusNeighborsClassifier::find_radius_neighbors(const MatrixXd& distances) const {
    int n_test = distances.rows();
    std::vector<std::vector<int>> neighbor_indices(n_test);

    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        double best_dist = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (int j = 0; j < distances.cols(); ++j) {
            double dist = distances(i, j);
            if (dist <= radius_) {
                neighbor_indices[i].push_back(j);
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        if (neighbor_indices[i].empty() && best_idx >= 0) {
            neighbor_indices[i].push_back(best_idx);
        }
    }

    return neighbor_indices;
}

VectorXi RadiusNeighborsClassifier::predict_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const {
    int n_test = neighbor_indices.size();
    VectorXi predictions(n_test);

    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        std::map<int, int> class_counts;
        for (int neighbor_idx : neighbor_indices[i]) {
            int class_label = static_cast<int>(y_train_(neighbor_idx));
            class_counts[class_label]++;
        }
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

MatrixXd RadiusNeighborsClassifier::predict_proba_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const {
    int n_test = neighbor_indices.size();

    std::set<int> unique_classes;
    for (int i = 0; i < y_train_.size(); ++i) {
        unique_classes.insert(static_cast<int>(y_train_(i)));
    }
    int n_classes = unique_classes.size();
    std::vector<int> class_labels(unique_classes.begin(), unique_classes.end());

    MatrixXd probabilities = MatrixXd::Zero(n_test, n_classes);

    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        std::map<int, int> class_counts;
        for (int cls : class_labels) {
            class_counts[cls] = 0;
        }
        for (int neighbor_idx : neighbor_indices[i]) {
            int class_label = static_cast<int>(y_train_(neighbor_idx));
            class_counts[class_label]++;
        }
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

// RadiusNeighborsRegressor implementation
RadiusNeighborsRegressor::RadiusNeighborsRegressor(double radius, const std::string& weights,
                                                 const std::string& algorithm, const std::string& metric,
                                                 double p, int n_jobs)
    : X_train_(), y_train_(), fitted_(false), radius_(radius),
      weights_(weights), algorithm_(algorithm), metric_(metric), p_(p), n_jobs_(n_jobs) {
    if (radius <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
}

Estimator& RadiusNeighborsRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (radius_ <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
    X_train_ = X;
    y_train_ = y;
    fitted_ = true;
    return *this;
}

VectorXd RadiusNeighborsRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RadiusNeighborsRegressor must be fitted before predict");
    }
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd distances = compute_distances(X);
    std::vector<std::vector<int>> neighbor_indices = find_radius_neighbors(distances);
    return predict_from_neighbors(neighbor_indices);
}

Params RadiusNeighborsRegressor::get_params() const {
    return {
        {"radius", std::to_string(radius_)},
        {"weights", weights_},
        {"algorithm", algorithm_},
        {"metric", metric_},
        {"p", std::to_string(p_)},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& RadiusNeighborsRegressor::set_params(const Params& params) {
    radius_ = utils::get_param_double(params, "radius", radius_);
    weights_ = utils::get_param_string(params, "weights", weights_);
    algorithm_ = utils::get_param_string(params, "algorithm", algorithm_);
    metric_ = utils::get_param_string(params, "metric", metric_);
    p_ = utils::get_param_double(params, "p", p_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool RadiusNeighborsRegressor::is_fitted() const {
    return fitted_;
}

MatrixXd RadiusNeighborsRegressor::compute_distances(const MatrixXd& X) const {
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

std::vector<std::vector<int>> RadiusNeighborsRegressor::find_radius_neighbors(const MatrixXd& distances) const {
    int n_test = distances.rows();
    std::vector<std::vector<int>> neighbor_indices(n_test);

    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        double best_dist = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (int j = 0; j < distances.cols(); ++j) {
            double dist = distances(i, j);
            if (dist <= radius_) {
                neighbor_indices[i].push_back(j);
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        if (neighbor_indices[i].empty() && best_idx >= 0) {
            neighbor_indices[i].push_back(best_idx);
        }
    }

    return neighbor_indices;
}

VectorXd RadiusNeighborsRegressor::predict_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const {
    int n_test = neighbor_indices.size();
    VectorXd predictions(n_test);

    #pragma omp parallel for if(n_test > 16)
    for (int i = 0; i < n_test; ++i) {
        double sum = 0.0;
        for (int neighbor_idx : neighbor_indices[i]) {
            sum += y_train_(neighbor_idx);
        }
        if (!neighbor_indices[i].empty()) {
            predictions(i) = sum / static_cast<double>(neighbor_indices[i].size());
        } else {
            predictions(i) = 0.0;
        }
    }

    return predictions;
}

// NearestCentroid implementation
NearestCentroid::NearestCentroid(const std::string& metric, double p)
    : centroids_(), classes_(), fitted_(false), metric_(metric), p_(p) {}

Estimator& NearestCentroid::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);

    std::set<int> unique_classes;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes.insert(static_cast<int>(y(i)));
    }
    classes_ = std::vector<int>(unique_classes.begin(), unique_classes.end());
    int n_classes = classes_.size();
    centroids_ = MatrixXd::Zero(n_classes, X.cols());

    for (int c = 0; c < n_classes; ++c) {
        int class_label = classes_[c];
        int count = 0;
        VectorXd sum = VectorXd::Zero(X.cols());
        for (int i = 0; i < X.rows(); ++i) {
            if (static_cast<int>(y(i)) == class_label) {
                sum += X.row(i).transpose();
                count++;
            }
        }
        if (count > 0) {
            centroids_.row(c) = (sum / static_cast<double>(count)).transpose();
        }
    }

    fitted_ = true;
    return *this;
}

VectorXi NearestCentroid::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("NearestCentroid not fitted");
    }
    if (X.cols() != centroids_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd distances = compute_distances(X);
    VectorXi predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        int best_idx = 0;
        double best_dist = distances(i, 0);
        for (int j = 1; j < distances.cols(); ++j) {
            if (distances(i, j) < best_dist) {
                best_dist = distances(i, j);
                best_idx = j;
            }
        }
        predictions(i) = classes_[best_idx];
    }
    return predictions;
}

MatrixXd NearestCentroid::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("NearestCentroid not fitted");
    }
    if (X.cols() != centroids_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd distances = compute_distances(X);
    MatrixXd probabilities = MatrixXd::Zero(X.rows(), centroids_.rows());
    for (int i = 0; i < distances.rows(); ++i) {
        double weight_sum = 0.0;
        for (int j = 0; j < distances.cols(); ++j) {
            double weight = 1.0 / (distances(i, j) + 1e-12);
            probabilities(i, j) = weight;
            weight_sum += weight;
        }
        if (weight_sum > 0.0) {
            probabilities.row(i) /= weight_sum;
        }
    }
    return probabilities;
}

VectorXd NearestCentroid::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("NearestCentroid not fitted");
    }
    if (X.cols() != centroids_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd distances = compute_distances(X);
    VectorXd scores(X.rows());
    for (int i = 0; i < distances.rows(); ++i) {
        scores(i) = -distances.row(i).minCoeff();
    }
    return scores;
}

Params NearestCentroid::get_params() const {
    return {
        {"metric", metric_},
        {"p", std::to_string(p_)}
    };
}

Estimator& NearestCentroid::set_params(const Params& params) {
    metric_ = utils::get_param_string(params, "metric", metric_);
    p_ = utils::get_param_double(params, "p", p_);
    return *this;
}

bool NearestCentroid::is_fitted() const {
    return fitted_;
}

MatrixXd NearestCentroid::compute_distances(const MatrixXd& X) const {
    MatrixXd distances(X.rows(), centroids_.rows());
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < centroids_.rows(); ++j) {
            if (metric_ == "minkowski") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    dist += std::pow(std::abs(X(i, k) - centroids_(j, k)), p_);
                }
                distances(i, j) = std::pow(dist, 1.0 / p_);
            } else if (metric_ == "manhattan") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    dist += std::abs(X(i, k) - centroids_(j, k));
                }
                distances(i, j) = dist;
            } else {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    double diff = X(i, k) - centroids_(j, k);
                    dist += diff * diff;
                }
                distances(i, j) = std::sqrt(dist);
            }
        }
    }
    return distances;
}

// NearestNeighbors implementation
NearestNeighbors::NearestNeighbors(int n_neighbors, double radius,
                                   const std::string& algorithm, const std::string& metric,
                                   double p, int n_jobs)
    : X_train_(), fitted_(false), n_neighbors_(n_neighbors), radius_(radius),
      algorithm_(algorithm), metric_(metric), p_(p), n_jobs_(n_jobs) {
    if (n_neighbors_ <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
    if (radius_ <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
}

Estimator& NearestNeighbors::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    if (n_neighbors_ <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
    if (n_neighbors_ > X.rows()) {
        throw std::invalid_argument("n_neighbors cannot be greater than number of samples");
    }
    if (radius_ <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
    X_train_ = X;
    fitted_ = true;
    return *this;
}

std::pair<MatrixXd, MatrixXi> NearestNeighbors::kneighbors(const MatrixXd& X, int n_neighbors) const {
    if (!fitted_) {
        throw std::runtime_error("NearestNeighbors must be fitted before kneighbors");
    }
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    int k = n_neighbors > 0 ? n_neighbors : n_neighbors_;
    if (k <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
    if (k > X_train_.rows()) {
        throw std::invalid_argument("n_neighbors cannot be greater than number of samples");
    }

    MatrixXd distances = compute_distances(X);
    MatrixXd out_dist(X.rows(), k);
    MatrixXi out_idx(X.rows(), k);

    #pragma omp parallel for if(X.rows() > 16)
    for (int i = 0; i < X.rows(); ++i) {
        std::vector<std::pair<double, int>> dist_idx;
        dist_idx.reserve(X_train_.rows());
        for (int j = 0; j < X_train_.rows(); ++j) {
            dist_idx.emplace_back(distances(i, j), j);
        }
        std::partial_sort(dist_idx.begin(), dist_idx.begin() + k, dist_idx.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
        for (int j = 0; j < k; ++j) {
            out_dist(i, j) = dist_idx[j].first;
            out_idx(i, j) = dist_idx[j].second;
        }
    }

    return {out_dist, out_idx};
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> NearestNeighbors::radius_neighbors(
    const MatrixXd& X, double radius) const {
    if (!fitted_) {
        throw std::runtime_error("NearestNeighbors must be fitted before radius_neighbors");
    }
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    double r = radius > 0.0 ? radius : radius_;
    if (r <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }

    MatrixXd distances = compute_distances(X);
    std::vector<std::vector<double>> out_dist(X.rows());
    std::vector<std::vector<int>> out_idx(X.rows());

    #pragma omp parallel for if(X.rows() > 16)
    for (int i = 0; i < X.rows(); ++i) {
        std::vector<double> dists;
        std::vector<int> idxs;
        for (int j = 0; j < X_train_.rows(); ++j) {
            double dist = distances(i, j);
            if (dist <= r) {
                dists.push_back(dist);
                idxs.push_back(j);
            }
        }
        out_dist[i] = std::move(dists);
        out_idx[i] = std::move(idxs);
    }

    return {out_dist, out_idx};
}

Params NearestNeighbors::get_params() const {
    return {
        {"n_neighbors", std::to_string(n_neighbors_)},
        {"radius", std::to_string(radius_)},
        {"algorithm", algorithm_},
        {"metric", metric_},
        {"p", std::to_string(p_)},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& NearestNeighbors::set_params(const Params& params) {
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", n_neighbors_);
    radius_ = utils::get_param_double(params, "radius", radius_);
    algorithm_ = utils::get_param_string(params, "algorithm", algorithm_);
    metric_ = utils::get_param_string(params, "metric", metric_);
    p_ = utils::get_param_double(params, "p", p_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool NearestNeighbors::is_fitted() const {
    return fitted_;
}

MatrixXd NearestNeighbors::compute_distances(const MatrixXd& X) const {
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
            } else if (metric_ == "manhattan") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    dist += std::abs(X(i, k) - X_train_(j, k));
                }
                distances(i, j) = dist;
            } else {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    double diff = X(i, k) - X_train_(j, k);
                    dist += diff * diff;
                }
                distances(i, j) = std::sqrt(dist);
            }
        }
    }

    return distances;
}

// KNeighborsTransformer implementation
KNeighborsTransformer::KNeighborsTransformer(int n_neighbors, const std::string& mode,
                                             const std::string& metric, double p, int n_jobs)
    : X_train_(), fitted_(false), n_neighbors_(n_neighbors), mode_(mode),
      metric_(metric), p_(p), n_jobs_(n_jobs) {
    if (n_neighbors_ <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
    if (mode_ != "distance" && mode_ != "connectivity") {
        throw std::invalid_argument("mode must be 'distance' or 'connectivity'");
    }
}

Estimator& KNeighborsTransformer::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    if (n_neighbors_ <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
    if (n_neighbors_ > X.rows()) {
        throw std::invalid_argument("n_neighbors cannot be greater than number of samples");
    }
    if (mode_ != "distance" && mode_ != "connectivity") {
        throw std::invalid_argument("mode must be 'distance' or 'connectivity'");
    }
    X_train_ = X;
    fitted_ = true;
    return *this;
}

MatrixXd KNeighborsTransformer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KNeighborsTransformer must be fitted before transform");
    }
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    if (n_neighbors_ <= 0) {
        throw std::invalid_argument("n_neighbors must be positive");
    }
    if (n_neighbors_ > X_train_.rows()) {
        throw std::invalid_argument("n_neighbors cannot be greater than number of samples");
    }

    MatrixXd distances = compute_distances(X);
    MatrixXd graph = MatrixXd::Zero(X.rows(), X_train_.rows());

    #pragma omp parallel for if(X.rows() > 16)
    for (int i = 0; i < X.rows(); ++i) {
        std::vector<std::pair<double, int>> dist_idx;
        dist_idx.reserve(X_train_.rows());
        for (int j = 0; j < X_train_.rows(); ++j) {
            dist_idx.emplace_back(distances(i, j), j);
        }
        std::partial_sort(dist_idx.begin(), dist_idx.begin() + n_neighbors_, dist_idx.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
        for (int j = 0; j < n_neighbors_; ++j) {
            int idx = dist_idx[j].second;
            graph(i, idx) = (mode_ == "connectivity") ? 1.0 : dist_idx[j].first;
        }
    }

    return graph;
}

MatrixXd KNeighborsTransformer::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("KNeighborsTransformer does not support inverse_transform");
}

MatrixXd KNeighborsTransformer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params KNeighborsTransformer::get_params() const {
    return {
        {"n_neighbors", std::to_string(n_neighbors_)},
        {"mode", mode_},
        {"metric", metric_},
        {"p", std::to_string(p_)},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& KNeighborsTransformer::set_params(const Params& params) {
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", n_neighbors_);
    mode_ = utils::get_param_string(params, "mode", mode_);
    metric_ = utils::get_param_string(params, "metric", metric_);
    p_ = utils::get_param_double(params, "p", p_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool KNeighborsTransformer::is_fitted() const {
    return fitted_;
}

MatrixXd KNeighborsTransformer::compute_distances(const MatrixXd& X) const {
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
            } else if (metric_ == "manhattan") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    dist += std::abs(X(i, k) - X_train_(j, k));
                }
                distances(i, j) = dist;
            } else {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    double diff = X(i, k) - X_train_(j, k);
                    dist += diff * diff;
                }
                distances(i, j) = std::sqrt(dist);
            }
        }
    }

    return distances;
}

// RadiusNeighborsTransformer implementation
RadiusNeighborsTransformer::RadiusNeighborsTransformer(double radius, const std::string& mode,
                                                       const std::string& metric, double p, int n_jobs)
    : X_train_(), fitted_(false), radius_(radius), mode_(mode),
      metric_(metric), p_(p), n_jobs_(n_jobs) {
    if (radius_ <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
    if (mode_ != "distance" && mode_ != "connectivity") {
        throw std::invalid_argument("mode must be 'distance' or 'connectivity'");
    }
}

Estimator& RadiusNeighborsTransformer::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    if (radius_ <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }
    if (mode_ != "distance" && mode_ != "connectivity") {
        throw std::invalid_argument("mode must be 'distance' or 'connectivity'");
    }
    X_train_ = X;
    fitted_ = true;
    return *this;
}

MatrixXd RadiusNeighborsTransformer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RadiusNeighborsTransformer must be fitted before transform");
    }
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    if (radius_ <= 0.0) {
        throw std::invalid_argument("radius must be positive");
    }

    MatrixXd distances = compute_distances(X);
    MatrixXd graph = MatrixXd::Zero(X.rows(), X_train_.rows());

    #pragma omp parallel for if(X.rows() > 16)
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X_train_.rows(); ++j) {
            double dist = distances(i, j);
            if (dist <= radius_) {
                graph(i, j) = (mode_ == "connectivity") ? 1.0 : dist;
            }
        }
    }

    return graph;
}

MatrixXd RadiusNeighborsTransformer::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("RadiusNeighborsTransformer does not support inverse_transform");
}

MatrixXd RadiusNeighborsTransformer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params RadiusNeighborsTransformer::get_params() const {
    return {
        {"radius", std::to_string(radius_)},
        {"mode", mode_},
        {"metric", metric_},
        {"p", std::to_string(p_)},
        {"n_jobs", std::to_string(n_jobs_)}
    };
}

Estimator& RadiusNeighborsTransformer::set_params(const Params& params) {
    radius_ = utils::get_param_double(params, "radius", radius_);
    mode_ = utils::get_param_string(params, "mode", mode_);
    metric_ = utils::get_param_string(params, "metric", metric_);
    p_ = utils::get_param_double(params, "p", p_);
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

bool RadiusNeighborsTransformer::is_fitted() const {
    return fitted_;
}

MatrixXd RadiusNeighborsTransformer::compute_distances(const MatrixXd& X) const {
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
            } else if (metric_ == "manhattan") {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    dist += std::abs(X(i, k) - X_train_(j, k));
                }
                distances(i, j) = dist;
            } else {
                double dist = 0.0;
                for (int k = 0; k < X.cols(); ++k) {
                    double diff = X(i, k) - X_train_(j, k);
                    dist += diff * diff;
                }
                distances(i, j) = std::sqrt(dist);
            }
        }
    }

    return distances;
}

} // namespace neighbors
} // namespace ingenuityml

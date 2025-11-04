#include "auroraml/cluster_extended.hpp"
#include "auroraml/base.hpp"
#include "auroraml/kmeans.hpp"
#include "auroraml/pca.hpp"
#include <cmath>
#include <random>
#include <algorithm>

namespace auroraml {
namespace cluster {

// MiniBatchKMeans implementation

MiniBatchKMeans::MiniBatchKMeans(
    int n_clusters,
    int max_iter,
    double tol,
    int batch_size,
    int random_state
) : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol),
    batch_size_(batch_size), random_state_(random_state), fitted_(false) {
}

void MiniBatchKMeans::init_centroids(const MatrixXd& X, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, X.rows() - 1);
    cluster_centers_ = MatrixXd::Zero(n_clusters_, X.cols());
    
    for (int i = 0; i < n_clusters_; ++i) {
        int idx = dist(rng);
        cluster_centers_.row(i) = X.row(idx);
    }
}

void MiniBatchKMeans::update_centroids_minibatch(const MatrixXd& X_batch, const VectorXi& assignments,
                                                 std::vector<int>& counts, std::mt19937& rng) {
    for (int i = 0; i < X_batch.rows(); ++i) {
        int cluster = assignments(i);
        counts[cluster]++;
        
        // Update centroid with learning rate
        double learning_rate = 1.0 / counts[cluster];
        cluster_centers_.row(cluster) = (1.0 - learning_rate) * cluster_centers_.row(cluster) +
                                         learning_rate * X_batch.row(i);
    }
}

Estimator& MiniBatchKMeans::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    std::mt19937 rng;
    if (random_state_ >= 0) {
        rng.seed(random_state_);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    init_centroids(X, rng);
    
    int n_samples = X.rows();
    std::vector<int> counts(n_clusters_, 0);
    
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Shuffle indices
        std::shuffle(indices.begin(), indices.end(), rng);
        
        // Process in batches
        for (int batch_start = 0; batch_start < n_samples; batch_start += batch_size_) {
            int batch_end = std::min(batch_start + batch_size_, n_samples);
            int current_batch_size = batch_end - batch_start;
            
            MatrixXd X_batch(current_batch_size, X.cols());
            for (int i = 0; i < current_batch_size; ++i) {
                X_batch.row(i) = X.row(indices[batch_start + i]);
            }
            
            // Assign to nearest centroid
            VectorXi assignments = VectorXi::Zero(current_batch_size);
            for (int i = 0; i < current_batch_size; ++i) {
                double min_dist = (X_batch.row(i) - cluster_centers_.row(0)).squaredNorm();
                int nearest = 0;
                
                for (int k = 1; k < n_clusters_; ++k) {
                    double dist = (X_batch.row(i) - cluster_centers_.row(k)).squaredNorm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest = k;
                    }
                }
                assignments(i) = nearest;
            }
            
            // Update centroids
            update_centroids_minibatch(X_batch, assignments, counts, rng);
        }
    }
    
    // Final assignment
    labels_ = VectorXi::Zero(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        double min_dist = (X.row(i) - cluster_centers_.row(0)).squaredNorm();
        int nearest = 0;
        
        for (int k = 1; k < n_clusters_; ++k) {
            double dist = (X.row(i) - cluster_centers_.row(k)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                nearest = k;
            }
        }
        labels_(i) = nearest;
    }
    
    fitted_ = true;
    return *this;
}

VectorXi MiniBatchKMeans::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return labels_;
}

VectorXi MiniBatchKMeans::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchKMeans must be fitted before predict");
    }
    
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        double min_dist = (X.row(i) - cluster_centers_.row(0)).squaredNorm();
        int nearest = 0;
        
        for (int k = 1; k < n_clusters_; ++k) {
            double dist = (X.row(i) - cluster_centers_.row(k)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                nearest = k;
            }
        }
        predictions(i) = nearest;
    }
    
    return predictions;
}

Params MiniBatchKMeans::get_params() const {
    Params params;
    params["n_clusters"] = std::to_string(n_clusters_);
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    params["batch_size"] = std::to_string(batch_size_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& MiniBatchKMeans::set_params(const Params& params) {
    n_clusters_ = utils::get_param_int(params, "n_clusters", n_clusters_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    batch_size_ = utils::get_param_int(params, "batch_size", batch_size_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// SpectralClustering implementation

SpectralClustering::SpectralClustering(
    int n_clusters,
    const std::string& affinity,
    double gamma,
    int n_neighbors,
    int random_state
) : n_clusters_(n_clusters), affinity_(affinity), gamma_(gamma),
    n_neighbors_(n_neighbors), random_state_(random_state), fitted_(false) {
}

MatrixXd SpectralClustering::build_affinity_matrix(const MatrixXd& X) const {
    int n_samples = X.rows();
    MatrixXd affinity = MatrixXd::Zero(n_samples, n_samples);
    
    if (affinity_ == "rbf") {
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_samples; ++j) {
                if (i == j) {
                    affinity(i, j) = 1.0;
                } else {
                    double dist_sq = (X.row(i) - X.row(j)).squaredNorm();
                    affinity(i, j) = std::exp(-gamma_ * dist_sq);
                }
            }
        }
    } else if (affinity_ == "nearest_neighbors") {
        // KNN-based affinity
        for (int i = 0; i < n_samples; ++i) {
            std::vector<std::pair<double, int>> distances;
            for (int j = 0; j < n_samples; ++j) {
                if (i != j) {
                    double dist = (X.row(i) - X.row(j)).norm();
                    distances.push_back({dist, j});
                }
            }
            std::sort(distances.begin(), distances.end());
            
            for (int k = 0; k < std::min(n_neighbors_, static_cast<int>(distances.size())); ++k) {
                int j = distances[k].second;
                affinity(i, j) = 1.0;
                affinity(j, i) = 1.0;
            }
        }
    }
    
    return affinity;
}

void SpectralClustering::perform_clustering(const MatrixXd& X) {
    // Build affinity matrix
    MatrixXd affinity = build_affinity_matrix(X);
    
    // Compute normalized Laplacian
    VectorXd degree = affinity.rowwise().sum();
    for (int i = 0; i < degree.size(); ++i) {
        if (degree(i) > 0) {
            degree(i) = 1.0 / std::sqrt(degree(i));
        }
    }
    
    MatrixXd normalized_affinity = affinity;
    for (int i = 0; i < normalized_affinity.rows(); ++i) {
        for (int j = 0; j < normalized_affinity.cols(); ++j) {
            normalized_affinity(i, j) *= degree(i) * degree(j);
        }
    }
    
    // Use PCA to reduce dimensionality (simplified spectral clustering)
    decomposition::PCA pca(n_clusters_);
    pca.fit(normalized_affinity, VectorXd());
    MatrixXd embeddings = pca.transform(normalized_affinity);
    
    // Apply KMeans on embeddings
    KMeans kmeans(n_clusters_, 100, 1e-4, "k-means++", random_state_);
    kmeans.fit(embeddings, VectorXd());
    labels_ = kmeans.predict_labels(embeddings);
}

Estimator& SpectralClustering::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    perform_clustering(X);
    
    fitted_ = true;
    return *this;
}

VectorXi SpectralClustering::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return labels_;
}

Params SpectralClustering::get_params() const {
    Params params;
    params["n_clusters"] = std::to_string(n_clusters_);
    params["affinity"] = affinity_;
    params["gamma"] = std::to_string(gamma_);
    params["n_neighbors"] = std::to_string(n_neighbors_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& SpectralClustering::set_params(const Params& params) {
    n_clusters_ = utils::get_param_int(params, "n_clusters", n_clusters_);
    affinity_ = utils::get_param_string(params, "affinity", affinity_);
    gamma_ = utils::get_param_double(params, "gamma", gamma_);
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", n_neighbors_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

} // namespace cluster
} // namespace auroraml


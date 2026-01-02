#include "auroraml/cluster_extended.hpp"
#include "auroraml/base.hpp"
#include "auroraml/kmeans.hpp"
#include "auroraml/pca.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <limits>

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

// MeanShift implementation
MeanShift::MeanShift(double bandwidth, const std::vector<VectorXd>& seeds, bool bin_seeding,
                     int min_bin_freq, bool cluster_all, int max_iter)
    : bandwidth_(bandwidth), seeds_(seeds), bin_seeding_(bin_seeding),
      min_bin_freq_(min_bin_freq), cluster_all_(cluster_all), max_iter_(max_iter), fitted_(false) {}

Estimator& MeanShift::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    int n_samples = X.rows();
    int n_features = X.cols();
    
    // Estimate bandwidth if not provided
    if (bandwidth_ <= 0) {
        bandwidth_ = estimate_bandwidth(X);
    }
    
    // Initialize seeds
    std::vector<VectorXd> seeds;
    if (seeds_.empty()) {
        seeds.reserve(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            seeds.push_back(X.row(i));
        }
    } else {
        seeds = seeds_;
    }
    
    // Perform mean shift for each seed
    std::vector<VectorXd> centers;
    for (const auto& seed : seeds) {
        auto shifted = mean_shift_single_seed(X, seed);
        centers.insert(centers.end(), shifted.begin(), shifted.end());
    }
    
    // Remove duplicate centers
    std::vector<VectorXd> unique_centers;
    for (const auto& center : centers) {
        bool is_duplicate = false;
        for (const auto& existing : unique_centers) {
            if ((center - existing).norm() < bandwidth_ / 10.0) {
                is_duplicate = true;
                break;
            }
        }
        if (!is_duplicate) {
            unique_centers.push_back(center);
        }
    }
    
    // Convert to matrix
    int n_clusters = unique_centers.size();
    cluster_centers_ = MatrixXd(n_clusters, n_features);
    for (int i = 0; i < n_clusters; ++i) {
        cluster_centers_.row(i) = unique_centers[i];
    }
    
    // Assign labels
    labels_ = VectorXi(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < n_clusters; ++j) {
            double dist = (X.row(i) - cluster_centers_.row(j)).norm();
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels_(i) = best_cluster;
    }
    
    fitted_ = true;
    return *this;
}

VectorXi MeanShift::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return labels_;
}

VectorXi MeanShift::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MeanShift must be fitted before predict");
    }
    
    int n_samples = X.rows();
    int n_clusters = cluster_centers_.rows();
    VectorXi predictions(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < n_clusters; ++j) {
            double dist = (X.row(i) - cluster_centers_.row(j)).norm();
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        predictions(i) = best_cluster;
    }
    
    return predictions;
}

double MeanShift::estimate_bandwidth(const MatrixXd& X) const {
    // Simple bandwidth estimation using median of pairwise distances
    int n_samples = X.rows();
    std::vector<double> distances;
    
    int max_samples = std::min(100, n_samples);
    for (int i = 0; i < max_samples; ++i) {
        for (int j = i + 1; j < max_samples; ++j) {
            double dist = (X.row(i) - X.row(j)).norm();
            distances.push_back(dist);
        }
    }
    
    std::sort(distances.begin(), distances.end());
    return distances[distances.size() / 2]; // median
}

std::vector<VectorXd> MeanShift::mean_shift_single_seed(const MatrixXd& X, const VectorXd& seed) const {
    VectorXd current = seed;
    
    for (int iter = 0; iter < max_iter_; ++iter) {
        VectorXd numerator = VectorXd::Zero(current.size());
        double denominator = 0.0;
        
        for (int i = 0; i < X.rows(); ++i) {
            double dist = (X.row(i).transpose() - current).norm();
            if (dist < bandwidth_) {
                double weight = 1.0; // uniform kernel
                numerator += weight * X.row(i).transpose();
                denominator += weight;
            }
        }
        
        if (denominator == 0.0) break;
        
        VectorXd new_current = numerator / denominator;
        if ((new_current - current).norm() < 1e-3) break;
        current = new_current;
    }
    
    return {current};
}

Params MeanShift::get_params() const {
    return {
        {"bandwidth", std::to_string(bandwidth_)},
        {"bin_seeding", bin_seeding_ ? "true" : "false"},
        {"min_bin_freq", std::to_string(min_bin_freq_)},
        {"cluster_all", cluster_all_ ? "true" : "false"},
        {"max_iter", std::to_string(max_iter_)}
    };
}

Estimator& MeanShift::set_params(const Params& params) {
    if (params.count("bandwidth")) bandwidth_ = std::stod(params.at("bandwidth"));
    if (params.count("bin_seeding")) bin_seeding_ = (params.at("bin_seeding") == "true");
    if (params.count("min_bin_freq")) min_bin_freq_ = std::stoi(params.at("min_bin_freq"));
    if (params.count("cluster_all")) cluster_all_ = (params.at("cluster_all") == "true");
    if (params.count("max_iter")) max_iter_ = std::stoi(params.at("max_iter"));
    return *this;
}

// OPTICS implementation
OPTICS::OPTICS(int min_samples, double max_eps, const std::string& metric, double eps)
    : min_samples_(min_samples), max_eps_(max_eps), metric_(metric), eps_(eps), fitted_(false) {}

Estimator& OPTICS::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    int n_samples = X.rows();
    reachability_ = VectorXd::Constant(n_samples, std::numeric_limits<double>::infinity());
    core_distances_ = VectorXd::Constant(n_samples, std::numeric_limits<double>::infinity());
    ordering_ = VectorXi::Constant(n_samples, -1);
    labels_ = VectorXi::Constant(n_samples, -1);
    
    std::vector<bool> processed(n_samples, false);
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>>> seeds;
    
    int order_idx = 0;
    
    for (int i = 0; i < n_samples; ++i) {
        if (!processed[i]) {
            auto neighbors = get_neighbors(X, i, max_eps_);
            processed[i] = true;
            ordering_(order_idx++) = i;
            
            if (neighbors.size() >= min_samples_) {
                core_distances_(i) = compute_core_distance(X, i, neighbors);
                
                for (int neighbor : neighbors) {
                    if (!processed[neighbor]) {
                        double new_reach = std::max(core_distances_(i), (X.row(i) - X.row(neighbor)).norm());
                        if (new_reach < reachability_(neighbor)) {
                            reachability_(neighbor) = new_reach;
                            seeds.push({new_reach, neighbor});
                        }
                    }
                }
                
                while (!seeds.empty()) {
                    auto current = seeds.top();
                    seeds.pop();
                    int curr_idx = current.second;
                    
                    if (!processed[curr_idx]) {
                        processed[curr_idx] = true;
                        ordering_(order_idx++) = curr_idx;
                        
                        auto curr_neighbors = get_neighbors(X, curr_idx, max_eps_);
                        if (curr_neighbors.size() >= min_samples_) {
                            core_distances_(curr_idx) = compute_core_distance(X, curr_idx, curr_neighbors);
                            
                            for (int neighbor : curr_neighbors) {
                                if (!processed[neighbor]) {
                                    double new_reach = std::max(core_distances_(curr_idx), (X.row(curr_idx) - X.row(neighbor)).norm());
                                    if (new_reach < reachability_(neighbor)) {
                                        reachability_(neighbor) = new_reach;
                                        seeds.push({new_reach, neighbor});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    extract_clusters();
    fitted_ = true;
    return *this;
}

VectorXi OPTICS::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return labels_;
}

VectorXi OPTICS::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OPTICS must be fitted before predict");
    }
    return labels_;
}

double OPTICS::compute_core_distance(const MatrixXd& X, int point_idx, const std::vector<int>& neighbors) const {
    if (neighbors.size() < min_samples_) {
        return std::numeric_limits<double>::infinity();
    }
    
    std::vector<double> distances;
    for (int neighbor : neighbors) {
        distances.push_back((X.row(point_idx) - X.row(neighbor)).norm());
    }
    
    std::sort(distances.begin(), distances.end());
    return distances[min_samples_ - 1];
}

std::vector<int> OPTICS::get_neighbors(const MatrixXd& X, int point_idx, double radius) const {
    std::vector<int> neighbors;
    for (int i = 0; i < X.rows(); ++i) {
        if (i != point_idx && (X.row(point_idx) - X.row(i)).norm() <= radius) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

void OPTICS::extract_clusters() {
    if (eps_ <= 0) return;
    
    int cluster_id = 0;
    for (int i = 0; i < reachability_.size(); ++i) {
        if (reachability_(i) <= eps_) {
            if (labels_(i) == -1) {
                labels_(i) = cluster_id++;
            }
        }
    }
}

Params OPTICS::get_params() const {
    return {
        {"min_samples", std::to_string(min_samples_)},
        {"max_eps", std::to_string(max_eps_)},
        {"metric", metric_},
        {"eps", std::to_string(eps_)}
    };
}

Estimator& OPTICS::set_params(const Params& params) {
    if (params.count("min_samples")) min_samples_ = std::stod(params.at("min_samples"));
    if (params.count("max_eps")) max_eps_ = std::stod(params.at("max_eps"));
    if (params.count("metric")) metric_ = params.at("metric");
    if (params.count("eps")) eps_ = std::stod(params.at("eps"));
    return *this;
}

// Birch implementation
Birch::Birch(int n_clusters, double threshold, int branching_factor)
    : n_clusters_(n_clusters), threshold_(threshold), branching_factor_(branching_factor), fitted_(false) {}

Estimator& Birch::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    int n_samples = X.rows();
    int n_features = X.cols();
    
    // Initialize root node
    root_ = std::make_unique<CFNode>(n_features);
    
    // Insert each point into CF-Tree
    for (int i = 0; i < n_samples; ++i) {
        insert_point(X.row(i), root_.get());
    }
    
    // Build subclusters from leaf nodes
    build_subclusters();
    
    // Assign labels based on closest subcluster
    labels_ = VectorXi(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < subcluster_centers_.rows(); ++j) {
            double dist = (X.row(i) - subcluster_centers_.row(j)).norm();
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels_(i) = best_cluster % n_clusters_;
    }
    
    fitted_ = true;
    return *this;
}

VectorXi Birch::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return labels_;
}

VectorXi Birch::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Birch must be fitted before predict");
    }
    
    int n_samples = X.rows();
    VectorXi predictions(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < subcluster_centers_.rows(); ++j) {
            double dist = (X.row(i) - subcluster_centers_.row(j)).norm();
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        predictions(i) = best_cluster % n_clusters_;
    }
    
    return predictions;
}

void Birch::insert_point(const VectorXd& point, CFNode* node) {
    if (node->is_leaf) {
        // Update CF for leaf node
        node->n_samples++;
        node->linear_sum += point;
        node->squared_sum += point.squaredNorm();
    } else {
        // Find closest child
        double min_dist = std::numeric_limits<double>::max();
        CFNode* best_child = nullptr;
        
        for (auto& child : node->children) {
            double dist = distance_to_centroid(point, *child);
            if (dist < min_dist) {
                min_dist = dist;
                best_child = child.get();
            }
        }
        
        if (best_child && min_dist < threshold_) {
            insert_point(point, best_child);
        } else if (node->children.size() < branching_factor_) {
            // Create new child
            auto new_child = std::make_unique<CFNode>(point.size());
            new_child->n_samples = 1;
            new_child->linear_sum = point;
            new_child->squared_sum = point.squaredNorm();
            node->children.push_back(std::move(new_child));
        }
    }
}

void Birch::build_subclusters() {
    std::vector<VectorXd> centroids;
    
    // Simple approach: collect leaf centroids
    std::function<void(CFNode*)> collect_leaves = [&](CFNode* node) {
        if (node->is_leaf && node->n_samples > 0) {
            VectorXd centroid = node->linear_sum / node->n_samples;
            centroids.push_back(centroid);
        } else {
            for (auto& child : node->children) {
                collect_leaves(child.get());
            }
        }
    };
    
    collect_leaves(root_.get());
    
    // Convert to matrix
    if (!centroids.empty()) {
        subcluster_centers_ = MatrixXd(centroids.size(), centroids[0].size());
        for (int i = 0; i < centroids.size(); ++i) {
            subcluster_centers_.row(i) = centroids[i];
        }
    } else {
        subcluster_centers_ = MatrixXd(0, 0);
    }
}

double Birch::distance_to_centroid(const VectorXd& point, const CFNode& node) const {
    if (node.n_samples == 0) return std::numeric_limits<double>::max();
    VectorXd centroid = node.linear_sum / node.n_samples;
    return (point - centroid).norm();
}

Params Birch::get_params() const {
    return {
        {"n_clusters", std::to_string(n_clusters_)},
        {"threshold", std::to_string(threshold_)},
        {"branching_factor", std::to_string(branching_factor_)}
    };
}

Estimator& Birch::set_params(const Params& params) {
    if (params.count("n_clusters")) n_clusters_ = std::stoi(params.at("n_clusters"));
    if (params.count("threshold")) threshold_ = std::stod(params.at("threshold"));
    if (params.count("branching_factor")) branching_factor_ = std::stoi(params.at("branching_factor"));
    return *this;
}

} // namespace cluster
} // namespace auroraml


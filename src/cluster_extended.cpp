#include "ingenuityml/cluster_extended.hpp"
#include "ingenuityml/base.hpp"
#include "ingenuityml/kmeans.hpp"
#include "ingenuityml/pca.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <limits>
#include <numeric>
#include <deque>
#include <unordered_map>

#include "ingenuityml/agglomerative.hpp"

namespace ingenuityml {
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

// BisectingKMeans implementation

static double cluster_sse(const MatrixXd& X, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    VectorXd centroid = VectorXd::Zero(X.cols());
    for (int idx : indices) {
        centroid += X.row(idx).transpose();
    }
    centroid /= static_cast<double>(indices.size());

    double sse = 0.0;
    for (int idx : indices) {
        VectorXd diff = X.row(idx).transpose() - centroid;
        sse += diff.squaredNorm();
    }
    return sse;
}

Estimator& BisectingKMeans::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    if (n_clusters_ <= 0 || n_clusters_ > X.rows()) {
        throw std::invalid_argument("n_clusters must be in (0, n_samples]");
    }

    std::vector<std::vector<int>> clusters;
    clusters.resize(1);
    clusters[0].resize(X.rows());
    std::iota(clusters[0].begin(), clusters[0].end(), 0);

    labels_ = VectorXi::Zero(X.rows());

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());

    while (static_cast<int>(clusters.size()) < n_clusters_) {
        int split_idx = -1;
        double best_sse = -1.0;
        for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
            if (clusters[i].size() < 2) {
                continue;
            }
            double sse = cluster_sse(X, clusters[i]);
            if (sse > best_sse) {
                best_sse = sse;
                split_idx = i;
            }
        }

        if (split_idx < 0) {
            break;
        }

        const auto& indices = clusters[split_idx];
        MatrixXd X_sub(indices.size(), X.cols());
        for (size_t i = 0; i < indices.size(); ++i) {
            X_sub.row(i) = X.row(indices[i]);
        }

        int split_seed = static_cast<int>(rng());
        KMeans kmeans(2, max_iter_, tol_, init_, split_seed);
        kmeans.fit(X_sub, VectorXd());
        VectorXi sub_labels = kmeans.predict_labels(X_sub);

        std::vector<int> left;
        std::vector<int> right;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (sub_labels(static_cast<int>(i)) == 0) {
                left.push_back(indices[i]);
            } else {
                right.push_back(indices[i]);
            }
        }

        if (left.empty() || right.empty()) {
            break;
        }

        clusters[split_idx] = left;
        clusters.push_back(right);
        int new_id = static_cast<int>(clusters.size()) - 1;

        for (int idx : left) {
            labels_(idx) = split_idx;
        }
        for (int idx : right) {
            labels_(idx) = new_id;
        }
    }

    cluster_centers_ = MatrixXd::Zero(clusters.size(), X.cols());
    for (int c = 0; c < static_cast<int>(clusters.size()); ++c) {
        if (clusters[c].empty()) continue;
        VectorXd centroid = VectorXd::Zero(X.cols());
        for (int idx : clusters[c]) {
            centroid += X.row(idx).transpose();
        }
        centroid /= static_cast<double>(clusters[c].size());
        cluster_centers_.row(c) = centroid.transpose();
    }

    fitted_ = true;
    return *this;
}

VectorXi BisectingKMeans::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return labels_;
}

VectorXi BisectingKMeans::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BisectingKMeans must be fitted before predict");
    }
    if (X.cols() != cluster_centers_.cols()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }

    VectorXi predictions = VectorXi::Zero(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double min_dist = (X.row(i) - cluster_centers_.row(0)).squaredNorm();
        int best_cluster = 0;
        for (int k = 1; k < cluster_centers_.rows(); ++k) {
            double dist = (X.row(i) - cluster_centers_.row(k)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        predictions(i) = best_cluster;
    }
    return predictions;
}

Params BisectingKMeans::get_params() const {
    Params params;
    params["n_clusters"] = std::to_string(n_clusters_);
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    params["init"] = init_;
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& BisectingKMeans::set_params(const Params& params) {
    n_clusters_ = utils::get_param_int(params, "n_clusters", n_clusters_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    init_ = utils::get_param_string(params, "init", init_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// AffinityPropagation implementation

Estimator& AffinityPropagation::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);

    int n_samples = X.rows();
    MatrixXd S = MatrixXd::Zero(n_samples, n_samples);
    for (int i = 0; i < n_samples; ++i) {
        for (int k = 0; k < n_samples; ++k) {
            double dist_sq = (X.row(i) - X.row(k)).squaredNorm();
            S(i, k) = -dist_sq;
        }
    }

    double preference = preference_;
    if (std::isnan(preference)) {
        std::vector<double> sims;
        sims.reserve(n_samples * n_samples);
        for (int i = 0; i < n_samples; ++i) {
            for (int k = 0; k < n_samples; ++k) {
                sims.push_back(S(i, k));
            }
        }
        size_t mid = sims.size() / 2;
        std::nth_element(sims.begin(), sims.begin() + mid, sims.end());
        preference = sims[mid];
    }

    for (int i = 0; i < n_samples; ++i) {
        S(i, i) = preference;
    }

    MatrixXd R = MatrixXd::Zero(n_samples, n_samples);
    MatrixXd A = MatrixXd::Zero(n_samples, n_samples);

    std::vector<int> last_exemplars;
    int stable_count = 0;

    for (int iter = 0; iter < max_iter_; ++iter) {
        // Update responsibilities
        MatrixXd AS = A + S;
        for (int i = 0; i < n_samples; ++i) {
            double max1 = -std::numeric_limits<double>::infinity();
            double max2 = -std::numeric_limits<double>::infinity();
            int idx1 = -1;
            for (int k = 0; k < n_samples; ++k) {
                double val = AS(i, k);
                if (val > max1) {
                    max2 = max1;
                    max1 = val;
                    idx1 = k;
                } else if (val > max2) {
                    max2 = val;
                }
            }
            for (int k = 0; k < n_samples; ++k) {
                double val = S(i, k) - ((k == idx1) ? max2 : max1);
                R(i, k) = damping_ * R(i, k) + (1.0 - damping_) * val;
            }
        }

        // Update availabilities
        for (int k = 0; k < n_samples; ++k) {
            double sum_pos = 0.0;
            for (int i = 0; i < n_samples; ++i) {
                if (i == k) continue;
                sum_pos += std::max(0.0, R(i, k));
            }
            for (int i = 0; i < n_samples; ++i) {
                double val;
                if (i == k) {
                    val = sum_pos;
                } else {
                    val = std::min(0.0, R(k, k) + sum_pos - std::max(0.0, R(i, k)));
                }
                A(i, k) = damping_ * A(i, k) + (1.0 - damping_) * val;
            }
        }

        // Check convergence
        std::vector<int> exemplars;
        for (int k = 0; k < n_samples; ++k) {
            if (A(k, k) + R(k, k) > 0.0) {
                exemplars.push_back(k);
            }
        }

        if (exemplars == last_exemplars && !exemplars.empty()) {
            stable_count++;
        } else {
            stable_count = 0;
            last_exemplars = exemplars;
        }

        if (stable_count >= convergence_iter_) {
            break;
        }
    }

    std::vector<int> exemplars = last_exemplars;
    if (exemplars.empty()) {
        int best_k = 0;
        double best_val = A(0, 0) + R(0, 0);
        for (int k = 1; k < n_samples; ++k) {
            double val = A(k, k) + R(k, k);
            if (val > best_val) {
                best_val = val;
                best_k = k;
            }
        }
        exemplars.push_back(best_k);
    }

    std::unordered_map<int, int> exemplar_to_label;
    for (int i = 0; i < static_cast<int>(exemplars.size()); ++i) {
        exemplar_to_label[exemplars[i]] = i;
    }

    labels_ = VectorXi::Zero(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        int best_exemplar = exemplars[0];
        double best_score = A(i, best_exemplar) + R(i, best_exemplar);
        for (int ex : exemplars) {
            double score = A(i, ex) + R(i, ex);
            if (score > best_score) {
                best_score = score;
                best_exemplar = ex;
            }
        }
        labels_(i) = exemplar_to_label[best_exemplar];
    }

    exemplar_indices_ = VectorXi::Zero(exemplars.size());
    cluster_centers_ = MatrixXd(exemplars.size(), X.cols());
    for (int i = 0; i < static_cast<int>(exemplars.size()); ++i) {
        exemplar_indices_(i) = exemplars[i];
        cluster_centers_.row(i) = X.row(exemplars[i]);
    }

    fitted_ = true;
    return *this;
}

VectorXi AffinityPropagation::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return labels_;
}

VectorXi AffinityPropagation::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("AffinityPropagation must be fitted before predict");
    }
    if (X.cols() != cluster_centers_.cols()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    if (cluster_centers_.rows() == 0) {
        throw std::runtime_error("AffinityPropagation has no cluster centers");
    }

    VectorXi predictions = VectorXi::Zero(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double min_dist = (X.row(i) - cluster_centers_.row(0)).squaredNorm();
        int best_cluster = 0;
        for (int k = 1; k < cluster_centers_.rows(); ++k) {
            double dist = (X.row(i) - cluster_centers_.row(k)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        predictions(i) = best_cluster;
    }
    return predictions;
}

Params AffinityPropagation::get_params() const {
    Params params;
    params["damping"] = std::to_string(damping_);
    params["max_iter"] = std::to_string(max_iter_);
    params["convergence_iter"] = std::to_string(convergence_iter_);
    if (std::isnan(preference_)) {
        params["preference"] = "nan";
    } else {
        params["preference"] = std::to_string(preference_);
    }
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& AffinityPropagation::set_params(const Params& params) {
    damping_ = utils::get_param_double(params, "damping", damping_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    convergence_iter_ = utils::get_param_int(params, "convergence_iter", convergence_iter_);
    if (params.count("preference")) {
        const std::string& val = params.at("preference");
        if (val == "nan" || val == "NaN") {
            preference_ = std::numeric_limits<double>::quiet_NaN();
        } else {
            preference_ = std::stod(val);
        }
    }
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// FeatureAgglomeration implementation

Estimator& FeatureAgglomeration::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    if (n_clusters_ <= 0 || n_clusters_ > X.cols()) {
        throw std::invalid_argument("n_clusters must be in (0, n_features]");
    }

    n_features_ = X.cols();
    MatrixXd X_t = X.transpose();
    AgglomerativeClustering agg(n_clusters_, linkage_, affinity_);
    agg.fit(X_t, VectorXd());
    feature_labels_ = agg.labels();

    fitted_ = true;
    return *this;
}

MatrixXd FeatureAgglomeration::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("FeatureAgglomeration must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd reduced = MatrixXd::Zero(X.rows(), n_clusters_);
    std::vector<int> counts(n_clusters_, 0);

    for (int j = 0; j < n_features_; ++j) {
        int cluster = feature_labels_(j);
        reduced.col(cluster) += X.col(j);
        counts[cluster]++;
    }

    for (int k = 0; k < n_clusters_; ++k) {
        if (counts[k] > 0) {
            reduced.col(k) /= static_cast<double>(counts[k]);
        }
    }

    return reduced;
}

MatrixXd FeatureAgglomeration::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("FeatureAgglomeration must be fitted before inverse_transform");
    }
    if (X.cols() != n_clusters_) {
        throw std::invalid_argument("X must have the same number of clusters as the model");
    }

    MatrixXd expanded = MatrixXd::Zero(X.rows(), n_features_);
    for (int j = 0; j < n_features_; ++j) {
        int cluster = feature_labels_(j);
        expanded.col(j) = X.col(cluster);
    }
    return expanded;
}

MatrixXd FeatureAgglomeration::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params FeatureAgglomeration::get_params() const {
    Params params;
    params["n_clusters"] = std::to_string(n_clusters_);
    params["linkage"] = linkage_;
    params["affinity"] = affinity_;
    return params;
}

Estimator& FeatureAgglomeration::set_params(const Params& params) {
    n_clusters_ = utils::get_param_int(params, "n_clusters", n_clusters_);
    linkage_ = utils::get_param_string(params, "linkage", linkage_);
    affinity_ = utils::get_param_string(params, "affinity", affinity_);
    return *this;
}

// SpectralBiclustering implementation

Estimator& SpectralBiclustering::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    if (n_clusters_ <= 0 || n_clusters_ > X.rows() || n_clusters_ > X.cols()) {
        throw std::invalid_argument("n_clusters must be in (0, min(n_samples, n_features)]");
    }

    decomposition::PCA pca_rows(n_clusters_);
    pca_rows.fit(X, VectorXd());
    MatrixXd row_embed = pca_rows.transform(X);

    KMeans row_kmeans(n_clusters_, 100, 1e-4, "k-means++", random_state_);
    row_kmeans.fit(row_embed, VectorXd());
    row_labels_ = row_kmeans.predict_labels(row_embed);

    MatrixXd X_t = X.transpose();
    decomposition::PCA pca_cols(n_clusters_);
    pca_cols.fit(X_t, VectorXd());
    MatrixXd col_embed = pca_cols.transform(X_t);

    KMeans col_kmeans(n_clusters_, 100, 1e-4, "k-means++", random_state_);
    col_kmeans.fit(col_embed, VectorXd());
    column_labels_ = col_kmeans.predict_labels(col_embed);

    fitted_ = true;
    return *this;
}

Params SpectralBiclustering::get_params() const {
    Params params;
    params["n_clusters"] = std::to_string(n_clusters_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& SpectralBiclustering::set_params(const Params& params) {
    n_clusters_ = utils::get_param_int(params, "n_clusters", n_clusters_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// SpectralCoclustering implementation

Estimator& SpectralCoclustering::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    if (n_clusters_ <= 0 || n_clusters_ > X.rows() || n_clusters_ > X.cols()) {
        throw std::invalid_argument("n_clusters must be in (0, min(n_samples, n_features)]");
    }

    decomposition::PCA pca_rows(n_clusters_);
    pca_rows.fit(X, VectorXd());
    MatrixXd row_embed = pca_rows.transform(X);

    KMeans row_kmeans(n_clusters_, 100, 1e-4, "k-means++", random_state_);
    row_kmeans.fit(row_embed, VectorXd());
    row_labels_ = row_kmeans.predict_labels(row_embed);

    MatrixXd X_t = X.transpose();
    decomposition::PCA pca_cols(n_clusters_);
    pca_cols.fit(X_t, VectorXd());
    MatrixXd col_embed = pca_cols.transform(X_t);

    KMeans col_kmeans(n_clusters_, 100, 1e-4, "k-means++", random_state_);
    col_kmeans.fit(col_embed, VectorXd());
    column_labels_ = col_kmeans.predict_labels(col_embed);

    fitted_ = true;
    return *this;
}

Params SpectralCoclustering::get_params() const {
    Params params;
    params["n_clusters"] = std::to_string(n_clusters_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& SpectralCoclustering::set_params(const Params& params) {
    n_clusters_ = utils::get_param_int(params, "n_clusters", n_clusters_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

} // namespace cluster
} // namespace ingenuityml

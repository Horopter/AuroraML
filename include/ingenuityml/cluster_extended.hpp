#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <random>
#include <limits>

namespace ingenuityml {
namespace cluster {

/**
 * SpectralClustering - Spectral clustering
 * 
 * Similar to scikit-learn's SpectralClustering, uses the spectrum of
 * the similarity matrix to perform dimensionality reduction before clustering.
 */
class SpectralClustering : public Estimator {
private:
    int n_clusters_;
    std::string affinity_;
    double gamma_;
    int n_neighbors_;
    int random_state_;
    bool fitted_;
    VectorXi labels_;

public:
    SpectralClustering(
        int n_clusters = 8,
        const std::string& affinity = "rbf",
        double gamma = 1.0,
        int n_neighbors = 10,
        int random_state = -1
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi fit_predict(const MatrixXd& X);
    VectorXi labels() const { return labels_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    MatrixXd build_affinity_matrix(const MatrixXd& X) const;
    void perform_clustering(const MatrixXd& X);
};

/**
 * MiniBatchKMeans - Mini-batch K-Means clustering
 * 
 * Similar to scikit-learn's MiniBatchKMeans, uses mini-batches to reduce
 * computation time while still achieving similar results.
 */
class MiniBatchKMeans : public Estimator {
private:
    int n_clusters_;
    int max_iter_;
    double tol_;
    int batch_size_;
    int random_state_;
    bool fitted_;
    MatrixXd cluster_centers_;
    VectorXi labels_;

public:
    MiniBatchKMeans(
        int n_clusters = 8,
        int max_iter = 100,
        double tol = 1e-4,
        int batch_size = 100,
        int random_state = -1
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi fit_predict(const MatrixXd& X);
    VectorXi predict(const MatrixXd& X) const;
    MatrixXd cluster_centers() const { return cluster_centers_; }
    VectorXi labels() const { return labels_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    void init_centroids(const MatrixXd& X, std::mt19937& rng);
    void update_centroids_minibatch(const MatrixXd& X_batch, const VectorXi& assignments, 
                                    std::vector<int>& counts, std::mt19937& rng);
};

/**
 * MeanShift - Mean shift clustering
 * 
 * Similar to scikit-learn's MeanShift, finds dense areas of data points
 * by shifting points towards the mode of the local density.
 */
class MeanShift : public Estimator {
private:
    double bandwidth_;
    std::vector<VectorXd> seeds_;
    bool bin_seeding_;
    int min_bin_freq_;
    bool cluster_all_;
    int max_iter_;
    bool fitted_;
    VectorXi labels_;
    MatrixXd cluster_centers_;

public:
    MeanShift(
        double bandwidth = -1.0,
        const std::vector<VectorXd>& seeds = std::vector<VectorXd>(),
        bool bin_seeding = false,
        int min_bin_freq = 1,
        bool cluster_all = true,
        int max_iter = 300
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi fit_predict(const MatrixXd& X);
    VectorXi predict(const MatrixXd& X) const;
    MatrixXd cluster_centers() const { return cluster_centers_; }
    VectorXi labels() const { return labels_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    double estimate_bandwidth(const MatrixXd& X) const;
    std::vector<VectorXd> mean_shift_single_seed(const MatrixXd& X, const VectorXd& seed) const;
    void remove_duplicate_centers();
};

/**
 * OPTICS - Ordering Points To Identify Clustering Structure
 * 
 * Similar to scikit-learn's OPTICS, extends DBSCAN to find clusters
 * of varying densities by ordering points and extracting cluster hierarchy.
 */
class OPTICS : public Estimator {
private:
    double min_samples_;
    double max_eps_;
    std::string metric_;
    double eps_;
    bool fitted_;
    VectorXi labels_;
    VectorXd reachability_;
    VectorXi ordering_;
    VectorXd core_distances_;

public:
    OPTICS(
        int min_samples = 5,
        double max_eps = std::numeric_limits<double>::infinity(),
        const std::string& metric = "euclidean",
        double eps = -1.0
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi fit_predict(const MatrixXd& X);
    VectorXi predict(const MatrixXd& X) const;
    VectorXi labels() const { return labels_; }
    VectorXd reachability() const { return reachability_; }
    VectorXi ordering() const { return ordering_; }
    VectorXd core_distances() const { return core_distances_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    double compute_core_distance(const MatrixXd& X, int point_idx, const std::vector<int>& neighbors) const;
    std::vector<int> get_neighbors(const MatrixXd& X, int point_idx, double radius) const;
    void extract_clusters();
};

/**
 * Birch - Balanced Iterative Reducing and Clustering using Hierarchies
 * 
 * Similar to scikit-learn's Birch, builds a CF-Tree (Clustering Feature Tree)
 * for efficient clustering of large datasets.
 */
class Birch : public Estimator {
private:
    int n_clusters_;
    double threshold_;
    int branching_factor_;
    bool fitted_;
    VectorXi labels_;
    MatrixXd subcluster_centers_;
    
    struct CFNode {
        int n_samples = 0;
        VectorXd linear_sum;
        double squared_sum = 0.0;
        std::vector<std::unique_ptr<CFNode>> children;
        bool is_leaf = true;
        
        CFNode() = default;
        CFNode(int n_features) : linear_sum(VectorXd::Zero(n_features)) {}
    };
    
    std::unique_ptr<CFNode> root_;

public:
    Birch(
        int n_clusters = 8,
        double threshold = 0.5,
        int branching_factor = 50
    );
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi fit_predict(const MatrixXd& X);
    VectorXi predict(const MatrixXd& X) const;
    VectorXi labels() const { return labels_; }
    MatrixXd subcluster_centers() const { return subcluster_centers_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    void insert_point(const VectorXd& point, CFNode* node);
    void build_subclusters();
    double distance_to_centroid(const VectorXd& point, const CFNode& node) const;
};

/**
 * BisectingKMeans - Hierarchical K-Means clustering
 *
 * Similar to scikit-learn's BisectingKMeans, recursively splits clusters
 * using k-means with k=2 until the desired number of clusters is reached.
 */
class BisectingKMeans : public Estimator {
private:
    int n_clusters_;
    int max_iter_;
    double tol_;
    std::string init_;
    int random_state_;
    bool fitted_ = false;
    MatrixXd cluster_centers_;
    VectorXi labels_;

public:
    BisectingKMeans(int n_clusters = 8, int max_iter = 300, double tol = 1e-4,
                    const std::string& init = "k-means++", int random_state = -1)
        : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol),
          init_(init), random_state_(random_state) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi fit_predict(const MatrixXd& X);
    VectorXi predict(const MatrixXd& X) const;
    MatrixXd cluster_centers() const { return cluster_centers_; }
    VectorXi labels() const { return labels_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

/**
 * AffinityPropagation - Affinity propagation clustering
 *
 * Similar to scikit-learn's AffinityPropagation, identifies exemplars via
 * message passing between points.
 */
class AffinityPropagation : public Estimator {
private:
    double damping_;
    int max_iter_;
    int convergence_iter_;
    double preference_;
    int random_state_;
    bool fitted_ = false;
    VectorXi labels_;
    MatrixXd cluster_centers_;
    VectorXi exemplar_indices_;

public:
    AffinityPropagation(double damping = 0.5, int max_iter = 200, int convergence_iter = 15,
                        double preference = std::numeric_limits<double>::quiet_NaN(),
                        int random_state = -1)
        : damping_(damping), max_iter_(max_iter), convergence_iter_(convergence_iter),
          preference_(preference), random_state_(random_state) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi fit_predict(const MatrixXd& X);
    VectorXi predict(const MatrixXd& X) const;
    VectorXi labels() const { return labels_; }
    MatrixXd cluster_centers() const { return cluster_centers_; }
    VectorXi exemplar_indices() const { return exemplar_indices_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

/**
 * FeatureAgglomeration - Agglomerative clustering on features
 *
 * Similar to scikit-learn's FeatureAgglomeration, clusters features and
 * reduces dimensionality by averaging within clusters.
 */
class FeatureAgglomeration : public Estimator, public Transformer {
private:
    int n_clusters_;
    std::string linkage_;
    std::string affinity_;
    bool fitted_ = false;
    int n_features_ = 0;
    VectorXi feature_labels_;

public:
    FeatureAgglomeration(int n_clusters = 2, const std::string& linkage = "single",
                         const std::string& affinity = "euclidean")
        : n_clusters_(n_clusters), linkage_(linkage), affinity_(affinity) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    VectorXi labels() const { return feature_labels_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

/**
 * SpectralBiclustering - Spectral biclustering
 *
 * Similar to scikit-learn's SpectralBiclustering, clusters rows and columns
 * of a data matrix simultaneously.
 */
class SpectralBiclustering : public Estimator {
private:
    int n_clusters_;
    int random_state_;
    bool fitted_ = false;
    VectorXi row_labels_;
    VectorXi column_labels_;

public:
    SpectralBiclustering(int n_clusters = 3, int random_state = -1)
        : n_clusters_(n_clusters), random_state_(random_state) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi row_labels() const { return row_labels_; }
    VectorXi column_labels() const { return column_labels_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

/**
 * SpectralCoclustering - Spectral co-clustering
 *
 * Similar to scikit-learn's SpectralCoclustering, finds row/column clusters
 * based on spectral embeddings.
 */
class SpectralCoclustering : public Estimator {
private:
    int n_clusters_;
    int random_state_;
    bool fitted_ = false;
    VectorXi row_labels_;
    VectorXi column_labels_;

public:
    SpectralCoclustering(int n_clusters = 3, int random_state = -1)
        : n_clusters_(n_clusters), random_state_(random_state) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi row_labels() const { return row_labels_; }
    VectorXi column_labels() const { return column_labels_; }
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
};

} // namespace cluster
} // namespace ingenuityml

#pragma once

#include "base.hpp"
#include <vector>
#include <string>
#include <random>

namespace auroraml {
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

} // namespace cluster
} // namespace auroraml


#ifndef CXML_KMEANS_HPP
#define CXML_KMEANS_HPP

#include "base.hpp"
#include <vector>
#include <string>
#include <limits>

namespace auroraml {
namespace cluster {

class KMeans : public Estimator, public Transformer {
private:
    int n_clusters_;
    int max_iter_;
    double tol_;
    std::string init_;
    int random_state_;
    bool fitted_ = false;

    MatrixXd centroids_;
    VectorXi labels_cache_;
    double inertia_cache_ = 0.0;

public:
    KMeans(int n_clusters = 8, int max_iter = 300, double tol = 1e-4, const std::string& init = "k-means++", int random_state = -1)
        : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol), init_(init), random_state_(random_state) {}

    // Estimator
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    // Transformer
    MatrixXd transform(const MatrixXd& X) const override; // distances to centroids
    MatrixXd inverse_transform(const MatrixXd& X) const override; // not meaningful, return X
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;

    // API
    VectorXi predict_labels(const MatrixXd& X) const;
    const MatrixXd& cluster_centers() const { 
        if (!fitted_) throw std::runtime_error("KMeans must be fitted before accessing cluster centers.");
        return centroids_; 
    }
    const VectorXi& labels() const { 
        if (!fitted_) throw std::runtime_error("KMeans must be fitted before accessing labels.");
        return labels_cache_; 
    }
    double inertia() const { 
        if (!fitted_) throw std::runtime_error("KMeans must be fitted before accessing inertia.");
        return inertia_cache_; 
    }
    bool is_fitted() const override { return fitted_; }

    // Params
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;

private:
    void init_centroids_pp(const MatrixXd& X);
    void init_centroids_random(const MatrixXd& X);
    double step_once(const MatrixXd& X, VectorXi& labels);
};

} // namespace cluster
} // namespace cxml

#endif // CXML_KMEANS_HPP



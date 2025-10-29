#include "auroraml/kmeans.hpp"
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace auroraml {
namespace cluster {

static double sqdist(const VectorXd& a, const VectorXd& b) {
    return (a - b).squaredNorm();
}

Estimator& KMeans::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    if (n_clusters_ <= 0 || n_clusters_ > X.rows()) throw std::invalid_argument("Invalid n_clusters");

    if (init_ == "k-means++") init_centroids_pp(X); else init_centroids_random(X);

    VectorXi labels = VectorXi::Zero(X.rows());
    double last_inertia = std::numeric_limits<double>::infinity();
    for (int it = 0; it < max_iter_; ++it) {
        double inertia = step_once(X, labels);
        if (std::abs(last_inertia - inertia) <= tol_ * (1.0 + last_inertia)) break;
        last_inertia = inertia;
    }
    // Ensure inertia is calculated even if no iterations were run
    if (last_inertia == std::numeric_limits<double>::infinity()) {
        last_inertia = step_once(X, labels);
    }
    labels_cache_ = labels;
    inertia_cache_ = last_inertia;
    fitted_ = true;
    return *this;
}

MatrixXd KMeans::transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("KMeans must be fitted before transform.");
    MatrixXd D(X.rows(), n_clusters_);
    
    #pragma omp parallel for if(X.rows() > 32)
    for (int i = 0; i < X.rows(); ++i) {
        for (int k = 0; k < n_clusters_; ++k) {
            D(i, k) = std::sqrt(sqdist(X.row(i), centroids_.row(k)));
        }
    }
    return D;
}

MatrixXd KMeans::inverse_transform(const MatrixXd& X) const {
    return X; // no-op
}

MatrixXd KMeans::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

VectorXi KMeans::predict_labels(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("KMeans must be fitted before predict_labels.");
    VectorXi labels(X.rows());
    
    #pragma omp parallel for if(X.rows() > 32)
    for (int i = 0; i < X.rows(); ++i) {
        int best_k = 0; double best_d = sqdist(X.row(i), centroids_.row(0));
        for (int k = 1; k < n_clusters_; ++k) {
            double d = sqdist(X.row(i), centroids_.row(k));
            if (d < best_d) { best_d = d; best_k = k; }
        }
        labels(i) = best_k;
    }
    return labels;
}

Params KMeans::get_params() const {
    Params p;
    p["n_clusters"] = std::to_string(n_clusters_);
    p["max_iter"] = std::to_string(max_iter_);
    p["tol"] = std::to_string(tol_);
    p["init"] = init_;
    p["random_state"] = std::to_string(random_state_);
    return p;
}

Estimator& KMeans::set_params(const Params& params) {
    n_clusters_ = utils::get_param_int(params, "n_clusters", n_clusters_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    init_ = utils::get_param_string(params, "init", init_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

void KMeans::init_centroids_random(const MatrixXd& X) {
    std::random_device rd;
    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? rd() : random_state_));
    std::uniform_int_distribution<int> dist(0, X.rows() - 1);
    centroids_.resize(n_clusters_, X.cols());
    for (int k = 0; k < n_clusters_; ++k) centroids_.row(k) = X.row(dist(rng));
}

void KMeans::init_centroids_pp(const MatrixXd& X) {
    std::random_device rd;
    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? rd() : random_state_));
    std::uniform_int_distribution<int> uni(0, X.rows() - 1);
    centroids_.resize(n_clusters_, X.cols());
    centroids_.row(0) = X.row(uni(rng));
    std::vector<double> dist2(X.rows(), std::numeric_limits<double>::infinity());
    for (int k = 1; k < n_clusters_; ++k) {
        for (int i = 0; i < X.rows(); ++i) {
            double d = sqdist(X.row(i), centroids_.row(k-1));
            if (d < dist2[i]) dist2[i] = d;
        }
        double sum = 0.0; for (double v : dist2) sum += v;
        std::uniform_real_distribution<double> ru(0.0, sum);
        double r = ru(rng);
        double acc = 0.0;
        int chosen = 0;
        for (int i = 0; i < X.rows(); ++i) { acc += dist2[i]; if (acc >= r) { chosen = i; break; } }
        centroids_.row(k) = X.row(chosen);
    }
}

double KMeans::step_once(const MatrixXd& X, VectorXi& labels) {
    // assign
    for (int i = 0; i < X.rows(); ++i) {
        int best_k = 0; double best_d = sqdist(X.row(i), centroids_.row(0));
        for (int k = 1; k < n_clusters_; ++k) {
            double d = sqdist(X.row(i), centroids_.row(k));
            if (d < best_d) { best_d = d; best_k = k; }
        }
        labels(i) = best_k;
    }
    // update
    MatrixXd new_centroids = MatrixXd::Zero(n_clusters_, X.cols());
    VectorXi counts = VectorXi::Zero(n_clusters_);
    double inertia = 0.0;
    for (int i = 0; i < X.rows(); ++i) {
        new_centroids.row(labels(i)) += X.row(i);
        counts(labels(i)) += 1;
        inertia += sqdist(X.row(i), centroids_.row(labels(i)));
    }
    for (int k = 0; k < n_clusters_; ++k) {
        if (counts(k) > 0) new_centroids.row(k) /= static_cast<double>(counts(k));
        else new_centroids.row(k) = centroids_.row(k); // keep old if empty
    }
    double shift = (centroids_ - new_centroids).rowwise().norm().sum();
    centroids_ = new_centroids;
    return inertia + shift;
}

} // namespace cluster
} // namespace cxml



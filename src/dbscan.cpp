#include "ingenuityml/dbscan.hpp"
#include "ingenuityml/base.hpp"
#include <cmath>
#include <queue>

namespace ingenuityml {
namespace cluster {

Estimator& DBSCAN::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    
    // Validate parameters
    if (eps_ <= 0.0) {
        throw std::invalid_argument("eps must be positive");
    }
    if (min_samples_ <= 0) {
        throw std::invalid_argument("min_samples must be positive");
    }
    
    const int n = X.rows();
    labels_ = VectorXi::Constant(n, -1); // -1 for noise
    std::vector<bool> visited(n, false);
    int cluster_id = 0;

    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;
        visited[i] = true;
        auto neighbors = region_query(X, i);
        if (static_cast<int>(neighbors.size()) < min_samples_) {
            labels_(i) = -1; // noise
        } else {
            labels_(i) = cluster_id;
            expand_cluster(X, i, cluster_id, neighbors, visited);
            cluster_id++;
        }
    }

    fitted_ = true;
    return *this;
}

VectorXi DBSCAN::fit_predict(const MatrixXd& X) {
    fit(X, VectorXd());
    return labels_;
}

Estimator& DBSCAN::set_params(const Params& params) {
    eps_ = utils::get_param_double(params, "eps", eps_);
    min_samples_ = utils::get_param_int(params, "min_samples", min_samples_);
    return *this;
}

std::vector<int> DBSCAN::region_query(const MatrixXd& X, int idx) const {
    std::vector<int> neighbors;
    const int n = X.rows();
    for (int j = 0; j < n; ++j) {
        double dist = (X.row(idx) - X.row(j)).norm();
        if (dist <= eps_) neighbors.push_back(j);
    }
    return neighbors;
}

void DBSCAN::expand_cluster(const MatrixXd& X, int idx, int cluster_id, std::vector<int>& neighbors, std::vector<bool>& visited) {
    std::queue<int> q;
    for (int nb : neighbors) q.push(nb);
    while (!q.empty()) {
        int current = q.front(); q.pop();
        if (!visited[current]) {
            visited[current] = true;
            auto current_neighbors = region_query(X, current);
            if (static_cast<int>(current_neighbors.size()) >= min_samples_) {
                for (int nb : current_neighbors) q.push(nb);
            }
        }
        if (labels_(current) == -1) {
            labels_(current) = cluster_id;
        }
    }
}

} // namespace cluster
} // namespace ingenuityml


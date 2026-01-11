#include "ingenuityml/agglomerative.hpp"
#include "ingenuityml/base.hpp"
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace ingenuityml {
namespace cluster {

Estimator& AgglomerativeClustering::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    if (n_clusters_ <= 0 || n_clusters_ > X.rows()) {
        throw std::invalid_argument("n_clusters must be in (0, n_samples]");
    }
    if (affinity_ != "euclidean") {
        throw std::invalid_argument("Only euclidean affinity is supported");
    }
    if (linkage_ != "single") {
        throw std::invalid_argument("Only single linkage is supported");
    }

    const int n = X.rows();
    labels_ = VectorXi::LinSpaced(n, 0, n - 1); // each point its own cluster

    auto dist = [&](int i, int j) {
        return (X.row(i) - X.row(j)).norm();
    };

    int num_clusters = n;
    // Simple O(n^3) agglomeration for clarity
    while (num_clusters > n_clusters_) {
        double best = std::numeric_limits<double>::infinity();
        int ca = -1, cb = -1;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (labels_(i) == labels_(j)) continue;
                double dmin = std::numeric_limits<double>::infinity();
                // single-linkage: min pairwise distance between clusters
                for (int a = 0; a < n; ++a) if (labels_(a) == labels_(i))
                    for (int b = 0; b < n; ++b) if (labels_(b) == labels_(j))
                        dmin = std::min(dmin, dist(a, b));
                if (dmin < best) { best = dmin; ca = labels_(i); cb = labels_(j); }
            }
        }
        if (ca == -1 || cb == -1) break;
        // merge cluster cb into ca
        for (int k = 0; k < n; ++k) if (labels_(k) == cb) labels_(k) = ca;
        num_clusters--;
    }
    // relabel to 0..C-1 compactly
    std::vector<int> uniq;
    uniq.reserve(n);
    for (int i = 0; i < n; ++i) uniq.push_back(labels_(i));
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    for (int i = 0; i < n; ++i) labels_(i) = static_cast<int>(std::lower_bound(uniq.begin(), uniq.end(), labels_(i)) - uniq.begin());

    fitted_ = true;
    return *this;
}

Estimator& AgglomerativeClustering::set_params(const Params& params) {
    n_clusters_ = utils::get_param_int(params, "n_clusters", n_clusters_);
    linkage_ = utils::get_param_string(params, "linkage", linkage_);
    affinity_ = utils::get_param_string(params, "affinity", affinity_);
    return *this;
}

} // namespace cluster
} // namespace ingenuityml



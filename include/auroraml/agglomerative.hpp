#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace auroraml {
namespace cluster {

// Agglomerative Clustering (single-linkage, euclidean)
class AgglomerativeClustering : public Estimator {
private:
    int n_clusters_;
    std::string linkage_;
    std::string affinity_;
    bool fitted_ = false;
    VectorXi labels_;

public:
    AgglomerativeClustering(int n_clusters = 2, const std::string& linkage = "single", const std::string& affinity = "euclidean")
        : n_clusters_(n_clusters), linkage_(linkage), affinity_(affinity) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    Params get_params() const override {
        return {{"n_clusters", std::to_string(n_clusters_)}, {"linkage", linkage_}, {"affinity", affinity_}};
    }
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const VectorXi& labels() const { 
        if (!fitted_) throw std::runtime_error("AgglomerativeClustering must be fitted before accessing labels.");
        return labels_; 
    }
};

} // namespace cluster
} // namespace cxml



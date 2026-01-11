#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>

namespace ingenuityml {
namespace cluster {

// DBSCAN clustering (basic implementation with euclidean distance)
class DBSCAN : public Estimator {
private:
    double eps_;
    int min_samples_;
    bool fitted_ = false;
    VectorXi labels_;

public:
    DBSCAN(double eps = 0.5, int min_samples = 5)
        : eps_(eps), min_samples_(min_samples) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    VectorXi fit_predict(const MatrixXd& X);
    Params get_params() const override {
        return {{"eps", std::to_string(eps_)}, {"min_samples", std::to_string(min_samples_)}};
    }
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const VectorXi& labels() const { 
        if (!fitted_) throw std::runtime_error("DBSCAN must be fitted before accessing labels.");
        return labels_; 
    }

private:
    std::vector<int> region_query(const MatrixXd& X, int idx) const;
    void expand_cluster(const MatrixXd& X, int idx, int cluster_id, std::vector<int>& neighbors, std::vector<bool>& visited);
};

} // namespace cluster
} // namespace ingenuityml


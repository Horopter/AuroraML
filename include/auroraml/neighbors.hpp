#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <set>

namespace auroraml {
namespace neighbors {

/**
 * K-Nearest Neighbors Classifier
 */
class KNeighborsClassifier : public Estimator, public Classifier {
private:
    MatrixXd X_train_;
    VectorXd y_train_;
    bool fitted_;
    int n_neighbors_;
    std::string weights_;
    std::string algorithm_;
    std::string metric_;
    double p_;
    int n_jobs_;

public:
    KNeighborsClassifier(int n_neighbors = 5, const std::string& weights = "uniform",
                        const std::string& algorithm = "auto", const std::string& metric = "euclidean",
                        double p = 2, int n_jobs = 1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    MatrixXd compute_distances(const MatrixXd& X) const;
    std::vector<std::vector<int>> find_k_neighbors(const MatrixXd& distances) const;
    VectorXi predict_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const;
    MatrixXd predict_proba_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const;
};

/**
 * K-Nearest Neighbors Regressor
 */
class KNeighborsRegressor : public Estimator, public Regressor {
private:
    MatrixXd X_train_;
    VectorXd y_train_;
    bool fitted_;
    int n_neighbors_;
    std::string weights_;
    std::string algorithm_;
    std::string metric_;
    double p_;
    int n_jobs_;

public:
    KNeighborsRegressor(int n_neighbors = 5, const std::string& weights = "uniform",
                       const std::string& algorithm = "auto", const std::string& metric = "euclidean",
                       double p = 2, int n_jobs = 1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    MatrixXd compute_distances(const MatrixXd& X) const;
    std::vector<std::vector<int>> find_k_neighbors(const MatrixXd& distances) const;
    VectorXd predict_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const;
};

} // namespace neighbors
} // namespace cxml

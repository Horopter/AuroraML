#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include <utility>

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

/**
 * Radius Neighbors Classifier
 */
class RadiusNeighborsClassifier : public Estimator, public Classifier {
private:
    MatrixXd X_train_;
    VectorXd y_train_;
    bool fitted_;
    double radius_;
    std::string weights_;
    std::string algorithm_;
    std::string metric_;
    double p_;
    int n_jobs_;

public:
    RadiusNeighborsClassifier(double radius = 1.0, const std::string& weights = "uniform",
                             const std::string& algorithm = "auto", const std::string& metric = "euclidean",
                             double p = 2, int n_jobs = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

private:
    MatrixXd compute_distances(const MatrixXd& X) const;
    std::vector<std::vector<int>> find_radius_neighbors(const MatrixXd& distances) const;
    VectorXi predict_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const;
    MatrixXd predict_proba_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const;
};

/**
 * Radius Neighbors Regressor
 */
class RadiusNeighborsRegressor : public Estimator, public Regressor {
private:
    MatrixXd X_train_;
    VectorXd y_train_;
    bool fitted_;
    double radius_;
    std::string weights_;
    std::string algorithm_;
    std::string metric_;
    double p_;
    int n_jobs_;

public:
    RadiusNeighborsRegressor(double radius = 1.0, const std::string& weights = "uniform",
                            const std::string& algorithm = "auto", const std::string& metric = "euclidean",
                            double p = 2, int n_jobs = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

private:
    MatrixXd compute_distances(const MatrixXd& X) const;
    std::vector<std::vector<int>> find_radius_neighbors(const MatrixXd& distances) const;
    VectorXd predict_from_neighbors(const std::vector<std::vector<int>>& neighbor_indices) const;
};

/**
 * Nearest Centroid Classifier
 */
class NearestCentroid : public Estimator, public Classifier {
private:
    MatrixXd centroids_;
    std::vector<int> classes_;
    bool fitted_;
    std::string metric_;
    double p_;

public:
    NearestCentroid(const std::string& metric = "euclidean", double p = 2.0);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

private:
    MatrixXd compute_distances(const MatrixXd& X) const;
};

/**
 * Nearest Neighbors (unsupervised)
 */
class NearestNeighbors : public Estimator {
private:
    MatrixXd X_train_;
    bool fitted_;
    int n_neighbors_;
    double radius_;
    std::string algorithm_;
    std::string metric_;
    double p_;
    int n_jobs_;

public:
    NearestNeighbors(int n_neighbors = 5, double radius = 1.0,
                     const std::string& algorithm = "auto", const std::string& metric = "euclidean",
                     double p = 2.0, int n_jobs = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    std::pair<MatrixXd, MatrixXi> kneighbors(const MatrixXd& X, int n_neighbors = -1) const;
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> radius_neighbors(
        const MatrixXd& X, double radius = -1.0) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

private:
    MatrixXd compute_distances(const MatrixXd& X) const;
};

/**
 * KNeighborsTransformer - Transform data into a k-nearest neighbors graph
 */
class KNeighborsTransformer : public Estimator, public Transformer {
private:
    MatrixXd X_train_;
    bool fitted_;
    int n_neighbors_;
    std::string mode_;
    std::string metric_;
    double p_;
    int n_jobs_;

public:
    KNeighborsTransformer(int n_neighbors = 5, const std::string& mode = "distance",
                          const std::string& metric = "euclidean", double p = 2.0, int n_jobs = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

private:
    MatrixXd compute_distances(const MatrixXd& X) const;
};

/**
 * RadiusNeighborsTransformer - Transform data into a radius-neighbors graph
 */
class RadiusNeighborsTransformer : public Estimator, public Transformer {
private:
    MatrixXd X_train_;
    bool fitted_;
    double radius_;
    std::string mode_;
    std::string metric_;
    double p_;
    int n_jobs_;

public:
    RadiusNeighborsTransformer(double radius = 1.0, const std::string& mode = "distance",
                               const std::string& metric = "euclidean", double p = 2.0, int n_jobs = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override;

private:
    MatrixXd compute_distances(const MatrixXd& X) const;
};

} // namespace neighbors
} // namespace cxml

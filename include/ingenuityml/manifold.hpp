#pragma once

#include "base.hpp"
#include <string>

namespace ingenuityml {
namespace manifold {

/**
 * MDS - Multidimensional Scaling (classical)
 */
class MDS : public Estimator, public Transformer {
private:
    int n_components_;
    bool fitted_ = false;
    MatrixXd embedding_;

public:
    MDS(int n_components = 2) : n_components_(n_components) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    MatrixXd embedding() const { return embedding_; }
};

/**
 * Isomap - Manifold learning via geodesic distances
 */
class Isomap : public Estimator, public Transformer {
private:
    int n_components_;
    int n_neighbors_;
    bool fitted_ = false;
    MatrixXd embedding_;

public:
    Isomap(int n_components = 2, int n_neighbors = 5)
        : n_components_(n_components), n_neighbors_(n_neighbors) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    MatrixXd embedding() const { return embedding_; }
};

/**
 * LocallyLinearEmbedding - LLE manifold learning
 */
class LocallyLinearEmbedding : public Estimator, public Transformer {
private:
    int n_components_;
    int n_neighbors_;
    bool fitted_ = false;
    MatrixXd embedding_;

public:
    LocallyLinearEmbedding(int n_components = 2, int n_neighbors = 5)
        : n_components_(n_components), n_neighbors_(n_neighbors) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    MatrixXd embedding() const { return embedding_; }
};

/**
 * SpectralEmbedding - Laplacian eigenmaps
 */
class SpectralEmbedding : public Estimator, public Transformer {
private:
    int n_components_;
    int n_neighbors_;
    bool fitted_ = false;
    MatrixXd embedding_;

public:
    SpectralEmbedding(int n_components = 2, int n_neighbors = 5)
        : n_components_(n_components), n_neighbors_(n_neighbors) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    MatrixXd embedding() const { return embedding_; }
};

} // namespace manifold
} // namespace ingenuityml

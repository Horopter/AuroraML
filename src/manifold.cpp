#include "ingenuityml/manifold.hpp"
#include "ingenuityml/base.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace ingenuityml {
namespace manifold {

static MatrixXd pairwise_distances(const MatrixXd& X) {
    int n = X.rows();
    MatrixXd D = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dist = (X.row(i) - X.row(j)).norm();
            D(i, j) = D(j, i) = dist;
        }
    }
    return D;
}

static MatrixXd classical_mds_from_distances(const MatrixXd& distances, int n_components) {
    int n = distances.rows();
    if (n_components <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }
    int comps = std::min(n_components, n);

    MatrixXd D2 = distances.array().square().matrix();
    MatrixXd J = MatrixXd::Identity(n, n) - MatrixXd::Constant(n, n, 1.0 / n);
    MatrixXd B = -0.5 * J * D2 * J;

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(B);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed in MDS");
    }

    VectorXd evals = solver.eigenvalues();
    MatrixXd evecs = solver.eigenvectors();

    MatrixXd embedding = MatrixXd::Zero(n, comps);
    for (int i = 0; i < comps; ++i) {
        int idx = n - 1 - i;
        double val = evals(idx);
        double scale = val > 0.0 ? std::sqrt(val) : 0.0;
        embedding.col(comps - 1 - i) = evecs.col(idx) * scale;
    }
    return embedding;
}

static std::vector<int> k_nearest_indices(const MatrixXd& X, int idx, int k) {
    int n = X.rows();
    std::vector<std::pair<double, int>> distances;
    distances.reserve(n - 1);
    for (int j = 0; j < n; ++j) {
        if (j == idx) continue;
        double dist = (X.row(idx) - X.row(j)).norm();
        distances.push_back({dist, j});
    }
    std::nth_element(distances.begin(), distances.begin() + std::min(k, static_cast<int>(distances.size())) - 1, distances.end());
    std::sort(distances.begin(), distances.begin() + std::min(k, static_cast<int>(distances.size())));

    std::vector<int> neighbors;
    int limit = std::min(k, static_cast<int>(distances.size()));
    neighbors.reserve(limit);
    for (int i = 0; i < limit; ++i) {
        neighbors.push_back(distances[i].second);
    }
    return neighbors;
}

// MDS implementation

Estimator& MDS::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);

    MatrixXd distances = pairwise_distances(X);
    embedding_ = classical_mds_from_distances(distances, n_components_);
    fitted_ = true;
    return *this;
}

MatrixXd MDS::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MDS must be fitted before transform");
    }
    (void)X;
    return embedding_;
}

MatrixXd MDS::inverse_transform(const MatrixXd& X) const {
    (void)X;
    throw std::runtime_error("MDS does not support inverse_transform");
}

MatrixXd MDS::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return embedding_;
}

Params MDS::get_params() const {
    return {{"n_components", std::to_string(n_components_)}};
}

Estimator& MDS::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    return *this;
}

// Isomap implementation

Estimator& Isomap::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    int n = X.rows();
    if (n_neighbors_ <= 0 || n_neighbors_ >= n) {
        throw std::invalid_argument("n_neighbors must be in (0, n_samples)");
    }

    MatrixXd distances = MatrixXd::Constant(n, n, std::numeric_limits<double>::infinity());
    for (int i = 0; i < n; ++i) {
        distances(i, i) = 0.0;
        auto neighbors = k_nearest_indices(X, i, n_neighbors_);
        for (int j : neighbors) {
            double dist = (X.row(i) - X.row(j)).norm();
            distances(i, j) = dist;
            distances(j, i) = dist;
        }
    }

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            if (distances(i, k) == std::numeric_limits<double>::infinity()) continue;
            for (int j = 0; j < n; ++j) {
                double alt = distances(i, k) + distances(k, j);
                if (alt < distances(i, j)) {
                    distances(i, j) = alt;
                }
            }
        }
    }

    double max_finite = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::isfinite(distances(i, j)) && distances(i, j) > max_finite) {
                max_finite = distances(i, j);
            }
        }
    }
    if (max_finite == 0.0) {
        max_finite = 1.0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!std::isfinite(distances(i, j))) {
                distances(i, j) = max_finite * 10.0;
            }
        }
    }

    embedding_ = classical_mds_from_distances(distances, n_components_);
    fitted_ = true;
    return *this;
}

MatrixXd Isomap::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Isomap must be fitted before transform");
    }
    (void)X;
    return embedding_;
}

MatrixXd Isomap::inverse_transform(const MatrixXd& X) const {
    (void)X;
    throw std::runtime_error("Isomap does not support inverse_transform");
}

MatrixXd Isomap::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return embedding_;
}

Params Isomap::get_params() const {
    return {{"n_components", std::to_string(n_components_)},
            {"n_neighbors", std::to_string(n_neighbors_)}};
}

Estimator& Isomap::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", n_neighbors_);
    return *this;
}

// LocallyLinearEmbedding implementation

Estimator& LocallyLinearEmbedding::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    int n = X.rows();
    int d = X.cols();
    if (n_neighbors_ <= 0 || n_neighbors_ >= n) {
        throw std::invalid_argument("n_neighbors must be in (0, n_samples)");
    }
    if (n_components_ <= 0 || n_components_ >= n) {
        throw std::invalid_argument("n_components must be in (0, n_samples)");
    }

    MatrixXd W = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        auto neighbors = k_nearest_indices(X, i, n_neighbors_);
        MatrixXd Z(neighbors.size(), d);
        for (size_t j = 0; j < neighbors.size(); ++j) {
            Z.row(j) = X.row(neighbors[j]) - X.row(i);
        }

        MatrixXd C = Z * Z.transpose();
        double trace = C.trace();
        if (trace <= 0.0) trace = 1.0;
        C += MatrixXd::Identity(C.rows(), C.cols()) * (1e-3 * trace);

        VectorXd ones = VectorXd::Ones(C.rows());
        VectorXd w = C.ldlt().solve(ones);
        double w_sum = w.sum();
        if (w_sum != 0.0) {
            w /= w_sum;
        }
        for (size_t j = 0; j < neighbors.size(); ++j) {
            W(i, neighbors[j]) = w(j);
        }
    }

    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd M = (I - W).transpose() * (I - W);

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(M);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed in LLE");
    }

    MatrixXd evecs = solver.eigenvectors();
    int comps = std::min(n_components_, n - 1);
    embedding_ = evecs.block(0, 1, n, comps);

    fitted_ = true;
    return *this;
}

MatrixXd LocallyLinearEmbedding::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LocallyLinearEmbedding must be fitted before transform");
    }
    (void)X;
    return embedding_;
}

MatrixXd LocallyLinearEmbedding::inverse_transform(const MatrixXd& X) const {
    (void)X;
    throw std::runtime_error("LocallyLinearEmbedding does not support inverse_transform");
}

MatrixXd LocallyLinearEmbedding::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return embedding_;
}

Params LocallyLinearEmbedding::get_params() const {
    return {{"n_components", std::to_string(n_components_)},
            {"n_neighbors", std::to_string(n_neighbors_)}};
}

Estimator& LocallyLinearEmbedding::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", n_neighbors_);
    return *this;
}

// SpectralEmbedding implementation

Estimator& SpectralEmbedding::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    int n = X.rows();
    if (n_neighbors_ <= 0 || n_neighbors_ >= n) {
        throw std::invalid_argument("n_neighbors must be in (0, n_samples)");
    }
    if (n_components_ <= 0 || n_components_ >= n) {
        throw std::invalid_argument("n_components must be in (0, n_samples)");
    }

    MatrixXd A = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        auto neighbors = k_nearest_indices(X, i, n_neighbors_);
        for (int j : neighbors) {
            double dist = (X.row(i) - X.row(j)).norm();
            double weight = std::exp(-dist * dist);
            A(i, j) = weight;
            A(j, i) = weight;
        }
    }

    VectorXd degree = A.rowwise().sum();
    VectorXd inv_sqrt_degree = VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) {
        if (degree(i) > 0.0) {
            inv_sqrt_degree(i) = 1.0 / std::sqrt(degree(i));
        }
    }

    MatrixXd D_inv_sqrt = inv_sqrt_degree.asDiagonal();
    MatrixXd L = MatrixXd::Identity(n, n) - D_inv_sqrt * A * D_inv_sqrt;

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(L);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed in SpectralEmbedding");
    }

    MatrixXd evecs = solver.eigenvectors();
    int comps = std::min(n_components_, n - 1);
    embedding_ = evecs.block(0, 1, n, comps);

    fitted_ = true;
    return *this;
}

MatrixXd SpectralEmbedding::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SpectralEmbedding must be fitted before transform");
    }
    (void)X;
    return embedding_;
}

MatrixXd SpectralEmbedding::inverse_transform(const MatrixXd& X) const {
    (void)X;
    throw std::runtime_error("SpectralEmbedding does not support inverse_transform");
}

MatrixXd SpectralEmbedding::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return embedding_;
}

Params SpectralEmbedding::get_params() const {
    return {{"n_components", std::to_string(n_components_)},
            {"n_neighbors", std::to_string(n_neighbors_)}};
}

Estimator& SpectralEmbedding::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    n_neighbors_ = utils::get_param_int(params, "n_neighbors", n_neighbors_);
    return *this;
}

} // namespace manifold
} // namespace ingenuityml

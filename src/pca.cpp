#include "ingenuityml/pca.hpp"
#include <numeric>
#include <functional>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>

namespace ingenuityml {
namespace decomposition {

Estimator& PCA::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    // Validate n_components parameter
    int n_samples = X.rows();
    int n_features = X.cols();
    if (n_components_ <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }
    if (n_components_ > n_features) {
        throw std::invalid_argument("n_components cannot be greater than n_features");
    }
    if (n_components_ > n_samples) {
        throw std::invalid_argument("n_components cannot be greater than n_samples");
    }
    
    mean_ = X.colwise().mean();
    MatrixXd Xc = X.rowwise() - mean_.transpose();
    // covariance matrix (biased, divide by n_samples)
    MatrixXd C = (Xc.transpose() * Xc) / static_cast<double>(n_samples);
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(C);
    VectorXd evals = es.eigenvalues();
    MatrixXd evecs = es.eigenvectors();
    // sort descending
    std::vector<int> idx(n_features);
    for (int i = 0; i < n_features; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return evals(a) > evals(b); });

    int k = n_components_;
    explained_variance_.resize(k);
    components_.resize(k, n_features);
    for (int i = 0; i < k; ++i) {
        explained_variance_(i) = evals(idx[i]);
        components_.row(i) = evecs.col(idx[i]).transpose();
    }
    double total = evals.sum();
    explained_variance_ratio_sum_ = (k > 0 && total > 0) ? explained_variance_.sum() / total : 0.0;
    fitted_ = true;
    return *this;
}

MatrixXd PCA::transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("PCA must be fitted before transform.");
    validation::check_X(X);
    if (X.cols() != mean_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd Xc = X.rowwise() - mean_.transpose();
    MatrixXd T = Xc * components_.transpose();
    if (whiten_) {
        for (int i = 0; i < explained_variance_.size(); ++i) {
            double s = std::sqrt(std::max(explained_variance_(i), 1e-12));
            T.col(i) /= s;
        }
    }
    return T;
}

MatrixXd PCA::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("PCA must be fitted before inverse_transform.");
    MatrixXd R = X * components_;
    if (whiten_) {
        for (int i = 0; i < explained_variance_.size(); ++i) {
            double s = std::sqrt(std::max(explained_variance_(i), 1e-12));
            R.col(i) *= s;
        }
    }
    R = R.rowwise() + mean_.transpose();
    return R;
}

MatrixXd PCA::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params PCA::get_params() const {
    Params p; p["n_components"] = std::to_string(n_components_); p["whiten"] = whiten_ ? "true" : "false"; return p;
}

Estimator& PCA::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    whiten_ = utils::get_param_bool(params, "whiten", whiten_);
    return *this;
}

// KernelPCA implementation
KernelPCA::KernelPCA(int n_components, const std::string& kernel, double gamma, double degree, double coef0)
    : n_components_(n_components), kernel_(kernel), gamma_(gamma), degree_(degree), coef0_(coef0), fitted_(false) {}

Estimator& KernelPCA::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    int n_samples = X.rows();
    int n_features = X.cols();
    
    if (n_components_ <= 0) {
        n_components_ = std::min(n_samples, n_features);
    }
    
    X_fit_ = X;
    
    // Compute kernel matrix
    MatrixXd K = compute_kernel_matrix(X, X);
    
    // Center kernel matrix
    VectorXd ones = VectorXd::Ones(n_samples);
    MatrixXd ones_mat = ones * ones.transpose() / n_samples;
    K = K - ones_mat * K - K * ones_mat + ones_mat * K * ones_mat;
    
    // Eigendecomposition
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(K);
    VectorXd eigenvalues = solver.eigenvalues().reverse();
    MatrixXd eigenvectors = solver.eigenvectors().rowwise().reverse();
    
    // Keep only positive eigenvalues and corresponding eigenvectors
    int n_components = std::min(n_components_, static_cast<int>((eigenvalues.array() > 1e-8).count()));
    
    lambdas_ = eigenvalues.head(n_components);
    alphas_ = eigenvectors.leftCols(n_components);
    
    // Normalize eigenvectors
    for (int i = 0; i < n_components; ++i) {
        if (lambdas_(i) > 1e-8) {
            alphas_.col(i) /= std::sqrt(lambdas_(i));
        }
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd KernelPCA::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KernelPCA must be fitted before transform");
    }
    
    MatrixXd K = compute_kernel_matrix(X, X_fit_);
    
    // Center with respect to training data
    int n_train = X_fit_.rows();
    VectorXd ones_train = VectorXd::Ones(n_train);
    VectorXd k_mean = K * ones_train / n_train;
    VectorXd k_train_mean = VectorXd::Ones(X.rows()) * (ones_train.transpose() * compute_kernel_matrix(X_fit_, X_fit_) * ones_train) / (n_train * n_train);
    
    for (int i = 0; i < K.cols(); ++i) {
        K.col(i) -= k_mean;
        K.col(i) -= k_train_mean;
    }
    
    return K * alphas_;
}

MatrixXd KernelPCA::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

MatrixXd KernelPCA::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("KernelPCA does not support inverse_transform");
}

double KernelPCA::kernel_function(const VectorXd& x1, const VectorXd& x2) const {
    if (kernel_ == "linear") {
        return x1.dot(x2);
    } else if (kernel_ == "rbf") {
        double diff_norm = (x1 - x2).squaredNorm();
        return std::exp(-gamma_ * diff_norm);
    } else if (kernel_ == "poly") {
        return std::pow(gamma_ * x1.dot(x2) + coef0_, degree_);
    }
    return x1.dot(x2); // default to linear
}

MatrixXd KernelPCA::compute_kernel_matrix(const MatrixXd& X1, const MatrixXd& X2) const {
    MatrixXd K(X1.rows(), X2.rows());
    for (int i = 0; i < X1.rows(); ++i) {
        for (int j = 0; j < X2.rows(); ++j) {
            K(i, j) = kernel_function(X1.row(i), X2.row(j));
        }
    }
    return K;
}

Params KernelPCA::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"kernel", kernel_},
        {"gamma", std::to_string(gamma_)},
        {"degree", std::to_string(degree_)},
        {"coef0", std::to_string(coef0_)}
    };
}

Estimator& KernelPCA::set_params(const Params& params) {
    if (params.count("n_components")) n_components_ = std::stoi(params.at("n_components"));
    if (params.count("kernel")) kernel_ = params.at("kernel");
    if (params.count("gamma")) gamma_ = std::stod(params.at("gamma"));
    if (params.count("degree")) degree_ = std::stod(params.at("degree"));
    if (params.count("coef0")) coef0_ = std::stod(params.at("coef0"));
    return *this;
}

// FastICA implementation
FastICA::FastICA(int n_components, const std::string& algorithm, const std::string& fun,
                 int max_iter, double tol, int random_state)
    : n_components_(n_components), algorithm_(algorithm), fun_(fun), max_iter_(max_iter),
      tol_(tol), random_state_(random_state), fitted_(false) {}

Estimator& FastICA::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    int n_samples = X.rows();
    int n_features = X.cols();
    
    if (n_components_ <= 0) {
        n_components_ = n_features;
    }
    
    mean_ = X.colwise().mean();
    MatrixXd X_centered = X.rowwise() - mean_.transpose();
    
    // Whitening
    whiten(X_centered);
    MatrixXd X_white = whitening_ * X_centered.transpose();
    
    // Initialize random components
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::normal_distribution<double> normal(0.0, 1.0);
    
    components_ = MatrixXd(n_components_, n_features);
    for (int i = 0; i < n_components_; ++i) {
        for (int j = 0; j < n_features; ++j) {
            components_(i, j) = normal(rng);
        }
        components_.row(i).normalize();
    }
    
    // FastICA algorithm
    for (int comp = 0; comp < n_components_; ++comp) {
        VectorXd w = components_.row(comp);
        
        for (int iter = 0; iter < max_iter_; ++iter) {
            VectorXd w_old = w;
            
            // FastICA update rule
            VectorXd wtx = w.transpose() * X_white;
            VectorXd g_wtx = g_function(wtx.transpose()).transpose();
            VectorXd g_prime_wtx = g_prime_function(wtx.transpose()).transpose();
            
            VectorXd temp = (X_white * g_wtx) / n_samples;
            w = temp - g_prime_wtx.mean() * w;
            
            // Gram-Schmidt orthogonalization
            for (int j = 0; j < comp; ++j) {
                w -= w.dot(components_.row(j)) * components_.row(j).transpose();
            }
            
            w.normalize();
            
            if (std::abs(std::abs(w.dot(w_old)) - 1.0) < tol_) {
                break;
            }
        }
        
        components_.row(comp) = w;
    }
    
    // Compute mixing matrix (pseudo-inverse of components)
    mixing_ = components_.completeOrthogonalDecomposition().pseudoInverse();
    
    fitted_ = true;
    return *this;
}

MatrixXd FastICA::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("FastICA must be fitted before transform");
    }
    
    MatrixXd X_centered = X.rowwise() - mean_.transpose();
    MatrixXd X_white = whitening_ * X_centered.transpose();
    MatrixXd transformed = components_ * X_white;
    return transformed.transpose();
}

MatrixXd FastICA::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

MatrixXd FastICA::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("FastICA must be fitted before inverse_transform");
    }
    
    MatrixXd X_mixed = mixing_ * X.transpose();
    MatrixXd whitening_inv = whitening_.completeOrthogonalDecomposition().pseudoInverse();
    MatrixXd X_unwhite = whitening_inv * X_mixed;
    MatrixXd result = X_unwhite.transpose();
    return result.rowwise() + mean_.transpose();
}

void FastICA::whiten(const MatrixXd& X) {
    // PCA whitening
    MatrixXd cov = (X.transpose() * X) / (X.rows() - 1);
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(cov);
    
    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXd eigenvectors = solver.eigenvectors();
    
    // Sort eigenvalues and eigenvectors in descending order
    std::vector<int> indices(eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return eigenvalues(i) > eigenvalues(j);
    });
    
    MatrixXd sorted_vectors(eigenvectors.rows(), eigenvectors.cols());
    VectorXd sorted_values(eigenvalues.size());
    for (int i = 0; i < indices.size(); ++i) {
        sorted_vectors.col(i) = eigenvectors.col(indices[i]);
        sorted_values(i) = eigenvalues(indices[i]);
    }
    
    // Whitening matrix
    VectorXd inv_sqrt_vals = sorted_values.array().sqrt().inverse();
    whitening_ = inv_sqrt_vals.asDiagonal() * sorted_vectors.transpose();
}

MatrixXd FastICA::g_function(const MatrixXd& X) const {
    if (fun_ == "logcosh") {
        return X.array().tanh();
    } else if (fun_ == "exp") {
        return X.cwiseProduct((-0.5 * X.array().square()).exp().matrix());
    } else if (fun_ == "cube") {
        return X.array().cube();
    }
    return X.array().tanh(); // default
}

MatrixXd FastICA::g_prime_function(const MatrixXd& X) const {
    if (fun_ == "logcosh") {
        return 1.0 - X.array().tanh().square();
    } else if (fun_ == "exp") {
        return (1.0 - X.array().square()).cwiseProduct((-0.5 * X.array().square()).exp().matrix().array());
    } else if (fun_ == "cube") {
        return 3.0 * X.array().square();
    }
    return 1.0 - X.array().tanh().square(); // default
}

Params FastICA::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"algorithm", algorithm_},
        {"fun", fun_},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& FastICA::set_params(const Params& params) {
    if (params.count("n_components")) n_components_ = std::stoi(params.at("n_components"));
    if (params.count("algorithm")) algorithm_ = params.at("algorithm");
    if (params.count("fun")) fun_ = params.at("fun");
    if (params.count("max_iter")) max_iter_ = std::stoi(params.at("max_iter"));
    if (params.count("tol")) tol_ = std::stod(params.at("tol"));
    if (params.count("random_state")) random_state_ = std::stoi(params.at("random_state"));
    return *this;
}

// FactorAnalysis implementation
FactorAnalysis::FactorAnalysis(int n_components, double tol, int max_iter, int random_state)
    : n_components_(n_components),
      tol_(tol),
      max_iter_(max_iter),
      random_state_(random_state),
      components_(),
      noise_variance_(),
      mean_(),
      loglike_(0.0),
      fitted_(false) {}

Estimator& FactorAnalysis::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);

    int n_samples = X.rows();
    int n_features = X.cols();
    if (n_samples == 0 || n_features == 0) {
        throw std::invalid_argument("X cannot be empty");
    }
    if (n_components_ <= 0) {
        n_components_ = std::min(n_samples, n_features);
    }
    if (n_components_ > n_features) {
        throw std::invalid_argument("n_components cannot be greater than n_features");
    }

    mean_ = X.colwise().mean();
    MatrixXd Xc = X.rowwise() - mean_.transpose();
    fit_em_algorithm(Xc);
    fitted_ = true;
    return *this;
}

void FactorAnalysis::fit_em_algorithm(const MatrixXd& X) {
    const int n_samples = X.rows();
    const int n_features = X.cols();
    const int k = n_components_;
    const double eps = 1e-6;

    MatrixXd cov = (X.transpose() * X) / static_cast<double>(n_samples);
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(cov);
    VectorXd evals = es.eigenvalues();
    MatrixXd evecs = es.eigenvectors();

    std::vector<int> idx(n_features);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return evals(a) > evals(b); });

    MatrixXd W = MatrixXd::Zero(n_features, k);
    for (int i = 0; i < k; ++i) {
        double val = std::max(evals(idx[i]) - eps, eps);
        W.col(i) = evecs.col(idx[i]) * std::sqrt(val);
    }

    double noise = eps;
    if (n_features > k) {
        double sum = 0.0;
        for (int i = k; i < n_features; ++i) {
            sum += evals(idx[i]);
        }
        noise = sum / static_cast<double>(n_features - k);
    }
    noise = std::max(noise, eps);
    VectorXd psi = VectorXd::Constant(n_features, noise);

    double prev_loglike = -std::numeric_limits<double>::infinity();
    const double kTwoPi = 6.28318530717958647692;

    for (int iter = 0; iter < max_iter_; ++iter) {
        MatrixXd psi_inv = psi.cwiseInverse().asDiagonal();
        MatrixXd M = W.transpose() * psi_inv * W + MatrixXd::Identity(k, k);
        MatrixXd M_inv = M.inverse();

        MatrixXd Ez = X * psi_inv * W * M_inv;
        MatrixXd Ezz = Ez.transpose() * Ez + static_cast<double>(n_samples) * M_inv;

        MatrixXd W_new = (X.transpose() * Ez) * Ezz.inverse();
        MatrixXd S = (X.transpose() * X) / static_cast<double>(n_samples);
        MatrixXd W_EzX = W_new * (Ez.transpose() * X) / static_cast<double>(n_samples);
        VectorXd psi_new = (S - W_EzX).diagonal();
        for (int i = 0; i < psi_new.size(); ++i) {
            psi_new(i) = std::max(psi_new(i), eps);
        }

        MatrixXd cov_new = W_new * W_new.transpose();
        cov_new.diagonal() += psi_new;
        Eigen::SelfAdjointEigenSolver<MatrixXd> es_cov(cov_new);
        VectorXd cov_evals = es_cov.eigenvalues();
        double log_det = 0.0;
        for (int i = 0; i < cov_evals.size(); ++i) {
            log_det += std::log(std::max(cov_evals(i), eps));
        }
        MatrixXd cov_inv = cov_new.ldlt().solve(MatrixXd::Identity(n_features, n_features));
        double trace_term = (cov_inv * S).trace();
        double loglike = -0.5 * static_cast<double>(n_samples) * (n_features * std::log(kTwoPi) + log_det + trace_term);

        if (std::abs(loglike - prev_loglike) < tol_) {
            W = W_new;
            psi = psi_new;
            prev_loglike = loglike;
            break;
        }

        W = W_new;
        psi = psi_new;
        prev_loglike = loglike;
    }

    components_ = W.transpose();
    noise_variance_ = psi;
    loglike_ = prev_loglike;
}

MatrixXd FactorAnalysis::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("FactorAnalysis must be fitted before transform");
    }
    if (X.cols() != mean_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd Xc = X.rowwise() - mean_.transpose();
    MatrixXd W = components_.transpose();
    MatrixXd psi_inv = noise_variance_.cwiseInverse().asDiagonal();
    MatrixXd M = W.transpose() * psi_inv * W + MatrixXd::Identity(n_components_, n_components_);
    MatrixXd M_inv = M.inverse();
    return Xc * psi_inv * W * M_inv;
}

MatrixXd FactorAnalysis::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

MatrixXd FactorAnalysis::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("FactorAnalysis must be fitted before inverse_transform");
    }
    MatrixXd W = components_.transpose();
    MatrixXd reconstructed = X * W.transpose();
    return reconstructed.rowwise() + mean_.transpose();
}

Params FactorAnalysis::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"tol", std::to_string(tol_)},
        {"max_iter", std::to_string(max_iter_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& FactorAnalysis::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// TSNE implementation
TSNE::TSNE(int n_components, double perplexity, double early_exaggeration,
           double learning_rate, int max_iter, int random_state)
    : n_components_(n_components), perplexity_(perplexity), early_exaggeration_(early_exaggeration),
      learning_rate_(learning_rate), max_iter_(max_iter), random_state_(random_state), fitted_(false) {}

Estimator& TSNE::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    int n_samples = X.rows();
    
    // Compute pairwise affinities
    MatrixXd P = compute_pairwise_affinities(X);
    
    // Initialize embedding randomly
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::normal_distribution<double> normal(0.0, 1e-4);
    
    embedding_ = MatrixXd(n_samples, n_components_);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_components_; ++j) {
            embedding_(i, j) = normal(rng);
        }
    }
    
    // Perform gradient descent
    gradient_descent(P);
    
    fitted_ = true;
    return *this;
}

MatrixXd TSNE::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("TSNE must be fitted before transform");
    }
    return embedding_;
}

MatrixXd TSNE::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return embedding_;
}

MatrixXd TSNE::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("TSNE does not support inverse_transform");
}

MatrixXd TSNE::compute_pairwise_affinities(const MatrixXd& X) const {
    int n_samples = X.rows();
    MatrixXd P = MatrixXd::Zero(n_samples, n_samples);
    
    // Compute Euclidean distances
    MatrixXd distances = MatrixXd::Zero(n_samples, n_samples);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = i + 1; j < n_samples; ++j) {
            double dist = (X.row(i) - X.row(j)).norm();
            distances(i, j) = distances(j, i) = dist * dist;
        }
    }
    
    // Convert distances to probabilities using Gaussian kernel
    double sigma = 1.0; // Simplified: use fixed bandwidth
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            if (i != j) {
                P(i, j) = std::exp(-distances(i, j) / (2.0 * sigma * sigma));
            }
        }
        
        // Normalize row
        double row_sum = P.row(i).sum();
        if (row_sum > 0) {
            P.row(i) /= row_sum;
        }
    }
    
    // Make symmetric
    P = (P + P.transpose()) / (2.0 * n_samples);
    
    // Ensure minimum probability
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            P(i, j) = std::max(P(i, j), 1e-12);
        }
    }
    
    return P;
}

void TSNE::gradient_descent(const MatrixXd& P) {
    int n_samples = P.rows();
    MatrixXd gains = MatrixXd::Ones(n_samples, n_components_);
    MatrixXd momentum = MatrixXd::Zero(n_samples, n_components_);
    
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Compute Q (low-dimensional affinities)
        MatrixXd Q = MatrixXd::Zero(n_samples, n_samples);
        double sum_Q = 0.0;
        
        for (int i = 0; i < n_samples; ++i) {
            for (int j = i + 1; j < n_samples; ++j) {
                double dist_sq = (embedding_.row(i) - embedding_.row(j)).squaredNorm();
                double q_ij = 1.0 / (1.0 + dist_sq);
                Q(i, j) = Q(j, i) = q_ij;
                sum_Q += 2.0 * q_ij;
            }
        }
        
        // Normalize Q
        if (sum_Q > 0) {
            Q /= sum_Q;
        }
        
        // Ensure minimum probability
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_samples; ++j) {
                Q(i, j) = std::max(Q(i, j), 1e-12);
            }
        }
        
        // Compute gradient
        MatrixXd gradient = MatrixXd::Zero(n_samples, n_components_);
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_samples; ++j) {
                if (i != j) {
                    double multiplier = (P(i, j) - Q(i, j)) * (1.0 / (1.0 + (embedding_.row(i) - embedding_.row(j)).squaredNorm()));
                    for (int d = 0; d < n_components_; ++d) {
                        gradient(i, d) += 4.0 * multiplier * (embedding_(i, d) - embedding_(j, d));
                    }
                }
            }
        }
        
        // Update gains
        for (int i = 0; i < n_samples; ++i) {
            for (int d = 0; d < n_components_; ++d) {
                gains(i, d) = (std::signbit(gradient(i, d)) != std::signbit(momentum(i, d))) ? gains(i, d) + 0.2 : gains(i, d) * 0.8;
                gains(i, d) = std::max(gains(i, d), 0.01);
            }
        }
        
        // Update momentum and embedding
        double lr = (iter < 250) ? learning_rate_ * early_exaggeration_ : learning_rate_;
        momentum = 0.8 * momentum - lr * gains.cwiseProduct(gradient);
        embedding_ += momentum;
        
        // Center embedding
        VectorXd mean = embedding_.colwise().mean();
        embedding_ = embedding_.rowwise() - mean.transpose();
    }
}

Params TSNE::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"perplexity", std::to_string(perplexity_)},
        {"early_exaggeration", std::to_string(early_exaggeration_)},
        {"learning_rate", std::to_string(learning_rate_)},
        {"max_iter", std::to_string(max_iter_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& TSNE::set_params(const Params& params) {
    if (params.count("n_components")) n_components_ = std::stoi(params.at("n_components"));
    if (params.count("perplexity")) perplexity_ = std::stod(params.at("perplexity"));
    if (params.count("early_exaggeration")) early_exaggeration_ = std::stod(params.at("early_exaggeration"));
    if (params.count("learning_rate")) learning_rate_ = std::stod(params.at("learning_rate"));
    if (params.count("max_iter")) max_iter_ = std::stoi(params.at("max_iter"));
    if (params.count("random_state")) random_state_ = std::stoi(params.at("random_state"));
    return *this;
}

} // namespace decomposition
} // namespace ingenuityml

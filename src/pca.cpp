#include "auroraml/pca.hpp"

namespace auroraml {
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

} // namespace decomposition
} // namespace cxml



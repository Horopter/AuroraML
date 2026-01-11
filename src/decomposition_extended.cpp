#include "ingenuityml/decomposition_extended.hpp"
#include "ingenuityml/base.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <limits>

namespace ingenuityml {
namespace decomposition {

namespace {

void check_nonnegative(const MatrixXd& X, const std::string& name) {
    if (X.size() == 0) {
        throw std::invalid_argument(name + " cannot be empty");
    }
    if (X.minCoeff() < 0.0) {
        throw std::invalid_argument(name + " must be non-negative");
    }
}

MatrixXd soft_threshold(const MatrixXd& X, double alpha) {
    if (alpha <= 0.0) {
        return X;
    }
    MatrixXd out = X;
    for (int i = 0; i < out.rows(); ++i) {
        for (int j = 0; j < out.cols(); ++j) {
            double v = out(i, j);
            if (v > alpha) {
                out(i, j) = v - alpha;
            } else if (v < -alpha) {
                out(i, j) = v + alpha;
            } else {
                out(i, j) = 0.0;
            }
        }
    }
    return out;
}

void normalize_rows(MatrixXd& X) {
    for (int i = 0; i < X.rows(); ++i) {
        double norm = X.row(i).norm();
        if (norm > 0.0) {
            X.row(i) /= norm;
        }
    }
}

void compute_pca_components(const MatrixXd& X, int n_components,
                            VectorXd& mean, MatrixXd& components, VectorXd& explained_variance) {
    validation::check_X(X);
    int n_samples = X.rows();
    int n_features = X.cols();
    int k = n_components;
    if (k <= 0) {
        k = std::min(n_samples, n_features);
    }
    if (k > n_features || k > n_samples) {
        throw std::invalid_argument("n_components cannot be greater than n_samples or n_features");
    }

    mean = X.colwise().mean();
    MatrixXd Xc = X.rowwise() - mean.transpose();
    MatrixXd C = (Xc.transpose() * Xc) / static_cast<double>(n_samples);
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(C);
    VectorXd evals = es.eigenvalues();
    MatrixXd evecs = es.eigenvectors();

    std::vector<int> idx(n_features);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return evals(a) > evals(b); });

    explained_variance.resize(k);
    components.resize(k, n_features);
    for (int i = 0; i < k; ++i) {
        explained_variance(i) = evals(idx[i]);
        components.row(i) = evecs.col(idx[i]).transpose();
    }
}

} // namespace

IncrementalPCA::IncrementalPCA(int n_components, bool whiten, int batch_size)
    : n_components_(n_components),
      whiten_(whiten),
      batch_size_(batch_size),
      fitted_(false),
      n_samples_seen_(0),
      mean_(),
      components_(),
      explained_variance_(),
      buffer_() {}

Estimator& IncrementalPCA::fit(const MatrixXd& X, const VectorXd& y) {
    compute_pca_components(X, n_components_, mean_, components_, explained_variance_);
    if (n_components_ <= 0) {
        n_components_ = components_.rows();
    }
    fitted_ = true;
    n_samples_seen_ = static_cast<int>(X.rows());
    return *this;
}

Estimator& IncrementalPCA::partial_fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    if (buffer_.size() == 0) {
        buffer_ = X;
    } else {
        MatrixXd combined(buffer_.rows() + X.rows(), buffer_.cols());
        combined.topRows(buffer_.rows()) = buffer_;
        combined.bottomRows(X.rows()) = X;
        buffer_.swap(combined);
    }
    n_samples_seen_ += static_cast<int>(X.rows());
    return fit(buffer_, VectorXd());
}

MatrixXd IncrementalPCA::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("IncrementalPCA must be fitted before transform");
    }
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

MatrixXd IncrementalPCA::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("IncrementalPCA must be fitted before inverse_transform");
    }
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

MatrixXd IncrementalPCA::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params IncrementalPCA::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"whiten", whiten_ ? "true" : "false"},
        {"batch_size", std::to_string(batch_size_)}
    };
}

Estimator& IncrementalPCA::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    whiten_ = utils::get_param_bool(params, "whiten", whiten_);
    batch_size_ = utils::get_param_int(params, "batch_size", batch_size_);
    return *this;
}

const MatrixXd& IncrementalPCA::components() const {
    if (!fitted_) {
        throw std::runtime_error("IncrementalPCA must be fitted before accessing components");
    }
    return components_;
}

const VectorXd& IncrementalPCA::explained_variance() const {
    if (!fitted_) {
        throw std::runtime_error("IncrementalPCA must be fitted before accessing explained variance");
    }
    return explained_variance_;
}

VectorXd IncrementalPCA::explained_variance_ratio() const {
    if (!fitted_) {
        throw std::runtime_error("IncrementalPCA must be fitted before accessing explained variance ratio");
    }
    if (explained_variance_.size() == 0) {
        return VectorXd();
    }
    double sum = explained_variance_.sum();
    if (sum <= 0) {
        return VectorXd::Zero(explained_variance_.size());
    }
    return explained_variance_ / sum;
}

const VectorXd& IncrementalPCA::mean() const {
    if (!fitted_) {
        throw std::runtime_error("IncrementalPCA must be fitted before accessing mean");
    }
    return mean_;
}

SparsePCA::SparsePCA(int n_components, double alpha, int max_iter, double tol)
    : n_components_(n_components),
      alpha_(alpha),
      max_iter_(max_iter),
      tol_(tol),
      fitted_(false),
      mean_(),
      components_() {}

Estimator& SparsePCA::fit(const MatrixXd& X, const VectorXd& y) {
    VectorXd mean;
    MatrixXd components;
    VectorXd explained_variance;
    compute_pca_components(X, n_components_, mean, components, explained_variance);

    mean_ = mean;
    components_ = soft_threshold(components, alpha_);
    normalize_rows(components_);
    fitted_ = true;
    return *this;
}

MatrixXd SparsePCA::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SparsePCA must be fitted before transform");
    }
    if (X.cols() != mean_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd Xc = X.rowwise() - mean_.transpose();
    return Xc * components_.transpose();
}

MatrixXd SparsePCA::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SparsePCA must be fitted before inverse_transform");
    }
    MatrixXd R = X * components_;
    return R.rowwise() + mean_.transpose();
}

MatrixXd SparsePCA::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SparsePCA::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"alpha", std::to_string(alpha_)},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)}
    };
}

Estimator& SparsePCA::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    return *this;
}

const MatrixXd& SparsePCA::components() const {
    if (!fitted_) {
        throw std::runtime_error("SparsePCA must be fitted before accessing components");
    }
    return components_;
}

MiniBatchSparsePCA::MiniBatchSparsePCA(int n_components, double alpha, int max_iter,
                                       int batch_size, double tol, int random_state)
    : n_components_(n_components),
      alpha_(alpha),
      max_iter_(max_iter),
      batch_size_(batch_size),
      tol_(tol),
      random_state_(random_state),
      fitted_(false),
      mean_(),
      components_() {}

Estimator& MiniBatchSparsePCA::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    int n_samples = X.rows();
    int n_features = X.cols();
    mean_ = X.colwise().mean();

    int batch = batch_size_ > 0 ? std::min(batch_size_, n_samples) : n_samples;
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    MatrixXd X_batch(batch, n_features);
    for (int i = 0; i < batch; ++i) {
        X_batch.row(i) = X.row(indices[i]);
    }

    VectorXd mean;
    MatrixXd components;
    VectorXd explained_variance;
    compute_pca_components(X_batch, n_components_, mean, components, explained_variance);
    components_ = soft_threshold(components, alpha_);
    normalize_rows(components_);
    fitted_ = true;
    return *this;
}

MatrixXd MiniBatchSparsePCA::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchSparsePCA must be fitted before transform");
    }
    if (X.cols() != mean_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd Xc = X.rowwise() - mean_.transpose();
    return Xc * components_.transpose();
}

MatrixXd MiniBatchSparsePCA::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchSparsePCA must be fitted before inverse_transform");
    }
    MatrixXd R = X * components_;
    return R.rowwise() + mean_.transpose();
}

MatrixXd MiniBatchSparsePCA::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params MiniBatchSparsePCA::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"alpha", std::to_string(alpha_)},
        {"max_iter", std::to_string(max_iter_)},
        {"batch_size", std::to_string(batch_size_)},
        {"tol", std::to_string(tol_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& MiniBatchSparsePCA::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    batch_size_ = utils::get_param_int(params, "batch_size", batch_size_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

const MatrixXd& MiniBatchSparsePCA::components() const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchSparsePCA must be fitted before accessing components");
    }
    return components_;
}

NMF::NMF(int n_components, int max_iter, double tol, double alpha, int random_state)
    : n_components_(n_components),
      max_iter_(max_iter),
      tol_(tol),
      alpha_(alpha),
      random_state_(random_state),
      fitted_(false),
      components_(),
      W_() {}

Estimator& NMF::fit(const MatrixXd& X, const VectorXd& y) {
    check_nonnegative(X, "X");
    int n_samples = X.rows();
    int n_features = X.cols();
    if (n_components_ <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    MatrixXd W = MatrixXd::NullaryExpr(n_samples, n_components_, [&]() { return dist(rng) + 1e-3; });
    MatrixXd H = MatrixXd::NullaryExpr(n_components_, n_features, [&]() { return dist(rng) + 1e-3; });

    double prev_error = std::numeric_limits<double>::infinity();
    const double eps = 1e-9;
    for (int iter = 0; iter < max_iter_; ++iter) {
        MatrixXd numerator = W.transpose() * X;
        MatrixXd denominator = (W.transpose() * W * H).array() + alpha_ + eps;
        H = H.array() * numerator.array() / denominator.array();

        numerator = X * H.transpose();
        denominator = (W * H * H.transpose()).array() + alpha_ + eps;
        W = W.array() * numerator.array() / denominator.array();

        double error = (X - W * H).norm();
        if (std::abs(prev_error - error) / (prev_error + eps) < tol_) {
            break;
        }
        prev_error = error;
    }

    W_ = W;
    components_ = H;
    fitted_ = true;
    return *this;
}

MatrixXd NMF::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("NMF must be fitted before transform");
    }
    check_nonnegative(X, "X");
    if (X.cols() != components_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    MatrixXd W = MatrixXd::NullaryExpr(X.rows(), n_components_, [&]() { return dist(rng) + 1e-3; });

    const double eps = 1e-9;
    for (int iter = 0; iter < max_iter_; ++iter) {
        MatrixXd numerator = X * components_.transpose();
        MatrixXd denominator = (W * components_ * components_.transpose()).array() + alpha_ + eps;
        W = W.array() * numerator.array() / denominator.array();
    }
    return W;
}

MatrixXd NMF::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("NMF must be fitted before inverse_transform");
    }
    if (X.cols() != components_.rows()) {
        throw std::invalid_argument("X must have the same number of components as training data");
    }
    return X * components_;
}

MatrixXd NMF::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return W_;
}

Params NMF::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"alpha", std::to_string(alpha_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& NMF::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

const MatrixXd& NMF::components() const {
    if (!fitted_) {
        throw std::runtime_error("NMF must be fitted before accessing components");
    }
    return components_;
}

MiniBatchNMF::MiniBatchNMF(int n_components, int max_iter, int batch_size,
                           double tol, double alpha, int random_state)
    : n_components_(n_components),
      max_iter_(max_iter),
      batch_size_(batch_size),
      tol_(tol),
      alpha_(alpha),
      random_state_(random_state),
      fitted_(false),
      components_(),
      W_() {}

Estimator& MiniBatchNMF::fit(const MatrixXd& X, const VectorXd& y) {
    check_nonnegative(X, "X");
    int n_samples = X.rows();
    int n_features = X.cols();
    if (n_components_ <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    MatrixXd W = MatrixXd::NullaryExpr(n_samples, n_components_, [&]() { return dist(rng) + 1e-3; });
    MatrixXd H = MatrixXd::NullaryExpr(n_components_, n_features, [&]() { return dist(rng) + 1e-3; });

    const double eps = 1e-9;
    double prev_error = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < max_iter_; ++iter) {
        int batch = batch_size_ > 0 ? std::min(batch_size_, n_samples) : n_samples;
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        MatrixXd X_batch(batch, n_features);
        MatrixXd W_batch(batch, n_components_);
        for (int i = 0; i < batch; ++i) {
            X_batch.row(i) = X.row(indices[i]);
            W_batch.row(i) = W.row(indices[i]);
        }

        MatrixXd numerator = W_batch.transpose() * X_batch;
        MatrixXd denominator = (W_batch.transpose() * W_batch * H).array() + alpha_ + eps;
        H = H.array() * numerator.array() / denominator.array();

        numerator = X_batch * H.transpose();
        denominator = (W_batch * H * H.transpose()).array() + alpha_ + eps;
        W_batch = W_batch.array() * numerator.array() / denominator.array();

        for (int i = 0; i < batch; ++i) {
            W.row(indices[i]) = W_batch.row(i);
        }

        double error = (X_batch - W_batch * H).norm();
        if (std::abs(prev_error - error) / (prev_error + eps) < tol_) {
            break;
        }
        prev_error = error;
    }

    W_ = W;
    components_ = H;
    fitted_ = true;
    return *this;
}

MatrixXd MiniBatchNMF::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchNMF must be fitted before transform");
    }
    check_nonnegative(X, "X");
    if (X.cols() != components_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    MatrixXd W = MatrixXd::NullaryExpr(X.rows(), n_components_, [&]() { return dist(rng) + 1e-3; });
    const double eps = 1e-9;
    for (int iter = 0; iter < max_iter_; ++iter) {
        MatrixXd numerator = X * components_.transpose();
        MatrixXd denominator = (W * components_ * components_.transpose()).array() + alpha_ + eps;
        W = W.array() * numerator.array() / denominator.array();
    }
    return W;
}

MatrixXd MiniBatchNMF::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchNMF must be fitted before inverse_transform");
    }
    if (X.cols() != components_.rows()) {
        throw std::invalid_argument("X must have the same number of components as training data");
    }
    return X * components_;
}

MatrixXd MiniBatchNMF::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return W_;
}

Params MiniBatchNMF::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"max_iter", std::to_string(max_iter_)},
        {"batch_size", std::to_string(batch_size_)},
        {"tol", std::to_string(tol_)},
        {"alpha", std::to_string(alpha_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& MiniBatchNMF::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    batch_size_ = utils::get_param_int(params, "batch_size", batch_size_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

const MatrixXd& MiniBatchNMF::components() const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchNMF must be fitted before accessing components");
    }
    return components_;
}

DictionaryLearning::DictionaryLearning(int n_components, double alpha, int max_iter, double tol, int random_state)
    : n_components_(n_components),
      alpha_(alpha),
      max_iter_(max_iter),
      tol_(tol),
      random_state_(random_state),
      fitted_(false),
      components_(),
      codes_() {}

Estimator& DictionaryLearning::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    int n_samples = X.rows();
    int n_features = X.cols();
    if (n_components_ <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    components_.resize(n_components_, n_features);
    for (int i = 0; i < n_components_; ++i) {
        components_.row(i) = X.row(indices[i % n_samples]);
    }
    normalize_rows(components_);

    double prev_error = std::numeric_limits<double>::infinity();
    const double eps = 1e-9;
    for (int iter = 0; iter < max_iter_; ++iter) {
        MatrixXd codes = X * components_.transpose();
        codes = soft_threshold(codes, alpha_);

        MatrixXd gram = codes.transpose() * codes;
        MatrixXd XtC = codes.transpose() * X;
        gram += eps * MatrixXd::Identity(gram.rows(), gram.cols());
        MatrixXd new_components = gram.ldlt().solve(XtC);
        normalize_rows(new_components);

        double error = (X - codes * new_components).norm();
        if (std::abs(prev_error - error) / (prev_error + eps) < tol_) {
            components_ = new_components;
            codes_ = codes;
            break;
        }
        prev_error = error;
        components_ = new_components;
        codes_ = codes;
    }

    fitted_ = true;
    return *this;
}

MatrixXd DictionaryLearning::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("DictionaryLearning must be fitted before transform");
    }
    if (X.cols() != components_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd codes = X * components_.transpose();
    return soft_threshold(codes, alpha_);
}

MatrixXd DictionaryLearning::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("DictionaryLearning must be fitted before inverse_transform");
    }
    if (X.cols() != components_.rows()) {
        throw std::invalid_argument("X must have the same number of components as training data");
    }
    return X * components_;
}

MatrixXd DictionaryLearning::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return codes_;
}

Params DictionaryLearning::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"alpha", std::to_string(alpha_)},
        {"max_iter", std::to_string(max_iter_)},
        {"tol", std::to_string(tol_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& DictionaryLearning::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

const MatrixXd& DictionaryLearning::components() const {
    if (!fitted_) {
        throw std::runtime_error("DictionaryLearning must be fitted before accessing components");
    }
    return components_;
}

const MatrixXd& DictionaryLearning::codes() const {
    if (!fitted_) {
        throw std::runtime_error("DictionaryLearning must be fitted before accessing codes");
    }
    return codes_;
}

MiniBatchDictionaryLearning::MiniBatchDictionaryLearning(int n_components, double alpha, int max_iter,
                                                         int batch_size, double tol, int random_state)
    : n_components_(n_components),
      alpha_(alpha),
      max_iter_(max_iter),
      batch_size_(batch_size),
      tol_(tol),
      random_state_(random_state),
      fitted_(false),
      components_() {}

Estimator& MiniBatchDictionaryLearning::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    int n_samples = X.rows();
    int n_features = X.cols();
    if (n_components_ <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    components_ = MatrixXd::NullaryExpr(n_components_, n_features, [&]() { return dist(rng); });
    normalize_rows(components_);

    const double eps = 1e-9;
    double prev_error = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < max_iter_; ++iter) {
        int batch = batch_size_ > 0 ? std::min(batch_size_, n_samples) : n_samples;
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        MatrixXd X_batch(batch, n_features);
        for (int i = 0; i < batch; ++i) {
            X_batch.row(i) = X.row(indices[i]);
        }

        MatrixXd codes = X_batch * components_.transpose();
        codes = soft_threshold(codes, alpha_);

        MatrixXd gram = codes.transpose() * codes;
        MatrixXd XtC = codes.transpose() * X_batch;
        gram += eps * MatrixXd::Identity(gram.rows(), gram.cols());
        MatrixXd new_components = gram.ldlt().solve(XtC);
        normalize_rows(new_components);

        double error = (X_batch - codes * new_components).norm();
        components_ = new_components;
        if (std::abs(prev_error - error) / (prev_error + eps) < tol_) {
            break;
        }
        prev_error = error;
    }

    fitted_ = true;
    return *this;
}

MatrixXd MiniBatchDictionaryLearning::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchDictionaryLearning must be fitted before transform");
    }
    if (X.cols() != components_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd codes = X * components_.transpose();
    return soft_threshold(codes, alpha_);
}

MatrixXd MiniBatchDictionaryLearning::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchDictionaryLearning must be fitted before inverse_transform");
    }
    if (X.cols() != components_.rows()) {
        throw std::invalid_argument("X must have the same number of components as training data");
    }
    return X * components_;
}

MatrixXd MiniBatchDictionaryLearning::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params MiniBatchDictionaryLearning::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"alpha", std::to_string(alpha_)},
        {"max_iter", std::to_string(max_iter_)},
        {"batch_size", std::to_string(batch_size_)},
        {"tol", std::to_string(tol_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& MiniBatchDictionaryLearning::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    batch_size_ = utils::get_param_int(params, "batch_size", batch_size_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

const MatrixXd& MiniBatchDictionaryLearning::components() const {
    if (!fitted_) {
        throw std::runtime_error("MiniBatchDictionaryLearning must be fitted before accessing components");
    }
    return components_;
}

LatentDirichletAllocation::LatentDirichletAllocation(int n_components, int max_iter,
                                                     double doc_topic_prior, double topic_word_prior,
                                                     int random_state)
    : n_components_(n_components),
      max_iter_(max_iter),
      doc_topic_prior_(doc_topic_prior),
      topic_word_prior_(topic_word_prior),
      random_state_(random_state),
      fitted_(false),
      components_(),
      doc_topic_() {}

Estimator& LatentDirichletAllocation::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    check_nonnegative(X, "X");
    int n_docs = X.rows();
    int n_words = X.cols();
    if (n_components_ <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }

    std::vector<std::vector<int>> docs(n_docs);
    docs.reserve(n_docs);
    for (int d = 0; d < n_docs; ++d) {
        for (int w = 0; w < n_words; ++w) {
            int count = static_cast<int>(std::round(X(d, w)));
            for (int c = 0; c < count; ++c) {
                docs[d].push_back(w);
            }
        }
    }

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::uniform_int_distribution<int> topic_dist(0, n_components_ - 1);

    Eigen::MatrixXi doc_topic_counts = Eigen::MatrixXi::Zero(n_docs, n_components_);
    Eigen::MatrixXi topic_word_counts = Eigen::MatrixXi::Zero(n_components_, n_words);
    VectorXi topic_counts = VectorXi::Zero(n_components_);
    std::vector<std::vector<int>> assignments(n_docs);

    for (int d = 0; d < n_docs; ++d) {
        assignments[d].resize(docs[d].size());
        for (size_t i = 0; i < docs[d].size(); ++i) {
            int topic = topic_dist(rng);
            assignments[d][i] = topic;
            doc_topic_counts(d, topic) += 1;
            topic_word_counts(topic, docs[d][i]) += 1;
            topic_counts(topic) += 1;
        }
    }

    for (int iter = 0; iter < max_iter_; ++iter) {
        for (int d = 0; d < n_docs; ++d) {
            for (size_t i = 0; i < docs[d].size(); ++i) {
                int word = docs[d][i];
                int topic = assignments[d][i];

                doc_topic_counts(d, topic) -= 1;
                topic_word_counts(topic, word) -= 1;
                topic_counts(topic) -= 1;

                std::vector<double> probs(n_components_);
                double sum = 0.0;
                for (int k = 0; k < n_components_; ++k) {
                    double term1 = doc_topic_counts(d, k) + doc_topic_prior_;
                    double term2 = topic_word_counts(k, word) + topic_word_prior_;
                    double term3 = topic_counts(k) + n_words * topic_word_prior_;
                    double p = term1 * term2 / term3;
                    probs[k] = p;
                    sum += p;
                }
                if (sum <= 0.0) {
                    std::fill(probs.begin(), probs.end(), 1.0 / n_components_);
                } else {
                    for (double& p : probs) {
                        p /= sum;
                    }
                }
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int new_topic = dist(rng);

                assignments[d][i] = new_topic;
                doc_topic_counts(d, new_topic) += 1;
                topic_word_counts(new_topic, word) += 1;
                topic_counts(new_topic) += 1;
            }
        }
    }

    components_ = MatrixXd::Zero(n_components_, n_words);
    for (int k = 0; k < n_components_; ++k) {
        for (int w = 0; w < n_words; ++w) {
            components_(k, w) = (topic_word_counts(k, w) + topic_word_prior_) /
                                (topic_counts(k) + n_words * topic_word_prior_);
        }
    }

    doc_topic_ = MatrixXd::Zero(n_docs, n_components_);
    for (int d = 0; d < n_docs; ++d) {
        double denom = docs[d].size() + n_components_ * doc_topic_prior_;
        for (int k = 0; k < n_components_; ++k) {
            doc_topic_(d, k) = (doc_topic_counts(d, k) + doc_topic_prior_) / std::max(denom, 1.0);
        }
    }

    fitted_ = true;
    return *this;
}

MatrixXd LatentDirichletAllocation::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LatentDirichletAllocation must be fitted before transform");
    }
    check_nonnegative(X, "X");
    if (X.cols() != components_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    MatrixXd doc_topic = X * components_.transpose();
    for (int i = 0; i < doc_topic.rows(); ++i) {
        double sum = doc_topic.row(i).sum();
        if (sum > 0.0) {
            doc_topic.row(i) /= sum;
        } else {
            doc_topic.row(i).setConstant(1.0 / n_components_);
        }
    }
    return doc_topic;
}

MatrixXd LatentDirichletAllocation::inverse_transform(const MatrixXd& X) const {
    throw std::runtime_error("LatentDirichletAllocation does not support inverse_transform");
}

MatrixXd LatentDirichletAllocation::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return doc_topic_;
}

Params LatentDirichletAllocation::get_params() const {
    return {
        {"n_components", std::to_string(n_components_)},
        {"max_iter", std::to_string(max_iter_)},
        {"doc_topic_prior", std::to_string(doc_topic_prior_)},
        {"topic_word_prior", std::to_string(topic_word_prior_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& LatentDirichletAllocation::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    doc_topic_prior_ = utils::get_param_double(params, "doc_topic_prior", doc_topic_prior_);
    topic_word_prior_ = utils::get_param_double(params, "topic_word_prior", topic_word_prior_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

const MatrixXd& LatentDirichletAllocation::components() const {
    if (!fitted_) {
        throw std::runtime_error("LatentDirichletAllocation must be fitted before accessing components");
    }
    return components_;
}

const MatrixXd& LatentDirichletAllocation::doc_topic() const {
    if (!fitted_) {
        throw std::runtime_error("LatentDirichletAllocation must be fitted before accessing doc_topic");
    }
    return doc_topic_;
}

} // namespace decomposition
} // namespace ingenuityml

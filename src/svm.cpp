#include "auroraml/svm.hpp"
#include "auroraml/base.hpp"
#include <random>
#include <set>
#include <algorithm>
#include <numeric>

namespace auroraml {
namespace svm {

Estimator& LinearSVC::fit(const MatrixXd& X, const VectorXd& y_in) {
    validation::check_X_y(X, y_in);
    
    // Validate parameters
    if (C_ <= 0.0) {
        throw std::invalid_argument("C must be positive");
    }
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (lr_ <= 0.0) {
        throw std::invalid_argument("learning rate must be positive");
    }
    
    // Convert y to {-1, +1}
    VectorXd y = y_in; for (int i = 0; i < y.size(); ++i) y(i) = (y(i) > 0.5) ? 1.0 : -1.0;

    w_ = VectorXd::Zero(X.cols());
    b_ = 0.0;
    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);

    const double lambda = 1.0 / std::max(C_, 1e-12);
    for (int it = 0; it < max_iter_; ++it) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double lr = lr_ / (1.0 + 0.01 * it);
        for (int idx : indices) {
            double margin = y(idx) * (X.row(idx).dot(w_) + b_);
            if (margin < 1.0) {
                // gradient: w <- (1 - lr*lambda)w + lr*y_i*x_i; b <- b + lr*y_i
                w_ = (1.0 - lr * lambda) * w_ + lr * y(idx) * X.row(idx).transpose();
                b_ = b_ + lr * y(idx);
            } else {
                w_ = (1.0 - lr * lambda) * w_;
            }
        }
    }
    fitted_ = true;
    return *this;
}

VectorXi LinearSVC::predict_classes(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("LinearSVC must be fitted before predict");
    validation::check_X(X);
    if (X.cols() != w_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    VectorXi y(X.rows());
    for (int i = 0; i < X.rows(); ++i) y(i) = (X.row(i).dot(w_) + b_ >= 0.0) ? 1 : 0;
    return y;
}

MatrixXd LinearSVC::predict_proba(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("LinearSVC must be fitted before predict_proba");
    // Not probabilistic; use a logistic squashing on decision function as a proxy
    MatrixXd P(X.rows(), 2);
    for (int i = 0; i < X.rows(); ++i) {
        double s = X.row(i).dot(w_) + b_;
        double p1 = 1.0 / (1.0 + std::exp(-s));
        P(i, 1) = p1;
        P(i, 0) = 1.0 - p1;
    }
    return P;
}

VectorXd LinearSVC::decision_function(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("LinearSVC must be fitted before decision_function");
    VectorXd s(X.rows());
    for (int i = 0; i < X.rows(); ++i) s(i) = X.row(i).dot(w_) + b_;
    return s;
}

Params LinearSVC::get_params() const {
    return {{"C", std::to_string(C_)}, {"max_iter", std::to_string(max_iter_)}, {"lr", std::to_string(lr_)}, {"random_state", std::to_string(random_state_)}};
}

Estimator& LinearSVC::set_params(const Params& params) {
    C_ = utils::get_param_double(params, "C", C_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    lr_ = utils::get_param_double(params, "lr", lr_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// LinearSVC save/load implementation
void LinearSVC::save(const std::string& filepath) const {
    if (!fitted_) {
        throw std::runtime_error("LinearSVC must be fitted before saving");
    }
    
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&C_), sizeof(C_));
    ofs.write(reinterpret_cast<const char*>(&max_iter_), sizeof(max_iter_));
    ofs.write(reinterpret_cast<const char*>(&lr_), sizeof(lr_));
    ofs.write(reinterpret_cast<const char*>(&random_state_), sizeof(random_state_));
    
    // Save weights and bias
    int weights_size = w_.size();
    ofs.write(reinterpret_cast<const char*>(&weights_size), sizeof(weights_size));
    if (weights_size > 0) {
        ofs.write(reinterpret_cast<const char*>(w_.data()), weights_size * sizeof(double));
    }
    ofs.write(reinterpret_cast<const char*>(&b_), sizeof(b_));
    
    ofs.close();
}

void LinearSVC::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&C_), sizeof(C_));
    ifs.read(reinterpret_cast<char*>(&max_iter_), sizeof(max_iter_));
    ifs.read(reinterpret_cast<char*>(&lr_), sizeof(lr_));
    ifs.read(reinterpret_cast<char*>(&random_state_), sizeof(random_state_));
    
    // Load weights and bias
    int weights_size;
    ifs.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));
    w_.resize(weights_size);
    if (weights_size > 0) {
        ifs.read(reinterpret_cast<char*>(w_.data()), weights_size * sizeof(double));
    }
    ifs.read(reinterpret_cast<char*>(&b_), sizeof(b_));
    
    ifs.close();
}

// SVR implementation
SVR::SVR(double C, double epsilon, int max_iter, double lr, int random_state)
    : C_(C), epsilon_(epsilon), max_iter_(max_iter), lr_(lr), random_state_(random_state) {}

Estimator& SVR::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    if (C_ <= 0.0) {
        throw std::invalid_argument("C must be positive");
    }
    if (max_iter_ <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (lr_ <= 0.0) {
        throw std::invalid_argument("learning rate must be positive");
    }
    if (epsilon_ < 0.0) {
        throw std::invalid_argument("epsilon must be non-negative");
    }
    
    const double lambda = 1.0 / std::max(C_, 1e-12);
    int n_samples = X.rows();
    int n_features = X.cols();
    MatrixXd X_aug(n_samples, n_features + 1);
    X_aug.leftCols(n_features) = X;
    X_aug.col(n_features) = VectorXd::Ones(n_samples);
    
    MatrixXd XtX = X_aug.transpose() * X_aug;
    MatrixXd reg = MatrixXd::Identity(n_features + 1, n_features + 1);
    reg(n_features, n_features) = 0.0; // Don't regularize intercept
    MatrixXd A = XtX + lambda * reg;
    VectorXd Xty = X_aug.transpose() * y;
    VectorXd coeffs = A.ldlt().solve(Xty);
    
    w_ = coeffs.head(n_features);
    b_ = coeffs(n_features);
    
    fitted_ = true;
    return *this;
}

VectorXd SVR::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SVR must be fitted before predict");
    }
    validation::check_X(X);
    
    if (X.cols() != w_.size()) {
        throw std::runtime_error("X must have the same number of features as training data");
    }
    
    VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = X.row(i).dot(w_) + b_;
    }
    
    return predictions;
}

Params SVR::get_params() const {
    return {
        {"C", std::to_string(C_)},
        {"epsilon", std::to_string(epsilon_)},
        {"max_iter", std::to_string(max_iter_)},
        {"lr", std::to_string(lr_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& SVR::set_params(const Params& params) {
    C_ = utils::get_param_double(params, "C", C_);
    epsilon_ = utils::get_param_double(params, "epsilon", epsilon_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    lr_ = utils::get_param_double(params, "lr", lr_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// LinearSVR implementation
LinearSVR::LinearSVR(double C, double epsilon, int max_iter, double lr, int random_state)
    : C_(C), epsilon_(epsilon), max_iter_(max_iter), lr_(lr), random_state_(random_state) {}

Estimator& LinearSVR::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    if (C_ <= 0.0) {
        throw std::invalid_argument("C must be positive");
    }
    if (epsilon_ < 0.0) {
        throw std::invalid_argument("epsilon must be non-negative");
    }
    
    w_ = VectorXd::Zero(X.cols());
    b_ = 0.0;
    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    
    const double lambda = 1.0 / std::max(C_, 1e-12);
    for (int it = 0; it < max_iter_; ++it) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double lr = lr_ / (1.0 + 0.01 * it);
        for (int idx : indices) {
            double pred = X.row(idx).dot(w_) + b_;
            double error = y(idx) - pred;
            double loss = std::max(0.0, std::abs(error) - epsilon_);
            
            if (loss > 0.0) {
                double sign = (error > 0.0) ? 1.0 : -1.0;
                w_ = (1.0 - lr * lambda) * w_ + lr * sign * X.row(idx).transpose();
                b_ = b_ + lr * sign;
            } else {
                w_ = (1.0 - lr * lambda) * w_;
            }
        }
    }
    
    fitted_ = true;
    return *this;
}

VectorXd LinearSVR::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LinearSVR must be fitted before predict");
    }
    validation::check_X(X);
    
    if (X.cols() != w_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = X.row(i).dot(w_) + b_;
    }
    
    return predictions;
}

Params LinearSVR::get_params() const {
    return {
        {"C", std::to_string(C_)},
        {"epsilon", std::to_string(epsilon_)},
        {"max_iter", std::to_string(max_iter_)},
        {"lr", std::to_string(lr_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& LinearSVR::set_params(const Params& params) {
    C_ = utils::get_param_double(params, "C", C_);
    epsilon_ = utils::get_param_double(params, "epsilon", epsilon_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    lr_ = utils::get_param_double(params, "lr", lr_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// NuSVC implementation
NuSVC::NuSVC(double nu, int max_iter, double lr, int random_state)
    : nu_(nu), max_iter_(max_iter), lr_(lr), random_state_(random_state) {}

Estimator& NuSVC::fit(const MatrixXd& X, const VectorXd& y_in) {
    validation::check_X_y(X, y_in);
    
    if (nu_ <= 0.0 || nu_ > 1.0) {
        throw std::invalid_argument("nu must be in (0, 1]");
    }
    
    // Extract unique classes
    std::set<int> unique_classes;
    for (int i = 0; i < y_in.size(); ++i) {
        unique_classes.insert(static_cast<int>(y_in(i)));
    }
    classes_ = std::vector<int>(unique_classes.begin(), unique_classes.end());
    
    // Convert y to {-1, +1}
    VectorXd y = y_in; 
    for (int i = 0; i < y.size(); ++i) {
        y(i) = (y(i) == classes_[1]) ? 1.0 : -1.0;
    }

    w_ = VectorXd::Zero(X.cols());
    b_ = 0.0;
    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);

    // Nu-SVM uses nu to control the trade-off
    const double lambda = nu_ / (2.0 * X.rows());
    for (int it = 0; it < max_iter_; ++it) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double lr = lr_ / (1.0 + 0.01 * it);
        for (int idx : indices) {
            double margin = y(idx) * (X.row(idx).dot(w_) + b_);
            if (margin < 1.0) {
                w_ = (1.0 - lr * lambda) * w_ + lr * y(idx) * X.row(idx).transpose();
                b_ = b_ + lr * y(idx);
            } else {
                w_ = (1.0 - lr * lambda) * w_;
            }
        }
    }
    fitted_ = true;
    return *this;
}

VectorXi NuSVC::predict_classes(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("NuSVC must be fitted before predict");
    validation::check_X(X);
    
    VectorXi predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(w_) + b_;
        predictions(i) = (score > 0.0) ? classes_[1] : classes_[0];
    }
    return predictions;
}

MatrixXd NuSVC::predict_proba(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("NuSVC must be fitted before predict_proba");
    validation::check_X(X);
    
    MatrixXd probabilities(X.rows(), 2);
    for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(w_) + b_;
        double prob = 1.0 / (1.0 + std::exp(-score));
        probabilities(i, 0) = 1.0 - prob;
        probabilities(i, 1) = prob;
    }
    return probabilities;
}

VectorXd NuSVC::decision_function(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("NuSVC must be fitted before decision_function");
    validation::check_X(X);
    
    VectorXd scores(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        scores(i) = X.row(i).dot(w_) + b_;
    }
    return scores;
}

Params NuSVC::get_params() const {
    return {
        {"nu", std::to_string(nu_)},
        {"max_iter", std::to_string(max_iter_)},
        {"lr", std::to_string(lr_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& NuSVC::set_params(const Params& params) {
    nu_ = utils::get_param_double(params, "nu", nu_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    lr_ = utils::get_param_double(params, "lr", lr_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// NuSVR implementation
NuSVR::NuSVR(double nu, double C, int max_iter, double lr, int random_state)
    : nu_(nu), C_(C), max_iter_(max_iter), lr_(lr), random_state_(random_state) {}

Estimator& NuSVR::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    if (nu_ <= 0.0 || nu_ > 1.0) {
        throw std::invalid_argument("nu must be in (0, 1]");
    }
    if (C_ <= 0.0) {
        throw std::invalid_argument("C must be positive");
    }
    
    w_ = VectorXd::Zero(X.cols());
    b_ = 0.0;
    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Estimate epsilon automatically based on nu
    double epsilon = nu_ * 0.1; // Simple heuristic
    const double lambda = nu_ / (2.0 * C_ * X.rows());
    
    for (int it = 0; it < max_iter_; ++it) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double lr = lr_ / (1.0 + 0.01 * it);
        for (int idx : indices) {
            double pred = X.row(idx).dot(w_) + b_;
            double error = y(idx) - pred;
            double loss = std::max(0.0, std::abs(error) - epsilon);
            
            if (loss > 0.0) {
                double sign = (error > 0.0) ? 1.0 : -1.0;
                w_ = (1.0 - lr * lambda) * w_ + lr * sign * X.row(idx).transpose();
                b_ = b_ + lr * sign;
            } else {
                w_ = (1.0 - lr * lambda) * w_;
            }
        }
    }
    
    fitted_ = true;
    return *this;
}

VectorXd NuSVR::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("NuSVR must be fitted before predict");
    }
    validation::check_X(X);
    
    if (X.cols() != w_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = X.row(i).dot(w_) + b_;
    }
    
    return predictions;
}

Params NuSVR::get_params() const {
    return {
        {"nu", std::to_string(nu_)},
        {"C", std::to_string(C_)},
        {"max_iter", std::to_string(max_iter_)},
        {"lr", std::to_string(lr_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& NuSVR::set_params(const Params& params) {
    nu_ = utils::get_param_double(params, "nu", nu_);
    C_ = utils::get_param_double(params, "C", C_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    lr_ = utils::get_param_double(params, "lr", lr_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// OneClassSVM implementation
OneClassSVM::OneClassSVM(double nu, double gamma, const std::string& kernel,
                         int max_iter, double lr, int random_state)
    : nu_(nu), gamma_(gamma), max_iter_(max_iter), lr_(lr), random_state_(random_state) {
    if (kernel == "linear") {
        kernel_ = Kernel::LINEAR;
    } else if (kernel == "rbf") {
        kernel_ = Kernel::RBF;
    } else {
        kernel_ = Kernel::RBF;  // default to RBF
    }
}

double OneClassSVM::linear_kernel(const VectorXd& x1, const VectorXd& x2) const {
    return x1.dot(x2);
}

double OneClassSVM::rbf_kernel(const VectorXd& x1, const VectorXd& x2) const {
    double diff_norm = (x1 - x2).squaredNorm();
    return std::exp(-gamma_ * diff_norm);
}

double OneClassSVM::kernel_function(const VectorXd& x1, const VectorXd& x2) const {
    switch (kernel_) {
        case Kernel::LINEAR:
            return linear_kernel(x1, x2);
        case Kernel::RBF:
            return rbf_kernel(x1, x2);
        default:
            return rbf_kernel(x1, x2);
    }
}

Estimator& OneClassSVM::fit(const MatrixXd& X, const VectorXd& y) {
    int n_samples = X.rows();
    int n_features = X.cols();
    
    // Initialize random number generator
    std::mt19937 rng;
    if (random_state_ >= 0) {
        rng.seed(static_cast<unsigned>(random_state_));
    } else {
        rng.seed(std::random_device{}());
    }
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    // Initialize parameters for linear kernel case
    if (kernel_ == Kernel::LINEAR) {
        // Simple approach: use data statistics
        // Calculate mean of data as the center
        center_ = X.colwise().mean();
        
        // Calculate distances from center
        std::vector<double> distances;
        for (int i = 0; i < n_samples; ++i) {
            double dist = (X.row(i).transpose() - center_).norm();
            distances.push_back(dist);
        }
        
        // Set threshold as quantile of distances
        std::sort(distances.begin(), distances.end());
        int threshold_idx = static_cast<int>((1.0 - nu_) * distances.size());
        threshold_idx = std::max(0, std::min(threshold_idx, static_cast<int>(distances.size()) - 1));
        rho_ = distances[threshold_idx];
    } else {
        // For RBF kernel, use a simplified approach
        // Store a subset of training data as support vectors
        int n_support = std::min(n_samples, static_cast<int>(n_samples * nu_));
        
        // Simple random selection of support vectors
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        // Use first n_support samples as support vectors
        support_vectors_.resize(n_support, n_features);
        for (int i = 0; i < n_support; ++i) {
            support_vectors_.row(i) = X.row(indices[i]);
        }
        
        // Estimate threshold based on kernel values
        std::vector<double> scores;
        for (int i = 0; i < n_samples; ++i) {
            double score = 0.0;
            for (int j = 0; j < n_support; ++j) {
                score += kernel_function(X.row(i), support_vectors_.row(j));
            }
            scores.push_back(score / n_support);
        }
        
        // Set threshold as quantile (nu is the expected outlier fraction)
        std::sort(scores.begin(), scores.end());
        int threshold_idx = static_cast<int>(nu_ * scores.size());
        rho_ = scores[std::max(0, std::min(threshold_idx, static_cast<int>(scores.size()) - 1))];
    }
    
    fitted_ = true;
    return *this;
}

VectorXi OneClassSVM::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneClassSVM must be fitted before prediction");
    }
    
    VectorXd scores = decision_function(X);
    VectorXi predictions(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = scores(i) >= 0 ? 1 : -1;  // +1 for inliers, -1 for outliers
    }
    
    return predictions;
}

VectorXd OneClassSVM::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneClassSVM must be fitted before prediction");
    }
    
    VectorXd scores(X.rows());
    
    if (kernel_ == Kernel::LINEAR) {
        // For linear kernel, use distance from center approach
        for (int i = 0; i < X.rows(); ++i) {
            double dist = (X.row(i).transpose() - center_).norm();
            scores(i) = rho_ - dist;  // Positive if within threshold
        }
    } else {
        // RBF kernel case
        for (int i = 0; i < X.rows(); ++i) {
            double score = 0.0;
            for (int j = 0; j < support_vectors_.rows(); ++j) {
                score += kernel_function(X.row(i), support_vectors_.row(j));
            }
            scores(i) = score / support_vectors_.rows() - rho_;
        }
    }
    
    return scores;
}

VectorXd OneClassSVM::score_samples(const MatrixXd& X) const {
    return decision_function(X);  // For one-class SVM, scores are the same as decision function
}

Params OneClassSVM::get_params() const {
    std::string kernel_str = (kernel_ == Kernel::LINEAR) ? "linear" : "rbf";
    return {
        {"nu", std::to_string(nu_)},
        {"gamma", std::to_string(gamma_)},
        {"kernel", kernel_str},
        {"max_iter", std::to_string(max_iter_)},
        {"lr", std::to_string(lr_)},
        {"random_state", std::to_string(random_state_)}
    };
}

Estimator& OneClassSVM::set_params(const Params& params) {
    nu_ = utils::get_param_double(params, "nu", nu_);
    gamma_ = utils::get_param_double(params, "gamma", gamma_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    lr_ = utils::get_param_double(params, "lr", lr_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    
    // Handle kernel parameter
    auto it = params.find("kernel");
    if (it != params.end()) {
        if (it->second == "linear") {
            kernel_ = Kernel::LINEAR;
        } else if (it->second == "rbf") {
            kernel_ = Kernel::RBF;
        }
    }
    
    return *this;
}

} // namespace svm
} // namespace auroraml

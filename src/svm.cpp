#include "auroraml/svm.hpp"
#include "auroraml/base.hpp"
#include <random>

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
    std::uniform_int_distribution<int> uni(0, X.rows()-1);

    const double lambda = 1.0 / std::max(C_, 1e-12);
    for (int it = 0; it < max_iter_; ++it) {
        int i = uni(rng);
        double margin = y(i) * (X.row(i).dot(w_) + b_);
        if (margin < 1.0) {
            // gradient: w <- (1 - lr*lambda)w + lr*y_i*x_i; b <- b + lr*y_i
            w_ = (1.0 - lr_ * lambda) * w_ + lr_ * y(i) * X.row(i).transpose();
            b_ = b_ + lr_ * y(i);
        } else {
            w_ = (1.0 - lr_ * lambda) * w_;
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

} // namespace svm
} // namespace cxml



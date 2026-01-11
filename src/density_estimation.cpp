#include "ingenuityml/density_estimation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ingenuityml {
namespace density {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

} // namespace

KernelDensity::KernelDensity(double bandwidth, const std::string& kernel)
    : bandwidth_(bandwidth),
      kernel_(kernel),
      fitted_(false) {}

Estimator& KernelDensity::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;

    if (bandwidth_ <= 0.0) {
        throw std::invalid_argument("bandwidth must be positive");
    }
    X_train_ = X;
    fitted_ = true;
    return *this;
}

VectorXd KernelDensity::score_samples(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KernelDensity must be fitted before score_samples");
    }
    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    if (kernel_ != "gaussian") {
        throw std::invalid_argument("Only gaussian kernel is supported");
    }

    int n_train = X_train_.rows();
    int n_features = X_train_.cols();
    double bw2 = bandwidth_ * bandwidth_;
    double log_norm = -0.5 * n_features * std::log(kTwoPi) - n_features * std::log(bandwidth_) - std::log(static_cast<double>(n_train));

    VectorXd log_density(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        double max_val = -std::numeric_limits<double>::infinity();
        std::vector<double> vals;
        vals.reserve(n_train);
        for (int j = 0; j < n_train; ++j) {
            double dist = (X.row(i) - X_train_.row(j)).squaredNorm();
            double val = -0.5 * dist / bw2;
            vals.push_back(val);
            if (val > max_val) {
                max_val = val;
            }
        }

        double sum_exp = 0.0;
        for (double val : vals) {
            sum_exp += std::exp(val - max_val);
        }
        log_density(i) = log_norm + max_val + std::log(std::max(sum_exp, 1e-12));
    }
    return log_density;
}

double KernelDensity::score(const MatrixXd& X) const {
    VectorXd log_density = score_samples(X);
    return log_density.mean();
}

Params KernelDensity::get_params() const {
    Params params;
    params["bandwidth"] = std::to_string(bandwidth_);
    params["kernel"] = kernel_;
    return params;
}

Estimator& KernelDensity::set_params(const Params& params) {
    bandwidth_ = utils::get_param_double(params, "bandwidth", bandwidth_);
    kernel_ = utils::get_param_string(params, "kernel", kernel_);
    return *this;
}

} // namespace density
} // namespace ingenuityml

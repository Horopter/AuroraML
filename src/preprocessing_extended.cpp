#include "auroraml/preprocessing_extended.hpp"
#include "auroraml/base.hpp"

namespace auroraml {
namespace preprocessing {

// MaxAbsScaler implementation

MaxAbsScaler::MaxAbsScaler() : fitted_(false) {
}

Estimator& MaxAbsScaler::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    
    max_abs_ = X.cwiseAbs().colwise().maxCoeff();
    
    // Handle zero max_abs
    for (int i = 0; i < max_abs_.size(); ++i) {
        if (max_abs_(i) == 0.0) {
            max_abs_(i) = 1.0;
        }
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd MaxAbsScaler::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MaxAbsScaler must be fitted before transform");
    }
    
    MatrixXd X_scaled = X;
    for (int j = 0; j < X.cols(); ++j) {
        X_scaled.col(j) /= max_abs_(j);
    }
    
    return X_scaled;
}

MatrixXd MaxAbsScaler::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MaxAbsScaler must be fitted before inverse_transform");
    }
    
    MatrixXd X_original = X;
    for (int j = 0; j < X.cols(); ++j) {
        X_original.col(j) *= max_abs_(j);
    }
    
    return X_original;
}

MatrixXd MaxAbsScaler::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params MaxAbsScaler::get_params() const {
    return Params();
}

Estimator& MaxAbsScaler::set_params(const Params& params) {
    return *this;
}

// Binarizer implementation

Binarizer::Binarizer(double threshold) : threshold_(threshold), fitted_(false) {
}

Estimator& Binarizer::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    fitted_ = true;
    return *this;
}

MatrixXd Binarizer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Binarizer must be fitted before transform");
    }
    
    MatrixXd X_binary = X;
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            X_binary(i, j) = (X(i, j) > threshold_) ? 1.0 : 0.0;
        }
    }
    
    return X_binary;
}

MatrixXd Binarizer::inverse_transform(const MatrixXd& X) const {
    // Inverse transform not meaningful for binarizer
    return X;
}

MatrixXd Binarizer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params Binarizer::get_params() const {
    Params params;
    params["threshold"] = std::to_string(threshold_);
    return params;
}

Estimator& Binarizer::set_params(const Params& params) {
    threshold_ = utils::get_param_double(params, "threshold", threshold_);
    return *this;
}

} // namespace preprocessing
} // namespace auroraml


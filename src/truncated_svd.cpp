#include "ingenuityml/truncated_svd.hpp"
#include "ingenuityml/base.hpp"
#include <Eigen/SVD>
#include <stdexcept>

namespace ingenuityml {
namespace decomposition {

Estimator& TruncatedSVD::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    if (n_components_ <= 0 || n_components_ > std::min(X.rows(), X.cols())) {
        throw std::invalid_argument("n_components must be in (0, min(n_samples, n_features)]");
    }
    Eigen::BDCSVD<MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    singular_values_ = svd.singularValues().head(n_components_);
    components_ = svd.matrixV().leftCols(n_components_).transpose();
    
    // Calculate explained variance (singular values squared)
    explained_variance_ = singular_values_.array().square();
    
    fitted_ = true;
    return *this;
}

MatrixXd TruncatedSVD::transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("TruncatedSVD must be fitted before transform");
    validation::check_X(X);
    if (X.cols() != components_.cols()) throw std::runtime_error("Feature mismatch in TruncatedSVD::transform");
    return X * components_.transpose();
}

MatrixXd TruncatedSVD::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("TruncatedSVD must be fitted before inverse_transform");
    validation::check_X(X);
    if (X.cols() != components_.rows()) throw std::runtime_error("Component mismatch in TruncatedSVD::inverse_transform");
    return X * components_;
}

MatrixXd TruncatedSVD::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Estimator& TruncatedSVD::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    return *this;
}

} // namespace decomposition
} // namespace ingenuityml


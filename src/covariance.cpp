#include "auroraml/covariance.hpp"
#include "auroraml/utils.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace auroraml {
namespace covariance {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

void compute_location_covariance(const MatrixXd& X, bool assume_centered,
                                 VectorXd& location, MatrixXd& covariance) {
    validation::check_X(X);
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (assume_centered) {
        location = VectorXd::Zero(X.cols());
        covariance = (X.transpose() * X) / static_cast<double>(X.rows());
        return;
    }

    location = X.colwise().mean();
    MatrixXd Xc = X.rowwise() - location.transpose();
    covariance = (Xc.transpose() * Xc) / static_cast<double>(X.rows());
}

MatrixXd compute_precision(const MatrixXd& covariance) {
    if (covariance.rows() != covariance.cols()) {
        throw std::invalid_argument("Covariance must be a square matrix");
    }
    const int n = covariance.rows();
    MatrixXd identity = MatrixXd::Identity(n, n);
    Eigen::LDLT<MatrixXd> ldlt(covariance);
    if (ldlt.info() == Eigen::Success) {
        MatrixXd inv = ldlt.solve(identity);
        if (inv.allFinite()) {
            return inv;
        }
    }
    Eigen::CompleteOrthogonalDecomposition<MatrixXd> cod(covariance);
    return cod.pseudoInverse();
}

VectorXd compute_mahalanobis(const MatrixXd& X, const VectorXd& location, const MatrixXd& precision) {
    if (X.cols() != location.size()) {
        throw std::invalid_argument("X must have the same number of features as the fitted data");
    }
    VectorXd distances(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        VectorXd diff = X.row(i).transpose() - location;
        distances(i) = diff.transpose() * precision * diff;
    }
    return distances;
}

MatrixXd shrink_covariance(const MatrixXd& covariance, double shrinkage) {
    if (shrinkage < 0.0 || shrinkage > 1.0) {
        throw std::invalid_argument("shrinkage must be between 0 and 1");
    }
    const int p = covariance.rows();
    double mu = covariance.trace() / static_cast<double>(p);
    MatrixXd identity = MatrixXd::Identity(p, p);
    return (1.0 - shrinkage) * covariance + shrinkage * mu * identity;
}

double compute_log_det(const MatrixXd& covariance) {
    if (covariance.rows() != covariance.cols()) {
        throw std::invalid_argument("Covariance must be a square matrix");
    }
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(covariance);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Failed to compute eigenvalues for log determinant");
    }
    const double eps = 1e-12;
    double log_det = 0.0;
    for (int i = 0; i < es.eigenvalues().size(); ++i) {
        double val = std::max(es.eigenvalues()(i), eps);
        log_det += std::log(val);
    }
    return log_det;
}

double ledoit_wolf_shrinkage(const MatrixXd& X, bool assume_centered, VectorXd& location, MatrixXd& covariance) {
    compute_location_covariance(X, assume_centered, location, covariance);
    const int n_samples = X.rows();
    const int n_features = X.cols();

    MatrixXd Xc = assume_centered ? X : (X.rowwise() - location.transpose());
    MatrixXd sample_cov = covariance;
    double mu = sample_cov.trace() / static_cast<double>(n_features);

    MatrixXd X2 = Xc.array().square().matrix();
    MatrixXd beta_matrix = (X2.transpose() * X2) / static_cast<double>(n_samples) - sample_cov.array().square().matrix();
    double beta = beta_matrix.sum();

    MatrixXd diff = sample_cov - mu * MatrixXd::Identity(n_features, n_features);
    double delta = diff.array().square().sum();

    if (delta <= 0.0) {
        return 0.0;
    }
    double shrinkage = std::min(beta / delta, 1.0);
    if (shrinkage < 0.0) {
        shrinkage = 0.0;
    }
    return shrinkage;
}

double oas_shrinkage(const MatrixXd& covariance, int n_samples) {
    const int p = covariance.rows();
    double trace = covariance.trace();
    double trace2 = (covariance * covariance).trace();

    double mu = trace / static_cast<double>(p);
    double numerator = (1.0 - 2.0 / static_cast<double>(p)) * trace2 + trace * trace;
    double denominator = (n_samples + 1.0 - 2.0 / static_cast<double>(p)) * (trace2 - (trace * trace) / static_cast<double>(p));

    if (denominator <= 0.0) {
        return 0.0;
    }

    double shrinkage = numerator / denominator;
    if (shrinkage < 0.0) {
        shrinkage = 0.0;
    } else if (shrinkage > 1.0) {
        shrinkage = 1.0;
    }
    return shrinkage;
}

double select_quantile(std::vector<double>& values, double quantile) {
    if (values.empty()) {
        throw std::invalid_argument("values cannot be empty");
    }
    if (quantile < 0.0 || quantile > 1.0) {
        throw std::invalid_argument("quantile must be between 0 and 1");
    }
    size_t idx = static_cast<size_t>(std::floor(quantile * (values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

} // namespace

EmpiricalCovariance::EmpiricalCovariance(bool assume_centered)
    : assume_centered_(assume_centered), fitted_(false), covariance_(), location_(), precision_() {}

Estimator& EmpiricalCovariance::fit(const MatrixXd& X, const VectorXd& y) {
    compute_location_covariance(X, assume_centered_, location_, covariance_);
    precision_ = compute_precision(covariance_);
    fitted_ = true;
    return *this;
}

VectorXd EmpiricalCovariance::mahalanobis(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("EmpiricalCovariance must be fitted before calling mahalanobis");
    }
    return compute_mahalanobis(X, location_, precision_);
}

VectorXd EmpiricalCovariance::score_samples(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("EmpiricalCovariance must be fitted before calling score_samples");
    }
    VectorXd distances = mahalanobis(X);
    double log_det = compute_log_det(covariance_);
    double norm = static_cast<double>(covariance_.rows()) * std::log(kTwoPi) + log_det;
    return (-0.5 * (distances.array() + norm)).matrix();
}

Params EmpiricalCovariance::get_params() const {
    Params params;
    params["assume_centered"] = assume_centered_ ? "true" : "false";
    return params;
}

Estimator& EmpiricalCovariance::set_params(const Params& params) {
    assume_centered_ = utils::get_param_bool(params, "assume_centered", assume_centered_);
    return *this;
}

const MatrixXd& EmpiricalCovariance::covariance() const {
    if (!fitted_) {
        throw std::runtime_error("EmpiricalCovariance must be fitted before accessing covariance");
    }
    return covariance_;
}

const VectorXd& EmpiricalCovariance::location() const {
    if (!fitted_) {
        throw std::runtime_error("EmpiricalCovariance must be fitted before accessing location");
    }
    return location_;
}

const MatrixXd& EmpiricalCovariance::precision() const {
    if (!fitted_) {
        throw std::runtime_error("EmpiricalCovariance must be fitted before accessing precision");
    }
    return precision_;
}

ShrunkCovariance::ShrunkCovariance(double shrinkage, bool assume_centered)
    : shrinkage_(shrinkage), assume_centered_(assume_centered), fitted_(false), covariance_(), location_(), precision_() {}

Estimator& ShrunkCovariance::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd empirical;
    compute_location_covariance(X, assume_centered_, location_, empirical);
    covariance_ = shrink_covariance(empirical, shrinkage_);
    precision_ = compute_precision(covariance_);
    fitted_ = true;
    return *this;
}

VectorXd ShrunkCovariance::mahalanobis(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ShrunkCovariance must be fitted before calling mahalanobis");
    }
    return compute_mahalanobis(X, location_, precision_);
}

VectorXd ShrunkCovariance::score_samples(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ShrunkCovariance must be fitted before calling score_samples");
    }
    VectorXd distances = mahalanobis(X);
    double log_det = compute_log_det(covariance_);
    double norm = static_cast<double>(covariance_.rows()) * std::log(kTwoPi) + log_det;
    return (-0.5 * (distances.array() + norm)).matrix();
}

Params ShrunkCovariance::get_params() const {
    Params params;
    params["shrinkage"] = std::to_string(shrinkage_);
    params["assume_centered"] = assume_centered_ ? "true" : "false";
    return params;
}

Estimator& ShrunkCovariance::set_params(const Params& params) {
    shrinkage_ = utils::get_param_double(params, "shrinkage", shrinkage_);
    assume_centered_ = utils::get_param_bool(params, "assume_centered", assume_centered_);
    return *this;
}

const MatrixXd& ShrunkCovariance::covariance() const {
    if (!fitted_) {
        throw std::runtime_error("ShrunkCovariance must be fitted before accessing covariance");
    }
    return covariance_;
}

const VectorXd& ShrunkCovariance::location() const {
    if (!fitted_) {
        throw std::runtime_error("ShrunkCovariance must be fitted before accessing location");
    }
    return location_;
}

const MatrixXd& ShrunkCovariance::precision() const {
    if (!fitted_) {
        throw std::runtime_error("ShrunkCovariance must be fitted before accessing precision");
    }
    return precision_;
}

LedoitWolf::LedoitWolf(bool assume_centered)
    : assume_centered_(assume_centered), fitted_(false), shrinkage_(0.0), covariance_(), location_(), precision_() {}

Estimator& LedoitWolf::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd empirical;
    shrinkage_ = ledoit_wolf_shrinkage(X, assume_centered_, location_, empirical);
    covariance_ = shrink_covariance(empirical, shrinkage_);
    precision_ = compute_precision(covariance_);
    fitted_ = true;
    return *this;
}

VectorXd LedoitWolf::mahalanobis(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LedoitWolf must be fitted before calling mahalanobis");
    }
    return compute_mahalanobis(X, location_, precision_);
}

VectorXd LedoitWolf::score_samples(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LedoitWolf must be fitted before calling score_samples");
    }
    VectorXd distances = mahalanobis(X);
    double log_det = compute_log_det(covariance_);
    double norm = static_cast<double>(covariance_.rows()) * std::log(kTwoPi) + log_det;
    return (-0.5 * (distances.array() + norm)).matrix();
}

Params LedoitWolf::get_params() const {
    Params params;
    params["assume_centered"] = assume_centered_ ? "true" : "false";
    return params;
}

Estimator& LedoitWolf::set_params(const Params& params) {
    assume_centered_ = utils::get_param_bool(params, "assume_centered", assume_centered_);
    return *this;
}

const MatrixXd& LedoitWolf::covariance() const {
    if (!fitted_) {
        throw std::runtime_error("LedoitWolf must be fitted before accessing covariance");
    }
    return covariance_;
}

const VectorXd& LedoitWolf::location() const {
    if (!fitted_) {
        throw std::runtime_error("LedoitWolf must be fitted before accessing location");
    }
    return location_;
}

const MatrixXd& LedoitWolf::precision() const {
    if (!fitted_) {
        throw std::runtime_error("LedoitWolf must be fitted before accessing precision");
    }
    return precision_;
}

OAS::OAS(bool assume_centered)
    : assume_centered_(assume_centered), fitted_(false), shrinkage_(0.0), covariance_(), location_(), precision_() {}

Estimator& OAS::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd empirical;
    compute_location_covariance(X, assume_centered_, location_, empirical);
    shrinkage_ = oas_shrinkage(empirical, X.rows());
    covariance_ = shrink_covariance(empirical, shrinkage_);
    precision_ = compute_precision(covariance_);
    fitted_ = true;
    return *this;
}

VectorXd OAS::mahalanobis(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OAS must be fitted before calling mahalanobis");
    }
    return compute_mahalanobis(X, location_, precision_);
}

VectorXd OAS::score_samples(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OAS must be fitted before calling score_samples");
    }
    VectorXd distances = mahalanobis(X);
    double log_det = compute_log_det(covariance_);
    double norm = static_cast<double>(covariance_.rows()) * std::log(kTwoPi) + log_det;
    return (-0.5 * (distances.array() + norm)).matrix();
}

Params OAS::get_params() const {
    Params params;
    params["assume_centered"] = assume_centered_ ? "true" : "false";
    return params;
}

Estimator& OAS::set_params(const Params& params) {
    assume_centered_ = utils::get_param_bool(params, "assume_centered", assume_centered_);
    return *this;
}

const MatrixXd& OAS::covariance() const {
    if (!fitted_) {
        throw std::runtime_error("OAS must be fitted before accessing covariance");
    }
    return covariance_;
}

const VectorXd& OAS::location() const {
    if (!fitted_) {
        throw std::runtime_error("OAS must be fitted before accessing location");
    }
    return location_;
}

const MatrixXd& OAS::precision() const {
    if (!fitted_) {
        throw std::runtime_error("OAS must be fitted before accessing precision");
    }
    return precision_;
}

MinCovDet::MinCovDet(double support_fraction, bool assume_centered, int max_iter, double tol, int random_state)
    : support_fraction_(support_fraction),
      assume_centered_(assume_centered),
      max_iter_(max_iter),
      tol_(tol),
      random_state_(random_state),
      fitted_(false),
      covariance_(),
      location_(),
      precision_(),
      support_() {}

Estimator& MinCovDet::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    const int n_samples = X.rows();
    const int n_features = X.cols();
    if (n_samples == 0 || n_features == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (support_fraction_ <= 0.0 || support_fraction_ > 1.0) {
        throw std::invalid_argument("support_fraction must be in (0, 1]");
    }

    int h = static_cast<int>(std::floor(support_fraction_ * n_samples));
    int min_h = std::min(n_samples, n_features + 1);
    if (h < min_h) {
        h = min_h;
    }

    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<int> support_indices(indices.begin(), indices.begin() + h);
    std::vector<int> prev_support;

    VectorXd current_location;
    MatrixXd current_covariance;

    auto compute_subset = [&](const std::vector<int>& subset, VectorXd& loc, MatrixXd& cov) {
        MatrixXd X_subset(subset.size(), n_features);
        for (size_t i = 0; i < subset.size(); ++i) {
            X_subset.row(i) = X.row(subset[i]);
        }
        compute_location_covariance(X_subset, assume_centered_, loc, cov);
    };

    compute_subset(support_indices, current_location, current_covariance);

    for (int iter = 0; iter < max_iter_; ++iter) {
        MatrixXd current_precision = compute_precision(current_covariance);
        VectorXd distances = compute_mahalanobis(X, current_location, current_precision);

        std::vector<int> order(n_samples);
        std::iota(order.begin(), order.end(), 0);
        std::nth_element(order.begin(), order.begin() + h, order.end(),
                         [&](int a, int b) { return distances(a) < distances(b); });
        order.resize(h);
        std::sort(order.begin(), order.end());

        if (!prev_support.empty() && order == prev_support) {
            break;
        }

        VectorXd new_location;
        MatrixXd new_covariance;
        compute_subset(order, new_location, new_covariance);

        double loc_diff = (new_location - current_location).norm();
        double cov_norm = std::max(current_covariance.norm(), 1e-12);
        double cov_diff = (new_covariance - current_covariance).norm() / cov_norm;

        current_location = new_location;
        current_covariance = new_covariance;
        prev_support = order;

        if (loc_diff < tol_ && cov_diff < tol_) {
            break;
        }
    }

    location_ = current_location;
    covariance_ = current_covariance;
    precision_ = compute_precision(covariance_);

    support_ = VectorXi::Zero(n_samples);
    for (int idx : prev_support) {
        support_(idx) = 1;
    }

    fitted_ = true;
    return *this;
}

VectorXd MinCovDet::mahalanobis(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MinCovDet must be fitted before calling mahalanobis");
    }
    return compute_mahalanobis(X, location_, precision_);
}

VectorXd MinCovDet::score_samples(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MinCovDet must be fitted before calling score_samples");
    }
    VectorXd distances = mahalanobis(X);
    double log_det = compute_log_det(covariance_);
    double norm = static_cast<double>(covariance_.rows()) * std::log(kTwoPi) + log_det;
    return (-0.5 * (distances.array() + norm)).matrix();
}

Params MinCovDet::get_params() const {
    Params params;
    params["support_fraction"] = std::to_string(support_fraction_);
    params["assume_centered"] = assume_centered_ ? "true" : "false";
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& MinCovDet::set_params(const Params& params) {
    support_fraction_ = utils::get_param_double(params, "support_fraction", support_fraction_);
    assume_centered_ = utils::get_param_bool(params, "assume_centered", assume_centered_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

const MatrixXd& MinCovDet::covariance() const {
    if (!fitted_) {
        throw std::runtime_error("MinCovDet must be fitted before accessing covariance");
    }
    return covariance_;
}

const VectorXd& MinCovDet::location() const {
    if (!fitted_) {
        throw std::runtime_error("MinCovDet must be fitted before accessing location");
    }
    return location_;
}

const MatrixXd& MinCovDet::precision() const {
    if (!fitted_) {
        throw std::runtime_error("MinCovDet must be fitted before accessing precision");
    }
    return precision_;
}

const VectorXi& MinCovDet::support() const {
    if (!fitted_) {
        throw std::runtime_error("MinCovDet must be fitted before accessing support");
    }
    return support_;
}

EllipticEnvelope::EllipticEnvelope(double contamination, double support_fraction,
                                   int max_iter, double tol, int random_state)
    : contamination_(contamination),
      mcd_(support_fraction, false, max_iter, tol, random_state),
      fitted_(false),
      threshold_(0.0) {
    if (contamination_ <= 0.0 || contamination_ >= 0.5) {
        throw std::invalid_argument("contamination must be in (0, 0.5)");
    }
}

Estimator& EllipticEnvelope::fit(const MatrixXd& X, const VectorXd& y) {
    mcd_.fit(X, y);
    VectorXd distances = mcd_.mahalanobis(X);
    std::vector<double> dist_vec(distances.data(), distances.data() + distances.size());
    threshold_ = select_quantile(dist_vec, 1.0 - contamination_);
    fitted_ = true;
    return *this;
}

VectorXi EllipticEnvelope::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("EllipticEnvelope must be fitted before predict");
    }
    VectorXd distances = mcd_.mahalanobis(X);
    VectorXi labels(X.rows());
    for (int i = 0; i < distances.size(); ++i) {
        labels(i) = (distances(i) <= threshold_) ? 1 : -1;
    }
    return labels;
}

VectorXd EllipticEnvelope::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("EllipticEnvelope must be fitted before decision_function");
    }
    VectorXd distances = mcd_.mahalanobis(X);
    return (threshold_ - distances.array()).matrix();
}

VectorXd EllipticEnvelope::score_samples(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("EllipticEnvelope must be fitted before score_samples");
    }
    VectorXd distances = mcd_.mahalanobis(X);
    return (-distances.array()).matrix();
}

Params EllipticEnvelope::get_params() const {
    Params params = mcd_.get_params();
    params["contamination"] = std::to_string(contamination_);
    return params;
}

Estimator& EllipticEnvelope::set_params(const Params& params) {
    contamination_ = utils::get_param_double(params, "contamination", contamination_);
    mcd_.set_params(params);
    return *this;
}

} // namespace covariance
} // namespace auroraml

#include "ingenuityml/preprocessing_extended.hpp"
#include "ingenuityml/base.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace ingenuityml {
namespace preprocessing {

namespace {

constexpr double kEps = 1e-12;

double compute_quantile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    if (q <= 0.0) {
        return sorted.front();
    }
    if (q >= 1.0) {
        return sorted.back();
    }
    double pos = q * static_cast<double>(sorted.size() - 1);
    size_t idx = static_cast<size_t>(std::floor(pos));
    size_t idx_next = std::min(idx + 1, sorted.size() - 1);
    double frac = pos - static_cast<double>(idx);
    return sorted[idx] * (1.0 - frac) + sorted[idx_next] * frac;
}

double normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double inverse_normal_cdf(double p) {
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();

    static const double a1 = -39.69683028665376;
    static const double a2 = 220.9460984245205;
    static const double a3 = -275.9285104469687;
    static const double a4 = 138.3577518672690;
    static const double a5 = -30.66479806614716;
    static const double a6 = 2.506628277459239;

    static const double b1 = -54.47609879822406;
    static const double b2 = 161.5858368580409;
    static const double b3 = -155.6989798598866;
    static const double b4 = 66.80131188771972;
    static const double b5 = -13.28068155288572;

    static const double c1 = -0.007784894002430293;
    static const double c2 = -0.3223964580411365;
    static const double c3 = -2.400758277161838;
    static const double c4 = -2.549732539343734;
    static const double c5 = 4.374664141464968;
    static const double c6 = 2.938163982698783;

    static const double d1 = 0.007784695709041462;
    static const double d2 = 0.3224671290700398;
    static const double d3 = 2.445134137142996;
    static const double d4 = 3.754408661907416;

    double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
        double r = 0.180625 - q * q;
        double num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6);
        double den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        return q * num / den;
    }

    double r = (q > 0.0) ? 1.0 - p : p;
    r = std::sqrt(-std::log(r));
    double num = (((((c1 * r + c2) * r + c3) * r + c4) * r + c5) * r + c6);
    double den = ((((d1 * r + d2) * r + d3) * r + d4) * r + 1.0);
    double result = num / den;
    return (q < 0.0) ? -result : result;
}

double yeo_johnson_transform(double x, double lambda) {
    if (x >= 0.0) {
        if (std::abs(lambda) < kEps) {
            return std::log(x + 1.0);
        }
        return (std::pow(x + 1.0, lambda) - 1.0) / lambda;
    }
    if (std::abs(lambda - 2.0) < kEps) {
        return -std::log(-x + 1.0);
    }
    return -(std::pow(-x + 1.0, 2.0 - lambda) - 1.0) / (2.0 - lambda);
}

double yeo_johnson_inverse(double y, double lambda) {
    if (y >= 0.0) {
        if (std::abs(lambda) < kEps) {
            return std::exp(y) - 1.0;
        }
        return std::pow(lambda * y + 1.0, 1.0 / lambda) - 1.0;
    }
    double z = -y;
    if (std::abs(lambda - 2.0) < kEps) {
        return 1.0 - std::exp(z);
    }
    return 1.0 - std::pow((2.0 - lambda) * z + 1.0, 1.0 / (2.0 - lambda));
}

double box_cox_transform(double x, double lambda) {
    if (std::abs(lambda) < kEps) {
        return std::log(x);
    }
    return (std::pow(x, lambda) - 1.0) / lambda;
}

double box_cox_inverse(double y, double lambda) {
    if (std::abs(lambda) < kEps) {
        return std::exp(y);
    }
    return std::pow(lambda * y + 1.0, 1.0 / lambda);
}

MatrixXd apply_function(const MatrixXd& X, const std::string& name) {
    if (name == "identity") {
        return X;
    }
    MatrixXd out = X;
    if (name == "log1p") {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                if (X(i, j) <= -1.0) {
                    throw std::invalid_argument("log1p requires all entries > -1");
                }
                out(i, j) = std::log1p(X(i, j));
            }
        }
        return out;
    }
    if (name == "expm1") {
        return X.array().exp().matrix() - MatrixXd::Ones(X.rows(), X.cols());
    }
    if (name == "log") {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                if (X(i, j) <= 0.0) {
                    throw std::invalid_argument("log requires all entries > 0");
                }
                out(i, j) = std::log(X(i, j));
            }
        }
        return out;
    }
    if (name == "exp") {
        return X.array().exp().matrix();
    }
    if (name == "sqrt") {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                if (X(i, j) < 0.0) {
                    throw std::invalid_argument("sqrt requires all entries >= 0");
                }
                out(i, j) = std::sqrt(X(i, j));
            }
        }
        return out;
    }
    if (name == "square") {
        return X.array().square().matrix();
    }
    if (name == "abs") {
        return X.cwiseAbs();
    }
    if (name == "negative") {
        return -X;
    }
    throw std::invalid_argument("Unsupported function: " + name);
}

double bspline_basis(int i, int degree, double x, const VectorXd& knots) {
    if (degree == 0) {
        bool in_interval = (knots(i) <= x && x < knots(i + 1));
        bool at_end = (x == knots(knots.size() - 1) && i + 1 == knots.size() - 1);
        return (in_interval || at_end) ? 1.0 : 0.0;
    }
    double denom1 = knots(i + degree) - knots(i);
    double denom2 = knots(i + degree + 1) - knots(i + 1);
    double term1 = 0.0;
    double term2 = 0.0;
    if (denom1 > kEps) {
        term1 = (x - knots(i)) / denom1 * bspline_basis(i, degree - 1, x, knots);
    }
    if (denom2 > kEps) {
        term2 = (knots(i + degree + 1) - x) / denom2 * bspline_basis(i + 1, degree - 1, x, knots);
    }
    return term1 + term2;
}

} // namespace

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

// LabelBinarizer implementation

LabelBinarizer::LabelBinarizer(int neg_label, int pos_label)
    : neg_label_(neg_label), pos_label_(pos_label), fitted_(false) {}

Estimator& LabelBinarizer::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    std::set<double> unique_labels;
    for (int i = 0; i < y.size(); ++i) {
        unique_labels.insert(y(i));
    }
    if (unique_labels.empty()) {
        throw std::invalid_argument("LabelBinarizer requires at least one label");
    }
    classes_.resize(unique_labels.size());
    class_to_index_.clear();
    int idx = 0;
    for (double label : unique_labels) {
        classes_(idx) = label;
        class_to_index_[label] = idx;
        idx++;
    }
    fitted_ = true;
    return *this;
}

MatrixXd LabelBinarizer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelBinarizer must be fitted before transform");
    }
    if (X.cols() < 1) {
        throw std::invalid_argument("X must have at least one column");
    }
    VectorXd labels = X.col(0);
    int n_samples = labels.size();
    int n_classes = classes_.size();

    if (n_classes == 2) {
        MatrixXd out = MatrixXd::Constant(n_samples, 1, static_cast<double>(neg_label_));
        for (int i = 0; i < n_samples; ++i) {
            auto it = class_to_index_.find(labels(i));
            if (it == class_to_index_.end()) {
                throw std::invalid_argument("Unknown label");
            }
            if (it->second == 1) {
                out(i, 0) = static_cast<double>(pos_label_);
            }
        }
        return out;
    }

    MatrixXd out = MatrixXd::Constant(n_samples, n_classes, static_cast<double>(neg_label_));
    for (int i = 0; i < n_samples; ++i) {
        auto it = class_to_index_.find(labels(i));
        if (it == class_to_index_.end()) {
            throw std::invalid_argument("Unknown label");
        }
        out(i, it->second) = static_cast<double>(pos_label_);
    }
    return out;
}

MatrixXd LabelBinarizer::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelBinarizer must be fitted before inverse_transform");
    }
    int n_samples = X.rows();
    int n_classes = classes_.size();
    MatrixXd labels(n_samples, 1);

    if (n_classes == 2) {
        double threshold = (static_cast<double>(pos_label_) + static_cast<double>(neg_label_)) / 2.0;
        for (int i = 0; i < n_samples; ++i) {
            labels(i, 0) = (X(i, 0) >= threshold) ? classes_(1) : classes_(0);
        }
        return labels;
    }

    if (X.cols() != n_classes) {
        throw std::invalid_argument("X must have the same number of columns as classes");
    }
    for (int i = 0; i < n_samples; ++i) {
        Eigen::Index max_idx = 0;
        X.row(i).maxCoeff(&max_idx);
        labels(i, 0) = classes_(static_cast<int>(max_idx));
    }
    return labels;
}

MatrixXd LabelBinarizer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    MatrixXd y_mat(y.size(), 1);
    y_mat.col(0) = y;
    return transform(y_mat);
}

Params LabelBinarizer::get_params() const {
    Params params;
    params["neg_label"] = std::to_string(neg_label_);
    params["pos_label"] = std::to_string(pos_label_);
    return params;
}

Estimator& LabelBinarizer::set_params(const Params& params) {
    neg_label_ = utils::get_param_int(params, "neg_label", neg_label_);
    pos_label_ = utils::get_param_int(params, "pos_label", pos_label_);
    return *this;
}

// MultiLabelBinarizer implementation

MultiLabelBinarizer::MultiLabelBinarizer() : fitted_(false) {}

Estimator& MultiLabelBinarizer::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;

    std::set<double> unique_labels;
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            double label = X(i, j);
            if (std::isnan(label)) {
                continue;
            }
            unique_labels.insert(label);
        }
    }
    if (unique_labels.empty()) {
        throw std::invalid_argument("MultiLabelBinarizer requires at least one label");
    }
    classes_.resize(unique_labels.size());
    class_to_index_.clear();
    int idx = 0;
    for (double label : unique_labels) {
        classes_(idx) = label;
        class_to_index_[label] = idx;
        idx++;
    }
    fitted_ = true;
    return *this;
}

MatrixXd MultiLabelBinarizer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultiLabelBinarizer must be fitted before transform");
    }
    if (X.cols() == 0) {
        throw std::invalid_argument("X must have at least one column");
    }
    int n_samples = X.rows();
    int n_classes = classes_.size();
    MatrixXd out = MatrixXd::Zero(n_samples, n_classes);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            double label = X(i, j);
            if (std::isnan(label)) {
                continue;
            }
            auto it = class_to_index_.find(label);
            if (it == class_to_index_.end()) {
                throw std::invalid_argument("Unknown label");
            }
            out(i, it->second) = 1.0;
        }
    }
    return out;
}

MatrixXd MultiLabelBinarizer::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultiLabelBinarizer must be fitted before inverse_transform");
    }
    if (X.cols() != classes_.size()) {
        throw std::invalid_argument("X must have the same number of columns as classes");
    }

    std::vector<std::vector<double>> labels;
    labels.reserve(X.rows());
    size_t max_count = 0;
    for (int i = 0; i < X.rows(); ++i) {
        std::vector<double> row_labels;
        for (int c = 0; c < X.cols(); ++c) {
            if (X(i, c) > 0.5) {
                row_labels.push_back(classes_(c));
            }
        }
        max_count = std::max(max_count, row_labels.size());
        labels.push_back(std::move(row_labels));
    }

    int out_cols = static_cast<int>(std::max<size_t>(1, max_count));
    MatrixXd out = MatrixXd::Constant(X.rows(), out_cols, -1.0);
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
        for (int j = 0; j < static_cast<int>(labels[i].size()); ++j) {
            out(i, j) = labels[i][j];
        }
    }
    return out;
}

MatrixXd MultiLabelBinarizer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params MultiLabelBinarizer::get_params() const {
    return Params();
}

Estimator& MultiLabelBinarizer::set_params(const Params& params) {
    (void)params;
    return *this;
}

// KBinsDiscretizer implementation

KBinsDiscretizer::KBinsDiscretizer(int n_bins, const std::string& encode, const std::string& strategy)
    : n_bins_(n_bins),
      encode_(encode),
      strategy_(strategy),
      fitted_(false),
      n_features_(0),
      output_dim_(0) {}

Estimator& KBinsDiscretizer::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;
    if (n_bins_ <= 1) {
        throw std::invalid_argument("n_bins must be greater than 1");
    }
    if (encode_ != "ordinal" && encode_ != "onehot") {
        throw std::invalid_argument("encode must be 'ordinal' or 'onehot'");
    }
    if (strategy_ != "uniform" && strategy_ != "quantile") {
        throw std::invalid_argument("strategy must be 'uniform' or 'quantile'");
    }

    n_features_ = X.cols();
    bin_edges_.clear();
    bin_edges_.resize(n_features_);

    for (int j = 0; j < n_features_; ++j) {
        VectorXd col = X.col(j);
        double min_val = col.minCoeff();
        double max_val = col.maxCoeff();
        VectorXd edges(n_bins_ + 1);

        if (strategy_ == "uniform") {
            if (max_val <= min_val) {
                edges.setConstant(min_val);
            } else {
                for (int k = 0; k <= n_bins_; ++k) {
                    edges(k) = min_val + (max_val - min_val) * static_cast<double>(k) / n_bins_;
                }
            }
        } else {
            std::vector<double> values(col.data(), col.data() + col.size());
            std::sort(values.begin(), values.end());
            for (int k = 0; k <= n_bins_; ++k) {
                double q = static_cast<double>(k) / n_bins_;
                edges(k) = compute_quantile(values, q);
            }
        }
        bin_edges_[j] = edges;
    }

    output_dim_ = (encode_ == "onehot") ? n_features_ * n_bins_ : n_features_;
    fitted_ = true;
    return *this;
}

MatrixXd KBinsDiscretizer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KBinsDiscretizer must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    int n_samples = X.rows();
    if (encode_ == "ordinal") {
        MatrixXd out(n_samples, n_features_);
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features_; ++j) {
                const VectorXd& edges = bin_edges_[j];
                int n_edges = edges.size();
                int bin = 0;
                if (n_edges > 1) {
                    const double* data = edges.data();
                    auto it = std::upper_bound(data, data + n_edges, X(i, j));
                    int idx = static_cast<int>(it - data) - 1;
                    if (idx < 0) idx = 0;
                    if (idx >= n_bins_) idx = n_bins_ - 1;
                    bin = idx;
                }
                out(i, j) = static_cast<double>(bin);
            }
        }
        return out;
    }

    MatrixXd out = MatrixXd::Zero(n_samples, output_dim_);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features_; ++j) {
            const VectorXd& edges = bin_edges_[j];
            int n_edges = edges.size();
            int bin = 0;
            if (n_edges > 1) {
                const double* data = edges.data();
                auto it = std::upper_bound(data, data + n_edges, X(i, j));
                int idx = static_cast<int>(it - data) - 1;
                if (idx < 0) idx = 0;
                if (idx >= n_bins_) idx = n_bins_ - 1;
                bin = idx;
            }
            int col = j * n_bins_ + bin;
            out(i, col) = 1.0;
        }
    }
    return out;
}

MatrixXd KBinsDiscretizer::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("KBinsDiscretizer must be fitted before inverse_transform");
    }

    int n_samples = X.rows();
    MatrixXd out(n_samples, n_features_);

    if (encode_ == "ordinal") {
        if (X.cols() != n_features_) {
            throw std::invalid_argument("X must have the same number of features as training data");
        }
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features_; ++j) {
                const VectorXd& edges = bin_edges_[j];
                int bin = static_cast<int>(std::round(X(i, j)));
                bin = std::max(0, std::min(n_bins_ - 1, bin));
                if (edges.size() > 1) {
                    out(i, j) = 0.5 * (edges(bin) + edges(bin + 1));
                } else {
                    out(i, j) = edges(0);
                }
            }
        }
        return out;
    }

    if (X.cols() != output_dim_) {
        throw std::invalid_argument("X must have the same number of features as transformed data");
    }
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features_; ++j) {
            int offset = j * n_bins_;
            Eigen::Index max_idx = 0;
            X.row(i).segment(offset, n_bins_).maxCoeff(&max_idx);
            int bin = static_cast<int>(max_idx);
            const VectorXd& edges = bin_edges_[j];
            if (edges.size() > 1) {
                out(i, j) = 0.5 * (edges(bin) + edges(bin + 1));
            } else {
                out(i, j) = edges(0);
            }
        }
    }
    return out;
}

MatrixXd KBinsDiscretizer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params KBinsDiscretizer::get_params() const {
    Params params;
    params["n_bins"] = std::to_string(n_bins_);
    params["encode"] = encode_;
    params["strategy"] = strategy_;
    return params;
}

Estimator& KBinsDiscretizer::set_params(const Params& params) {
    n_bins_ = utils::get_param_int(params, "n_bins", n_bins_);
    encode_ = utils::get_param_string(params, "encode", encode_);
    strategy_ = utils::get_param_string(params, "strategy", strategy_);
    return *this;
}

// QuantileTransformer implementation

QuantileTransformer::QuantileTransformer(int n_quantiles, const std::string& output_distribution)
    : n_quantiles_(n_quantiles),
      output_distribution_(output_distribution),
      fitted_(false),
      n_features_(0) {}

Estimator& QuantileTransformer::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;
    if (n_quantiles_ <= 0) {
        throw std::invalid_argument("n_quantiles must be positive");
    }
    if (output_distribution_ != "uniform" && output_distribution_ != "normal") {
        throw std::invalid_argument("output_distribution must be 'uniform' or 'normal'");
    }

    n_features_ = X.cols();
    int n_samples = X.rows();
    n_quantiles_ = std::min(n_quantiles_, n_samples);
    if (n_quantiles_ <= 0) {
        n_quantiles_ = 1;
    }

    quantiles_.clear();
    quantiles_.resize(n_features_);

    for (int j = 0; j < n_features_; ++j) {
        std::vector<double> values(X.rows());
        for (int i = 0; i < X.rows(); ++i) {
            values[i] = X(i, j);
        }
        std::sort(values.begin(), values.end());
        VectorXd qvals(n_quantiles_);
        for (int k = 0; k < n_quantiles_; ++k) {
            double q = (n_quantiles_ == 1) ? 0.0 : static_cast<double>(k) / (n_quantiles_ - 1);
            qvals(k) = compute_quantile(values, q);
        }
        quantiles_[j] = qvals;
    }

    fitted_ = true;
    return *this;
}

MatrixXd QuantileTransformer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("QuantileTransformer must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd out(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < n_features_; ++j) {
            const VectorXd& qvals = quantiles_[j];
            int n_q = qvals.size();
            double value = X(i, j);
            double position = 0.0;

            if (n_q == 1) {
                position = 0.0;
            } else if (value <= qvals(0)) {
                position = 0.0;
            } else if (value >= qvals(n_q - 1)) {
                position = 1.0;
            } else {
                const double* data = qvals.data();
                auto it = std::upper_bound(data, data + n_q, value);
                int idx = static_cast<int>(it - data) - 1;
                idx = std::max(0, std::min(n_q - 2, idx));
                double left = qvals(idx);
                double right = qvals(idx + 1);
                double t = (right - left) > kEps ? (value - left) / (right - left) : 0.0;
                position = (static_cast<double>(idx) + t) / (n_q - 1);
            }

            if (output_distribution_ == "uniform") {
                out(i, j) = position;
            } else {
                double clipped = std::min(1.0 - 1e-6, std::max(1e-6, position));
                out(i, j) = inverse_normal_cdf(clipped);
            }
        }
    }
    return out;
}

MatrixXd QuantileTransformer::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("QuantileTransformer must be fitted before inverse_transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd out(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < n_features_; ++j) {
            const VectorXd& qvals = quantiles_[j];
            int n_q = qvals.size();
            double u = X(i, j);
            if (output_distribution_ == "normal") {
                u = normal_cdf(u);
            }
            u = std::min(1.0, std::max(0.0, u));

            if (n_q == 1) {
                out(i, j) = qvals(0);
            } else {
                double pos = u * (n_q - 1);
                int idx = static_cast<int>(std::floor(pos));
                idx = std::max(0, std::min(n_q - 2, idx));
                double t = pos - idx;
                out(i, j) = qvals(idx) * (1.0 - t) + qvals(idx + 1) * t;
            }
        }
    }
    return out;
}

MatrixXd QuantileTransformer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params QuantileTransformer::get_params() const {
    Params params;
    params["n_quantiles"] = std::to_string(n_quantiles_);
    params["output_distribution"] = output_distribution_;
    return params;
}

Estimator& QuantileTransformer::set_params(const Params& params) {
    n_quantiles_ = utils::get_param_int(params, "n_quantiles", n_quantiles_);
    output_distribution_ = utils::get_param_string(params, "output_distribution", output_distribution_);
    return *this;
}

// PowerTransformer implementation

PowerTransformer::PowerTransformer(const std::string& method, bool standardize)
    : method_(method),
      standardize_(standardize),
      fitted_(false) {}

Estimator& PowerTransformer::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;
    if (method_ != "yeo-johnson" && method_ != "box-cox") {
        throw std::invalid_argument("method must be 'yeo-johnson' or 'box-cox'");
    }

    int n_features = X.cols();
    lambdas_.resize(n_features);
    mean_.resize(n_features);
    scale_.resize(n_features);

    for (int j = 0; j < n_features; ++j) {
        VectorXd col = X.col(j);
        if (method_ == "box-cox") {
            for (int i = 0; i < col.size(); ++i) {
                if (col(i) <= 0.0) {
                    throw std::invalid_argument("Box-Cox requires all features to be positive");
                }
            }
        }

        double best_lambda = 1.0;
        double best_ll = -std::numeric_limits<double>::infinity();
        for (int step = 0; step <= 40; ++step) {
            double lambda = -2.0 + 0.1 * step;
            double sum_log = 0.0;
            VectorXd transformed(col.size());
            for (int i = 0; i < col.size(); ++i) {
                double x = col(i);
                if (method_ == "yeo-johnson") {
                    transformed(i) = yeo_johnson_transform(x, lambda);
                    sum_log += (x >= 0.0 ? 1.0 : -1.0) * std::log(std::abs(x) + 1.0);
                } else {
                    transformed(i) = box_cox_transform(x, lambda);
                    sum_log += std::log(x);
                }
            }
            double mean = transformed.mean();
            double var = (transformed.array() - mean).square().sum() / transformed.size();
            if (var < kEps) {
                var = kEps;
            }
            double loglik = -0.5 * transformed.size() * std::log(var) + (lambda - 1.0) * sum_log;
            if (loglik > best_ll) {
                best_ll = loglik;
                best_lambda = lambda;
            }
        }

        lambdas_(j) = best_lambda;

        VectorXd transformed(col.size());
        for (int i = 0; i < col.size(); ++i) {
            double x = col(i);
            transformed(i) = (method_ == "yeo-johnson")
                ? yeo_johnson_transform(x, best_lambda)
                : box_cox_transform(x, best_lambda);
        }
        mean_(j) = transformed.mean();
        double var = (transformed.array() - mean_(j)).square().sum() / transformed.size();
        scale_(j) = std::sqrt(std::max(var, kEps));
        if (!standardize_) {
            mean_(j) = 0.0;
            scale_(j) = 1.0;
        }
    }

    fitted_ = true;
    return *this;
}

MatrixXd PowerTransformer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PowerTransformer must be fitted before transform");
    }
    if (X.cols() != lambdas_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd out(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            double x = X(i, j);
            if (method_ == "box-cox" && x <= 0.0) {
                throw std::invalid_argument("Box-Cox requires all features to be positive");
            }
            double y = (method_ == "yeo-johnson")
                ? yeo_johnson_transform(x, lambdas_(j))
                : box_cox_transform(x, lambdas_(j));
            out(i, j) = (y - mean_(j)) / scale_(j);
        }
    }
    return out;
}

MatrixXd PowerTransformer::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("PowerTransformer must be fitted before inverse_transform");
    }
    if (X.cols() != lambdas_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd out(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            double y = X(i, j) * scale_(j) + mean_(j);
            double x = (method_ == "yeo-johnson")
                ? yeo_johnson_inverse(y, lambdas_(j))
                : box_cox_inverse(y, lambdas_(j));
            out(i, j) = x;
        }
    }
    return out;
}

MatrixXd PowerTransformer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params PowerTransformer::get_params() const {
    Params params;
    params["method"] = method_;
    params["standardize"] = standardize_ ? "true" : "false";
    return params;
}

Estimator& PowerTransformer::set_params(const Params& params) {
    method_ = utils::get_param_string(params, "method", method_);
    standardize_ = utils::get_param_bool(params, "standardize", standardize_);
    return *this;
}

// FunctionTransformer implementation

FunctionTransformer::FunctionTransformer(const std::string& func, const std::string& inverse_func, bool validate)
    : func_(func), inverse_func_(inverse_func), validate_(validate), fitted_(false) {}

Estimator& FunctionTransformer::fit(const MatrixXd& X, const VectorXd& y) {
    if (validate_) {
        validation::check_X(X);
    }
    (void)y;
    fitted_ = true;
    return *this;
}

MatrixXd FunctionTransformer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("FunctionTransformer must be fitted before transform");
    }
    if (validate_) {
        validation::check_X(X);
    }
    return apply_function(X, func_);
}

MatrixXd FunctionTransformer::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("FunctionTransformer must be fitted before inverse_transform");
    }
    if (validate_) {
        validation::check_X(X);
    }
    return apply_function(X, inverse_func_);
}

MatrixXd FunctionTransformer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params FunctionTransformer::get_params() const {
    Params params;
    params["func"] = func_;
    params["inverse_func"] = inverse_func_;
    params["validate"] = validate_ ? "true" : "false";
    return params;
}

Estimator& FunctionTransformer::set_params(const Params& params) {
    func_ = utils::get_param_string(params, "func", func_);
    inverse_func_ = utils::get_param_string(params, "inverse_func", inverse_func_);
    validate_ = utils::get_param_bool(params, "validate", validate_);
    return *this;
}

// SplineTransformer implementation

SplineTransformer::SplineTransformer(int n_knots, int degree, bool include_bias)
    : n_knots_(n_knots),
      degree_(degree),
      include_bias_(include_bias),
      fitted_(false),
      n_features_(0),
      n_splines_(0),
      output_dim_(0) {}

Estimator& SplineTransformer::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;
    if (n_knots_ < 2) {
        throw std::invalid_argument("n_knots must be at least 2");
    }
    if (degree_ < 1) {
        throw std::invalid_argument("degree must be at least 1");
    }

    n_features_ = X.cols();
    n_splines_ = n_knots_ + degree_ - 1;
    int basis_count = include_bias_ ? n_splines_ : (n_splines_ - 1);
    output_dim_ = n_features_ * basis_count;

    knot_vectors_.clear();
    greville_.clear();
    knot_vectors_.resize(n_features_);
    greville_.resize(n_features_);

    for (int j = 0; j < n_features_; ++j) {
        VectorXd col = X.col(j);
        double min_val = col.minCoeff();
        double max_val = col.maxCoeff();

        VectorXd base_knots(n_knots_);
        if (n_knots_ == 1 || max_val <= min_val) {
            base_knots.setConstant(min_val);
        } else {
            for (int k = 0; k < n_knots_; ++k) {
                base_knots(k) = min_val + (max_val - min_val) * static_cast<double>(k) / (n_knots_ - 1);
            }
        }

        int total_knots = n_knots_ + 2 * degree_;
        VectorXd knots(total_knots);
        for (int k = 0; k < degree_; ++k) {
            knots(k) = base_knots(0);
        }
        for (int k = 0; k < n_knots_; ++k) {
            knots(k + degree_) = base_knots(k);
        }
        for (int k = 0; k < degree_; ++k) {
            knots(k + degree_ + n_knots_) = base_knots(n_knots_ - 1);
        }
        knot_vectors_[j] = knots;

        VectorXd grev(n_splines_);
        for (int i = 0; i < n_splines_; ++i) {
            double sum = 0.0;
            for (int k = 1; k <= degree_; ++k) {
                sum += knots(i + k);
            }
            grev(i) = sum / static_cast<double>(degree_);
        }
        greville_[j] = grev;
    }

    fitted_ = true;
    return *this;
}

MatrixXd SplineTransformer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SplineTransformer must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    int basis_count = include_bias_ ? n_splines_ : (n_splines_ - 1);
    MatrixXd out = MatrixXd::Zero(X.rows(), output_dim_);

    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < n_features_; ++j) {
            const VectorXd& knots = knot_vectors_[j];
            int offset = j * basis_count;
            for (int b = 0; b < basis_count; ++b) {
                out(i, offset + b) = bspline_basis(b, degree_, X(i, j), knots);
            }
        }
    }
    return out;
}

MatrixXd SplineTransformer::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("SplineTransformer must be fitted before inverse_transform");
    }
    int basis_count = include_bias_ ? n_splines_ : (n_splines_ - 1);
    if (X.cols() != output_dim_) {
        throw std::invalid_argument("X must have the same number of features as transformed data");
    }
    MatrixXd out(X.rows(), n_features_);

    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < n_features_; ++j) {
            const VectorXd& grev = greville_[j];
            int offset = j * basis_count;
            double sum_w = 0.0;
            double sum_x = 0.0;
            for (int b = 0; b < basis_count; ++b) {
                double w = X(i, offset + b);
                sum_w += w;
                sum_x += w * grev(b);
            }
            if (sum_w <= kEps) {
                out(i, j) = grev.mean();
            } else {
                out(i, j) = sum_x / sum_w;
            }
        }
    }
    return out;
}

MatrixXd SplineTransformer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params SplineTransformer::get_params() const {
    Params params;
    params["n_knots"] = std::to_string(n_knots_);
    params["degree"] = std::to_string(degree_);
    params["include_bias"] = include_bias_ ? "true" : "false";
    return params;
}

Estimator& SplineTransformer::set_params(const Params& params) {
    n_knots_ = utils::get_param_int(params, "n_knots", n_knots_);
    degree_ = utils::get_param_int(params, "degree", degree_);
    include_bias_ = utils::get_param_bool(params, "include_bias", include_bias_);
    return *this;
}

} // namespace preprocessing
} // namespace ingenuityml

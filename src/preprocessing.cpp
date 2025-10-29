#include "auroraml/preprocessing.hpp"
#include "auroraml/base.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <limits>

namespace auroraml {
namespace preprocessing {

// StandardScaler implementation
StandardScaler::StandardScaler(bool with_mean, bool with_std)
    : mean_(), scale_(), fitted_(false), with_mean_(with_mean), with_std_(with_std) {}

Estimator& StandardScaler::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    int n_features = X.cols();
    mean_ = VectorXd::Zero(n_features);
    scale_ = VectorXd::Ones(n_features);
    
    if (with_mean_) {
        mean_ = X.colwise().mean();
    }
    
    if (with_std_) {
        VectorXd variance = VectorXd::Zero(n_features);
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < n_features; ++j) {
                double diff = X(i, j) - mean_(j);
                variance(j) += diff * diff;
            }
        }
        variance /= X.rows();
        
        for (int j = 0; j < n_features; ++j) {
            scale_(j) = std::sqrt(variance(j));
            if (scale_(j) == 0.0) {
                scale_(j) = 1.0;  // Avoid division by zero
            }
        }
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd StandardScaler::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("StandardScaler must be fitted before transform");
    }
    
    if (X.cols() != mean_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd X_scaled = X;
    
    if (with_mean_) {
        X_scaled.rowwise() -= mean_.transpose();
    }
    
    if (with_std_) {
        X_scaled.array().rowwise() /= scale_.transpose().array();
    }
    
    return X_scaled;
}

MatrixXd StandardScaler::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("StandardScaler must be fitted before inverse_transform");
    }
    
    if (X.cols() != mean_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd X_original = X;
    
    if (with_std_) {
        X_original.array().rowwise() *= scale_.transpose().array();
    }
    
    if (with_mean_) {
        X_original.rowwise() += mean_.transpose();
    }
    
    return X_original;
}

MatrixXd StandardScaler::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params StandardScaler::get_params() const {
    return {
        {"with_mean", with_mean_ ? "true" : "false"},
        {"with_std", with_std_ ? "true" : "false"}
    };
}

Estimator& StandardScaler::set_params(const Params& params) {
    with_mean_ = utils::get_param_bool(params, "with_mean", with_mean_);
    with_std_ = utils::get_param_bool(params, "with_std", with_std_);
    return *this;
}

bool StandardScaler::is_fitted() const {
    return fitted_;
}

// MinMaxScaler implementation
MinMaxScaler::MinMaxScaler(double feature_range_min, double feature_range_max)
    : data_min_(), data_max_(), scale_(), min_(), fitted_(false),
      feature_range_min_(feature_range_min), feature_range_max_(feature_range_max) {}

Estimator& MinMaxScaler::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    int n_features = X.cols();
    data_min_ = X.colwise().minCoeff();
    data_max_ = X.colwise().maxCoeff();
    
    scale_ = VectorXd::Ones(n_features);
    min_ = VectorXd::Zero(n_features);
    
    for (int j = 0; j < n_features; ++j) {
        double data_range = data_max_(j) - data_min_(j);
        if (data_range > 0) {
            scale_(j) = (feature_range_max_ - feature_range_min_) / data_range;
            min_(j) = feature_range_min_ - data_min_(j) * scale_(j);
        } else {
            scale_(j) = 1.0;
            min_(j) = feature_range_min_;
        }
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd MinMaxScaler::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MinMaxScaler must be fitted before transform");
    }
    
    if (X.cols() != data_min_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd X_scaled = X;
    X_scaled.array().rowwise() *= scale_.transpose().array();
    X_scaled.rowwise() += min_.transpose();
    
    return X_scaled;
}

MatrixXd MinMaxScaler::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MinMaxScaler must be fitted before inverse_transform");
    }
    
    if (X.cols() != data_min_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd X_original = X;
    X_original.rowwise() -= min_.transpose();
    X_original.array().rowwise() /= scale_.transpose().array();
    
    return X_original;
}

MatrixXd MinMaxScaler::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params MinMaxScaler::get_params() const {
    return {
        {"feature_range_min", std::to_string(feature_range_min_)},
        {"feature_range_max", std::to_string(feature_range_max_)}
    };
}

Estimator& MinMaxScaler::set_params(const Params& params) {
    feature_range_min_ = utils::get_param_double(params, "feature_range_min", feature_range_min_);
    feature_range_max_ = utils::get_param_double(params, "feature_range_max", feature_range_max_);
    return *this;
}

bool MinMaxScaler::is_fitted() const {
    return fitted_;
}

// LabelEncoder implementation
LabelEncoder::LabelEncoder() : fitted_(false), n_classes_(0) {}

Estimator& LabelEncoder::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    std::set<double> unique_labels;
    for (int i = 0; i < y.size(); ++i) {
        unique_labels.insert(y(i));
    }
    
    n_classes_ = unique_labels.size();
    label_to_index_.clear();
    index_to_label_.clear();
    
    int index = 0;
    for (double label : unique_labels) {
        label_to_index_[label] = index;
        index_to_label_[index] = label;
        index++;
    }
    
    fitted_ = true;
    return *this;
}

VectorXi LabelEncoder::transform(const VectorXd& y) const {
    if (!fitted_) {
        throw std::runtime_error("LabelEncoder must be fitted before transform");
    }
    
    VectorXi encoded(y.size());
    for (int i = 0; i < y.size(); ++i) {
        auto it = label_to_index_.find(y(i));
        if (it != label_to_index_.end()) {
            encoded(i) = it->second;
        } else {
            throw std::invalid_argument("Unknown label: " + std::to_string(y(i)));
        }
    }
    
    return encoded;
}

VectorXd LabelEncoder::inverse_transform(const VectorXi& y) const {
    if (!fitted_) {
        throw std::runtime_error("LabelEncoder must be fitted before inverse_transform");
    }
    
    VectorXd decoded(y.size());
    for (int i = 0; i < y.size(); ++i) {
        auto it = index_to_label_.find(y(i));
        if (it != index_to_label_.end()) {
            decoded(i) = it->second;
        } else {
            throw std::invalid_argument("Unknown index: " + std::to_string(y(i)));
        }
    }
    
    return decoded;
}

MatrixXd LabelEncoder::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelEncoder must be fitted before transform");
    }
    
    // For matrix input, assume we want to transform the first column
    VectorXd first_col = X.col(0);
    VectorXi encoded = transform(first_col);
    
    MatrixXd result(X.rows(), X.cols());
    result.col(0) = encoded.cast<double>();
    result.rightCols(X.cols() - 1) = X.rightCols(X.cols() - 1);
    
    return result;
}

MatrixXd LabelEncoder::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelEncoder must be fitted before inverse_transform");
    }
    
    // For matrix input, assume we want to inverse transform the first column
    VectorXi first_col = X.col(0).cast<int>();
    VectorXd decoded = inverse_transform(first_col);
    
    MatrixXd result(X.rows(), X.cols());
    result.col(0) = decoded;
    result.rightCols(X.cols() - 1) = X.rightCols(X.cols() - 1);
    
    return result;
}

MatrixXd LabelEncoder::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params LabelEncoder::get_params() const {
    return {};
}

Estimator& LabelEncoder::set_params(const Params& params) {
    return *this;
}

bool LabelEncoder::is_fitted() const {
    return fitted_;
}

} // namespace preprocessing
} // namespace cxml

// ---- RobustScaler implementation (in cxml::preprocessing)
namespace auroraml {
namespace preprocessing {

static double _median(VectorXd v) {
    std::vector<double> s(v.data(), v.data() + v.size());
    std::nth_element(s.begin(), s.begin() + s.size()/2, s.end());
    if (s.size() % 2 == 1) return s[s.size()/2];
    auto max_it = std::max_element(s.begin(), s.begin() + s.size()/2);
    return (*max_it + s[s.size()/2]) / 2.0;
}

static double _quantile(VectorXd v, double q) {
    std::vector<double> s(v.data(), v.data() + v.size());
    std::sort(s.begin(), s.end());
    double pos = q * (s.size() - 1);
    size_t idx = static_cast<size_t>(pos);
    double frac = pos - idx;
    if (idx + 1 < s.size()) return s[idx] * (1 - frac) + s[idx + 1] * frac;
    return s[idx];
}

Estimator& RobustScaler::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    int n_features = X.cols();
    center_.resize(n_features);
    scale_.resize(n_features);
    for (int j = 0; j < n_features; ++j) {
        VectorXd col = X.col(j);
        double med = _median(col);
        double q1 = _quantile(col, 0.25);
        double q3 = _quantile(col, 0.75);
        double iqr = q3 - q1;
        if (iqr < 1e-12) iqr = 1.0;
        center_(j) = med;
        scale_(j) = iqr;
    }
    fitted_ = true;
    return *this;
}

MatrixXd RobustScaler::transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("RobustScaler must be fitted before transform");
    if (X.cols() != center_.size()) throw std::invalid_argument("X must have same features as fit");
    MatrixXd Y = X;
    if (with_centering_) Y = Y.rowwise() - center_.transpose();
    if (with_scaling_) {
        for (int j = 0; j < Y.cols(); ++j) Y.col(j) = Y.col(j).array() / scale_(j);
    }
    return Y;
}

MatrixXd RobustScaler::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("RobustScaler must be fitted before inverse_transform");
    if (X.cols() != center_.size()) throw std::invalid_argument("X must have same features as fit");
    MatrixXd Y = X;
    if (with_scaling_) {
        for (int j = 0; j < Y.cols(); ++j) Y.col(j) = Y.col(j).array() * scale_(j);
    }
    if (with_centering_) Y = Y.rowwise() + center_.transpose();
    return Y;
}

MatrixXd RobustScaler::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params RobustScaler::get_params() const {
    return {
        {"with_centering", with_centering_ ? "true" : "false"},
        {"with_scaling", with_scaling_ ? "true" : "false"}
    };
}

Estimator& RobustScaler::set_params(const Params& params) {
    with_centering_ = utils::get_param_bool(params, "with_centering", with_centering_);
    with_scaling_ = utils::get_param_bool(params, "with_scaling", with_scaling_);
    return *this;
}

} // namespace preprocessing
} // namespace cxml

// ---- OneHotEncoder implementation
namespace auroraml {
namespace preprocessing {

Estimator& OneHotEncoder::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    int n_features = X.cols();
    categories_.clear();
    categories_.resize(n_features);
    for (int j = 0; j < n_features; ++j) {
        std::set<double> uniq;
        for (int i = 0; i < X.rows(); ++i) uniq.insert(X(i, j));
        categories_[j] = std::vector<double>(uniq.begin(), uniq.end());
    }
    // offsets
    col_offsets_.clear(); col_offsets_.resize(n_features);
    output_dim_ = 0;
    for (int j = 0; j < n_features; ++j) {
        col_offsets_[j] = output_dim_;
        output_dim_ += static_cast<int>(categories_[j].size());
    }
    fitted_ = true;
    return *this;
}

MatrixXd OneHotEncoder::transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("OneHotEncoder must be fitted before transform");
    validation::check_X(X);
    if (X.cols() != static_cast<int>(categories_.size())) throw std::runtime_error("Feature mismatch in OneHotEncoder::transform");
    MatrixXd Y = MatrixXd::Zero(X.rows(), output_dim_);
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            const auto& cats = categories_[j];
            auto it = std::lower_bound(cats.begin(), cats.end(), X(i, j));
            if (it != cats.end() && *it == X(i, j)) {
                int idx = static_cast<int>(it - cats.begin());
                Y(i, col_offsets_[j] + idx) = 1.0;
            } else {
                // unseen category -> skip (all zeros)
            }
        }
    }
    return Y;
}

MatrixXd OneHotEncoder::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("OneHotEncoder must be fitted before inverse_transform");
    if (X.cols() != output_dim_) throw std::runtime_error("Dim mismatch in OneHotEncoder::inverse_transform");
    int n_features = static_cast<int>(categories_.size());
    MatrixXd Y(X.rows(), n_features);
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < n_features; ++j) {
            int start = col_offsets_[j];
            int width = static_cast<int>(categories_[j].size());
            // pick argmax
            int arg = 0; double best = X(i, start);
            for (int k = 1; k < width; ++k) { if (X(i, start + k) > best) { best = X(i, start + k); arg = k; } }
            Y(i, j) = categories_[j][arg];
        }
    }
    return Y;
}

MatrixXd OneHotEncoder::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

// OrdinalEncoder implementation
Estimator& OrdinalEncoder::fit(const MatrixXd& X, const VectorXd& y) {
    (void)y;
    validation::check_X(X);
    
    int n_features = X.cols();
    categories_.clear();
    categories_.resize(n_features);
    category_to_int_.clear();
    category_to_int_.resize(n_features);
    
    for (int j = 0; j < n_features; ++j) {
        std::set<double> unique_values;
        for (int i = 0; i < X.rows(); ++i) {
            unique_values.insert(X(i, j));
        }
        
        // Store categories in sorted order
        categories_[j] = std::vector<double>(unique_values.begin(), unique_values.end());
        
        // Create mapping from category to integer
        for (size_t k = 0; k < categories_[j].size(); ++k) {
            category_to_int_[j][categories_[j][k]] = static_cast<int>(k);
        }
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd OrdinalEncoder::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OrdinalEncoder must be fitted before transform");
    }
    validation::check_X(X);
    
    if (X.cols() != static_cast<int>(categories_.size())) {
        throw std::invalid_argument("Number of features in X must match number of features seen during fit");
    }
    
    MatrixXd result(X.rows(), X.cols());
    
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            double value = X(i, j);
            auto it = category_to_int_[j].find(value);
            
            if (it != category_to_int_[j].end()) {
                result(i, j) = static_cast<double>(it->second);
            } else {
                // Unknown category - assign -1 or throw error
                // For now, assign -1 to indicate unknown category
                result(i, j) = -1.0;
            }
        }
    }
    
    return result;
}

MatrixXd OrdinalEncoder::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OrdinalEncoder must be fitted before inverse_transform");
    }
    validation::check_X(X);
    
    if (X.cols() != static_cast<int>(categories_.size())) {
        throw std::invalid_argument("Number of features in X must match number of features seen during fit");
    }
    
    MatrixXd result(X.rows(), X.cols());
    
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            int encoded_value = static_cast<int>(X(i, j));
            
            if (encoded_value >= 0 && encoded_value < static_cast<int>(categories_[j].size())) {
                result(i, j) = categories_[j][encoded_value];
            } else {
                // Invalid encoded value - assign NaN or throw error
                // For now, assign NaN to indicate invalid encoding
                result(i, j) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    
    return result;
}

MatrixXd OrdinalEncoder::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

} // namespace preprocessing
} // namespace cxml
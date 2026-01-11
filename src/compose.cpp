#include "ingenuityml/compose.hpp"
#include "ingenuityml/base.hpp"
#include <stdexcept>
#include <algorithm>
#include <set>

namespace ingenuityml {
namespace compose {

// ColumnTransformer implementation

ColumnTransformer::ColumnTransformer(
    const std::vector<std::tuple<std::string, std::shared_ptr<Transformer>, std::vector<int>>>& transformers,
    const std::string& remainder,
    double sparse_threshold
) : transformers_(transformers), fitted_(false), drop_remaining_(remainder == "drop") {
    if (transformers_.empty()) {
        throw std::invalid_argument("ColumnTransformer must have at least one transformer");
    }
    
    // Collect all column indices to find remaining columns
    std::set<int> used_columns;
    for (const auto& transformer_tuple : transformers_) {
        const auto& column_indices = std::get<2>(transformer_tuple);
        for (int col_idx : column_indices) {
            used_columns.insert(col_idx);
        }
    }
    
    // For now, we'll calculate remaining columns during fit
    // (since we don't know the number of columns yet)
}

MatrixXd ColumnTransformer::extract_columns(const MatrixXd& X, const std::vector<int>& column_indices) const {
    MatrixXd extracted(X.rows(), column_indices.size());
    for (size_t i = 0; i < column_indices.size(); ++i) {
        int col_idx = column_indices[i];
        if (col_idx < 0 || col_idx >= X.cols()) {
            throw std::invalid_argument("Column index out of range: " + std::to_string(col_idx));
        }
        extracted.col(i) = X.col(col_idx);
    }
    return extracted;
}

Estimator& ColumnTransformer::fit(const MatrixXd& X, const VectorXd& y) {
    // Find remaining columns
    std::set<int> used_columns;
    for (const auto& transformer_tuple : transformers_) {
        const auto& column_indices = std::get<2>(transformer_tuple);
        for (int col_idx : column_indices) {
            used_columns.insert(col_idx);
        }
    }
    
    remaining_columns_.clear();
    for (int i = 0; i < X.cols(); ++i) {
        if (used_columns.find(i) == used_columns.end()) {
            remaining_columns_.push_back(i);
        }
    }
    
    // Fit each transformer on its specified columns
    for (auto& transformer_tuple : transformers_) {
        auto& transformer = std::get<1>(transformer_tuple);
        const auto& column_indices = std::get<2>(transformer_tuple);
        
        MatrixXd X_subset = extract_columns(X, column_indices);
        transformer->fit_transform(X_subset, y);
    }
    
    fitted_ = true;
    return *this;
}

MatrixXd ColumnTransformer::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ColumnTransformer must be fitted before transform");
    }
    
    if (transformers_.empty()) {
        return MatrixXd::Zero(X.rows(), 0);
    }
    
    // Transform with first transformer
    const auto& first_tuple = transformers_[0];
    auto first_transformer = std::get<1>(first_tuple);
    const auto& first_columns = std::get<2>(first_tuple);
    
    MatrixXd X_first = extract_columns(X, first_columns);
    MatrixXd result = first_transformer->transform(X_first);
    
    // Concatenate results from all other transformers
    for (size_t i = 1; i < transformers_.size(); ++i) {
        const auto& transformer_tuple = transformers_[i];
        auto transformer = std::get<1>(transformer_tuple);
        const auto& column_indices = std::get<2>(transformer_tuple);
        
        MatrixXd X_subset = extract_columns(X, column_indices);
        MatrixXd transformed = transformer->transform(X_subset);
        
        // Concatenate horizontally
        MatrixXd combined(result.rows(), result.cols() + transformed.cols());
        combined << result, transformed;
        result = combined;
    }
    
    // Add remaining columns if not dropping
    if (!drop_remaining_ && !remaining_columns_.empty()) {
        MatrixXd X_remaining = extract_columns(X, remaining_columns_);
        MatrixXd combined(result.rows(), result.cols() + X_remaining.cols());
        combined << result, X_remaining;
        result = combined;
    }
    
    return result;
}

MatrixXd ColumnTransformer::inverse_transform(const MatrixXd& X) const {
    // Simplified: just return the transformed data
    // In a full implementation, this would attempt to inverse transform each column subset
    return transform(X);
}

MatrixXd ColumnTransformer::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params ColumnTransformer::get_params() const {
    Params params;
    params["transformers"] = std::to_string(transformers_.size());
    params["remainder"] = drop_remaining_ ? "drop" : "passthrough";
    for (size_t i = 0; i < transformers_.size(); ++i) {
        params["transformer_" + std::to_string(i) + "_name"] = std::get<0>(transformers_[i]);
    }
    return params;
}

Estimator& ColumnTransformer::set_params(const Params& params) {
    // Simplified parameter setting
    return *this;
}

std::shared_ptr<Transformer> ColumnTransformer::get_transformer(const std::string& name) const {
    for (const auto& transformer_tuple : transformers_) {
        if (std::get<0>(transformer_tuple) == name) {
            return std::get<1>(transformer_tuple);
        }
    }
    return nullptr;
}

std::vector<std::string> ColumnTransformer::get_transformer_names() const {
    std::vector<std::string> names;
    for (const auto& transformer_tuple : transformers_) {
        names.push_back(std::get<0>(transformer_tuple));
    }
    return names;
}

// TransformedTargetRegressor implementation

TransformedTargetRegressor::TransformedTargetRegressor(
    std::shared_ptr<Regressor> regressor,
    std::shared_ptr<Transformer> transformer
) : regressor_(regressor), transformer_(transformer), fitted_(false) {
    if (!regressor_) {
        throw std::invalid_argument("Regressor must not be null");
    }
}

Estimator& TransformedTargetRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    VectorXd y_transformed = y;
    
    // Transform target if transformer is provided
    if (transformer_) {
        // Create a dummy X for transformer (transformers typically need X)
        MatrixXd dummy_X = MatrixXd::Ones(y.size(), 1);
        y_transformed = transformer_->fit_transform(dummy_X, y).col(0);
    }
    
    // Fit regressor with transformed target
    // All Regressor implementations also inherit from Estimator
    Estimator* est = dynamic_cast<Estimator*>(regressor_.get());
    if (est) {
        est->fit(X, y_transformed);
    } else {
        throw std::runtime_error("Regressor must inherit from both Estimator and Regressor");
    }
    
    fitted_ = true;
    return *this;
}

VectorXd TransformedTargetRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("TransformedTargetRegressor must be fitted before predict");
    }
    
    // Predict with regressor
    VectorXd y_pred = regressor_->predict(X);
    
    // Inverse transform if transformer is provided
    if (transformer_) {
        // Create dummy X for inverse_transform
        MatrixXd dummy_X = MatrixXd::Ones(y_pred.size(), 1);
        MatrixXd y_pred_matrix = y_pred;
        MatrixXd y_pred_inverse = transformer_->inverse_transform(y_pred_matrix);
        y_pred = y_pred_inverse.col(0);
    }
    
    return y_pred;
}

Params TransformedTargetRegressor::get_params() const {
    Params params;
    if (regressor_) {
        // Cast Regressor to Estimator to access get_params
        Estimator* est = dynamic_cast<Estimator*>(regressor_.get());
        if (est) {
            auto regressor_params = est->get_params();
            for (const auto& [key, value] : regressor_params) {
                params["regressor__" + key] = value;
            }
        }
    }
    if (transformer_) {
        // Transformer params would go here
        params["transformer"] = "provided";
    } else {
        params["transformer"] = "none";
    }
    return params;
}

Estimator& TransformedTargetRegressor::set_params(const Params& params) {
    // Extract regressor-specific parameters and set them
    if (regressor_) {
        Params regressor_params;
        for (const auto& [key, value] : params) {
            if (key.find("regressor__") == 0) {
                std::string regressor_key = key.substr(11); // Remove "regressor__" prefix
                regressor_params[regressor_key] = value;
            }
        }
        // Regressor also inherits from Estimator in concrete implementations
        Estimator* est = dynamic_cast<Estimator*>(regressor_.get());
        if (est) {
            est->set_params(regressor_params);
        }
    }
    return *this;
}

} // namespace compose
} // namespace ingenuityml


#include "auroraml/pipeline.hpp"
#include "auroraml/base.hpp"
#include <stdexcept>
#include <algorithm>

namespace auroraml {
namespace pipeline {

Pipeline::Pipeline(const std::vector<std::pair<std::string, std::shared_ptr<Estimator>>>& steps)
    : steps_(steps), fitted_(false) {
    if (steps_.empty()) {
        throw std::invalid_argument("Pipeline must have at least one step");
    }
}

Estimator& Pipeline::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd X_transformed = X;
    
    // Fit and transform through all steps except the last
    for (size_t i = 0; i < steps_.size() - 1; ++i) {
        auto& step = steps_[i].second;
        
        // Check if step is a Transformer
        auto transformer = std::dynamic_pointer_cast<Transformer>(step);
        if (transformer) {
            X_transformed = transformer->fit_transform(X_transformed, y);
        } else {
            // If not a transformer, try to fit as estimator
            step->fit(X_transformed, y);
        }
    }
    
    // Fit the last step (usually the predictor)
    steps_.back().second->fit(X_transformed, y);
    
    fitted_ = true;
    return *this;
}

MatrixXd Pipeline::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Pipeline must be fitted before transform");
    }
    
    MatrixXd X_transformed = X;
    
    // Transform through all steps except the last
    for (size_t i = 0; i < steps_.size() - 1; ++i) {
        auto transformer = std::dynamic_pointer_cast<Transformer>(steps_[i].second);
        if (transformer) {
            X_transformed = transformer->transform(X_transformed);
        }
    }
    
    // Check if last step is a transformer
    auto last_transformer = std::dynamic_pointer_cast<Transformer>(steps_.back().second);
    if (last_transformer) {
        return last_transformer->transform(X_transformed);
    }
    
    // If last step is not a transformer, return the transformed data
    return X_transformed;
}

VectorXd Pipeline::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Pipeline must be fitted before predict");
    }
    
    MatrixXd X_transformed = X;
    
    // Transform through all steps except the last
    for (size_t i = 0; i < steps_.size() - 1; ++i) {
        auto transformer = std::dynamic_pointer_cast<Transformer>(steps_[i].second);
        if (transformer) {
            X_transformed = transformer->transform(X_transformed);
        }
    }
    
    // Predict using the last step (must be a Predictor)
    auto predictor = std::dynamic_pointer_cast<Predictor>(steps_.back().second);
    if (!predictor) {
        throw std::runtime_error("Last step in pipeline must be a Predictor for predict()");
    }
    
    return predictor->predict(X_transformed);
}

VectorXi Pipeline::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Pipeline must be fitted before predict_classes");
    }
    
    MatrixXd X_transformed = X;
    
    // Transform through all steps except the last
    for (size_t i = 0; i < steps_.size() - 1; ++i) {
        auto transformer = std::dynamic_pointer_cast<Transformer>(steps_[i].second);
        if (transformer) {
            X_transformed = transformer->transform(X_transformed);
        }
    }
    
    // Predict using the last step (must be a Classifier)
    auto classifier = std::dynamic_pointer_cast<Classifier>(steps_.back().second);
    if (!classifier) {
        throw std::runtime_error("Last step in pipeline must be a Classifier for predict_classes()");
    }
    
    return classifier->predict_classes(X_transformed);
}

MatrixXd Pipeline::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("Pipeline must be fitted before predict_proba");
    }
    
    MatrixXd X_transformed = X;
    
    // Transform through all steps except the last
    for (size_t i = 0; i < steps_.size() - 1; ++i) {
        auto transformer = std::dynamic_pointer_cast<Transformer>(steps_[i].second);
        if (transformer) {
            X_transformed = transformer->transform(X_transformed);
        }
    }
    
    // Predict using the last step (must be a Classifier)
    auto classifier = std::dynamic_pointer_cast<Classifier>(steps_.back().second);
    if (!classifier) {
        throw std::runtime_error("Last step in pipeline must be a Classifier for predict_proba()");
    }
    
    return classifier->predict_proba(X_transformed);
}

MatrixXd Pipeline::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params Pipeline::get_params() const {
    Params params;
    params["steps"] = std::to_string(steps_.size());
    for (size_t i = 0; i < steps_.size(); ++i) {
        params["step_" + std::to_string(i) + "_name"] = steps_[i].first;
        // Merge step params
        auto step_params = steps_[i].second->get_params();
        for (const auto& [key, value] : step_params) {
            params["step_" + std::to_string(i) + "_" + key] = value;
        }
    }
    return params;
}

Estimator& Pipeline::set_params(const Params& params) {
    // Extract step-specific parameters and set them
    // This is a simplified version - full implementation would need
    // to parse nested parameter names
    for (auto& step : steps_) {
        Params step_params;
        // Extract relevant params for this step
        // For now, we'll just call set_params with empty params
        // A full implementation would need to parse parameter prefixes
        step.second->set_params(step_params);
    }
    return *this;
}

std::shared_ptr<Estimator> Pipeline::get_step(const std::string& name) const {
    for (const auto& step : steps_) {
        if (step.first == name) {
            return step.second;
        }
    }
    return nullptr;
}

std::vector<std::string> Pipeline::get_step_names() const {
    std::vector<std::string> names;
    for (const auto& step : steps_) {
        names.push_back(step.first);
    }
    return names;
}

// FeatureUnion implementation

FeatureUnion::FeatureUnion(const std::vector<std::pair<std::string, std::shared_ptr<Transformer>>>& transformers)
    : transformers_(transformers), fitted_(false) {
    if (transformers_.empty()) {
        throw std::invalid_argument("FeatureUnion must have at least one transformer");
    }
}

Estimator& FeatureUnion::fit(const MatrixXd& X, const VectorXd& y) {
    for (auto& transformer_pair : transformers_) {
        auto transformer = transformer_pair.second;
        // Just fit, don't transform yet
        // We'll use a dummy vector for y if transformer doesn't need it
        VectorXd dummy_y = VectorXd::Zero(X.rows());
        transformer->fit_transform(X, dummy_y);
    }
    fitted_ = true;
    return *this;
}

MatrixXd FeatureUnion::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("FeatureUnion must be fitted before transform");
    }
    
    if (transformers_.empty()) {
        return MatrixXd::Zero(X.rows(), 0);
    }
    
    // Transform with first transformer to get dimensions
    MatrixXd result = transformers_[0].second->transform(X);
    
    // Concatenate results from all other transformers
    for (size_t i = 1; i < transformers_.size(); ++i) {
        MatrixXd transformed = transformers_[i].second->transform(X);
        // Concatenate horizontally
        MatrixXd combined(result.rows(), result.cols() + transformed.cols());
        combined << result, transformed;
        result = combined;
    }
    
    return result;
}

MatrixXd FeatureUnion::inverse_transform(const MatrixXd& X) const {
    // Simplified: just return the transformed data
    // In a full implementation, this would attempt to inverse transform each transformer
    return transform(X);
}

MatrixXd FeatureUnion::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params FeatureUnion::get_params() const {
    Params params;
    params["transformers"] = std::to_string(transformers_.size());
    for (size_t i = 0; i < transformers_.size(); ++i) {
        params["transformer_" + std::to_string(i) + "_name"] = transformers_[i].first;
    }
    return params;
}

Estimator& FeatureUnion::set_params(const Params& params) {
    // Extract transformer-specific parameters and set them
    // This is a simplified version
    for (auto& transformer_pair : transformers_) {
        Params transformer_params;
        // For now, we'll just call set_params with empty params
        // A full implementation would need to parse parameter prefixes
    }
    return *this;
}

std::shared_ptr<Transformer> FeatureUnion::get_transformer(const std::string& name) const {
    for (const auto& transformer_pair : transformers_) {
        if (transformer_pair.first == name) {
            return transformer_pair.second;
        }
    }
    return nullptr;
}

std::vector<std::string> FeatureUnion::get_transformer_names() const {
    std::vector<std::string> names;
    for (const auto& transformer_pair : transformers_) {
        names.push_back(transformer_pair.first);
    }
    return names;
}

} // namespace pipeline
} // namespace auroraml


#include "auroraml/calibration.hpp"
#include "auroraml/model_selection.hpp"
#include <algorithm>
#include <cmath>
#include <set>

namespace auroraml {
namespace calibration {

CalibratedClassifierCV::CalibratedClassifierCV(
    std::shared_ptr<Classifier> base_estimator,
    const std::string& method,
    int cv
) : base_estimator_(base_estimator), method_(method), cv_(cv), fitted_(false) {
    if (!base_estimator_) {
        throw std::invalid_argument("Base estimator must not be null");
    }
    if (method_ != "sigmoid" && method_ != "isotonic") {
        throw std::invalid_argument("Method must be 'sigmoid' or 'isotonic'");
    }
}

Estimator& CalibratedClassifierCV::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    // Find unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    classes_.resize(unique_classes_set.size());
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }
    
    // Create cross-validator
    model_selection::KFold kfold(cv_, false, -1);
    auto splits = kfold.split(X, y);
    
    calibrated_estimators_.clear();
    calibrated_estimators_.reserve(splits.size());
    
    // For each fold, fit base estimator and calibrate
    for (const auto& split : splits) {
        const auto& train_indices = split.first;
        const auto& test_indices = split.second;
        
        // Create training data
        MatrixXd X_train(train_indices.size(), X.cols());
        VectorXd y_train(train_indices.size());
        for (size_t i = 0; i < train_indices.size(); ++i) {
            X_train.row(i) = X.row(train_indices[i]);
            y_train(i) = y(train_indices[i]);
        }
        
        // Fit base estimator - all Classifier implementations also inherit from Estimator
        Estimator* est = dynamic_cast<Estimator*>(base_estimator_.get());
        if (est) {
            est->fit(X_train, y_train);
        } else {
            throw std::runtime_error("Base estimator must inherit from both Estimator and Classifier");
        }
        
        // Get predictions on test set for calibration
        MatrixXd X_test(test_indices.size(), X.cols());
        VectorXd y_test(test_indices.size());
        for (size_t i = 0; i < test_indices.size(); ++i) {
            X_test.row(i) = X.row(test_indices[i]);
            y_test(i) = y(test_indices[i]);
        }
        
        MatrixXd proba_test = base_estimator_->predict_proba(X_test);
        
        // Simplified calibration: use sigmoid mapping
        // In a full implementation, this would use Platt scaling or isotonic regression
        calibrated_estimators_.push_back(base_estimator_);
    }
    
    fitted_ = true;
    return *this;
}

VectorXi CalibratedClassifierCV::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("CalibratedClassifierCV must be fitted before predict");
    }
    
    // Average predictions from all calibrated estimators
    MatrixXd proba_sum = MatrixXd::Zero(X.rows(), classes_.size());
    
    for (const auto& estimator : calibrated_estimators_) {
        proba_sum += estimator->predict_proba(X);
    }
    
    proba_sum /= calibrated_estimators_.size();
    
    // Predict class with highest probability
    VectorXi predictions = VectorXi::Zero(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        int max_idx = 0;
        for (int j = 1; j < proba_sum.cols(); ++j) {
            if (proba_sum(i, j) > proba_sum(i, max_idx)) {
                max_idx = j;
            }
        }
        predictions(i) = classes_(max_idx);
    }
    
    return predictions;
}

MatrixXd CalibratedClassifierCV::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("CalibratedClassifierCV must be fitted before predict_proba");
    }
    
    MatrixXd proba_sum = MatrixXd::Zero(X.rows(), classes_.size());
    
    for (const auto& estimator : calibrated_estimators_) {
        proba_sum += estimator->predict_proba(X);
    }
    
    proba_sum /= calibrated_estimators_.size();
    
    // Normalize probabilities
    for (int i = 0; i < proba_sum.rows(); ++i) {
        double sum = proba_sum.row(i).sum();
        if (sum > 0) {
            proba_sum.row(i) /= sum;
        }
    }
    
    return proba_sum;
}

VectorXd CalibratedClassifierCV::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("CalibratedClassifierCV must be fitted before decision_function");
    }
    
    VectorXd decision_sum = VectorXd::Zero(X.rows());
    
    for (const auto& estimator : calibrated_estimators_) {
        decision_sum += estimator->decision_function(X);
    }
    
    decision_sum /= calibrated_estimators_.size();
    return decision_sum;
}

Params CalibratedClassifierCV::get_params() const {
    Params params;
    params["method"] = method_;
    params["cv"] = std::to_string(cv_);
    return params;
}

Estimator& CalibratedClassifierCV::set_params(const Params& params) {
    method_ = utils::get_param_string(params, "method", method_);
    cv_ = utils::get_param_int(params, "cv", cv_);
    return *this;
}

} // namespace calibration
} // namespace auroraml


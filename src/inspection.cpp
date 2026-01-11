#include "ingenuityml/inspection.hpp"
#include "ingenuityml/metrics.hpp"
#include <random>
#include <algorithm>
#include <numeric>

namespace ingenuityml {
namespace inspection {

// PermutationImportance implementation

PermutationImportance::PermutationImportance(
    std::shared_ptr<Estimator> estimator,
    const std::string& scoring,
    int n_repeats,
    int random_state
) : estimator_(estimator), scoring_(scoring), n_repeats_(n_repeats), random_state_(random_state) {
    if (!estimator_) {
        throw std::invalid_argument("Estimator must not be null");
    }
}

void PermutationImportance::fit(const MatrixXd& X, const VectorXd& y) {
    // Fit estimator on original data
    estimator_->fit(X, y);
    
    // Get baseline score
    double baseline_score = 0.0;
    if (scoring_ == "accuracy") {
        auto classifier = std::dynamic_pointer_cast<Classifier>(estimator_);
        if (classifier) {
            VectorXi y_pred = classifier->predict_classes(X);
            VectorXi y_int = y.cast<int>();
            baseline_score = metrics::accuracy_score(y_int, y_pred);
        }
    } else if (scoring_ == "r2") {
        auto regressor = std::dynamic_pointer_cast<Regressor>(estimator_);
        if (regressor) {
            VectorXd y_pred = regressor->predict(X);
            baseline_score = metrics::r2_score(y, y_pred);
        }
    }
    
    // Compute importance for each feature
    importances_.clear();
    importances_std_.clear();
    importances_.resize(X.cols());
    importances_std_.resize(X.cols());
    
    std::mt19937 rng;
    if (random_state_ >= 0) {
        rng.seed(random_state_);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    for (int feature_idx = 0; feature_idx < X.cols(); ++feature_idx) {
        std::vector<double> scores;
        
        for (int repeat = 0; repeat < n_repeats_; ++repeat) {
            // Permute feature
            MatrixXd X_permuted = X;
            std::vector<int> indices(X.rows());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            
            for (int i = 0; i < X.rows(); ++i) {
                X_permuted(i, feature_idx) = X(indices[i], feature_idx);
            }
            
            // Compute score with permuted feature
            double score = 0.0;
            if (scoring_ == "accuracy") {
                auto classifier = std::dynamic_pointer_cast<Classifier>(estimator_);
                if (classifier) {
                    VectorXi y_pred = classifier->predict_classes(X_permuted);
                    VectorXi y_int = y.cast<int>();
                    score = metrics::accuracy_score(y_int, y_pred);
                }
            } else if (scoring_ == "r2") {
                auto regressor = std::dynamic_pointer_cast<Regressor>(estimator_);
                if (regressor) {
                    VectorXd y_pred = regressor->predict(X_permuted);
                    score = metrics::r2_score(y, y_pred);
                }
            }
            
            scores.push_back(baseline_score - score);
        }
        
        // Calculate mean and std
        double mean_score = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        importances_[feature_idx] = mean_score;
        
        double variance = 0.0;
        for (double s : scores) {
            variance += (s - mean_score) * (s - mean_score);
        }
        double std_score = std::sqrt(variance / scores.size());
        importances_std_[feature_idx] = {std_score};
    }
}

// PartialDependence implementation

PartialDependence::PartialDependence(
    std::shared_ptr<Predictor> estimator,
    const std::vector<int>& features
) : estimator_(estimator), features_(features) {
    if (!estimator_) {
        throw std::invalid_argument("Estimator must not be null");
    }
}

void PartialDependence::compute(const MatrixXd& X) {
    if (features_.empty()) {
        throw std::invalid_argument("At least one feature must be specified");
    }
    
    // Create grid of values for each feature
    int n_grid_points = 50;
    grid_ = MatrixXd::Zero(n_grid_points, features_.size());
    
    for (size_t f_idx = 0; f_idx < features_.size(); ++f_idx) {
        int feature_idx = features_[f_idx];
        if (feature_idx < 0 || feature_idx >= X.cols()) {
            throw std::invalid_argument("Feature index out of range");
        }
        
        double min_val = X.col(feature_idx).minCoeff();
        double max_val = X.col(feature_idx).maxCoeff();
        
        for (int i = 0; i < n_grid_points; ++i) {
            grid_(i, f_idx) = min_val + (max_val - min_val) * i / (n_grid_points - 1.0);
        }
    }
    
    // Compute partial dependence
    partial_dependence_ = VectorXd::Zero(n_grid_points);
    
    for (int grid_idx = 0; grid_idx < n_grid_points; ++grid_idx) {
        // Create data with grid values
        MatrixXd X_pd = X;
        for (size_t f_idx = 0; f_idx < features_.size(); ++f_idx) {
            int feature_idx = features_[f_idx];
            double grid_value = grid_(grid_idx, f_idx);
            
            for (int i = 0; i < X.rows(); ++i) {
                X_pd(i, feature_idx) = grid_value;
            }
        }
        
        // Predict
        VectorXd y_pred = estimator_->predict(X_pd);
        partial_dependence_(grid_idx) = y_pred.mean();
    }
}

} // namespace inspection
} // namespace ingenuityml


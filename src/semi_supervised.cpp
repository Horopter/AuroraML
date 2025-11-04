#include "auroraml/semi_supervised.hpp"
#include "auroraml/base.hpp"
#include <set>
#include <cmath>
#include <algorithm>

namespace auroraml {
namespace semi_supervised {

// LabelPropagation implementation

LabelPropagation::LabelPropagation(
    double gamma,
    int max_iter,
    double tol,
    const std::string& kernel
) : gamma_(gamma), max_iter_(max_iter), tol_(tol), kernel_(kernel), fitted_(false) {
}

MatrixXd LabelPropagation::build_affinity_matrix(const MatrixXd& X) const {
    int n_samples = X.rows();
    MatrixXd affinity = MatrixXd::Zero(n_samples, n_samples);
    
    if (kernel_ == "rbf") {
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_samples; ++j) {
                if (i == j) {
                    affinity(i, j) = 1.0;
                } else {
                    double dist_sq = (X.row(i) - X.row(j)).squaredNorm();
                    affinity(i, j) = std::exp(-gamma_ * dist_sq);
                }
            }
        }
    } else if (kernel_ == "knn") {
        // KNN-based affinity
        int k = 7;
        for (int i = 0; i < n_samples; ++i) {
            std::vector<std::pair<double, int>> distances;
            for (int j = 0; j < n_samples; ++j) {
                if (i != j) {
                    double dist = (X.row(i) - X.row(j)).norm();
                    distances.push_back({dist, j});
                }
            }
            std::sort(distances.begin(), distances.end());
            
            for (int ki = 0; ki < std::min(k, static_cast<int>(distances.size())); ++ki) {
                int j = distances[ki].second;
                affinity(i, j) = 1.0;
            }
        }
    }
    
    return affinity;
}

void LabelPropagation::propagate_labels() {
    int n_samples = X_fitted_.rows();
    int n_classes = classes_.size();
    
    // Initialize label distributions
    label_distributions_ = MatrixXd::Zero(n_samples, n_classes);
    
    // Set labeled samples
    for (int i = 0; i < n_samples; ++i) {
        if (y_fitted_(i) >= 0) { // Labeled sample
            for (int c = 0; c < n_classes; ++c) {
                if (y_fitted_(i) == classes_(c)) {
                    label_distributions_(i, c) = 1.0;
                    break;
                }
            }
        } else {
            // Unlabeled sample - initialize uniformly
            label_distributions_.row(i).fill(1.0 / n_classes);
        }
    }
    
    // Build affinity matrix
    MatrixXd affinity = build_affinity_matrix(X_fitted_);
    
    // Normalize affinity matrix
    VectorXd row_sums = affinity.rowwise().sum();
    for (int i = 0; i < n_samples; ++i) {
        if (row_sums(i) > 0) {
            affinity.row(i) /= row_sums(i);
        }
    }
    
    // Propagate labels
    for (int iter = 0; iter < max_iter_; ++iter) {
        MatrixXd label_distributions_old = label_distributions_;
        
        // Propagate: Y = T * Y
        label_distributions_ = affinity * label_distributions_;
        
        // Clamp labeled samples
        for (int i = 0; i < n_samples; ++i) {
            if (y_fitted_(i) >= 0) {
                for (int c = 0; c < n_classes; ++c) {
                    if (y_fitted_(i) == classes_(c)) {
                        label_distributions_(i, c) = 1.0;
                    } else {
                        label_distributions_(i, c) = 0.0;
                    }
                }
            }
        }
        
        // Normalize
        for (int i = 0; i < n_samples; ++i) {
            double sum = label_distributions_.row(i).sum();
            if (sum > 0) {
                label_distributions_.row(i) /= sum;
            }
        }
        
        // Check convergence
        double change = (label_distributions_ - label_distributions_old).cwiseAbs().maxCoeff();
        if (change < tol_) {
            break;
        }
    }
}

Estimator& LabelPropagation::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    X_fitted_ = X;
    y_fitted_ = y.cast<int>();
    
    // Find unique classes (excluding -1 for unlabeled)
    std::set<int> unique_classes_set;
    for (int i = 0; i < y_fitted_.size(); ++i) {
        if (y_fitted_(i) >= 0) {
            unique_classes_set.insert(y_fitted_(i));
        }
    }
    
    classes_.resize(unique_classes_set.size());
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }
    
    propagate_labels();
    
    fitted_ = true;
    return *this;
}

VectorXi LabelPropagation::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelPropagation must be fitted before predict");
    }
    
    // For new samples, use nearest neighbor approach
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        // Find nearest labeled sample
        int nearest_idx = 0;
        double min_dist = (X.row(i) - X_fitted_.row(0)).squaredNorm();
        
        for (int j = 1; j < X_fitted_.rows(); ++j) {
            double dist = (X.row(i) - X_fitted_.row(j)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = j;
            }
        }
        
        // Predict based on label distribution of nearest sample
        int max_idx = 0;
        for (int c = 1; c < classes_.size(); ++c) {
            if (label_distributions_(nearest_idx, c) > label_distributions_(nearest_idx, max_idx)) {
                max_idx = c;
            }
        }
        predictions(i) = classes_(max_idx);
    }
    
    return predictions;
}

MatrixXd LabelPropagation::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelPropagation must be fitted before predict_proba");
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), classes_.size());
    
    for (int i = 0; i < X.rows(); ++i) {
        int nearest_idx = 0;
        double min_dist = (X.row(i) - X_fitted_.row(0)).squaredNorm();
        
        for (int j = 1; j < X_fitted_.rows(); ++j) {
            double dist = (X.row(i) - X_fitted_.row(j)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = j;
            }
        }
        
        proba.row(i) = label_distributions_.row(nearest_idx);
    }
    
    return proba;
}

VectorXd LabelPropagation::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelPropagation must be fitted before decision_function");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXd decision = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        decision(i) = proba.row(i).maxCoeff();
    }
    
    return decision;
}

Params LabelPropagation::get_params() const {
    Params params;
    params["gamma"] = std::to_string(gamma_);
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    params["kernel"] = kernel_;
    return params;
}

Estimator& LabelPropagation::set_params(const Params& params) {
    gamma_ = utils::get_param_double(params, "gamma", gamma_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    kernel_ = utils::get_param_string(params, "kernel", kernel_);
    return *this;
}

// LabelSpreading implementation

LabelSpreading::LabelSpreading(
    double alpha,
    double gamma,
    int max_iter,
    double tol,
    const std::string& kernel
) : alpha_(alpha), gamma_(gamma), max_iter_(max_iter), tol_(tol), kernel_(kernel), fitted_(false) {
}

MatrixXd LabelSpreading::build_affinity_matrix(const MatrixXd& X) const {
    int n_samples = X.rows();
    MatrixXd affinity = MatrixXd::Zero(n_samples, n_samples);
    
    if (kernel_ == "rbf") {
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_samples; ++j) {
                if (i == j) {
                    affinity(i, j) = 1.0;
                } else {
                    double dist_sq = (X.row(i) - X.row(j)).squaredNorm();
                    affinity(i, j) = std::exp(-gamma_ * dist_sq);
                }
            }
        }
    }
    
    // Normalize
    VectorXd row_sums = affinity.rowwise().sum();
    for (int i = 0; i < n_samples; ++i) {
        if (row_sums(i) > 0) {
            affinity.row(i) /= row_sums(i);
        }
    }
    
    return affinity;
}

void LabelSpreading::propagate_labels() {
    int n_samples = X_fitted_.rows();
    int n_classes = classes_.size();
    
    label_distributions_ = MatrixXd::Zero(n_samples, n_classes);
    
    // Initialize labeled samples
    for (int i = 0; i < n_samples; ++i) {
        if (y_fitted_(i) >= 0) {
            for (int c = 0; c < n_classes; ++c) {
                if (y_fitted_(i) == classes_(c)) {
                    label_distributions_(i, c) = 1.0;
                    break;
                }
            }
        } else {
            label_distributions_.row(i).fill(1.0 / n_classes);
        }
    }
    
    MatrixXd affinity = build_affinity_matrix(X_fitted_);
    
    // Label spreading: Y = alpha * T * Y + (1 - alpha) * Y0
    for (int iter = 0; iter < max_iter_; ++iter) {
        MatrixXd label_distributions_old = label_distributions_;
        
        // Compute new distributions
        MatrixXd Y0 = label_distributions_;
        for (int i = 0; i < n_samples; ++i) {
            if (y_fitted_(i) < 0) {
                Y0.row(i).fill(1.0 / n_classes);
            }
        }
        
        label_distributions_ = alpha_ * affinity * label_distributions_ + (1.0 - alpha_) * Y0;
        
        // Clamp labeled samples
        for (int i = 0; i < n_samples; ++i) {
            if (y_fitted_(i) >= 0) {
                for (int c = 0; c < n_classes; ++c) {
                    if (y_fitted_(i) == classes_(c)) {
                        label_distributions_(i, c) = 1.0;
                    } else {
                        label_distributions_(i, c) = 0.0;
                    }
                }
            }
        }
        
        // Normalize
        for (int i = 0; i < n_samples; ++i) {
            double sum = label_distributions_.row(i).sum();
            if (sum > 0) {
                label_distributions_.row(i) /= sum;
            }
        }
        
        double change = (label_distributions_ - label_distributions_old).cwiseAbs().maxCoeff();
        if (change < tol_) {
            break;
        }
    }
}

Estimator& LabelSpreading::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    X_fitted_ = X;
    y_fitted_ = y.cast<int>();
    
    std::set<int> unique_classes_set;
    for (int i = 0; i < y_fitted_.size(); ++i) {
        if (y_fitted_(i) >= 0) {
            unique_classes_set.insert(y_fitted_(i));
        }
    }
    
    classes_.resize(unique_classes_set.size());
    int idx = 0;
    for (int cls : unique_classes_set) {
        classes_(idx++) = cls;
    }
    
    propagate_labels();
    
    fitted_ = true;
    return *this;
}

VectorXi LabelSpreading::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelSpreading must be fitted before predict");
    }
    
    VectorXi predictions = VectorXi::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        int nearest_idx = 0;
        double min_dist = (X.row(i) - X_fitted_.row(0)).squaredNorm();
        
        for (int j = 1; j < X_fitted_.rows(); ++j) {
            double dist = (X.row(i) - X_fitted_.row(j)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = j;
            }
        }
        
        int max_idx = 0;
        for (int c = 1; c < classes_.size(); ++c) {
            if (label_distributions_(nearest_idx, c) > label_distributions_(nearest_idx, max_idx)) {
                max_idx = c;
            }
        }
        predictions(i) = classes_(max_idx);
    }
    
    return predictions;
}

MatrixXd LabelSpreading::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelSpreading must be fitted before predict_proba");
    }
    
    MatrixXd proba = MatrixXd::Zero(X.rows(), classes_.size());
    
    for (int i = 0; i < X.rows(); ++i) {
        int nearest_idx = 0;
        double min_dist = (X.row(i) - X_fitted_.row(0)).squaredNorm();
        
        for (int j = 1; j < X_fitted_.rows(); ++j) {
            double dist = (X.row(i) - X_fitted_.row(j)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = j;
            }
        }
        
        proba.row(i) = label_distributions_.row(nearest_idx);
    }
    
    return proba;
}

VectorXd LabelSpreading::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("LabelSpreading must be fitted before decision_function");
    }
    
    MatrixXd proba = predict_proba(X);
    VectorXd decision = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        decision(i) = proba.row(i).maxCoeff();
    }
    
    return decision;
}

Params LabelSpreading::get_params() const {
    Params params;
    params["alpha"] = std::to_string(alpha_);
    params["gamma"] = std::to_string(gamma_);
    params["max_iter"] = std::to_string(max_iter_);
    params["tol"] = std::to_string(tol_);
    params["kernel"] = kernel_;
    return params;
}

Estimator& LabelSpreading::set_params(const Params& params) {
    alpha_ = utils::get_param_double(params, "alpha", alpha_);
    gamma_ = utils::get_param_double(params, "gamma", gamma_);
    max_iter_ = utils::get_param_int(params, "max_iter", max_iter_);
    tol_ = utils::get_param_double(params, "tol", tol_);
    kernel_ = utils::get_param_string(params, "kernel", kernel_);
    return *this;
}

} // namespace semi_supervised
} // namespace auroraml


#include "auroraml/lda.hpp"
#include "auroraml/base.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <map>
#include <set>

namespace auroraml {
namespace decomposition {

Estimator& LDA::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    int n_samples = X.rows();
    int n_features = X.cols();
    
    // Validate n_components parameter
    if (n_components_ <= 0) {
        n_components_ = std::min(n_features, static_cast<int>(std::set<int>(y.data(), y.data() + y.size()).size()) - 1);
    }
    if (n_components_ > n_features) {
        throw std::invalid_argument("n_components cannot be greater than n_features");
    }
    
    // Find unique classes and their counts
    std::set<int> unique_classes_set;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes_set.insert(static_cast<int>(y(i)));
    }
    
    class_labels_.assign(unique_classes_set.begin(), unique_classes_set.end());
    int n_classes = class_labels_.size();
    
    if (n_classes < 2) {
        throw std::invalid_argument("LDA requires at least 2 classes");
    }
    
    if (n_components_ >= n_classes) {
        n_components_ = n_classes - 1;
    }
    
    // Create label to index mapping
    label_to_index_.clear();
    for (int i = 0; i < n_classes; ++i) {
        label_to_index_[class_labels_[i]] = i;
    }
    
    // Calculate overall mean
    mean_ = X.colwise().mean();
    
    // Calculate class means and counts
    class_means_.resize(n_classes);
    std::vector<int> class_counts(n_classes, 0);
    
    for (int i = 0; i < n_classes; ++i) {
        class_means_[i] = VectorXd::Zero(n_features);
    }
    
    for (int i = 0; i < n_samples; ++i) {
        int class_idx = label_to_index_[static_cast<int>(y(i))];
        class_means_[class_idx] += X.row(i).transpose();
        class_counts[class_idx]++;
    }
    
    for (int i = 0; i < n_classes; ++i) {
        class_means_[i] /= class_counts[i];
    }
    
    // Calculate within-class scatter matrix (Sw)
    MatrixXd Sw = MatrixXd::Zero(n_features, n_features);
    for (int i = 0; i < n_samples; ++i) {
        int class_idx = label_to_index_[static_cast<int>(y(i))];
        VectorXd diff = X.row(i).transpose() - class_means_[class_idx];
        Sw += diff * diff.transpose();
    }
    
    // Calculate between-class scatter matrix (Sb)
    MatrixXd Sb = MatrixXd::Zero(n_features, n_features);
    for (int i = 0; i < n_classes; ++i) {
        VectorXd diff = class_means_[i] - mean_;
        Sb += class_counts[i] * diff * diff.transpose();
    }
    
    // Solve generalized eigenvalue problem: Sb * v = lambda * Sw * v
    // This is equivalent to solving: Sw^(-1) * Sb * v = lambda * v
    
    // Use generalized eigenvalue decomposition
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> solver(Sb, Sw);
    
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve generalized eigenvalue problem in LDA");
    }
    
    // Get eigenvalues and eigenvectors
    VectorXd eigenvalues = solver.eigenvalues().real();
    MatrixXd eigenvectors = solver.eigenvectors().real();
    
    // Sort eigenvalues in descending order
    std::vector<std::pair<double, int>> eigen_pairs;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigen_pairs.push_back(std::make_pair(eigenvalues(i), i));
    }
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), 
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });
    
    // Select top n_components eigenvectors
    components_ = MatrixXd(n_components_, n_features);
    explained_variance_ = VectorXd(n_components_);
    
    for (int i = 0; i < n_components_; ++i) {
        int idx = eigen_pairs[i].second;
        components_.row(i) = eigenvectors.col(idx).transpose();
        explained_variance_(i) = eigen_pairs[i].first;
    }
    
    fitted_ = true;
    return static_cast<Estimator&>(*this);
}

MatrixXd LDA::transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("LDA must be fitted before transform.");
    validation::check_X(X);
    if (X.cols() != mean_.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    MatrixXd Xc = X.rowwise() - mean_.transpose();
    MatrixXd T = Xc * components_.transpose();
    return T;
}

MatrixXd LDA::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("LDA must be fitted before inverse_transform.");
    validation::check_X(X);
    if (X.cols() != components_.rows()) {
        throw std::invalid_argument("X must have the same number of components as LDA");
    }
    
    MatrixXd X_reconstructed = X * components_;
    MatrixXd X_original = X_reconstructed.rowwise() + mean_.transpose();
    return X_original;
}

MatrixXd LDA::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Estimator& LDA::set_params(const Params& params) {
    if (params.find("n_components") != params.end()) {
        n_components_ = std::stoi(params.at("n_components"));
    }
    return static_cast<Estimator&>(*this);
}

} // namespace decomposition
} // namespace cxml

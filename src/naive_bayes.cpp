#include "ingenuityml/naive_bayes.hpp"
#include <cmath>
#include <limits>

namespace ingenuityml {
namespace naive_bayes {

Estimator& GaussianNB::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);

    // collect classes
    std::set<int> cls;
    for (int i = 0; i < y.size(); ++i) cls.insert(static_cast<int>(y(i)));
    classes_.assign(cls.begin(), cls.end());

    // compute class priors, means and variances per feature
    int n_samples = X.rows();
    int n_features = X.cols();

    class_prior_.clear();
    theta_.clear();
    sigma_.clear();

    for (int c : classes_) {
        // indices for class c
        std::vector<int> idx;
        idx.reserve(n_samples);
        for (int i = 0; i < n_samples; ++i) if (static_cast<int>(y(i)) == c) idx.push_back(i);

        double prior = static_cast<double>(idx.size()) / static_cast<double>(n_samples);
        class_prior_[c] = prior;

        VectorXd mean = VectorXd::Zero(n_features);
        VectorXd var = VectorXd::Zero(n_features);

        if (!idx.empty()) {
            for (int j = 0; j < n_features; ++j) {
                double s = 0.0;
                for (int i : idx) s += X(i, j);
                mean(j) = s / static_cast<double>(idx.size());
            }
            for (int j = 0; j < n_features; ++j) {
                double ss = 0.0;
                for (int i : idx) {
                    double d = X(i, j) - mean(j);
                    ss += d * d;
                }
                var(j) = ss / static_cast<double>(idx.size());
                // smoothing
                var(j) = std::max(var(j), var_smoothing_);
            }
        }

        theta_[c] = mean;
        sigma_[c] = var;
    }

    fitted_ = true;
    return *this;
}

double GaussianNB::log_gaussian_likelihood(double x, double mean, double var) const {
    // log N(x | mean, var)
    const double two_pi = 2.0 * M_PI;
    return -0.5 * (std::log(two_pi * var) + (x - mean) * (x - mean) / var);
}

VectorXi GaussianNB::predict_classes(const MatrixXd& X) const {
    MatrixXd proba = predict_proba(X);
    VectorXi pred(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        // argmax
        int best_k = 0;
        double best_v = proba(i, 0);
        for (int k = 1; k < proba.cols(); ++k) {
            if (proba(i, k) > best_v) { best_v = proba(i, k); best_k = k; }
        }
        pred(i) = classes_[best_k];
    }
    return pred;
}

MatrixXd GaussianNB::predict_proba(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("GaussianNB must be fitted before predict_proba.");
    validation::check_X(X);

    int n_classes = static_cast<int>(classes_.size());
    MatrixXd log_proba(X.rows(), n_classes);

    for (int i = 0; i < X.rows(); ++i) {
        for (int k = 0; k < n_classes; ++k) {
            int c = classes_[k];
            double lp = std::log(std::max(class_prior_.at(c), std::numeric_limits<double>::min()));
            const VectorXd& mean = theta_.at(c);
            const VectorXd& var = sigma_.at(c);
            for (int j = 0; j < X.cols(); ++j) {
                lp += log_gaussian_likelihood(X(i, j), mean(j), var(j));
            }
            log_proba(i, k) = lp;
        }
        // log-sum-exp normalization
        double max_log = log_proba.row(i).maxCoeff();
        double sum_exp = (log_proba.row(i).array() - max_log).exp().sum();
        for (int k = 0; k < n_classes; ++k) {
            log_proba(i, k) = std::exp(log_proba(i, k) - max_log) / sum_exp;
        }
    }
    return log_proba;
}

VectorXd GaussianNB::decision_function(const MatrixXd& X) const {
    // For binary case, return probability of positive class (second class)
    MatrixXd proba = predict_proba(X);
    if (proba.cols() == 2) return proba.col(1);
    // For multiclass, return max probability
    VectorXd d(X.rows());
    for (int i = 0; i < X.rows(); ++i) d(i) = proba.row(i).maxCoeff();
    return d;
}

// GaussianNB save/load implementation
void GaussianNB::save(const std::string& filepath) const {
    if (!fitted_) {
        throw std::runtime_error("GaussianNB must be fitted before saving");
    }
    
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&var_smoothing_), sizeof(var_smoothing_));
    
    // Save class priors
    int priors_size = class_prior_.size();
    ofs.write(reinterpret_cast<const char*>(&priors_size), sizeof(priors_size));
    for (const auto& pair : class_prior_) {
        int class_label = pair.first;
        double prior = pair.second;
        ofs.write(reinterpret_cast<const char*>(&class_label), sizeof(class_label));
        ofs.write(reinterpret_cast<const char*>(&prior), sizeof(prior));
    }
    
    // Save class means (theta_)
    int means_size = theta_.size();
    ofs.write(reinterpret_cast<const char*>(&means_size), sizeof(means_size));
    for (const auto& pair : theta_) {
        int class_label = pair.first;
        int mean_size = pair.second.size();
        ofs.write(reinterpret_cast<const char*>(&class_label), sizeof(class_label));
        ofs.write(reinterpret_cast<const char*>(&mean_size), sizeof(mean_size));
        if (mean_size > 0) {
            ofs.write(reinterpret_cast<const char*>(pair.second.data()), mean_size * sizeof(double));
        }
    }
    
    // Save class variances (sigma_)
    int vars_size = sigma_.size();
    ofs.write(reinterpret_cast<const char*>(&vars_size), sizeof(vars_size));
    for (const auto& pair : sigma_) {
        int class_label = pair.first;
        int var_size = pair.second.size();
        ofs.write(reinterpret_cast<const char*>(&class_label), sizeof(class_label));
        ofs.write(reinterpret_cast<const char*>(&var_size), sizeof(var_size));
        if (var_size > 0) {
            ofs.write(reinterpret_cast<const char*>(pair.second.data()), var_size * sizeof(double));
        }
    }
    
    // Save classes
    int classes_size = classes_.size();
    ofs.write(reinterpret_cast<const char*>(&classes_size), sizeof(classes_size));
    if (classes_size > 0) {
        ofs.write(reinterpret_cast<const char*>(classes_.data()), classes_size * sizeof(int));
    }
    
    ofs.close();
}

void GaussianNB::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&var_smoothing_), sizeof(var_smoothing_));
    
    // Load class priors
    int priors_size;
    ifs.read(reinterpret_cast<char*>(&priors_size), sizeof(priors_size));
    class_prior_.clear();
    for (int i = 0; i < priors_size; ++i) {
        int class_label;
        double prior;
        ifs.read(reinterpret_cast<char*>(&class_label), sizeof(class_label));
        ifs.read(reinterpret_cast<char*>(&prior), sizeof(prior));
        class_prior_[class_label] = prior;
    }
    
    // Load class means (theta_)
    int means_size;
    ifs.read(reinterpret_cast<char*>(&means_size), sizeof(means_size));
    theta_.clear();
    for (int i = 0; i < means_size; ++i) {
        int class_label;
        int mean_size;
        ifs.read(reinterpret_cast<char*>(&class_label), sizeof(class_label));
        ifs.read(reinterpret_cast<char*>(&mean_size), sizeof(mean_size));
        VectorXd means(mean_size);
        if (mean_size > 0) {
            ifs.read(reinterpret_cast<char*>(means.data()), mean_size * sizeof(double));
        }
        theta_[class_label] = means;
    }
    
    // Load class variances (sigma_)
    int vars_size;
    ifs.read(reinterpret_cast<char*>(&vars_size), sizeof(vars_size));
    sigma_.clear();
    for (int i = 0; i < vars_size; ++i) {
        int class_label;
        int var_size;
        ifs.read(reinterpret_cast<char*>(&class_label), sizeof(class_label));
        ifs.read(reinterpret_cast<char*>(&var_size), sizeof(var_size));
        VectorXd vars(var_size);
        if (var_size > 0) {
            ifs.read(reinterpret_cast<char*>(vars.data()), var_size * sizeof(double));
        }
        sigma_[class_label] = vars;
    }
    
    // Load classes
    int classes_size;
    ifs.read(reinterpret_cast<char*>(&classes_size), sizeof(classes_size));
    classes_.resize(classes_size);
    if (classes_size > 0) {
        ifs.read(reinterpret_cast<char*>(classes_.data()), classes_size * sizeof(int));
    }
    
    ifs.close();
}

} // namespace naive_bayes
} // namespace ingenuityml



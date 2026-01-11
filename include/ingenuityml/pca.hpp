#ifndef CXML_PCA_HPP
#define CXML_PCA_HPP

#include "base.hpp"

namespace ingenuityml {
namespace decomposition {

class PCA : public Estimator, public Transformer {
private:
    int n_components_; // -1 means all
    bool whiten_;
    bool fitted_ = false;

    VectorXd mean_;
    MatrixXd components_; // rows = n_components, cols = n_features
    VectorXd explained_variance_;
    double explained_variance_ratio_sum_ = 0.0;

public:
    PCA(int n_components = -1, bool whiten = false)
        : n_components_(n_components), whiten_(whiten) {}

    // Estimator
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    // Transformer
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;

    // API
    const MatrixXd& components() const { 
        if (!fitted_) throw std::runtime_error("PCA must be fitted before accessing components.");
        return components_; 
    }
    const VectorXd& explained_variance() const { 
        if (!fitted_) throw std::runtime_error("PCA must be fitted before accessing explained variance.");
        return explained_variance_; 
    }
    VectorXd explained_variance_ratio() const {
        if (!fitted_) throw std::runtime_error("PCA must be fitted before accessing explained variance ratio.");
        if (explained_variance_.size() == 0) return VectorXd();
        double sum = explained_variance_.sum();
        if (sum <= 0) return VectorXd::Zero(explained_variance_.size());
        VectorXd r = explained_variance_ / sum;
        return r;
    }
    const VectorXd& mean() const { 
        if (!fitted_) throw std::runtime_error("PCA must be fitted before accessing mean.");
        return mean_; 
    }
    bool is_fitted() const override { return fitted_; }

    // Params
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
};

/**
 * KernelPCA - Kernel Principal Component Analysis
 * 
 * Non-linear dimensionality reduction using kernel trick.
 */
class KernelPCA : public Estimator, public Transformer {
private:
    int n_components_;
    std::string kernel_;
    double gamma_;
    double degree_;
    double coef0_;
    MatrixXd alphas_;
    MatrixXd X_fit_;
    VectorXd lambdas_;
    bool fitted_;

public:
    KernelPCA(int n_components = -1, const std::string& kernel = "rbf", 
              double gamma = 1.0, double degree = 3, double coef0 = 1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

private:
    double kernel_function(const VectorXd& x1, const VectorXd& x2) const;
    MatrixXd compute_kernel_matrix(const MatrixXd& X1, const MatrixXd& X2) const;
};

/**
 * FastICA - Fast Independent Component Analysis
 * 
 * Separates multivariate signal into additive independent components.
 */
class FastICA : public Estimator, public Transformer {
private:
    int n_components_;
    std::string algorithm_;
    std::string fun_;
    int max_iter_;
    double tol_;
    int random_state_;
    MatrixXd components_;
    MatrixXd mixing_;
    VectorXd mean_;
    MatrixXd whitening_;
    bool fitted_;

public:
    FastICA(int n_components = -1, const std::string& algorithm = "parallel",
            const std::string& fun = "logcosh", int max_iter = 200,
            double tol = 1e-4, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    MatrixXd components() const { return components_; }
    MatrixXd mixing() const { return mixing_; }

private:
    void whiten(const MatrixXd& X);
    MatrixXd g_function(const MatrixXd& X) const;
    MatrixXd g_prime_function(const MatrixXd& X) const;
};

/**
 * TSNE - t-Distributed Stochastic Neighbor Embedding
 * 
 * Non-linear dimensionality reduction technique for visualization.
 */
class TSNE : public Estimator, public Transformer {
private:
    int n_components_;
    double perplexity_;
    double early_exaggeration_;
    double learning_rate_;
    int max_iter_;
    int random_state_;
    MatrixXd embedding_;
    bool fitted_;

public:
    TSNE(int n_components = 2, double perplexity = 30.0, double early_exaggeration = 12.0,
         double learning_rate = 200.0, int max_iter = 1000, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    MatrixXd embedding() const { return embedding_; }

private:
    MatrixXd compute_pairwise_affinities(const MatrixXd& X) const;
    MatrixXd compute_conditional_probabilities(const MatrixXd& distances) const;
    double compute_perplexity(const VectorXd& prob_row) const;
    void gradient_descent(const MatrixXd& P);
};

/**
 * FactorAnalysis - Factor Analysis
 * 
 * A simple linear generative model with Gaussian latent variables.
 */
class FactorAnalysis : public Estimator, public Transformer {
private:
    int n_components_;
    double tol_;
    int max_iter_;
    int random_state_;
    MatrixXd components_;
    VectorXd noise_variance_;
    VectorXd mean_;
    double loglike_;
    bool fitted_;

public:
    FactorAnalysis(int n_components = -1, double tol = 1e-2, int max_iter = 1000, int random_state = -1);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y = VectorXd()) override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    MatrixXd components() const { return components_; }
    VectorXd noise_variance() const { return noise_variance_; }
    double loglike() const { return loglike_; }

private:
    void fit_em_algorithm(const MatrixXd& X);
};

} // namespace decomposition
} // namespace ingenuityml

#endif // CXML_PCA_HPP



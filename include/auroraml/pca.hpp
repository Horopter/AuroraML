#ifndef CXML_PCA_HPP
#define CXML_PCA_HPP

#include "base.hpp"

namespace auroraml {
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

} // namespace decomposition
} // namespace cxml

#endif // CXML_PCA_HPP



/**
 * Gaussian Naive Bayes classifier
 */
#ifndef CXML_NAIVE_BAYES_HPP
#define CXML_NAIVE_BAYES_HPP

#include "base.hpp"
#include <map>
#include <set>
#include <string>
#include <fstream>

namespace auroraml {
namespace naive_bayes {

class GaussianNB : public Estimator, public Classifier {
private:
    // class -> prior
    std::map<int, double> class_prior_;
    // class -> mean vector
    std::map<int, VectorXd> theta_;
    // class -> variance vector
    std::map<int, VectorXd> sigma_;
    std::vector<int> classes_;
    bool fitted_ = false;
    double var_smoothing_ = 1e-9;

public:
    GaussianNB(double var_smoothing = 1e-9) : var_smoothing_(var_smoothing) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override {
        Params p; p["var_smoothing"] = std::to_string(var_smoothing_); return p;
    }
    Estimator& set_params(const Params& params) override {
        var_smoothing_ = utils::get_param_double(params, "var_smoothing", 1e-9);
        return *this;
    }
    bool is_fitted() const override { return fitted_; }
    
    // Model persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    double log_gaussian_likelihood(double x, double mean, double var) const;
};

} // namespace naive_bayes
} // namespace cxml

#endif // CXML_NAIVE_BAYES_HPP



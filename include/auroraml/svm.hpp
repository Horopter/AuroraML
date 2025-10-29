#pragma once

#include "base.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>

namespace auroraml {
namespace svm {

// Linear SVM classifier (hinge loss) optimized via SGD
class LinearSVC : public Estimator, public Classifier {
private:
    VectorXd w_;
    double b_ = 0.0;
    bool fitted_ = false;

    // params
    double C_;          // regularization strength (1/lambda)
    int max_iter_;
    double lr_;         // learning rate
    int random_state_;

public:
    LinearSVC(double C = 1.0, int max_iter = 1000, double lr = 0.01, int random_state = -1)
        : C_(C), max_iter_(max_iter), lr_(lr), random_state_(random_state) {}

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;

    // Classifier API
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;

    // Estimator API
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    
    // Model persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
};

} // namespace svm
} // namespace cxml



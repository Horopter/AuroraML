#pragma once

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <memory>

namespace auroraml {

using MatrixXd = Eigen::MatrixXd;
using MatrixXi = Eigen::MatrixXi;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;

// Parameter map type
using Params = std::unordered_map<std::string, std::string>;

// Utility functions for parameter management
namespace utils {
    std::string get_param_string(const Params& params, const std::string& key, const std::string& default_val);
    int get_param_int(const Params& params, const std::string& key, int default_val);
    double get_param_double(const Params& params, const std::string& key, double default_val);
    bool get_param_bool(const Params& params, const std::string& key, bool default_val);
}

// Data validation utilities
namespace validation {
    void check_X_y(const MatrixXd& X, const VectorXd& y);
    void check_X(const MatrixXd& X);
}

/**
 * Base Estimator class
 */
class Estimator {
public:
    virtual ~Estimator() = default;
    
    virtual Estimator& fit(const MatrixXd& X, const VectorXd& y) = 0;
    virtual Params get_params() const = 0;
    virtual Estimator& set_params(const Params& params) = 0;
    virtual bool is_fitted() const = 0;
};

/**
 * Base Predictor class
 */
class Predictor {
public:
    virtual ~Predictor() = default;
    virtual VectorXd predict(const MatrixXd& X) const = 0;
};

/**
 * Classifier interface
 */
class Classifier : public Predictor {
public:
    virtual VectorXi predict_classes(const MatrixXd& X) const = 0;
    virtual MatrixXd predict_proba(const MatrixXd& X) const = 0;
    virtual VectorXd decision_function(const MatrixXd& X) const = 0;
    
    // Override the base predict method to call the integer version
    VectorXd predict(const MatrixXd& X) const override {
        VectorXi int_pred = predict_classes(X);
        return int_pred.cast<double>();
    }
};

/**
 * Regressor interface
 */
class Regressor : public Predictor {
public:
    virtual VectorXd predict(const MatrixXd& X) const = 0;
};

/**
 * Base Transformer class
 */
class Transformer {
public:
    virtual ~Transformer() = default;
    virtual MatrixXd transform(const MatrixXd& X) const = 0;
    virtual MatrixXd inverse_transform(const MatrixXd& X) const = 0;
    virtual MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) = 0;
};

} // namespace cxml
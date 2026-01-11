#include "ingenuityml/dummy.hpp"
#include "ingenuityml/utils.hpp"
#include "ingenuityml/base.hpp"
#include <algorithm>
#include <vector>
#include <limits>

namespace ingenuityml {
namespace ensemble {

DummyClassifier::DummyClassifier(const std::string& strategy)
    : strategy_(strategy),
      fitted_(false),
      classes_(),
      most_frequent_class_(0),
      class_counts_(),
      n_features_(0) {
    if (strategy_ != "most_frequent" && strategy_ != "uniform") {
        throw std::invalid_argument("Unsupported strategy: " + strategy_);
    }
}

Estimator& DummyClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (strategy_ != "most_frequent" && strategy_ != "uniform") {
        throw std::invalid_argument("Unsupported strategy: " + strategy_);
    }
    n_features_ = static_cast<int>(X.cols());
    VectorXi y_int = y.cast<int>();
    std::set<int> unique_classes;
    class_counts_.clear();

    for (int i = 0; i < y_int.size(); ++i) {
        int cls = y_int(i);
        unique_classes.insert(cls);
        class_counts_[cls]++;
    }

    if (unique_classes.empty()) {
        throw std::runtime_error("No classes found");
    }
    classes_ = VectorXi(static_cast<int>(unique_classes.size()));
    int idx = 0;
    for (int cls : unique_classes) {
        classes_(idx++) = cls;
    }

    most_frequent_class_ = classes_(0);
    int max_count = class_counts_[most_frequent_class_];
    for (const auto& pair : class_counts_) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_frequent_class_ = pair.first;
        }
    }

    fitted_ = true;
    return *this;
}

VectorXi DummyClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("DummyClassifier not fitted");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    VectorXi predictions = VectorXi::Zero(X.rows());
    if (strategy_ == "most_frequent") {
        predictions.array() = most_frequent_class_;
    } else if (strategy_ == "uniform") {
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, classes_.size() - 1);
        for (int i = 0; i < predictions.size(); ++i) {
            predictions(i) = classes_(dist(gen));
        }
    } else {
        throw std::invalid_argument("Unsupported strategy: " + strategy_);
    }

    return predictions;
}

MatrixXd DummyClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("DummyClassifier not fitted");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd proba = MatrixXd::Zero(X.rows(), classes_.size());
    if (strategy_ == "most_frequent") {
        int most_freq_idx = 0;
        for (int i = 0; i < classes_.size(); ++i) {
            if (classes_(i) == most_frequent_class_) {
                most_freq_idx = i;
                break;
            }
        }
        for (int i = 0; i < proba.rows(); ++i) {
            proba(i, most_freq_idx) = 1.0;
        }
    } else if (strategy_ == "uniform") {
        double prob = 1.0 / static_cast<double>(classes_.size());
        proba.array() = prob;
    } else {
        throw std::invalid_argument("Unsupported strategy: " + strategy_);
    }

    return proba;
}

VectorXd DummyClassifier::decision_function(const MatrixXd& X) const {
    MatrixXd proba = predict_proba(X);
    return proba.rowwise().sum();
}

Params DummyClassifier::get_params() const {
    Params p;
    p["strategy"] = strategy_;
    return p;
}

Estimator& DummyClassifier::set_params(const Params& params) {
    std::string strategy = utils::get_param_string(params, "strategy", strategy_);
    if (strategy != "most_frequent" && strategy != "uniform") {
        throw std::invalid_argument("Unsupported strategy: " + strategy);
    }
    strategy_ = strategy;
    return *this;
}

DummyRegressor::DummyRegressor(const std::string& strategy, double quantile, double constant)
    : strategy_(strategy),
      quantile_(quantile),
      constant_(constant),
      statistic_(0.0),
      fitted_(false),
      n_features_(0) {
    if (strategy_ != "mean" && strategy_ != "median" && strategy_ != "quantile" && strategy_ != "constant") {
        throw std::invalid_argument("Unsupported strategy: " + strategy_);
    }
}

Estimator& DummyRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    if (strategy_ != "mean" && strategy_ != "median" && strategy_ != "quantile" && strategy_ != "constant") {
        throw std::invalid_argument("Unsupported strategy: " + strategy_);
    }
    n_features_ = static_cast<int>(X.cols());

    std::vector<double> values(y.data(), y.data() + y.size());
    if (values.empty()) {
        throw std::runtime_error("y cannot be empty");
    }

    if (strategy_ == "mean") {
        statistic_ = y.mean();
    } else if (strategy_ == "median") {
        std::sort(values.begin(), values.end());
        size_t mid = values.size() / 2;
        if (values.size() % 2 == 0) {
            statistic_ = 0.5 * (values[mid - 1] + values[mid]);
        } else {
            statistic_ = values[mid];
        }
    } else if (strategy_ == "quantile") {
        if (quantile_ < 0.0 || quantile_ > 1.0) {
            throw std::invalid_argument("quantile must be between 0 and 1");
        }
        std::sort(values.begin(), values.end());
        double pos = quantile_ * (values.size() - 1);
        size_t idx = static_cast<size_t>(std::floor(pos));
        size_t idx_hi = std::min(idx + 1, values.size() - 1);
        double frac = pos - static_cast<double>(idx);
        statistic_ = values[idx] + frac * (values[idx_hi] - values[idx]);
    } else if (strategy_ == "constant") {
        statistic_ = constant_;
    }

    fitted_ = true;
    return *this;
}

VectorXd DummyRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("DummyRegressor not fitted");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    return VectorXd::Constant(X.rows(), statistic_);
}

Params DummyRegressor::get_params() const {
    Params p;
    p["strategy"] = strategy_;
    p["quantile"] = std::to_string(quantile_);
    p["constant"] = std::to_string(constant_);
    return p;
}

Estimator& DummyRegressor::set_params(const Params& params) {
    std::string strategy = utils::get_param_string(params, "strategy", strategy_);
    if (strategy != "mean" && strategy != "median" && strategy != "quantile" && strategy != "constant") {
        throw std::invalid_argument("Unsupported strategy: " + strategy);
    }
    strategy_ = strategy;
    quantile_ = utils::get_param_double(params, "quantile", quantile_);
    constant_ = utils::get_param_double(params, "constant", constant_);
    return *this;
}

} // namespace ensemble
} // namespace ingenuityml

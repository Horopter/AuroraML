#include "auroraml/meta_estimators.hpp"
#include "auroraml/utils.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <sstream>
#include <cmath>
#include <limits>

namespace auroraml {
namespace meta {

namespace {

std::shared_ptr<Classifier> create_classifier(const std::function<std::shared_ptr<Classifier>()>& factory) {
    if (!factory) {
        throw std::invalid_argument("estimator_factory must be provided");
    }
    auto estimator = factory();
    if (!estimator) {
        throw std::runtime_error("estimator_factory returned null classifier");
    }
    return estimator;
}

std::shared_ptr<Regressor> create_regressor(const std::function<std::shared_ptr<Regressor>()>& factory) {
    if (!factory) {
        throw std::invalid_argument("estimator_factory must be provided");
    }
    auto estimator = factory();
    if (!estimator) {
        throw std::runtime_error("estimator_factory returned null regressor");
    }
    return estimator;
}

Estimator& fit_classifier(const std::shared_ptr<Classifier>& classifier, const MatrixXd& X, const VectorXd& y) {
    auto est = dynamic_cast<Estimator*>(classifier.get());
    if (!est) {
        throw std::runtime_error("Base classifier must inherit from Estimator and Classifier");
    }
    return est->fit(X, y);
}

Estimator& fit_regressor(const std::shared_ptr<Regressor>& regressor, const MatrixXd& X, const VectorXd& y) {
    auto est = dynamic_cast<Estimator*>(regressor.get());
    if (!est) {
        throw std::runtime_error("Base regressor must inherit from Estimator and Regressor");
    }
    return est->fit(X, y);
}

MatrixXd softmax_rows(const MatrixXd& scores) {
    MatrixXd probs(scores.rows(), scores.cols());
    for (int i = 0; i < scores.rows(); ++i) {
        double max_val = scores.row(i).maxCoeff();
        VectorXd exps = (scores.row(i).array() - max_val).exp();
        double sum = exps.sum();
        if (sum <= 0.0) {
            probs.row(i).setConstant(1.0 / static_cast<double>(scores.cols()));
        } else {
            probs.row(i) = (exps / sum).transpose();
        }
    }
    return probs;
}

std::string join_indices(const std::vector<int>& indices) {
    std::ostringstream oss;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << indices[i];
    }
    return oss.str();
}

std::vector<int> parse_indices(const std::string& value) {
    std::vector<int> indices;
    if (value.empty()) {
        return indices;
    }
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            indices.push_back(std::stoi(token));
        }
    }
    return indices;
}

void validate_order(const std::vector<int>& order, int n_outputs) {
    if (order.size() != static_cast<size_t>(n_outputs)) {
        throw std::invalid_argument("order must include each output exactly once");
    }
    std::vector<int> sorted = order;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < n_outputs; ++i) {
        if (sorted[i] != i) {
            throw std::invalid_argument("order must be a permutation of output indices");
        }
    }
}

MatrixXd build_augmented(const MatrixXd& X, const MatrixXd& extra) {
    if (extra.cols() == 0) {
        return X;
    }
    MatrixXd augmented(X.rows(), X.cols() + extra.cols());
    augmented << X, extra;
    return augmented;
}

} // namespace

OneVsRestClassifier::OneVsRestClassifier(std::function<std::shared_ptr<Classifier>()> estimator_factory, int n_jobs)
    : estimator_factory_(std::move(estimator_factory)), n_jobs_(n_jobs), fitted_(false), classes_(), estimators_() {}

Estimator& OneVsRestClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    VectorXi y_int = y.cast<int>();
    classes_ = utils::multiclass::unique_labels(y_int);
    estimators_.clear();

    for (int i = 0; i < classes_.size(); ++i) {
        int cls = classes_(i);
        VectorXd y_binary = (y_int.array() == cls).cast<double>();
        auto estimator = create_classifier(estimator_factory_);
        fit_classifier(estimator, X, y_binary);
        estimators_.push_back(estimator);
    }

    fitted_ = true;
    return *this;
}

VectorXi OneVsRestClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneVsRestClassifier must be fitted before predict");
    }
    MatrixXd scores(X.rows(), classes_.size());
    for (int i = 0; i < classes_.size(); ++i) {
        VectorXd decision = estimators_[i]->decision_function(X);
        if (decision.size() != X.rows()) {
            throw std::runtime_error("Base estimator returned unexpected decision size");
        }
        scores.col(i) = decision;
    }

    VectorXi predictions(X.rows());
    for (int i = 0; i < scores.rows(); ++i) {
        Eigen::Index max_idx = 0;
        scores.row(i).maxCoeff(&max_idx);
        predictions(i) = classes_(static_cast<int>(max_idx));
    }
    return predictions;
}

MatrixXd OneVsRestClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneVsRestClassifier must be fitted before predict_proba");
    }
    MatrixXd scores(X.rows(), classes_.size());
    for (int i = 0; i < classes_.size(); ++i) {
        VectorXd decision = estimators_[i]->decision_function(X);
        scores.col(i) = decision;
    }
    return softmax_rows(scores);
}

VectorXd OneVsRestClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneVsRestClassifier must be fitted before decision_function");
    }
    MatrixXd scores(X.rows(), classes_.size());
    for (int i = 0; i < classes_.size(); ++i) {
        scores.col(i) = estimators_[i]->decision_function(X);
    }
    VectorXd max_scores(X.rows());
    for (int i = 0; i < scores.rows(); ++i) {
        max_scores(i) = scores.row(i).maxCoeff();
    }
    return max_scores;
}

Params OneVsRestClassifier::get_params() const {
    Params params;
    params["n_jobs"] = std::to_string(n_jobs_);
    return params;
}

Estimator& OneVsRestClassifier::set_params(const Params& params) {
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

OneVsOneClassifier::OneVsOneClassifier(std::function<std::shared_ptr<Classifier>()> estimator_factory, int n_jobs)
    : estimator_factory_(std::move(estimator_factory)), n_jobs_(n_jobs), fitted_(false), classes_(), estimators_(), class_pairs_() {}

Estimator& OneVsOneClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    VectorXi y_int = y.cast<int>();
    classes_ = utils::multiclass::unique_labels(y_int);

    estimators_.clear();
    class_pairs_.clear();

    for (int i = 0; i < classes_.size(); ++i) {
        for (int j = i + 1; j < classes_.size(); ++j) {
            int class_i = classes_(i);
            int class_j = classes_(j);

            std::vector<int> indices;
            for (int k = 0; k < y_int.size(); ++k) {
                if (y_int(k) == class_i || y_int(k) == class_j) {
                    indices.push_back(k);
                }
            }

            MatrixXd X_subset(indices.size(), X.cols());
            VectorXd y_subset(indices.size());
            for (size_t idx = 0; idx < indices.size(); ++idx) {
                X_subset.row(idx) = X.row(indices[idx]);
                y_subset(idx) = (y_int(indices[idx]) == class_j) ? 1.0 : 0.0;
            }

            auto estimator = create_classifier(estimator_factory_);
            fit_classifier(estimator, X_subset, y_subset);
            estimators_.push_back(estimator);
            class_pairs_.push_back({class_i, class_j});
        }
    }

    fitted_ = true;
    return *this;
}

VectorXi OneVsOneClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneVsOneClassifier must be fitted before predict");
    }

    std::unordered_map<int, int> class_index;
    for (int i = 0; i < classes_.size(); ++i) {
        class_index[classes_(i)] = i;
    }

    MatrixXi votes = MatrixXi::Zero(X.rows(), classes_.size());
    MatrixXd scores = MatrixXd::Zero(X.rows(), classes_.size());

    for (size_t k = 0; k < estimators_.size(); ++k) {
        int class_i = class_pairs_[k].first;
        int class_j = class_pairs_[k].second;
        int idx_i = class_index[class_i];
        int idx_j = class_index[class_j];

        VectorXi pred = estimators_[k]->predict_classes(X);
        VectorXd decision = estimators_[k]->decision_function(X);

        for (int i = 0; i < pred.size(); ++i) {
            int predicted_label = (pred(i) == 1) ? class_j : class_i;
            int target_idx = (predicted_label == class_j) ? idx_j : idx_i;
            votes(i, target_idx) += 1;
            scores(i, target_idx) += std::abs(decision(i));
        }
    }

    VectorXi predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        int best_class = 0;
        int best_votes = votes(i, 0);
        double best_score = scores(i, 0);
        for (int j = 1; j < classes_.size(); ++j) {
            if (votes(i, j) > best_votes || (votes(i, j) == best_votes && scores(i, j) > best_score)) {
                best_class = j;
                best_votes = votes(i, j);
                best_score = scores(i, j);
            }
        }
        predictions(i) = classes_(best_class);
    }

    return predictions;
}

MatrixXd OneVsOneClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneVsOneClassifier must be fitted before predict_proba");
    }
    VectorXi preds = predict_classes(X);
    MatrixXd proba = MatrixXd::Zero(X.rows(), classes_.size());
    std::unordered_map<int, int> class_index;
    for (int i = 0; i < classes_.size(); ++i) {
        class_index[classes_(i)] = i;
    }
    for (int i = 0; i < preds.size(); ++i) {
        proba(i, class_index[preds(i)]) = 1.0;
    }
    return proba;
}

VectorXd OneVsOneClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneVsOneClassifier must be fitted before decision_function");
    }
    VectorXi preds = predict_classes(X);
    VectorXd decision(X.rows());
    for (int i = 0; i < preds.size(); ++i) {
        decision(i) = static_cast<double>(preds(i));
    }
    return decision;
}

Params OneVsOneClassifier::get_params() const {
    Params params;
    params["n_jobs"] = std::to_string(n_jobs_);
    return params;
}

Estimator& OneVsOneClassifier::set_params(const Params& params) {
    n_jobs_ = utils::get_param_int(params, "n_jobs", n_jobs_);
    return *this;
}

OutputCodeClassifier::OutputCodeClassifier(std::function<std::shared_ptr<Classifier>()> estimator_factory,
                                           int code_size, int random_state)
    : estimator_factory_(std::move(estimator_factory)),
      code_size_(code_size),
      random_state_(random_state),
      fitted_(false),
      classes_(),
      code_book_(),
      estimators_() {}

Estimator& OutputCodeClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    VectorXi y_int = y.cast<int>();
    classes_ = utils::multiclass::unique_labels(y_int);

    int n_classes = classes_.size();
    int size = code_size_ > 0 ? code_size_ : std::max(2, n_classes);
    code_book_.resize(n_classes, size);

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 1);

    for (int i = 0; i < n_classes; ++i) {
        for (int j = 0; j < size; ++j) {
            code_book_(i, j) = dist(rng) == 1 ? 1 : -1;
        }
    }

    estimators_.clear();
    std::unordered_map<int, int> class_index;
    for (int i = 0; i < n_classes; ++i) {
        class_index[classes_(i)] = i;
    }

    for (int j = 0; j < size; ++j) {
        VectorXd y_bit(y_int.size());
        for (int i = 0; i < y_int.size(); ++i) {
            int idx = class_index[y_int(i)];
            y_bit(i) = code_book_(idx, j) == 1 ? 1.0 : 0.0;
        }
        auto estimator = create_classifier(estimator_factory_);
        fit_classifier(estimator, X, y_bit);
        estimators_.push_back(estimator);
    }

    fitted_ = true;
    return *this;
}

VectorXi OutputCodeClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OutputCodeClassifier must be fitted before predict");
    }
    const int n_samples = X.rows();
    const int n_classes = classes_.size();
    const int size = code_book_.cols();

    MatrixXi pred_code = MatrixXi::Zero(n_samples, size);
    for (int j = 0; j < size; ++j) {
        VectorXd decision = estimators_[j]->decision_function(X);
        for (int i = 0; i < n_samples; ++i) {
            pred_code(i, j) = decision(i) >= 0.0 ? 1 : -1;
        }
    }

    VectorXi predictions(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        int best_class = 0;
        int best_dist = std::numeric_limits<int>::max();
        for (int c = 0; c < n_classes; ++c) {
            int dist = 0;
            for (int j = 0; j < size; ++j) {
                if (pred_code(i, j) != code_book_(c, j)) {
                    dist += 1;
                }
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_class = c;
            }
        }
        predictions(i) = classes_(best_class);
    }
    return predictions;
}

MatrixXd OutputCodeClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OutputCodeClassifier must be fitted before predict_proba");
    }
    const int n_samples = X.rows();
    const int n_classes = classes_.size();
    const int size = code_book_.cols();

    MatrixXi pred_code = MatrixXi::Zero(n_samples, size);
    for (int j = 0; j < size; ++j) {
        VectorXd decision = estimators_[j]->decision_function(X);
        for (int i = 0; i < n_samples; ++i) {
            pred_code(i, j) = decision(i) >= 0.0 ? 1 : -1;
        }
    }

    MatrixXd distances = MatrixXd::Zero(n_samples, n_classes);
    for (int i = 0; i < n_samples; ++i) {
        for (int c = 0; c < n_classes; ++c) {
            int dist = 0;
            for (int j = 0; j < size; ++j) {
                if (pred_code(i, j) != code_book_(c, j)) {
                    dist += 1;
                }
            }
            distances(i, c) = static_cast<double>(dist);
        }
    }

    MatrixXd scores = -distances;
    return softmax_rows(scores);
}

VectorXd OutputCodeClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("OutputCodeClassifier must be fitted before decision_function");
    }
    VectorXi preds = predict_classes(X);
    VectorXd decision(X.rows());
    for (int i = 0; i < preds.size(); ++i) {
        decision(i) = static_cast<double>(preds(i));
    }
    return decision;
}

Params OutputCodeClassifier::get_params() const {
    Params params;
    params["code_size"] = std::to_string(code_size_);
    params["random_state"] = std::to_string(random_state_);
    return params;
}

Estimator& OutputCodeClassifier::set_params(const Params& params) {
    code_size_ = utils::get_param_int(params, "code_size", code_size_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

MultiOutputClassifier::MultiOutputClassifier(std::function<std::shared_ptr<Classifier>()> estimator_factory)
    : estimator_factory_(std::move(estimator_factory)), fitted_(false), n_outputs_(0), estimators_() {}

Estimator& MultiOutputClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd Y(X.rows(), 1);
    Y.col(0) = y;
    return fit(X, Y);
}

Estimator& MultiOutputClassifier::fit(const MatrixXd& X, const MatrixXd& Y) {
    if (X.rows() != Y.rows()) {
        throw std::invalid_argument("X and Y must have the same number of samples");
    }
    n_outputs_ = static_cast<int>(Y.cols());
    estimators_.clear();

    for (int j = 0; j < n_outputs_; ++j) {
        auto estimator = create_classifier(estimator_factory_);
        fit_classifier(estimator, X, Y.col(j));
        estimators_.push_back(estimator);
    }

    fitted_ = true;
    return *this;
}

MatrixXi MultiOutputClassifier::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultiOutputClassifier must be fitted before predict");
    }
    MatrixXi predictions(X.rows(), n_outputs_);
    for (int j = 0; j < n_outputs_; ++j) {
        predictions.col(j) = estimators_[j]->predict_classes(X);
    }
    return predictions;
}

std::vector<MatrixXd> MultiOutputClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultiOutputClassifier must be fitted before predict_proba");
    }
    std::vector<MatrixXd> probs;
    probs.reserve(n_outputs_);
    for (int j = 0; j < n_outputs_; ++j) {
        probs.push_back(estimators_[j]->predict_proba(X));
    }
    return probs;
}

Params MultiOutputClassifier::get_params() const {
    return Params{};
}

Estimator& MultiOutputClassifier::set_params(const Params& params) {
    (void)params;
    return *this;
}

MultiOutputRegressor::MultiOutputRegressor(std::function<std::shared_ptr<Regressor>()> estimator_factory)
    : estimator_factory_(std::move(estimator_factory)), fitted_(false), n_outputs_(0), estimators_() {}

Estimator& MultiOutputRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd Y(X.rows(), 1);
    Y.col(0) = y;
    return fit(X, Y);
}

Estimator& MultiOutputRegressor::fit(const MatrixXd& X, const MatrixXd& Y) {
    if (X.rows() != Y.rows()) {
        throw std::invalid_argument("X and Y must have the same number of samples");
    }
    n_outputs_ = static_cast<int>(Y.cols());
    estimators_.clear();

    for (int j = 0; j < n_outputs_; ++j) {
        auto estimator = create_regressor(estimator_factory_);
        fit_regressor(estimator, X, Y.col(j));
        estimators_.push_back(estimator);
    }

    fitted_ = true;
    return *this;
}

MatrixXd MultiOutputRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MultiOutputRegressor must be fitted before predict");
    }
    MatrixXd predictions(X.rows(), n_outputs_);
    for (int j = 0; j < n_outputs_; ++j) {
        predictions.col(j) = estimators_[j]->predict(X);
    }
    return predictions;
}

Params MultiOutputRegressor::get_params() const {
    return Params{};
}

Estimator& MultiOutputRegressor::set_params(const Params& params) {
    (void)params;
    return *this;
}

ClassifierChain::ClassifierChain(std::function<std::shared_ptr<Classifier>()> estimator_factory,
                                 const std::vector<int>& order)
    : estimator_factory_(std::move(estimator_factory)), fitted_(false), n_outputs_(0), order_(order), estimators_() {}

Estimator& ClassifierChain::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd Y(X.rows(), 1);
    Y.col(0) = y;
    return fit(X, Y);
}

Estimator& ClassifierChain::fit(const MatrixXd& X, const MatrixXd& Y) {
    if (X.rows() != Y.rows()) {
        throw std::invalid_argument("X and Y must have the same number of samples");
    }
    n_outputs_ = static_cast<int>(Y.cols());
    if (order_.empty()) {
        order_.resize(n_outputs_);
        std::iota(order_.begin(), order_.end(), 0);
    }
    validate_order(order_, n_outputs_);

    estimators_.clear();
    MatrixXd prev;
    for (int idx = 0; idx < n_outputs_; ++idx) {
        int output_idx = order_[idx];
        MatrixXd X_aug = build_augmented(X, prev);
        auto estimator = create_classifier(estimator_factory_);
        fit_classifier(estimator, X_aug, Y.col(output_idx));
        estimators_.push_back(estimator);

        MatrixXd new_prev(X.rows(), idx + 1);
        if (idx > 0) {
            new_prev.leftCols(idx) = prev;
        }
        new_prev.col(idx) = Y.col(output_idx);
        prev = new_prev;
    }

    fitted_ = true;
    return *this;
}

MatrixXi ClassifierChain::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ClassifierChain must be fitted before predict");
    }
    MatrixXi predictions = MatrixXi::Zero(X.rows(), n_outputs_);
    MatrixXd prev;

    for (int idx = 0; idx < n_outputs_; ++idx) {
        int output_idx = order_[idx];
        MatrixXd X_aug = build_augmented(X, prev);
        VectorXi pred = estimators_[idx]->predict_classes(X_aug);
        predictions.col(output_idx) = pred;

        MatrixXd new_prev(X.rows(), idx + 1);
        if (idx > 0) {
            new_prev.leftCols(idx) = prev;
        }
        new_prev.col(idx) = pred.cast<double>();
        prev = new_prev;
    }

    return predictions;
}

std::vector<MatrixXd> ClassifierChain::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("ClassifierChain must be fitted before predict_proba");
    }
    std::vector<MatrixXd> probs(n_outputs_);
    MatrixXd prev;

    for (int idx = 0; idx < n_outputs_; ++idx) {
        int output_idx = order_[idx];
        MatrixXd X_aug = build_augmented(X, prev);
        MatrixXd prob = estimators_[idx]->predict_proba(X_aug);
        probs[output_idx] = prob;

        VectorXi pred = estimators_[idx]->predict_classes(X_aug);
        MatrixXd new_prev(X.rows(), idx + 1);
        if (idx > 0) {
            new_prev.leftCols(idx) = prev;
        }
        new_prev.col(idx) = pred.cast<double>();
        prev = new_prev;
    }

    return probs;
}

Params ClassifierChain::get_params() const {
    Params params;
    params["order"] = join_indices(order_);
    return params;
}

Estimator& ClassifierChain::set_params(const Params& params) {
    std::string order_str = utils::get_param_string(params, "order", "");
    if (!order_str.empty()) {
        order_ = parse_indices(order_str);
    }
    return *this;
}

RegressorChain::RegressorChain(std::function<std::shared_ptr<Regressor>()> estimator_factory,
                               const std::vector<int>& order)
    : estimator_factory_(std::move(estimator_factory)), fitted_(false), n_outputs_(0), order_(order), estimators_() {}

Estimator& RegressorChain::fit(const MatrixXd& X, const VectorXd& y) {
    MatrixXd Y(X.rows(), 1);
    Y.col(0) = y;
    return fit(X, Y);
}

Estimator& RegressorChain::fit(const MatrixXd& X, const MatrixXd& Y) {
    if (X.rows() != Y.rows()) {
        throw std::invalid_argument("X and Y must have the same number of samples");
    }
    n_outputs_ = static_cast<int>(Y.cols());
    if (order_.empty()) {
        order_.resize(n_outputs_);
        std::iota(order_.begin(), order_.end(), 0);
    }
    validate_order(order_, n_outputs_);

    estimators_.clear();
    MatrixXd prev;
    for (int idx = 0; idx < n_outputs_; ++idx) {
        int output_idx = order_[idx];
        MatrixXd X_aug = build_augmented(X, prev);
        auto estimator = create_regressor(estimator_factory_);
        fit_regressor(estimator, X_aug, Y.col(output_idx));
        estimators_.push_back(estimator);

        MatrixXd new_prev(X.rows(), idx + 1);
        if (idx > 0) {
            new_prev.leftCols(idx) = prev;
        }
        new_prev.col(idx) = Y.col(output_idx);
        prev = new_prev;
    }

    fitted_ = true;
    return *this;
}

MatrixXd RegressorChain::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("RegressorChain must be fitted before predict");
    }
    MatrixXd predictions = MatrixXd::Zero(X.rows(), n_outputs_);
    MatrixXd prev;

    for (int idx = 0; idx < n_outputs_; ++idx) {
        int output_idx = order_[idx];
        MatrixXd X_aug = build_augmented(X, prev);
        VectorXd pred = estimators_[idx]->predict(X_aug);
        predictions.col(output_idx) = pred;

        MatrixXd new_prev(X.rows(), idx + 1);
        if (idx > 0) {
            new_prev.leftCols(idx) = prev;
        }
        new_prev.col(idx) = pred;
        prev = new_prev;
    }

    return predictions;
}

Params RegressorChain::get_params() const {
    Params params;
    params["order"] = join_indices(order_);
    return params;
}

Estimator& RegressorChain::set_params(const Params& params) {
    std::string order_str = utils::get_param_string(params, "order", "");
    if (!order_str.empty()) {
        order_ = parse_indices(order_str);
    }
    return *this;
}

} // namespace meta
} // namespace auroraml

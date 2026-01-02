#pragma once

#include "base.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace auroraml {
namespace meta {

class OneVsRestClassifier : public Estimator, public Classifier {
private:
    std::function<std::shared_ptr<Classifier>()> estimator_factory_;
    int n_jobs_;
    bool fitted_;
    VectorXi classes_;
    std::vector<std::shared_ptr<Classifier>> estimators_;

public:
    explicit OneVsRestClassifier(std::function<std::shared_ptr<Classifier>()> estimator_factory, int n_jobs = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

class OneVsOneClassifier : public Estimator, public Classifier {
private:
    std::function<std::shared_ptr<Classifier>()> estimator_factory_;
    int n_jobs_;
    bool fitted_;
    VectorXi classes_;
    std::vector<std::shared_ptr<Classifier>> estimators_;
    std::vector<std::pair<int, int>> class_pairs_;

public:
    explicit OneVsOneClassifier(std::function<std::shared_ptr<Classifier>()> estimator_factory, int n_jobs = 1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
};

class OutputCodeClassifier : public Estimator, public Classifier {
private:
    std::function<std::shared_ptr<Classifier>()> estimator_factory_;
    int code_size_;
    int random_state_;
    bool fitted_;
    VectorXi classes_;
    MatrixXi code_book_;
    std::vector<std::shared_ptr<Classifier>> estimators_;

public:
    OutputCodeClassifier(std::function<std::shared_ptr<Classifier>()> estimator_factory,
                         int code_size = 0, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    VectorXi classes() const { return classes_; }
    const MatrixXi& code_book() const { return code_book_; }
};

class MultiOutputClassifier : public Estimator {
private:
    std::function<std::shared_ptr<Classifier>()> estimator_factory_;
    bool fitted_;
    int n_outputs_;
    std::vector<std::shared_ptr<Classifier>> estimators_;

public:
    explicit MultiOutputClassifier(std::function<std::shared_ptr<Classifier>()> estimator_factory);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    Estimator& fit(const MatrixXd& X, const MatrixXd& Y);
    MatrixXi predict(const MatrixXd& X) const;
    std::vector<MatrixXd> predict_proba(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    int n_outputs() const { return n_outputs_; }
};

class MultiOutputRegressor : public Estimator {
private:
    std::function<std::shared_ptr<Regressor>()> estimator_factory_;
    bool fitted_;
    int n_outputs_;
    std::vector<std::shared_ptr<Regressor>> estimators_;

public:
    explicit MultiOutputRegressor(std::function<std::shared_ptr<Regressor>()> estimator_factory);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    Estimator& fit(const MatrixXd& X, const MatrixXd& Y);
    MatrixXd predict(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    int n_outputs() const { return n_outputs_; }
};

class ClassifierChain : public Estimator {
private:
    std::function<std::shared_ptr<Classifier>()> estimator_factory_;
    bool fitted_;
    int n_outputs_;
    std::vector<int> order_;
    std::vector<std::shared_ptr<Classifier>> estimators_;

public:
    explicit ClassifierChain(std::function<std::shared_ptr<Classifier>()> estimator_factory,
                             const std::vector<int>& order = {});

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    Estimator& fit(const MatrixXd& X, const MatrixXd& Y);
    MatrixXi predict(const MatrixXd& X) const;
    std::vector<MatrixXd> predict_proba(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    const std::vector<int>& order() const { return order_; }
};

class RegressorChain : public Estimator {
private:
    std::function<std::shared_ptr<Regressor>()> estimator_factory_;
    bool fitted_;
    int n_outputs_;
    std::vector<int> order_;
    std::vector<std::shared_ptr<Regressor>> estimators_;

public:
    explicit RegressorChain(std::function<std::shared_ptr<Regressor>()> estimator_factory,
                            const std::vector<int>& order = {});

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    Estimator& fit(const MatrixXd& X, const MatrixXd& Y);
    MatrixXd predict(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }
    const std::vector<int>& order() const { return order_; }
};

} // namespace meta
} // namespace auroraml

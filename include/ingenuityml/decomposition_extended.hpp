#pragma once

#include "base.hpp"
#include <vector>
#include <string>

namespace ingenuityml {
namespace decomposition {

class IncrementalPCA : public Estimator, public Transformer {
private:
    int n_components_;
    bool whiten_;
    int batch_size_;
    bool fitted_;
    int n_samples_seen_;
    VectorXd mean_;
    MatrixXd components_;
    VectorXd explained_variance_;
    MatrixXd buffer_;

public:
    IncrementalPCA(int n_components = -1, bool whiten = false, int batch_size = 0);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    Estimator& partial_fit(const MatrixXd& X, const VectorXd& y = VectorXd());
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const;
    const VectorXd& explained_variance() const;
    VectorXd explained_variance_ratio() const;
    const VectorXd& mean() const;
};

class SparsePCA : public Estimator, public Transformer {
private:
    int n_components_;
    double alpha_;
    int max_iter_;
    double tol_;
    bool fitted_;
    VectorXd mean_;
    MatrixXd components_;

public:
    SparsePCA(int n_components = -1, double alpha = 1.0, int max_iter = 1000, double tol = 1e-4);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const;
};

class MiniBatchSparsePCA : public Estimator, public Transformer {
private:
    int n_components_;
    double alpha_;
    int max_iter_;
    int batch_size_;
    double tol_;
    int random_state_;
    bool fitted_;
    VectorXd mean_;
    MatrixXd components_;

public:
    MiniBatchSparsePCA(int n_components = -1, double alpha = 1.0, int max_iter = 1000,
                       int batch_size = 100, double tol = 1e-4, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const;
};

class NMF : public Estimator, public Transformer {
private:
    int n_components_;
    int max_iter_;
    double tol_;
    double alpha_;
    int random_state_;
    bool fitted_;
    MatrixXd components_;
    MatrixXd W_;

public:
    NMF(int n_components = 2, int max_iter = 200, double tol = 1e-4, double alpha = 0.0, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const;
};

class MiniBatchNMF : public Estimator, public Transformer {
private:
    int n_components_;
    int max_iter_;
    int batch_size_;
    double tol_;
    double alpha_;
    int random_state_;
    bool fitted_;
    MatrixXd components_;
    MatrixXd W_;

public:
    MiniBatchNMF(int n_components = 2, int max_iter = 200, int batch_size = 100,
                 double tol = 1e-4, double alpha = 0.0, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const;
};

class DictionaryLearning : public Estimator, public Transformer {
private:
    int n_components_;
    double alpha_;
    int max_iter_;
    double tol_;
    int random_state_;
    bool fitted_;
    MatrixXd components_;
    MatrixXd codes_;

public:
    DictionaryLearning(int n_components = 2, double alpha = 1.0, int max_iter = 100, double tol = 1e-4,
                       int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const;
    const MatrixXd& codes() const;
};

class MiniBatchDictionaryLearning : public Estimator, public Transformer {
private:
    int n_components_;
    double alpha_;
    int max_iter_;
    int batch_size_;
    double tol_;
    int random_state_;
    bool fitted_;
    MatrixXd components_;

public:
    MiniBatchDictionaryLearning(int n_components = 2, double alpha = 1.0, int max_iter = 100,
                                int batch_size = 100, double tol = 1e-4, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const;
};

class LatentDirichletAllocation : public Estimator, public Transformer {
private:
    int n_components_;
    int max_iter_;
    double doc_topic_prior_;
    double topic_word_prior_;
    int random_state_;
    bool fitted_;
    MatrixXd components_;
    MatrixXd doc_topic_;

public:
    LatentDirichletAllocation(int n_components = 10, int max_iter = 10,
                              double doc_topic_prior = 0.1, double topic_word_prior = 0.01,
                              int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const;
    const MatrixXd& doc_topic() const;
};

} // namespace decomposition
} // namespace ingenuityml

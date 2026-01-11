#pragma once

#include "base.hpp"
#include <vector>
#include <string>

namespace ingenuityml {
namespace covariance {

class EmpiricalCovariance : public Estimator {
private:
    bool assume_centered_;
    bool fitted_;
    MatrixXd covariance_;
    VectorXd location_;
    MatrixXd precision_;

public:
    explicit EmpiricalCovariance(bool assume_centered = false);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd mahalanobis(const MatrixXd& X) const;
    VectorXd score_samples(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& covariance() const;
    const VectorXd& location() const;
    const MatrixXd& precision() const;
};

class ShrunkCovariance : public Estimator {
private:
    double shrinkage_;
    bool assume_centered_;
    bool fitted_;
    MatrixXd covariance_;
    VectorXd location_;
    MatrixXd precision_;

public:
    ShrunkCovariance(double shrinkage = 0.1, bool assume_centered = false);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd mahalanobis(const MatrixXd& X) const;
    VectorXd score_samples(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    double shrinkage() const { return shrinkage_; }
    const MatrixXd& covariance() const;
    const VectorXd& location() const;
    const MatrixXd& precision() const;
};

class LedoitWolf : public Estimator {
private:
    bool assume_centered_;
    bool fitted_;
    double shrinkage_;
    MatrixXd covariance_;
    VectorXd location_;
    MatrixXd precision_;

public:
    explicit LedoitWolf(bool assume_centered = false);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd mahalanobis(const MatrixXd& X) const;
    VectorXd score_samples(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    double shrinkage() const { return shrinkage_; }
    const MatrixXd& covariance() const;
    const VectorXd& location() const;
    const MatrixXd& precision() const;
};

class OAS : public Estimator {
private:
    bool assume_centered_;
    bool fitted_;
    double shrinkage_;
    MatrixXd covariance_;
    VectorXd location_;
    MatrixXd precision_;

public:
    explicit OAS(bool assume_centered = false);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd mahalanobis(const MatrixXd& X) const;
    VectorXd score_samples(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    double shrinkage() const { return shrinkage_; }
    const MatrixXd& covariance() const;
    const VectorXd& location() const;
    const MatrixXd& precision() const;
};

class MinCovDet : public Estimator {
private:
    double support_fraction_;
    bool assume_centered_;
    int max_iter_;
    double tol_;
    int random_state_;
    bool fitted_;
    MatrixXd covariance_;
    VectorXd location_;
    MatrixXd precision_;
    VectorXi support_;

public:
    MinCovDet(double support_fraction = 0.75, bool assume_centered = false,
              int max_iter = 100, double tol = 1e-3, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd mahalanobis(const MatrixXd& X) const;
    VectorXd score_samples(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& covariance() const;
    const VectorXd& location() const;
    const MatrixXd& precision() const;
    const VectorXi& support() const;
};

class EllipticEnvelope : public Estimator {
private:
    double contamination_;
    MinCovDet mcd_;
    bool fitted_;
    double threshold_;

public:
    EllipticEnvelope(double contamination = 0.1, double support_fraction = 0.75,
                     int max_iter = 100, double tol = 1e-3, int random_state = -1);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict(const MatrixXd& X) const;
    VectorXd decision_function(const MatrixXd& X) const;
    VectorXd score_samples(const MatrixXd& X) const;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& covariance() const { return mcd_.covariance(); }
    const VectorXd& location() const { return mcd_.location(); }
    const MatrixXd& precision() const { return mcd_.precision(); }
    double threshold() const { return threshold_; }
};

} // namespace covariance
} // namespace ingenuityml

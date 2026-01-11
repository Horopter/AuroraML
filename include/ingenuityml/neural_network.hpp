#pragma once

#include <vector>
#include <string>
#include <random>
#include <memory>
#include <functional>
#include "base.hpp"

namespace ingenuityml {
namespace neural_network {

// Activation functions
enum class ActivationFunction {
    RELU,
    TANH,
    LOGISTIC,
    IDENTITY
};

// Solvers for weight optimization
enum class Solver {
    LBFGS,
    SGD,
    ADAM
};

// Base class for activation functions
class Activation {
public:
    virtual ~Activation() = default;
    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual std::unique_ptr<Activation> clone() const = 0;
};

class ReLU : public Activation {
public:
    double activate(double x) const override { return std::max(0.0, x); }
    double derivative(double x) const override { return x > 0 ? 1.0 : 0.0; }
    std::unique_ptr<Activation> clone() const override { return std::make_unique<ReLU>(); }
};

class Tanh : public Activation {
public:
    double activate(double x) const override { return std::tanh(x); }
    double derivative(double x) const override { 
        double t = std::tanh(x);
        return 1.0 - t * t;
    }
    std::unique_ptr<Activation> clone() const override { return std::make_unique<Tanh>(); }
};

class Logistic : public Activation {
public:
    double activate(double x) const override { return 1.0 / (1.0 + std::exp(-x)); }
    double derivative(double x) const override {
        double s = activate(x);
        return s * (1.0 - s);
    }
    std::unique_ptr<Activation> clone() const override { return std::make_unique<Logistic>(); }
};

class Identity : public Activation {
public:
    double activate(double x) const override { return x; }
    double derivative(double x) const override { return 1.0; }
    std::unique_ptr<Activation> clone() const override { return std::make_unique<Identity>(); }
};

// Multi-Layer Perceptron base class
class MLPBase : public Estimator {
protected:
    std::vector<int> hidden_layer_sizes_;
    ActivationFunction activation_;
    Solver solver_;
    double alpha_;  // L2 regularization term
    int batch_size_;
    double learning_rate_;
    int max_iter_;
    int random_state_;
    double tol_;
    bool verbose_;
    bool warm_start_;
    double momentum_;
    bool nesterovs_momentum_;
    bool early_stopping_;
    double validation_fraction_;
    double beta_1_;  // Adam parameter
    double beta_2_;  // Adam parameter
    double epsilon_;  // Adam parameter
    int n_iter_no_change_;
    
    // Internal state
    std::vector<MatrixXd> weights_;
    std::vector<VectorXd> biases_;
    std::vector<std::unique_ptr<Activation>> activations_;
    std::mt19937 random_generator_;
    bool fitted_;
    int n_features_;
    int n_outputs_;
    std::vector<double> loss_curve_;
    
    // Training history
    std::vector<double> validation_scores_;
    int best_validation_score_iter_;
    
public:
    MLPBase(const std::vector<int>& hidden_layer_sizes = {100},
           ActivationFunction activation = ActivationFunction::RELU,
           Solver solver = Solver::ADAM,
           double alpha = 0.0001,
           int batch_size = 200,
           double learning_rate = 0.001,
           int max_iter = 200,
           int random_state = -1,
           double tol = 1e-4,
           bool verbose = false,
           bool warm_start = false,
           double momentum = 0.9,
           bool nesterovs_momentum = true,
           bool early_stopping = false,
           double validation_fraction = 0.1,
           double beta_1 = 0.9,
           double beta_2 = 0.999,
           double epsilon = 1e-8,
           int n_iter_no_change = 10);
           
    virtual ~MLPBase() = default;

protected:
    void initialize_weights(int n_features, int n_outputs);
    void forward_pass(const MatrixXd& X, std::vector<MatrixXd>& activations) const;
    void backward_pass(const MatrixXd& X, const VectorXd& y, 
                      const std::vector<MatrixXd>& activations,
                      std::vector<MatrixXd>& weight_gradients,
                      std::vector<VectorXd>& bias_gradients);
    double compute_loss(const MatrixXd& X, const VectorXd& y) const;
    virtual double compute_output_loss(const VectorXd& y_true, const VectorXd& y_pred) const = 0;
    virtual VectorXd compute_output_gradient(const VectorXd& y_true, const VectorXd& y_pred) const = 0;
    virtual VectorXd predict_output(const MatrixXd& activations) const = 0;
    
    void update_weights_sgd(const std::vector<MatrixXd>& weight_gradients,
                           const std::vector<VectorXd>& bias_gradients);
    void update_weights_adam(const std::vector<MatrixXd>& weight_gradients,
                            const std::vector<VectorXd>& bias_gradients,
                            int iteration);
    
    std::unique_ptr<Activation> create_activation(ActivationFunction func) const;
    
    // Adam optimizer state
    std::vector<MatrixXd> m_weights_, v_weights_;
    std::vector<VectorXd> m_biases_, v_biases_;
    
public:
    // Parameter getters and setters
    const std::vector<int>& hidden_layer_sizes() const { return hidden_layer_sizes_; }
    void set_hidden_layer_sizes(const std::vector<int>& sizes) { hidden_layer_sizes_ = sizes; }
    
    ActivationFunction activation() const { return activation_; }
    void set_activation(ActivationFunction activation) { activation_ = activation; }
    
    Solver solver() const { return solver_; }
    void set_solver(Solver solver) { solver_ = solver; }
    
    double alpha() const { return alpha_; }
    void set_alpha(double alpha) { alpha_ = alpha; }
    
    int batch_size() const { return batch_size_; }
    void set_batch_size(int batch_size) { batch_size_ = batch_size; }
    
    double learning_rate() const { return learning_rate_; }
    void set_learning_rate(double lr) { learning_rate_ = lr; }
    
    int max_iter() const { return max_iter_; }
    void set_max_iter(int max_iter) { max_iter_ = max_iter; }
    
    double tol() const { return tol_; }
    void set_tol(double tol) { tol_ = tol; }
    
    bool is_fitted() const override { return fitted_; }
    
    const std::vector<double>& loss_curve() const { return loss_curve_; }
    const std::vector<MatrixXd>& coefs() const { return weights_; }
    const std::vector<VectorXd>& intercepts() const { return biases_; }
    int n_iter() const { return static_cast<int>(loss_curve_.size()); }
    
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
};

// Multi-Layer Perceptron Classifier
class MLPClassifier : public MLPBase, public Classifier {
private:
    std::vector<int> classes_;
    int n_classes_;
    
public:
    MLPClassifier(const std::vector<int>& hidden_layer_sizes = {100},
                 ActivationFunction activation = ActivationFunction::RELU,
                 Solver solver = Solver::ADAM,
                 double alpha = 0.0001,
                 int batch_size = 200,
                 double learning_rate = 0.001,
                 int max_iter = 200,
                 int random_state = -1,
                 double tol = 1e-4,
                 bool verbose = false,
                 bool warm_start = false,
                 double momentum = 0.9,
                 bool nesterovs_momentum = true,
                 bool early_stopping = false,
                 double validation_fraction = 0.1,
                 double beta_1 = 0.9,
                 double beta_2 = 0.999,
                 double epsilon = 1e-8,
                 int n_iter_no_change = 10);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXi predict_classes(const MatrixXd& X) const override;
    MatrixXd predict_proba(const MatrixXd& X) const override;
    VectorXd decision_function(const MatrixXd& X) const override;
    
    const std::vector<int>& classes() const { return classes_; }
    int n_classes() const { return n_classes_; }
    
protected:
    double compute_output_loss(const VectorXd& y_true, const VectorXd& y_pred) const override;
    VectorXd compute_output_gradient(const VectorXd& y_true, const VectorXd& y_pred) const override;
    VectorXd predict_output(const MatrixXd& activations) const override;
    
private:
    void setup_output_layer();
    VectorXd encode_target(const VectorXd& y) const;
    MatrixXd encode_targets(const VectorXd& y) const;
    VectorXd softmax(const VectorXd& x) const;
};

// Multi-Layer Perceptron Regressor
class MLPRegressor : public MLPBase, public Regressor {
public:
    MLPRegressor(const std::vector<int>& hidden_layer_sizes = {100},
                ActivationFunction activation = ActivationFunction::RELU,
                Solver solver = Solver::ADAM,
                double alpha = 0.0001,
                int batch_size = 200,
                double learning_rate = 0.001,
                int max_iter = 200,
                int random_state = -1,
                double tol = 1e-4,
                bool verbose = false,
                bool warm_start = false,
                double momentum = 0.9,
                bool nesterovs_momentum = true,
                bool early_stopping = false,
                double validation_fraction = 0.1,
                double beta_1 = 0.9,
                double beta_2 = 0.999,
                double epsilon = 1e-8,
                int n_iter_no_change = 10);
    
    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    VectorXd predict(const MatrixXd& X) const override;
    
protected:
    double compute_output_loss(const VectorXd& y_true, const VectorXd& y_pred) const override;
    VectorXd compute_output_gradient(const VectorXd& y_true, const VectorXd& y_pred) const override;
    VectorXd predict_output(const MatrixXd& activations) const override;
};

/**
 * BernoulliRBM - Restricted Boltzmann Machine with Bernoulli visible units
 */
class BernoulliRBM : public Estimator, public Transformer {
private:
    int n_components_;
    double learning_rate_;
    int batch_size_;
    int n_iter_;
    int random_state_;
    bool verbose_;
    bool fitted_;
    int n_features_;
    MatrixXd components_;
    VectorXd intercept_visible_;
    VectorXd intercept_hidden_;
    std::mt19937 rng_;

    MatrixXd sigmoid(const MatrixXd& X) const;

public:
    BernoulliRBM(int n_components = 256, double learning_rate = 0.1,
                 int batch_size = 10, int n_iter = 10,
                 int random_state = -1, bool verbose = false);

    Estimator& fit(const MatrixXd& X, const VectorXd& y) override;
    MatrixXd transform(const MatrixXd& X) const override;
    MatrixXd inverse_transform(const MatrixXd& X) const override;
    MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y) override;
    Params get_params() const override;
    Estimator& set_params(const Params& params) override;
    bool is_fitted() const override { return fitted_; }

    const MatrixXd& components() const { return components_; }
    const VectorXd& intercept_visible() const { return intercept_visible_; }
    const VectorXd& intercept_hidden() const { return intercept_hidden_; }
    int n_iter() const { return n_iter_; }
};

} // namespace neural_network
} // namespace ingenuityml

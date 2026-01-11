#include "ingenuityml/neural_network.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <set>
#include <iostream>
#include "ingenuityml/utils.hpp"

namespace ingenuityml {
namespace neural_network {

// MLPBase implementation
MLPBase::MLPBase(const std::vector<int>& hidden_layer_sizes,
                ActivationFunction activation,
                Solver solver,
                double alpha,
                int batch_size,
                double learning_rate,
                int max_iter,
                int random_state,
                double tol,
                bool verbose,
                bool warm_start,
                double momentum,
                bool nesterovs_momentum,
                bool early_stopping,
                double validation_fraction,
                double beta_1,
                double beta_2,
                double epsilon,
                int n_iter_no_change)
    : hidden_layer_sizes_(hidden_layer_sizes),
      activation_(activation),
      solver_(solver),
      alpha_(alpha),
      batch_size_(batch_size),
      learning_rate_(learning_rate),
      max_iter_(max_iter),
      random_state_(random_state),
      tol_(tol),
      verbose_(verbose),
      warm_start_(warm_start),
      momentum_(momentum),
      nesterovs_momentum_(nesterovs_momentum),
      early_stopping_(early_stopping),
      validation_fraction_(validation_fraction),
      beta_1_(beta_1),
      beta_2_(beta_2),
      epsilon_(epsilon),
      n_iter_no_change_(n_iter_no_change),
      fitted_(false),
      n_features_(0),
      n_outputs_(0),
      best_validation_score_iter_(0) {
    
    if (random_state >= 0) {
        random_generator_.seed(random_state);
    } else {
        std::random_device rd;
        random_generator_.seed(rd());
    }
}

void MLPBase::initialize_weights(int n_features, int n_outputs) {
    n_features_ = n_features;
    n_outputs_ = n_outputs;
    
    // Build layer sizes: input -> hidden layers -> output
    std::vector<int> layer_sizes = {n_features};
    layer_sizes.insert(layer_sizes.end(), hidden_layer_sizes_.begin(), hidden_layer_sizes_.end());
    layer_sizes.push_back(n_outputs);
    
    weights_.clear();
    biases_.clear();
    activations_.clear();
    
    // Xavier initialization
    std::normal_distribution<double> normal(0.0, 1.0);
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        int fan_in = layer_sizes[i];
        int fan_out = layer_sizes[i + 1];
        
        // Xavier initialization scale
        double scale = std::sqrt(2.0 / (fan_in + fan_out));
        
        MatrixXd weight = MatrixXd::Zero(fan_out, fan_in);
        for (int row = 0; row < fan_out; ++row) {
            for (int col = 0; col < fan_in; ++col) {
                weight(row, col) = normal(random_generator_) * scale;
            }
        }
        weights_.push_back(weight);
        
        // Initialize biases to zero
        biases_.push_back(VectorXd::Zero(fan_out));
        
        // Create activation function (except for output layer)
        if (i < layer_sizes.size() - 2) {
            activations_.push_back(create_activation(activation_));
        }
    }
    
    // Initialize Adam optimizer state
    if (solver_ == Solver::ADAM) {
        m_weights_.clear();
        v_weights_.clear();
        m_biases_.clear();
        v_biases_.clear();
        
        for (const auto& weight : weights_) {
            m_weights_.push_back(MatrixXd::Zero(weight.rows(), weight.cols()));
            v_weights_.push_back(MatrixXd::Zero(weight.rows(), weight.cols()));
        }
        
        for (const auto& bias : biases_) {
            m_biases_.push_back(VectorXd::Zero(bias.size()));
            v_biases_.push_back(VectorXd::Zero(bias.size()));
        }
    }
}

void MLPBase::forward_pass(const MatrixXd& X, std::vector<MatrixXd>& layer_outputs) const {
    if (X.rows() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }
    if (n_features_ > 0 && X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    layer_outputs.clear();
    layer_outputs.push_back(X); // Input layer
    
    MatrixXd current_input = X;
    
    for (size_t i = 0; i < weights_.size(); ++i) {
        // Linear transformation: output = input * W^T + b
        MatrixXd linear_output = current_input * weights_[i].transpose();
        for (int row = 0; row < linear_output.rows(); ++row) {
            linear_output.row(row) += biases_[i].transpose();
        }
        
        // Apply activation function (except for output layer)
        MatrixXd activated_output;
        if (i < activations_.size()) {
            activated_output = MatrixXd::Zero(linear_output.rows(), linear_output.cols());
            for (int row = 0; row < linear_output.rows(); ++row) {
                for (int col = 0; col < linear_output.cols(); ++col) {
                    activated_output(row, col) = activations_[i]->activate(linear_output(row, col));
                }
            }
        } else {
            activated_output = linear_output; // Output layer (no activation for base class)
        }
        
        layer_outputs.push_back(activated_output);
        current_input = activated_output;
    }
}

void MLPBase::backward_pass(const MatrixXd& X, const VectorXd& y,
                           const std::vector<MatrixXd>& layer_outputs,
                           std::vector<MatrixXd>& weight_gradients,
                           std::vector<VectorXd>& bias_gradients) {
    
    int n_samples = X.rows();
    int n_layers = weights_.size();
    
    weight_gradients.resize(n_layers);
    bias_gradients.resize(n_layers);
    
    // Initialize gradients to zero
    for (int i = 0; i < n_layers; ++i) {
        weight_gradients[i] = MatrixXd::Zero(weights_[i].rows(), weights_[i].cols());
        bias_gradients[i] = VectorXd::Zero(biases_[i].size());
    }
    
    // Compute output layer error for all samples
    const MatrixXd& output_layer = layer_outputs.back();
    int output_size = output_layer.cols();
    MatrixXd current_error = MatrixXd::Zero(n_samples, output_size);
    
    for (int row = 0; row < n_samples; ++row) {
        VectorXd sample_pred = output_layer.row(row);
        VectorXd sample_y = VectorXd::Zero(1);
        sample_y(0) = y(row);
        VectorXd sample_error = compute_output_gradient(sample_y, sample_pred);
        
        if (sample_error.size() == 1 && output_size > 1) {
            // Binary classification case - expand to match output size
            current_error(row, 0) = sample_error(0);
        } else {
            current_error.row(row) = sample_error.transpose();
        }
    }
    
    // Backpropagate through layers
    for (int layer = n_layers - 1; layer >= 0; --layer) {
        const MatrixXd& layer_input = layer_outputs[layer];
        
        // Compute weight gradients: dW = (error^T * input) / n_samples
        weight_gradients[layer] = current_error.transpose() * layer_input / n_samples;
        
        // Compute bias gradients: db = mean(error, axis=0)
        bias_gradients[layer] = current_error.colwise().mean();
        
        // Add L2 regularization to weight gradients
        if (alpha_ > 0.0) {
            weight_gradients[layer] += alpha_ * weights_[layer];
        }
        
        // Propagate error to previous layer
        if (layer > 0) {
            // prev_error = error * W
            MatrixXd prev_error = current_error * weights_[layer];
            
            // Apply activation derivative to the previous layer's pre-activation values
            const MatrixXd& prev_layer_output = layer_outputs[layer];
            if (layer - 1 < static_cast<int>(activations_.size())) {
                for (int row = 0; row < prev_error.rows(); ++row) {
                    for (int col = 0; col < prev_error.cols(); ++col) {
                        // Use the pre-activation value from the forward pass
                        double activated_value = prev_layer_output(row, col);
                        prev_error(row, col) *= activations_[layer - 1]->derivative(activated_value);
                    }
                }
            }
            
            current_error = prev_error;
        }
    }
}

double MLPBase::compute_loss(const MatrixXd& X, const VectorXd& y) const {
    std::vector<MatrixXd> layer_outputs;
    forward_pass(X, layer_outputs);
    
    double total_loss = 0.0;
    int n_samples = X.rows();
    
    for (int i = 0; i < n_samples; ++i) {
        VectorXd pred = layer_outputs.back().row(i);
        VectorXd true_val = VectorXd::Zero(1);
        true_val(0) = y(i);
        total_loss += compute_output_loss(true_val, pred);
    }
    
    total_loss /= n_samples;
    
    // Add L2 regularization
    if (alpha_ > 0.0) {
        double reg_loss = 0.0;
        for (const auto& weight : weights_) {
            reg_loss += weight.squaredNorm();
        }
        total_loss += 0.5 * alpha_ * reg_loss;
    }
    
    return total_loss;
}

void MLPBase::update_weights_sgd(const std::vector<MatrixXd>& weight_gradients,
                                const std::vector<VectorXd>& bias_gradients) {
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate_ * weight_gradients[i];
        biases_[i] -= learning_rate_ * bias_gradients[i];
    }
}

void MLPBase::update_weights_adam(const std::vector<MatrixXd>& weight_gradients,
                                 const std::vector<VectorXd>& bias_gradients,
                                 int iteration) {
    double beta_1_t = std::pow(beta_1_, iteration + 1);
    double beta_2_t = std::pow(beta_2_, iteration + 1);
    
    for (size_t i = 0; i < weights_.size(); ++i) {
        // Update biased first moment estimate
        m_weights_[i] = beta_1_ * m_weights_[i] + (1.0 - beta_1_) * weight_gradients[i];
        m_biases_[i] = beta_1_ * m_biases_[i] + (1.0 - beta_1_) * bias_gradients[i];
        
        // Update biased second moment estimate
        v_weights_[i] = beta_2_ * v_weights_[i] + (1.0 - beta_2_) * weight_gradients[i].cwiseProduct(weight_gradients[i]);
        v_biases_[i] = beta_2_ * v_biases_[i] + (1.0 - beta_2_) * bias_gradients[i].cwiseProduct(bias_gradients[i]);
        
        // Compute bias-corrected first moment estimate
        MatrixXd m_hat_w = m_weights_[i] / (1.0 - beta_1_t);
        VectorXd m_hat_b = m_biases_[i] / (1.0 - beta_1_t);
        
        // Compute bias-corrected second moment estimate
        MatrixXd v_hat_w = v_weights_[i] / (1.0 - beta_2_t);
        VectorXd v_hat_b = v_biases_[i] / (1.0 - beta_2_t);
        
        // Update weights
        MatrixXd sqrt_v_w = v_hat_w.cwiseSqrt();
        for (int row = 0; row < sqrt_v_w.rows(); ++row) {
            for (int col = 0; col < sqrt_v_w.cols(); ++col) {
                sqrt_v_w(row, col) += epsilon_;
            }
        }
        weights_[i] -= learning_rate_ * m_hat_w.cwiseQuotient(sqrt_v_w);
        
        VectorXd sqrt_v_b = v_hat_b.cwiseSqrt();
        for (int j = 0; j < sqrt_v_b.size(); ++j) {
            sqrt_v_b(j) += epsilon_;
        }
        biases_[i] -= learning_rate_ * m_hat_b.cwiseQuotient(sqrt_v_b);
    }
}

std::unique_ptr<Activation> MLPBase::create_activation(ActivationFunction func) const {
    switch (func) {
        case ActivationFunction::RELU:
            return std::make_unique<ReLU>();
        case ActivationFunction::TANH:
            return std::make_unique<Tanh>();
        case ActivationFunction::LOGISTIC:
            return std::make_unique<Logistic>();
        case ActivationFunction::IDENTITY:
            return std::make_unique<Identity>();
        default:
            throw std::invalid_argument("Unknown activation function");
    }
}

Params MLPBase::get_params() const {
    Params params;
    
    // Convert hidden layer sizes to string
    std::ostringstream oss;
    for (size_t i = 0; i < hidden_layer_sizes_.size(); ++i) {
        if (i > 0) oss << ",";
        oss << hidden_layer_sizes_[i];
    }
    params["hidden_layer_sizes"] = oss.str();
    
    params["activation"] = std::to_string(static_cast<int>(activation_));
    params["solver"] = std::to_string(static_cast<int>(solver_));
    params["alpha"] = std::to_string(alpha_);
    params["batch_size"] = std::to_string(batch_size_);
    params["learning_rate"] = std::to_string(learning_rate_);
    params["max_iter"] = std::to_string(max_iter_);
    params["random_state"] = std::to_string(random_state_);
    params["tol"] = std::to_string(tol_);
    params["verbose"] = verbose_ ? "true" : "false";
    params["warm_start"] = warm_start_ ? "true" : "false";
    params["momentum"] = std::to_string(momentum_);
    params["nesterovs_momentum"] = nesterovs_momentum_ ? "true" : "false";
    params["early_stopping"] = early_stopping_ ? "true" : "false";
    params["validation_fraction"] = std::to_string(validation_fraction_);
    params["beta_1"] = std::to_string(beta_1_);
    params["beta_2"] = std::to_string(beta_2_);
    params["epsilon"] = std::to_string(epsilon_);
    params["n_iter_no_change"] = std::to_string(n_iter_no_change_);
    
    return params;
}

Estimator& MLPBase::set_params(const Params& params) {
    for (const auto& param : params) {
        const std::string& key = param.first;
        const std::string& value = param.second;
        
        if (key == "alpha") {
            alpha_ = std::stod(value);
        } else if (key == "batch_size") {
            batch_size_ = std::stoi(value);
        } else if (key == "learning_rate") {
            learning_rate_ = std::stod(value);
        } else if (key == "max_iter") {
            max_iter_ = std::stoi(value);
        } else if (key == "random_state") {
            random_state_ = std::stoi(value);
        } else if (key == "tol") {
            tol_ = std::stod(value);
        } else if (key == "verbose") {
            verbose_ = (value == "true");
        } else if (key == "warm_start") {
            warm_start_ = (value == "true");
        }
        // Add more parameter setters as needed
    }
    return *this;
}

// MLPClassifier implementation
MLPClassifier::MLPClassifier(const std::vector<int>& hidden_layer_sizes,
                            ActivationFunction activation,
                            Solver solver,
                            double alpha,
                            int batch_size,
                            double learning_rate,
                            int max_iter,
                            int random_state,
                            double tol,
                            bool verbose,
                            bool warm_start,
                            double momentum,
                            bool nesterovs_momentum,
                            bool early_stopping,
                            double validation_fraction,
                            double beta_1,
                            double beta_2,
                            double epsilon,
                            int n_iter_no_change)
    : MLPBase(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate,
             max_iter, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum,
             early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change),
      n_classes_(0) {}

Estimator& MLPClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    // Extract unique classes
    std::set<int> unique_classes;
    for (int i = 0; i < y.size(); ++i) {
        unique_classes.insert(static_cast<int>(y(i)));
    }
    classes_ = std::vector<int>(unique_classes.begin(), unique_classes.end());
    n_classes_ = classes_.size();
    
    // Initialize weights
    int n_outputs = (n_classes_ > 2) ? n_classes_ : 1;
    initialize_weights(X.cols(), n_outputs);
    
    loss_curve_.clear();
    
    // Training loop
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Compute gradients
        std::vector<MatrixXd> weight_gradients;
        std::vector<VectorXd> bias_gradients;
        std::vector<MatrixXd> layer_outputs;
        
        forward_pass(X, layer_outputs);
        backward_pass(X, y, layer_outputs, weight_gradients, bias_gradients);
        
        // Update weights
        if (solver_ == Solver::ADAM) {
            update_weights_adam(weight_gradients, bias_gradients, iter);
        } else {
            update_weights_sgd(weight_gradients, bias_gradients);
        }
        
        // Compute and store loss
        double current_loss = compute_loss(X, y);
        loss_curve_.push_back(current_loss);
        
        if (verbose_ && iter % 10 == 0) {
            std::cout << "Iteration " << iter << ", loss: " << current_loss << std::endl;
        }
        
        // Check convergence
        if (iter > 0 && std::abs(loss_curve_[iter] - loss_curve_[iter-1]) < tol_) {
            if (verbose_) {
                std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
            }
            break;
        }
    }
    
    fitted_ = true;
    return *this;
}

VectorXi MLPClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MLPClassifier not fitted");
    }
    
    std::vector<MatrixXd> layer_outputs;
    forward_pass(X, layer_outputs);
    
    VectorXi predictions(X.rows());
    
    if (n_classes_ == 2) {
        // Binary classification
        for (int i = 0; i < X.rows(); ++i) {
            double prob = 1.0 / (1.0 + std::exp(-layer_outputs.back()(i, 0)));
            predictions(i) = (prob > 0.5) ? classes_[1] : classes_[0];
        }
    } else {
        // Multiclass classification
        for (int i = 0; i < X.rows(); ++i) {
            VectorXd logits = layer_outputs.back().row(i);
            VectorXd probs = softmax(logits);
            int max_idx = 0;
            for (int j = 1; j < probs.size(); ++j) {
                if (probs(j) > probs(max_idx)) {
                    max_idx = j;
                }
            }
            predictions(i) = classes_[max_idx];
        }
    }
    
    return predictions;
}

MatrixXd MLPClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MLPClassifier not fitted");
    }
    
    std::vector<MatrixXd> layer_outputs;
    forward_pass(X, layer_outputs);
    
    MatrixXd probabilities(X.rows(), n_classes_);
    
    if (n_classes_ == 2) {
        // Binary classification
        for (int i = 0; i < X.rows(); ++i) {
            double prob_pos = 1.0 / (1.0 + std::exp(-layer_outputs.back()(i, 0)));
            probabilities(i, 0) = 1.0 - prob_pos;
            probabilities(i, 1) = prob_pos;
        }
    } else {
        // Multiclass classification
        for (int i = 0; i < X.rows(); ++i) {
            VectorXd logits = layer_outputs.back().row(i);
            VectorXd probs = softmax(logits);
            probabilities.row(i) = probs.transpose();
        }
    }
    
    return probabilities;
}

VectorXd MLPClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MLPClassifier not fitted");
    }
    
    std::vector<MatrixXd> layer_outputs;
    forward_pass(X, layer_outputs);
    
    if (n_classes_ == 2) {
        return layer_outputs.back().col(0);
    } else {
        // For multiclass, return the difference from mean
        MatrixXd scores = layer_outputs.back();
        VectorXd mean_scores = scores.rowwise().mean();
        return scores.col(0) - mean_scores;
    }
}

double MLPClassifier::compute_output_loss(const VectorXd& y_true, const VectorXd& y_pred) const {
    if (n_classes_ == 2) {
        // Binary cross-entropy
        double y = y_true(0);
        double p = 1.0 / (1.0 + std::exp(-y_pred(0)));
        p = std::max(1e-15, std::min(1.0 - 1e-15, p)); // Clip for numerical stability
        return -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
    } else {
        // Multiclass cross-entropy
        VectorXd probs = softmax(y_pred);
        int true_class = static_cast<int>(y_true(0));
        auto it = std::find(classes_.begin(), classes_.end(), true_class);
        if (it == classes_.end()) {
            throw std::runtime_error("Unknown class in y_true");
        }
        int class_idx = it - classes_.begin();
        double p = std::max(1e-15, probs(class_idx));
        return -std::log(p);
    }
}

VectorXd MLPClassifier::compute_output_gradient(const VectorXd& y_true, const VectorXd& y_pred) const {
    if (n_classes_ == 2) {
        // Binary classification gradient
        double y = y_true(0);
        double logit = y_pred.size() == 1 ? y_pred(0) : y_pred(1) - y_pred(0);
        double p = 1.0 / (1.0 + std::exp(-logit));
        VectorXd grad(1);
        grad(0) = p - y;
        return grad;
    } else {
        // Multiclass classification gradient
        VectorXd probs = softmax(y_pred);
        int true_class = static_cast<int>(y_true(0));
        auto it = std::find(classes_.begin(), classes_.end(), true_class);
        if (it == classes_.end()) {
            throw std::runtime_error("Unknown class in y_true");
        }
        int class_idx = it - classes_.begin();
        VectorXd grad = probs;
        grad(class_idx) -= 1.0;
        return grad;
    }
}

VectorXd MLPClassifier::predict_output(const MatrixXd& activations) const {
    return activations.row(0);
}

VectorXd MLPClassifier::softmax(const VectorXd& x) const {
    VectorXd shifted = x.array() - x.maxCoeff();
    VectorXd exp_values = shifted.array().exp();
    double sum_exp = exp_values.sum();
    return exp_values / sum_exp;
}

// MLPRegressor implementation
MLPRegressor::MLPRegressor(const std::vector<int>& hidden_layer_sizes,
                          ActivationFunction activation,
                          Solver solver,
                          double alpha,
                          int batch_size,
                          double learning_rate,
                          int max_iter,
                          int random_state,
                          double tol,
                          bool verbose,
                          bool warm_start,
                          double momentum,
                          bool nesterovs_momentum,
                          bool early_stopping,
                          double validation_fraction,
                          double beta_1,
                          double beta_2,
                          double epsilon,
                          int n_iter_no_change)
    : MLPBase(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate,
             max_iter, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum,
             early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change) {}

Estimator& MLPRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    // Initialize weights for single output regression
    initialize_weights(X.cols(), 1);
    
    loss_curve_.clear();
    
    // Training loop
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Compute gradients
        std::vector<MatrixXd> weight_gradients;
        std::vector<VectorXd> bias_gradients;
        std::vector<MatrixXd> layer_outputs;
        
        forward_pass(X, layer_outputs);
        backward_pass(X, y, layer_outputs, weight_gradients, bias_gradients);
        
        // Update weights
        if (solver_ == Solver::ADAM) {
            update_weights_adam(weight_gradients, bias_gradients, iter);
        } else {
            update_weights_sgd(weight_gradients, bias_gradients);
        }
        
        // Compute and store loss
        double current_loss = compute_loss(X, y);
        loss_curve_.push_back(current_loss);
        
        if (verbose_ && iter % 10 == 0) {
            std::cout << "Iteration " << iter << ", loss: " << current_loss << std::endl;
        }
        
        // Check convergence
        if (iter > 0 && std::abs(loss_curve_[iter] - loss_curve_[iter-1]) < tol_) {
            if (verbose_) {
                std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
            }
            break;
        }
    }
    
    fitted_ = true;
    return *this;
}

VectorXd MLPRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("MLPRegressor not fitted");
    }
    
    std::vector<MatrixXd> layer_outputs;
    forward_pass(X, layer_outputs);
    
    return layer_outputs.back().col(0);
}

double MLPRegressor::compute_output_loss(const VectorXd& y_true, const VectorXd& y_pred) const {
    // Mean squared error
    double diff = y_true(0) - y_pred(0);
    return 0.5 * diff * diff;
}

VectorXd MLPRegressor::compute_output_gradient(const VectorXd& y_true, const VectorXd& y_pred) const {
    // MSE gradient
    VectorXd grad(1);
    grad(0) = y_pred(0) - y_true(0);
    return grad;
}

VectorXd MLPRegressor::predict_output(const MatrixXd& activations) const {
    return activations.row(0);
}

// BernoulliRBM implementation

BernoulliRBM::BernoulliRBM(int n_components, double learning_rate, int batch_size,
                           int n_iter, int random_state, bool verbose)
    : n_components_(n_components),
      learning_rate_(learning_rate),
      batch_size_(batch_size),
      n_iter_(n_iter),
      random_state_(random_state),
      verbose_(verbose),
      fitted_(false),
      n_features_(0) {
    if (random_state_ >= 0) {
        rng_.seed(random_state_);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

MatrixXd BernoulliRBM::sigmoid(const MatrixXd& X) const {
    return (1.0 / (1.0 + (-X.array()).exp())).matrix();
}

Estimator& BernoulliRBM::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X(X);
    (void)y;

    n_features_ = X.cols();
    int n_samples = X.rows();

    std::normal_distribution<double> normal(0.0, 0.01);
    components_ = MatrixXd::Zero(n_components_, n_features_);
    for (int i = 0; i < n_components_; ++i) {
        for (int j = 0; j < n_features_; ++j) {
            components_(i, j) = normal(rng_);
        }
    }
    intercept_visible_ = VectorXd::Zero(n_features_);
    intercept_hidden_ = VectorXd::Zero(n_components_);

    MatrixXd X_clipped = X.cwiseMax(0.0).cwiseMin(1.0);

    for (int iter = 0; iter < n_iter_; ++iter) {
        MatrixXd pos_linear = X_clipped * components_.transpose();
        pos_linear.rowwise() += intercept_hidden_.transpose();
        MatrixXd pos_hidden = sigmoid(pos_linear);

        MatrixXd pos_assoc = pos_hidden.transpose() * X_clipped;

        MatrixXd neg_visible_linear = pos_hidden * components_;
        neg_visible_linear.rowwise() += intercept_visible_.transpose();
        MatrixXd neg_visible = sigmoid(neg_visible_linear);

        MatrixXd neg_hidden_linear = neg_visible * components_.transpose();
        neg_hidden_linear.rowwise() += intercept_hidden_.transpose();
        MatrixXd neg_hidden = sigmoid(neg_hidden_linear);

        MatrixXd neg_assoc = neg_hidden.transpose() * neg_visible;

        double lr = learning_rate_ / static_cast<double>(n_samples);
        components_ += lr * (pos_assoc - neg_assoc);

        VectorXd pos_vis_mean = X_clipped.colwise().mean().transpose();
        VectorXd neg_vis_mean = neg_visible.colwise().mean().transpose();
        intercept_visible_ += learning_rate_ * (pos_vis_mean - neg_vis_mean);

        VectorXd pos_hid_mean = pos_hidden.colwise().mean().transpose();
        VectorXd neg_hid_mean = neg_hidden.colwise().mean().transpose();
        intercept_hidden_ += learning_rate_ * (pos_hid_mean - neg_hid_mean);

        if (verbose_) {
            std::cout << "BernoulliRBM iter " << iter + 1 << "/" << n_iter_ << std::endl;
        }
    }

    fitted_ = true;
    return *this;
}

MatrixXd BernoulliRBM::transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BernoulliRBM must be fitted before transform");
    }
    if (X.cols() != n_features_) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }

    MatrixXd X_clipped = X.cwiseMax(0.0).cwiseMin(1.0);
    MatrixXd linear = X_clipped * components_.transpose();
    linear.rowwise() += intercept_hidden_.transpose();
    return sigmoid(linear);
}

MatrixXd BernoulliRBM::inverse_transform(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("BernoulliRBM must be fitted before inverse_transform");
    }
    if (X.cols() != n_components_) {
        throw std::invalid_argument("X must have the same number of components as the model");
    }

    MatrixXd linear = X * components_;
    linear.rowwise() += intercept_visible_.transpose();
    return sigmoid(linear);
}

MatrixXd BernoulliRBM::fit_transform(const MatrixXd& X, const VectorXd& y) {
    fit(X, y);
    return transform(X);
}

Params BernoulliRBM::get_params() const {
    Params params;
    params["n_components"] = std::to_string(n_components_);
    params["learning_rate"] = std::to_string(learning_rate_);
    params["batch_size"] = std::to_string(batch_size_);
    params["n_iter"] = std::to_string(n_iter_);
    params["random_state"] = std::to_string(random_state_);
    params["verbose"] = verbose_ ? "true" : "false";
    return params;
}

Estimator& BernoulliRBM::set_params(const Params& params) {
    n_components_ = utils::get_param_int(params, "n_components", n_components_);
    learning_rate_ = utils::get_param_double(params, "learning_rate", learning_rate_);
    batch_size_ = utils::get_param_int(params, "batch_size", batch_size_);
    n_iter_ = utils::get_param_int(params, "n_iter", n_iter_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    verbose_ = utils::get_param_bool(params, "verbose", verbose_);

    if (random_state_ >= 0) {
        rng_.seed(random_state_);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }

    return *this;
}

} // namespace neural_network
} // namespace ingenuityml

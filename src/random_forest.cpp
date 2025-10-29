#include "auroraml/random_forest.hpp"
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace auroraml {
namespace ensemble {

static std::vector<int> bootstrap_indices(int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> uni(0, n - 1);
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = uni(rng);
    return idx;
}

Estimator& RandomForestClassifier::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    n_features_ = X.cols();
    trees_.clear(); trees_.reserve(n_estimators_);
    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    for (int t = 0; t < n_estimators_; ++t) {
        auto idx = bootstrap_indices(X.rows(), rng);
        MatrixXd Xb(idx.size(), X.cols());
        VectorXd yb(idx.size());
        for (size_t i = 0; i < idx.size(); ++i) { Xb.row(i) = X.row(idx[i]); yb(i) = y(idx[i]); }
        tree::DecisionTreeClassifier dt("gini", max_depth_);
        dt.fit(Xb, yb);
        trees_.push_back(std::move(dt));
    }
    fitted_ = true;
    return *this;
}

VectorXi RandomForestClassifier::predict_classes(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("RandomForestClassifier not fitted");
    if (X.cols() != n_features_) {
        std::string msg = "X must have the same number of features as training data. Expected: " + 
                         std::to_string(n_features_) + ", got: " + std::to_string(X.cols());
        throw std::runtime_error(msg);
    }
    VectorXi pred(X.rows());
    
    #pragma omp parallel for if(X.rows() > 16)
    for (int i = 0; i < X.rows(); ++i) {
        std::unordered_map<int,int> votes;
        for (const auto& dt : trees_) {
            MatrixXd single_row = X.row(i);
            int c = dt.predict_classes(single_row)(0); // single row, get first element
            votes[c]++;
        }
        int best_c = 0, best_v = -1;
        for (auto& kv : votes) if (kv.second > best_v) { best_v = kv.second; best_c = kv.first; }
        pred(i) = best_c;
    }
    return pred;
}

MatrixXd RandomForestClassifier::predict_proba(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("RandomForestClassifier not fitted");
    // naive: average probabilities from trees that support predict_proba
    // For simplicity, convert votes to probabilities
    VectorXi cls = predict_classes(X);
    int max_label = cls.maxCoeff();
    MatrixXd P = MatrixXd::Zero(X.rows(), max_label + 1);
    for (int i = 0; i < X.rows(); ++i) P(i, cls(i)) = 1.0;
    return P;
}

VectorXd RandomForestClassifier::decision_function(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("RandomForestClassifier not fitted");
    VectorXi cls = predict_classes(X);
    return cls.cast<double>();
}

Params RandomForestClassifier::get_params() const {
    return {{"n_estimators", std::to_string(n_estimators_)}, {"max_depth", std::to_string(max_depth_)}, {"max_features", std::to_string(max_features_)}, {"random_state", std::to_string(random_state_)}};
}

Estimator& RandomForestClassifier::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    max_features_ = utils::get_param_int(params, "max_features", max_features_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

Estimator& RandomForestRegressor::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    n_features_ = X.cols();
    trees_.clear(); trees_.reserve(n_estimators_);
    std::mt19937 rng(static_cast<unsigned>(random_state_ == -1 ? std::random_device{}() : random_state_));
    for (int t = 0; t < n_estimators_; ++t) {
        auto idx = bootstrap_indices(X.rows(), rng);
        MatrixXd Xb(idx.size(), X.cols());
        VectorXd yb(idx.size());
        for (size_t i = 0; i < idx.size(); ++i) { Xb.row(i) = X.row(idx[i]); yb(i) = y(idx[i]); }
        tree::DecisionTreeRegressor dt("mse", max_depth_);
        dt.fit(Xb, yb);
        trees_.push_back(std::move(dt));
    }
    fitted_ = true;
    return *this;
}

VectorXd RandomForestRegressor::predict(const MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("RandomForestRegressor not fitted");
    if (X.cols() != n_features_) throw std::runtime_error("X must have the same number of features as training data");
    VectorXd y = VectorXd::Zero(X.rows());
    for (const auto& dt : trees_) y += dt.predict(X);
    y /= static_cast<double>(trees_.size());
    return y;
}

Params RandomForestRegressor::get_params() const {
    return {{"n_estimators", std::to_string(n_estimators_)}, {"max_depth", std::to_string(max_depth_)}, {"max_features", std::to_string(max_features_)}, {"random_state", std::to_string(random_state_)}};
}

Estimator& RandomForestRegressor::set_params(const Params& params) {
    n_estimators_ = utils::get_param_int(params, "n_estimators", n_estimators_);
    max_depth_ = utils::get_param_int(params, "max_depth", max_depth_);
    max_features_ = utils::get_param_int(params, "max_features", max_features_);
    random_state_ = utils::get_param_int(params, "random_state", random_state_);
    return *this;
}

// RandomForestClassifier save/load implementation
void RandomForestClassifier::save(const std::string& filepath) const {
    if (!fitted_) {
        throw std::runtime_error("RandomForestClassifier must be fitted before saving");
    }
    
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&n_estimators_), sizeof(n_estimators_));
    ofs.write(reinterpret_cast<const char*>(&max_depth_), sizeof(max_depth_));
    ofs.write(reinterpret_cast<const char*>(&max_features_), sizeof(max_features_));
    ofs.write(reinterpret_cast<const char*>(&random_state_), sizeof(random_state_));
    ofs.write(reinterpret_cast<const char*>(&n_features_), sizeof(n_features_));
    
    // Save number of trees
    int trees_size = trees_.size();
    ofs.write(reinterpret_cast<const char*>(&trees_size), sizeof(trees_size));
    
    ofs.close();
    
    // Save each tree to temporary files
    for (int i = 0; i < trees_size; ++i) {
        std::string temp_file = filepath + ".tree_" + std::to_string(i);
        trees_[i].save(temp_file);
    }
}

void RandomForestClassifier::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&n_estimators_), sizeof(n_estimators_));
    ifs.read(reinterpret_cast<char*>(&max_depth_), sizeof(max_depth_));
    ifs.read(reinterpret_cast<char*>(&max_features_), sizeof(max_features_));
    ifs.read(reinterpret_cast<char*>(&random_state_), sizeof(random_state_));
    ifs.read(reinterpret_cast<char*>(&n_features_), sizeof(n_features_));
    
    // Load number of trees
    int trees_size;
    ifs.read(reinterpret_cast<char*>(&trees_size), sizeof(trees_size));
    
    ifs.close();
    
    // Load each tree from temporary files
    trees_.clear();
    trees_.reserve(trees_size);
    for (int i = 0; i < trees_size; ++i) {
        std::string temp_file = filepath + ".tree_" + std::to_string(i);
        tree::DecisionTreeClassifier tree("gini", max_depth_);
        tree.load(temp_file);
        trees_.push_back(std::move(tree));
        // Clean up temporary file
        std::remove(temp_file.c_str());
    }
}

// RandomForestRegressor save/load implementation
void RandomForestRegressor::save(const std::string& filepath) const {
    if (!fitted_) {
        throw std::runtime_error("RandomForestRegressor must be fitted before saving");
    }
    
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }
    
    // Save basic parameters
    ofs.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
    ofs.write(reinterpret_cast<const char*>(&n_estimators_), sizeof(n_estimators_));
    ofs.write(reinterpret_cast<const char*>(&max_depth_), sizeof(max_depth_));
    ofs.write(reinterpret_cast<const char*>(&max_features_), sizeof(max_features_));
    ofs.write(reinterpret_cast<const char*>(&random_state_), sizeof(random_state_));
    ofs.write(reinterpret_cast<const char*>(&n_features_), sizeof(n_features_));
    
    // Save number of trees
    int trees_size = trees_.size();
    ofs.write(reinterpret_cast<const char*>(&trees_size), sizeof(trees_size));
    
    ofs.close();
    
    // Save each tree to temporary files
    for (int i = 0; i < trees_size; ++i) {
        std::string temp_file = filepath + ".tree_" + std::to_string(i);
        trees_[i].save(temp_file);
    }
}

void RandomForestRegressor::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }
    
    // Load basic parameters
    ifs.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
    ifs.read(reinterpret_cast<char*>(&n_estimators_), sizeof(n_estimators_));
    ifs.read(reinterpret_cast<char*>(&max_depth_), sizeof(max_depth_));
    ifs.read(reinterpret_cast<char*>(&max_features_), sizeof(max_features_));
    ifs.read(reinterpret_cast<char*>(&random_state_), sizeof(random_state_));
    ifs.read(reinterpret_cast<char*>(&n_features_), sizeof(n_features_));
    
    // Load number of trees
    int trees_size;
    ifs.read(reinterpret_cast<char*>(&trees_size), sizeof(trees_size));
    
    ifs.close();
    
    // Load each tree from temporary files
    trees_.clear();
    trees_.reserve(trees_size);
    for (int i = 0; i < trees_size; ++i) {
        std::string temp_file = filepath + ".tree_" + std::to_string(i);
        tree::DecisionTreeRegressor tree("mse", max_depth_);
        tree.load(temp_file);
        trees_.push_back(std::move(tree));
        // Clean up temporary file
        std::remove(temp_file.c_str());
    }
}

} // namespace ensemble
} // namespace cxml



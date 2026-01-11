#include "ingenuityml/base.hpp"
#include <stdexcept>

namespace ingenuityml {
namespace utils {

std::string get_param_string(const Params& params, const std::string& key, const std::string& default_val) {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
}

int get_param_int(const Params& params, const std::string& key, int default_val) {
    auto it = params.find(key);
    if (it != params.end()) {
        try {
            return std::stoi(it->second);
        } catch (const std::exception&) {
            return default_val;
        }
    }
    return default_val;
}

double get_param_double(const Params& params, const std::string& key, double default_val) {
    auto it = params.find(key);
    if (it != params.end()) {
        try {
            return std::stod(it->second);
        } catch (const std::exception&) {
            return default_val;
        }
    }
    return default_val;
}

bool get_param_bool(const Params& params, const std::string& key, bool default_val) {
    auto it = params.find(key);
    if (it != params.end()) {
        const std::string& value = it->second;
        if (value == "true" || value == "True" || value == "1") {
            return true;
        } else if (value == "false" || value == "False" || value == "0") {
            return false;
        }
    }
    return default_val;
}

} // namespace utils

namespace validation {

void check_X_y(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    if (X.rows() == 0) {
        throw std::invalid_argument("X and y cannot be empty");
    }
}

void check_X(const MatrixXd& X) {
    if (X.rows() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }
}

} // namespace validation

} // namespace ingenuityml

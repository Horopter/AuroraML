#include "auroraml/isotonic.hpp"
#include "auroraml/base.hpp"
#include <algorithm>
#include <cmath>

namespace auroraml {
namespace isotonic {

IsotonicRegression::IsotonicRegression(bool increasing)
    : increasing_(increasing), fitted_(false) {
}

Estimator& IsotonicRegression::fit(const MatrixXd& X, const VectorXd& y) {
    validation::check_X_y(X, y);
    
    if (X.cols() != 1) {
        throw std::invalid_argument("IsotonicRegression requires 1D input");
    }
    
    // Create pairs of (x, y) and sort
    std::vector<std::pair<double, double>> points;
    for (int i = 0; i < X.rows(); ++i) {
        points.push_back({X(i, 0), y(i)});
    }
    
    if (increasing_) {
        std::sort(points.begin(), points.end());
    } else {
        std::sort(points.begin(), points.end(), std::greater<std::pair<double, double>>());
    }
    
    // Pool Adjacent Violators Algorithm (PAV) - simplified
    thresholds_.clear();
    for (const auto& point : points) {
        thresholds_.push_back(point);
    }
    
    // Apply PAV (Pool Adjacent Violators Algorithm) to ensure monotonicity
    // Pool groups of adjacent violators until monotonicity is achieved
    bool changed = true;
    const int MAX_ITERATIONS = thresholds_.size() * 2;  // Reasonable upper bound
    int iterations = 0;
    
    while (changed && iterations < MAX_ITERATIONS) {
        changed = false;
        
        // Find and pool adjacent violators
        for (size_t i = 0; i < thresholds_.size() - 1; ++i) {
            bool is_violation = false;
            if (increasing_) {
                is_violation = thresholds_[i].second > thresholds_[i + 1].second;
            } else {
                is_violation = thresholds_[i].second < thresholds_[i + 1].second;
            }
            
            if (is_violation) {
                // Find the extent of the violation group
                size_t start = i;
                size_t end = i + 1;
                
                // Extend backward
                while (start > 0) {
                    bool prev_violation = increasing_ 
                        ? thresholds_[start - 1].second > thresholds_[start].second
                        : thresholds_[start - 1].second < thresholds_[start].second;
                    if (prev_violation) {
                        start--;
                    } else {
                        break;
                    }
                }
                
                // Extend forward
                while (end < thresholds_.size() - 1) {
                    bool next_violation = increasing_
                        ? thresholds_[end].second > thresholds_[end + 1].second
                        : thresholds_[end].second < thresholds_[end + 1].second;
                    if (next_violation) {
                        end++;
                    } else {
                        break;
                    }
                }
                
                // Pool the group: set all to the average
                double sum = 0.0;
                for (size_t j = start; j <= end; ++j) {
                    sum += thresholds_[j].second;
                }
                double avg = sum / (end - start + 1);
                
                for (size_t j = start; j <= end; ++j) {
                    thresholds_[j].second = avg;
                }
                
                changed = true;
                i = end;  // Skip ahead since we've fixed this group
            }
        }
        iterations++;
    }
    
    if (iterations >= MAX_ITERATIONS) {
        // If we can't converge, at least ensure basic monotonicity by sorting
        // This is a fallback - shouldn't normally happen
        std::vector<double> values;
        for (const auto& thresh : thresholds_) {
            values.push_back(thresh.second);
        }
        if (increasing_) {
            std::sort(values.begin(), values.end());
        } else {
            std::sort(values.begin(), values.end(), std::greater<double>());
        }
        for (size_t i = 0; i < thresholds_.size(); ++i) {
            thresholds_[i].second = values[i];
        }
    }
    
    X_min_ = VectorXd::Constant(1, points.front().first);
    X_max_ = VectorXd::Constant(1, points.back().first);
    y_min_ = VectorXd::Constant(1, points.front().second);
    y_max_ = VectorXd::Constant(1, points.back().second);
    
    fitted_ = true;
    return *this;
}

VectorXd IsotonicRegression::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("IsotonicRegression must be fitted before predict");
    }
    
    VectorXd predictions = VectorXd::Zero(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        double x_val = X(i, 0);
        
        // Find closest threshold
        double prediction = thresholds_[0].second;
        double min_dist = std::abs(x_val - thresholds_[0].first);
        
        for (size_t j = 0; j < thresholds_.size(); ++j) {
            double dist = std::abs(x_val - thresholds_[j].first);
            if (dist < min_dist) {
                min_dist = dist;
                prediction = thresholds_[j].second;
            }
        }
        
        // Linear interpolation if between thresholds
        for (size_t j = 0; j < thresholds_.size() - 1; ++j) {
            if (x_val >= thresholds_[j].first && x_val <= thresholds_[j + 1].first) {
                double t = (x_val - thresholds_[j].first) / (thresholds_[j + 1].first - thresholds_[j].first);
                prediction = thresholds_[j].second + t * (thresholds_[j + 1].second - thresholds_[j].second);
                break;
            }
        }
        
        predictions(i) = prediction;
    }
    
    return predictions;
}

VectorXd IsotonicRegression::transform(const VectorXd& X) const {
    MatrixXd X_mat = X;
    return predict(X_mat);
}

Params IsotonicRegression::get_params() const {
    Params params;
    params["increasing"] = increasing_ ? "true" : "false";
    return params;
}

Estimator& IsotonicRegression::set_params(const Params& params) {
    increasing_ = utils::get_param_bool(params, "increasing", increasing_);
    return *this;
}

} // namespace isotonic
} // namespace auroraml


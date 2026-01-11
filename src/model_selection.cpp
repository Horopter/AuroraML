#include "ingenuityml/model_selection.hpp"
#include "ingenuityml/base.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <numeric>
#include <map>
#include <set>
#include <cmath>
#include <limits>

namespace ingenuityml {
namespace model_selection {

namespace {
int compute_test_size(int n_samples, double test_size, double train_size) {
    if (n_samples <= 1) {
        throw std::runtime_error("Need at least 2 samples to split");
    }
    int n_test = static_cast<int>(std::round(n_samples * test_size));
    if (train_size > 0.0) {
        int n_train = static_cast<int>(std::round(n_samples * train_size));
        n_test = n_samples - n_train;
    }
    if (n_test <= 0 || n_test >= n_samples) {
        throw std::invalid_argument("Invalid train/test split sizes");
    }
    return n_test;
}

} // namespace

// train_test_split implementation
std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> train_test_split(
    const MatrixXd& X, const VectorXd& y, double test_size, double train_size,
    int random_state, bool shuffle, const VectorXd& stratify) {
    
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    int n_samples = X.rows();
    int n_test_samples = static_cast<int>(n_samples * test_size);
    int n_train_samples = n_samples - n_test_samples;
    
    if (train_size > 0) {
        n_train_samples = static_cast<int>(n_samples * train_size);
        n_test_samples = n_samples - n_train_samples;
    }
    
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle) {
        std::mt19937 rng(random_state >= 0 ? random_state : std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    std::vector<int> train_indices(indices.begin(), indices.begin() + n_train_samples);
    std::vector<int> test_indices(indices.begin() + n_train_samples, indices.end());
    
    // Create train and test sets
    MatrixXd X_train(n_train_samples, X.cols());
    MatrixXd X_test(n_test_samples, X.cols());
    VectorXd y_train(n_train_samples);
    VectorXd y_test(n_test_samples);
    
    for (int i = 0; i < n_train_samples; ++i) {
        X_train.row(i) = X.row(train_indices[i]);
        y_train(i) = y(train_indices[i]);
    }
    
    for (int i = 0; i < n_test_samples; ++i) {
        X_test.row(i) = X.row(test_indices[i]);
        y_test(i) = y(test_indices[i]);
    }
    
    return std::make_tuple(X_train, X_test, y_train, y_test);
}

// ParameterGrid implementation
ParameterGrid::ParameterGrid(const std::map<std::string, std::vector<std::string>>& param_grid) {
    if (param_grid.empty()) {
        throw std::invalid_argument("param_grid cannot be empty");
    }

    grid_.clear();
    grid_.push_back(Params{});

    for (const auto& [key, values] : param_grid) {
        if (values.empty()) {
            throw std::invalid_argument("param_grid values cannot be empty");
        }
        std::vector<Params> next_grid;
        next_grid.reserve(grid_.size() * values.size());
        for (const auto& params : grid_) {
            for (const auto& value : values) {
                Params updated = params;
                updated[key] = value;
                next_grid.push_back(updated);
            }
        }
        grid_.swap(next_grid);
    }
}

// ParameterSampler implementation
ParameterSampler::ParameterSampler(const std::map<std::string, std::vector<std::string>>& param_distributions,
                                   int n_iter, int random_state) {
    if (param_distributions.empty()) {
        throw std::invalid_argument("param_distributions cannot be empty");
    }
    if (n_iter <= 0) {
        throw std::invalid_argument("n_iter must be positive");
    }

    std::mt19937 rng(random_state >= 0 ? random_state : std::random_device{}());
    samples_.reserve(n_iter);

    for (int i = 0; i < n_iter; ++i) {
        Params params;
        for (const auto& [key, values] : param_distributions) {
            if (values.empty()) {
                throw std::invalid_argument("param_distributions values cannot be empty");
            }
            std::uniform_int_distribution<> dist(0, static_cast<int>(values.size()) - 1);
            params[key] = values[dist(rng)];
        }
        samples_.push_back(params);
    }
}

// KFold implementation
KFold::KFold(int n_splits, bool shuffle, int random_state)
    : n_splits_(n_splits), shuffle_(shuffle), random_state_(random_state) {
    if (n_splits < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> KFold::split(const MatrixXd& X, const VectorXd& y) const {
    int n_samples = X.rows();
    if (n_splits_ > n_samples) {
        throw std::runtime_error("n_splits cannot be greater than the number of samples");
    }
    int fold_size = n_samples / n_splits_;
    
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle_) {
        std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    
    for (int i = 0; i < n_splits_; ++i) {
        int start = i * fold_size;
        int end = (i == n_splits_ - 1) ? n_samples : (i + 1) * fold_size;
        
        std::vector<int> test_indices(indices.begin() + start, indices.begin() + end);
        std::vector<int> train_indices;
        
        train_indices.insert(train_indices.end(), indices.begin(), indices.begin() + start);
        train_indices.insert(train_indices.end(), indices.begin() + end, indices.end());
        
        splits.emplace_back(train_indices, test_indices);
    }
    
    return splits;
}

// RepeatedKFold implementation
RepeatedKFold::RepeatedKFold(int n_splits, int n_repeats, int random_state)
    : n_splits_(n_splits), n_repeats_(n_repeats), random_state_(random_state) {
    if (n_splits_ < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
    if (n_repeats_ < 1) {
        throw std::invalid_argument("n_repeats must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> RepeatedKFold::split(const MatrixXd& X, const VectorXd& y) const {
    if (n_splits_ > X.rows()) {
        throw std::runtime_error("n_splits cannot be greater than the number of samples");
    }

    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    splits.reserve(static_cast<size_t>(n_splits_) * n_repeats_);

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());

    for (int repeat = 0; repeat < n_repeats_; ++repeat) {
        int seed = random_state_ >= 0 ? (random_state_ + repeat) : static_cast<int>(rng());
        KFold kfold(n_splits_, true, seed);
        auto repeat_splits = kfold.split(X, y);
        splits.insert(splits.end(), repeat_splits.begin(), repeat_splits.end());
    }

    return splits;
}

// StratifiedKFold implementation
StratifiedKFold::StratifiedKFold(int n_splits, bool shuffle, int random_state)
    : n_splits_(n_splits), shuffle_(shuffle), random_state_(random_state) {
    if (n_splits < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> StratifiedKFold::split(
    const MatrixXd& X, const VectorXd& y) const {
    
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    if (n_splits_ > X.rows()) {
        throw std::runtime_error("n_splits cannot be greater than the number of samples");
    }
    
    // Group indices by class
    std::map<int, std::vector<int>> class_indices;
    for (int i = 0; i < y.size(); ++i) {
        int class_label = static_cast<int>(y(i));
        class_indices[class_label].push_back(i);
    }
    
    // Shuffle within each class if requested
    if (shuffle_) {
        std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
        for (auto& pair : class_indices) {
            std::shuffle(pair.second.begin(), pair.second.end(), rng);
        }
    }
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits(n_splits_);
    
    // Initialize splits
    for (int i = 0; i < n_splits_; ++i) {
        splits[i] = std::make_pair(std::vector<int>(), std::vector<int>());
    }
    
    // Distribute samples from each class across folds
    for (const auto& class_pair : class_indices) {
        const std::vector<int>& indices = class_pair.second;
        int n_class_samples = indices.size();
        int samples_per_fold = n_class_samples / n_splits_;
        int remainder = n_class_samples % n_splits_;
        
        int current_pos = 0;
        for (int fold = 0; fold < n_splits_; ++fold) {
            int fold_size = samples_per_fold + (fold < remainder ? 1 : 0);
            
            for (int i = 0; i < fold_size; ++i) {
                splits[fold].second.push_back(indices[current_pos + i]);
            }
            
            // Add remaining samples to training set
            for (int i = 0; i < current_pos; ++i) {
                splits[fold].first.push_back(indices[i]);
            }
            for (int i = current_pos + fold_size; i < n_class_samples; ++i) {
                splits[fold].first.push_back(indices[i]);
            }
            
            current_pos += fold_size;
        }
    }
    
    return splits;
}

// RepeatedStratifiedKFold implementation
RepeatedStratifiedKFold::RepeatedStratifiedKFold(int n_splits, int n_repeats, int random_state)
    : n_splits_(n_splits), n_repeats_(n_repeats), random_state_(random_state) {
    if (n_splits_ < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
    if (n_repeats_ < 1) {
        throw std::invalid_argument("n_repeats must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> RepeatedStratifiedKFold::split(
    const MatrixXd& X, const VectorXd& y) const {

    if (n_splits_ > X.rows()) {
        throw std::runtime_error("n_splits cannot be greater than the number of samples");
    }

    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    splits.reserve(static_cast<size_t>(n_splits_) * n_repeats_);

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());

    for (int repeat = 0; repeat < n_repeats_; ++repeat) {
        int seed = random_state_ >= 0 ? (random_state_ + repeat) : static_cast<int>(rng());
        StratifiedKFold skfold(n_splits_, true, seed);
        auto repeat_splits = skfold.split(X, y);
        splits.insert(splits.end(), repeat_splits.begin(), repeat_splits.end());
    }

    return splits;
}

// GroupKFold implementation
GroupKFold::GroupKFold(int n_splits) : n_splits_(n_splits) {
    if (n_splits < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> GroupKFold::split(
    const MatrixXd& X, const VectorXd& y, const VectorXd& groups) const {
    
    if (X.rows() == 0) {
        throw std::runtime_error("X cannot be empty");
    }
    if (X.rows() != y.size() || X.rows() != groups.size()) {
        throw std::invalid_argument("X, y, and groups must have the same number of samples");
    }
    
    // Group indices by group ID
    std::map<int, std::vector<int>> group_indices;
    for (int i = 0; i < groups.size(); ++i) {
        int group_id = static_cast<int>(groups(i));
        group_indices[group_id].push_back(i);
    }
    
    int n_groups = group_indices.size();
    if (n_splits_ > n_groups) {
        throw std::runtime_error("n_splits cannot be greater than the number of groups");
    }
    
    std::vector<int> group_ids;
    for (const auto& pair : group_indices) {
        group_ids.push_back(pair.first);
    }
    
    int groups_per_fold = n_groups / n_splits_;
    int remainder = n_groups % n_splits_;
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits(n_splits_);
    
    // Initialize splits
    for (int i = 0; i < n_splits_; ++i) {
        splits[i] = std::make_pair(std::vector<int>(), std::vector<int>());
    }
    
    // Distribute groups across folds
    int current_group = 0;
    for (int fold = 0; fold < n_splits_; ++fold) {
        int fold_groups = groups_per_fold + (fold < remainder ? 1 : 0);
        
        for (int g = 0; g < fold_groups; ++g) {
            int group_id = group_ids[current_group];
            const std::vector<int>& group_indices_vec = group_indices[group_id];
            
            splits[fold].second.insert(splits[fold].second.end(),
                                     group_indices_vec.begin(), group_indices_vec.end());
            current_group++;
        }
    }
    
    // Build training sets (all groups not in test set)
    for (int fold = 0; fold < n_splits_; ++fold) {
        std::set<int> test_groups;
        for (int idx : splits[fold].second) {
            int group_id = static_cast<int>(groups(idx));
            test_groups.insert(group_id);
        }
        
        for (int i = 0; i < groups.size(); ++i) {
            int group_id = static_cast<int>(groups(i));
            if (test_groups.find(group_id) == test_groups.end()) {
                splits[fold].first.push_back(i);
            }
        }
    }
    
    return splits;
}

// ShuffleSplit implementation
ShuffleSplit::ShuffleSplit(int n_splits, double test_size, double train_size, int random_state)
    : n_splits_(n_splits), test_size_(test_size), train_size_(train_size), random_state_(random_state) {
    if (n_splits_ < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> ShuffleSplit::split(const MatrixXd& X, const VectorXd& y) const {
    int n_samples = X.rows();
    int n_test = compute_test_size(n_samples, test_size_, train_size_);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    splits.reserve(n_splits_);

    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());

    for (int i = 0; i < n_splits_; ++i) {
        std::shuffle(indices.begin(), indices.end(), rng);
        std::vector<int> test_indices(indices.begin(), indices.begin() + n_test);
        std::vector<int> train_indices(indices.begin() + n_test, indices.end());
        splits.emplace_back(train_indices, test_indices);
    }

    return splits;
}

// StratifiedShuffleSplit implementation
StratifiedShuffleSplit::StratifiedShuffleSplit(int n_splits, double test_size, double train_size, int random_state)
    : n_splits_(n_splits), test_size_(test_size), train_size_(train_size), random_state_(random_state) {
    if (n_splits_ < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> StratifiedShuffleSplit::split(
    const MatrixXd& X, const VectorXd& y) const {

    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    int n_samples = X.rows();
    int n_test = compute_test_size(n_samples, test_size_, train_size_);

    std::map<int, std::vector<int>> class_indices;
    for (int i = 0; i < y.size(); ++i) {
        class_indices[static_cast<int>(y(i))].push_back(i);
    }

    if (class_indices.empty()) {
        throw std::runtime_error("y must contain at least one class");
    }

    for (const auto& entry : class_indices) {
        if (entry.second.size() < 2) {
            throw std::runtime_error("Each class must contain at least 2 samples for stratified shuffle split");
        }
    }

    std::vector<int> class_labels;
    std::vector<int> class_counts;
    std::vector<double> class_fracs;
    int n_classes = class_indices.size();
    class_labels.reserve(n_classes);
    class_counts.reserve(n_classes);
    class_fracs.reserve(n_classes);

    for (const auto& entry : class_indices) {
        class_labels.push_back(entry.first);
        class_counts.push_back(static_cast<int>(entry.second.size()));
        class_fracs.push_back(static_cast<double>(entry.second.size()) / n_samples);
    }

    std::vector<int> base_test_counts(n_classes, 0);
    std::vector<double> remainders(n_classes, 0.0);
    int assigned = 0;

    for (int i = 0; i < n_classes; ++i) {
        double exact = class_fracs[i] * n_test;
        int count = static_cast<int>(std::floor(exact));
        base_test_counts[i] = count;
        remainders[i] = exact - count;
        assigned += count;
    }

    int remaining = n_test - assigned;
    std::vector<int> order(n_classes);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return remainders[a] > remainders[b];
    });
    for (int i = 0; i < remaining; ++i) {
        base_test_counts[order[i % n_classes]]++;
    }

    for (int i = 0; i < n_classes; ++i) {
        if (base_test_counts[i] >= class_counts[i]) {
            base_test_counts[i] = class_counts[i] - 1;
        }
    }

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    splits.reserve(n_splits_);

    for (int split_idx = 0; split_idx < n_splits_; ++split_idx) {
        std::vector<int> train_indices;
        std::vector<int> test_indices;
        train_indices.reserve(n_samples - n_test);
        test_indices.reserve(n_test);

        for (int i = 0; i < n_classes; ++i) {
            int label = class_labels[i];
            std::vector<int> indices = class_indices[label];
            std::shuffle(indices.begin(), indices.end(), rng);

            int n_test_class = base_test_counts[i];
            test_indices.insert(test_indices.end(), indices.begin(), indices.begin() + n_test_class);
            train_indices.insert(train_indices.end(), indices.begin() + n_test_class, indices.end());
        }

        splits.emplace_back(train_indices, test_indices);
    }

    return splits;
}

// GroupShuffleSplit implementation
GroupShuffleSplit::GroupShuffleSplit(int n_splits, double test_size, double train_size, int random_state)
    : n_splits_(n_splits), test_size_(test_size), train_size_(train_size), random_state_(random_state) {
    if (n_splits_ < 1) {
        throw std::invalid_argument("n_splits must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> GroupShuffleSplit::split(
    const MatrixXd& X, const VectorXd& y, const VectorXd& groups) const {

    if (X.rows() == 0) {
        throw std::runtime_error("X cannot be empty");
    }
    if (X.rows() != y.size() || X.rows() != groups.size()) {
        throw std::invalid_argument("X, y, and groups must have the same number of samples");
    }

    std::map<int, std::vector<int>> group_indices;
    for (int i = 0; i < groups.size(); ++i) {
        int group_id = static_cast<int>(groups(i));
        group_indices[group_id].push_back(i);
    }

    int n_groups = group_indices.size();
    if (n_groups < 2) {
        throw std::runtime_error("Need at least 2 groups to split");
    }

    int n_test_groups = compute_test_size(n_groups, test_size_, train_size_);

    std::vector<int> group_ids;
    group_ids.reserve(n_groups);
    for (const auto& entry : group_indices) {
        group_ids.push_back(entry.first);
    }

    std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    splits.reserve(n_splits_);

    for (int split_idx = 0; split_idx < n_splits_; ++split_idx) {
        std::shuffle(group_ids.begin(), group_ids.end(), rng);

        std::set<int> test_groups(group_ids.begin(), group_ids.begin() + n_test_groups);

        std::vector<int> train_indices;
        std::vector<int> test_indices;

        for (const auto& entry : group_indices) {
            const std::vector<int>& indices = entry.second;
            if (test_groups.count(entry.first)) {
                test_indices.insert(test_indices.end(), indices.begin(), indices.end());
            } else {
                train_indices.insert(train_indices.end(), indices.begin(), indices.end());
            }
        }

        splits.emplace_back(train_indices, test_indices);
    }

    return splits;
}

// PredefinedSplit implementation
PredefinedSplit::PredefinedSplit(const std::vector<int>& test_fold)
    : test_fold_(test_fold) {
    std::set<int> fold_ids;
    for (int fold : test_fold_) {
        if (fold >= 0) {
            fold_ids.insert(fold);
        }
    }
    n_splits_ = static_cast<int>(fold_ids.size());
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> PredefinedSplit::split(
    const MatrixXd& X, const VectorXd& y) const {

    if (static_cast<int>(test_fold_.size()) != X.rows()) {
        throw std::invalid_argument("test_fold must have the same number of samples as X");
    }

    std::set<int> fold_ids;
    for (int fold : test_fold_) {
        if (fold >= 0) {
            fold_ids.insert(fold);
        }
    }

    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    splits.reserve(fold_ids.size());

    for (int fold_id : fold_ids) {
        std::vector<int> train_indices;
        std::vector<int> test_indices;
        for (int i = 0; i < static_cast<int>(test_fold_.size()); ++i) {
            if (test_fold_[i] == fold_id) {
                test_indices.push_back(i);
            } else {
                train_indices.push_back(i);
            }
        }
        splits.emplace_back(train_indices, test_indices);
    }

    return splits;
}

// LeaveOneOut implementation
std::vector<std::pair<std::vector<int>, std::vector<int>>> LeaveOneOut::split(
    const MatrixXd& X, const VectorXd& y) const {

    int n_samples = X.rows();
    if (n_samples < 1) {
        throw std::runtime_error("X cannot be empty");
    }

    n_splits_ = n_samples;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    splits.reserve(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        std::vector<int> train_indices;
        std::vector<int> test_indices = {i};
        train_indices.reserve(n_samples - 1);

        for (int j = 0; j < n_samples; ++j) {
            if (j != i) {
                train_indices.push_back(j);
            }
        }
        splits.emplace_back(train_indices, test_indices);
    }

    return splits;
}

// LeavePOut implementation
LeavePOut::LeavePOut(int p) : p_(p) {
    if (p_ < 1) {
        throw std::invalid_argument("p must be at least 1");
    }
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> LeavePOut::split(
    const MatrixXd& X, const VectorXd& y) const {

    int n_samples = X.rows();
    if (n_samples == 0) {
        throw std::runtime_error("X cannot be empty");
    }
    if (p_ > n_samples) {
        throw std::invalid_argument("p cannot be greater than number of samples");
    }

    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;

    std::vector<int> comb(p_);
    std::iota(comb.begin(), comb.end(), 0);

    while (true) {
        std::vector<int> test_indices = comb;
        std::vector<int> train_indices;
        train_indices.reserve(n_samples - p_);

        int comb_idx = 0;
        for (int i = 0; i < n_samples; ++i) {
            if (comb_idx < p_ && comb[comb_idx] == i) {
                ++comb_idx;
            } else {
                train_indices.push_back(i);
            }
        }
        splits.emplace_back(train_indices, test_indices);

        int i = p_ - 1;
        while (i >= 0 && comb[i] == n_samples - p_ + i) {
            --i;
        }
        if (i < 0) {
            break;
        }
        comb[i]++;
        for (int j = i + 1; j < p_; ++j) {
            comb[j] = comb[j - 1] + 1;
        }
    }

    n_splits_ = static_cast<int>(splits.size());
    return splits;
}

// cross_val_score implementation
VectorXd cross_val_score(Estimator& estimator, const MatrixXd& X, const VectorXd& y,
                        const BaseCrossValidator& cv, const std::string& scoring) {
    
    auto splits = cv.split(X, y);
    int n_splits = splits.size();
    VectorXd scores(n_splits);
    
    for (int i = 0; i < n_splits; ++i) {
        const auto& train_indices = splits[i].first;
        const auto& test_indices = splits[i].second;
        
        // Create train and test sets
        MatrixXd X_train(train_indices.size(), X.cols());
        VectorXd y_train(train_indices.size());
        MatrixXd X_test(test_indices.size(), X.cols());
        VectorXd y_test(test_indices.size());
        
        for (size_t j = 0; j < train_indices.size(); ++j) {
            X_train.row(j) = X.row(train_indices[j]);
            y_train(j) = y(train_indices[j]);
        }
        
        for (size_t j = 0; j < test_indices.size(); ++j) {
            X_test.row(j) = X.row(test_indices[j]);
            y_test(j) = y(test_indices[j]);
        }
        
        // Train estimator
        estimator.fit(X_train, y_train);
        
        // Make predictions
        VectorXd y_pred;
        if (auto* predictor = dynamic_cast<const Predictor*>(&estimator)) {
            y_pred = predictor->predict(X_test);
        } else {
            throw std::runtime_error("Estimator must be a Predictor to make predictions");
        }
        
        // Calculate score
        if (scoring == "accuracy") {
            VectorXi y_true_int = y_test.cast<int>();
            VectorXi y_pred_int = y_pred.cast<int>();
            scores(i) = metrics::accuracy_score(y_true_int, y_pred_int);
        } else if (scoring == "precision") {
            VectorXi y_true_int = y_test.cast<int>();
            VectorXi y_pred_int = y_pred.cast<int>();
            scores(i) = metrics::precision_score(y_true_int, y_pred_int);
        } else if (scoring == "recall") {
            VectorXi y_true_int = y_test.cast<int>();
            VectorXi y_pred_int = y_pred.cast<int>();
            scores(i) = metrics::recall_score(y_true_int, y_pred_int);
        } else if (scoring == "f1") {
            VectorXi y_true_int = y_test.cast<int>();
            VectorXi y_pred_int = y_pred.cast<int>();
            scores(i) = metrics::f1_score(y_true_int, y_pred_int);
        } else if (scoring == "mse") {
            scores(i) = -metrics::mean_squared_error(y_test, y_pred);  // Negative for maximization
        } else if (scoring == "mae") {
            scores(i) = -metrics::mean_absolute_error(y_test, y_pred);  // Negative for maximization
        } else if (scoring == "r2") {
            scores(i) = metrics::r2_score(y_test, y_pred);
        } else {
            throw std::invalid_argument("Unsupported scoring metric: " + scoring);
        }
    }
    
    return scores;
}

// GridSearchCV implementation
GridSearchCV::GridSearchCV(Estimator& estimator, const std::vector<Params>& param_grid,
                          const BaseCrossValidator& cv, const std::string& scoring,
                          int n_jobs, bool verbose)
    : estimator_(estimator), param_grid_(param_grid), cv_(cv), scoring_(scoring),
      n_jobs_(n_jobs), verbose_(verbose) {
    if (param_grid.empty()) {
        throw std::invalid_argument("param_grid cannot be empty");
    }
    if (n_jobs < 1) {
        throw std::invalid_argument("n_jobs must be at least 1");
    }
}

Estimator& GridSearchCV::fit(const MatrixXd& X, const VectorXd& y) {
    double best_score = -std::numeric_limits<double>::infinity();
    Params best_params;
    
    for (const auto& params : param_grid_) {
        estimator_.set_params(params);
        VectorXd scores = cross_val_score(estimator_, X, y, cv_, scoring_);
        double mean_score = scores.mean();
        
        if (mean_score > best_score) {
            best_score = mean_score;
            best_params = params;
        }
    }
    
    estimator_.set_params(best_params);
    estimator_.fit(X, y);
    
    return estimator_;
}

VectorXd GridSearchCV::predict(const MatrixXd& X) const {
    if (auto* predictor = dynamic_cast<const Predictor*>(&estimator_)) {
        return predictor->predict(X);
    } else {
        throw std::runtime_error("Estimator must be a Predictor to make predictions");
    }
}

Params GridSearchCV::best_params() const {
    return estimator_.get_params();
}

double GridSearchCV::best_score() const {
    return 0.0;  // Would need to store this during fit
}

// RandomizedSearchCV implementation
RandomizedSearchCV::RandomizedSearchCV(Estimator& estimator, const std::vector<Params>& param_distributions,
                                     const BaseCrossValidator& cv, const std::string& scoring,
                                     int n_iter, int n_jobs, bool verbose)
    : estimator_(estimator), param_distributions_(param_distributions), cv_(cv), scoring_(scoring),
      n_iter_(n_iter), n_jobs_(n_jobs), verbose_(verbose) {
    if (param_distributions.empty()) {
        throw std::invalid_argument("param_distributions cannot be empty");
    }
    if (n_iter <= 0) {
        throw std::invalid_argument("n_iter must be positive");
    }
    if (n_jobs < 1) {
        throw std::invalid_argument("n_jobs must be at least 1");
    }
}

Estimator& RandomizedSearchCV::fit(const MatrixXd& X, const VectorXd& y) {
    double best_score = -std::numeric_limits<double>::infinity();
    Params best_params;
    
    std::mt19937 rng(std::random_device{}());
    
    for (int i = 0; i < n_iter_; ++i) {
        // Randomly select parameter set
        std::uniform_int_distribution<> dist(0, param_distributions_.size() - 1);
        Params params = param_distributions_[dist(rng)];
        
        estimator_.set_params(params);
        VectorXd scores = cross_val_score(estimator_, X, y, cv_, scoring_);
        double mean_score = scores.mean();
        
        if (mean_score > best_score) {
            best_score = mean_score;
            best_params = params;
        }
    }
    
    estimator_.set_params(best_params);
    estimator_.fit(X, y);
    
    return estimator_;
}

VectorXd RandomizedSearchCV::predict(const MatrixXd& X) const {
    if (auto* predictor = dynamic_cast<const Predictor*>(&estimator_)) {
        return predictor->predict(X);
    } else {
        throw std::runtime_error("Estimator must be a Predictor to make predictions");
    }
}

Params RandomizedSearchCV::best_params() const {
    return estimator_.get_params();
}

double RandomizedSearchCV::best_score() const {
    return 0.0;  // Would need to store this during fit
}

// Parameter management implementations for KFold
Params KFold::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)},
            {"shuffle", shuffle_ ? "1" : "0"},
            {"random_state", std::to_string(random_state_)}};
}

void KFold::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
    if (params.find("shuffle") != params.end()) {
        shuffle_ = (params.at("shuffle") == "1");
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
}

// Parameter management implementations for StratifiedKFold
Params StratifiedKFold::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)},
            {"shuffle", shuffle_ ? "1" : "0"},
            {"random_state", std::to_string(random_state_)}};
}

void StratifiedKFold::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
    if (params.find("shuffle") != params.end()) {
        shuffle_ = (params.at("shuffle") == "1");
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
}

// Parameter management implementations for GroupKFold
Params GroupKFold::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)}};
}

void GroupKFold::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
}

// Parameter management implementations for RepeatedKFold
Params RepeatedKFold::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)},
            {"n_repeats", std::to_string(n_repeats_)},
            {"random_state", std::to_string(random_state_)}};
}

void RepeatedKFold::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
    if (params.find("n_repeats") != params.end()) {
        n_repeats_ = std::stoi(params.at("n_repeats"));
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
}

// Parameter management implementations for RepeatedStratifiedKFold
Params RepeatedStratifiedKFold::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)},
            {"n_repeats", std::to_string(n_repeats_)},
            {"random_state", std::to_string(random_state_)}};
}

void RepeatedStratifiedKFold::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
    if (params.find("n_repeats") != params.end()) {
        n_repeats_ = std::stoi(params.at("n_repeats"));
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
}

// Parameter management implementations for ShuffleSplit
Params ShuffleSplit::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)},
            {"test_size", std::to_string(test_size_)},
            {"train_size", std::to_string(train_size_)},
            {"random_state", std::to_string(random_state_)}};
}

void ShuffleSplit::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
    if (params.find("test_size") != params.end()) {
        test_size_ = std::stod(params.at("test_size"));
    }
    if (params.find("train_size") != params.end()) {
        train_size_ = std::stod(params.at("train_size"));
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
}

// Parameter management implementations for StratifiedShuffleSplit
Params StratifiedShuffleSplit::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)},
            {"test_size", std::to_string(test_size_)},
            {"train_size", std::to_string(train_size_)},
            {"random_state", std::to_string(random_state_)}};
}

void StratifiedShuffleSplit::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
    if (params.find("test_size") != params.end()) {
        test_size_ = std::stod(params.at("test_size"));
    }
    if (params.find("train_size") != params.end()) {
        train_size_ = std::stod(params.at("train_size"));
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
}

// Parameter management implementations for GroupShuffleSplit
Params GroupShuffleSplit::get_params() const {
    return {{"n_splits", std::to_string(n_splits_)},
            {"test_size", std::to_string(test_size_)},
            {"train_size", std::to_string(train_size_)},
            {"random_state", std::to_string(random_state_)}};
}

void GroupShuffleSplit::set_params(const Params& params) {
    if (params.find("n_splits") != params.end()) {
        n_splits_ = std::stoi(params.at("n_splits"));
    }
    if (params.find("test_size") != params.end()) {
        test_size_ = std::stod(params.at("test_size"));
    }
    if (params.find("train_size") != params.end()) {
        train_size_ = std::stod(params.at("train_size"));
    }
    if (params.find("random_state") != params.end()) {
        random_state_ = std::stoi(params.at("random_state"));
    }
}

// Parameter management implementations for LeavePOut
Params LeavePOut::get_params() const {
    return {{"p", std::to_string(p_)}};
}

void LeavePOut::set_params(const Params& params) {
    if (params.find("p") != params.end()) {
        p_ = std::stoi(params.at("p"));
    }
}

// Parameter management implementations for GridSearchCV
Params GridSearchCV::get_params() const {
    return {{"scoring", scoring_},
            {"n_jobs", std::to_string(n_jobs_)},
            {"verbose", verbose_ ? "1" : "0"}};
}

void GridSearchCV::set_params(const Params& params) {
    if (params.find("scoring") != params.end()) {
        scoring_ = params.at("scoring");
    }
    if (params.find("n_jobs") != params.end()) {
        n_jobs_ = std::stoi(params.at("n_jobs"));
    }
    if (params.find("verbose") != params.end()) {
        verbose_ = (params.at("verbose") == "1");
    }
}

int GridSearchCV::get_n_splits() const {
    return cv_.get_n_splits();
}

// Parameter management implementations for RandomizedSearchCV
Params RandomizedSearchCV::get_params() const {
    return {{"scoring", scoring_},
            {"n_iter", std::to_string(n_iter_)},
            {"n_jobs", std::to_string(n_jobs_)},
            {"verbose", verbose_ ? "1" : "0"}};
}

void RandomizedSearchCV::set_params(const Params& params) {
    if (params.find("scoring") != params.end()) {
        scoring_ = params.at("scoring");
    }
    if (params.find("n_iter") != params.end()) {
        n_iter_ = std::stoi(params.at("n_iter"));
    }
    if (params.find("n_jobs") != params.end()) {
        n_jobs_ = std::stoi(params.at("n_jobs"));
    }
    if (params.find("verbose") != params.end()) {
        verbose_ = (params.at("verbose") == "1");
    }
}

int RandomizedSearchCV::get_n_splits() const {
    return cv_.get_n_splits();
}

// TimeSeriesSplit implementation
TimeSeriesSplit::TimeSeriesSplit(int n_splits, int max_train_size, int test_size, int gap)
    : n_splits_(n_splits), max_train_size_(max_train_size), test_size_(test_size), gap_(gap) {}

std::vector<std::pair<std::vector<int>, std::vector<int>>> TimeSeriesSplit::split(const MatrixXd& X, const VectorXd& y) const {
    int n_samples = X.rows();
    std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
    
    if (n_splits_ >= n_samples) {
        throw std::invalid_argument("n_splits must be less than number of samples");
    }
    
    int test_size = (test_size_ > 0) ? test_size_ : n_samples / (n_splits_ + 1);
    
    for (int i = 0; i < n_splits_; ++i) {
        int test_start = (i + 1) * n_samples / (n_splits_ + 1);
        int test_end = std::min(test_start + test_size, n_samples);
        
        int train_end = test_start - gap_;
        int train_start = 0;
        if (max_train_size_ > 0) {
            train_start = std::max(0, train_end - max_train_size_);
        }
        
        std::vector<int> train_indices, test_indices;
        for (int j = train_start; j < train_end; ++j) {
            train_indices.push_back(j);
        }
        for (int j = test_start; j < test_end; ++j) {
            test_indices.push_back(j);
        }
        
        splits.emplace_back(train_indices, test_indices);
    }
    
    return splits;
}

Params TimeSeriesSplit::get_params() const {
    return {
        {"n_splits", std::to_string(n_splits_)},
        {"max_train_size", std::to_string(max_train_size_)},
        {"test_size", std::to_string(test_size_)},
        {"gap", std::to_string(gap_)}
    };
}

void TimeSeriesSplit::set_params(const Params& params) {
    if (params.count("n_splits")) n_splits_ = std::stoi(params.at("n_splits"));
    if (params.count("max_train_size")) max_train_size_ = std::stoi(params.at("max_train_size"));
    if (params.count("test_size")) test_size_ = std::stoi(params.at("test_size"));
    if (params.count("gap")) gap_ = std::stoi(params.at("gap"));
}

// HalvingGridSearchCV implementation
HalvingGridSearchCV::HalvingGridSearchCV(Estimator& estimator, const std::vector<Params>& param_grid,
                                         BaseCrossValidator& cv, const std::string& scoring,
                                         int factor, int min_resources, bool aggressive_elimination,
                                         int n_jobs, bool verbose)
    : estimator_(estimator), param_grid_(param_grid), cv_(cv), scoring_(scoring),
      factor_(factor), min_resources_(min_resources), aggressive_elimination_(aggressive_elimination),
      n_jobs_(n_jobs), verbose_(verbose), best_score_(-std::numeric_limits<double>::infinity()), fitted_(false) {
    if (param_grid_.empty()) {
        throw std::invalid_argument("param_grid cannot be empty");
    }
    if (factor_ < 2) {
        throw std::invalid_argument("factor must be at least 2");
    }
    if (min_resources_ < 1) {
        throw std::invalid_argument("min_resources must be at least 1");
    }
}

Estimator& HalvingGridSearchCV::fit(const MatrixXd& X, const VectorXd& y) {
    int n_samples = X.rows();
    int resources = std::min(min_resources_, n_samples);

    std::vector<Params> candidates = param_grid_;
    std::vector<std::pair<Params, double>> scored;

    while (candidates.size() > 1 && resources <= n_samples) {
        scored = evaluate_candidates(candidates, X, y, resources);

        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        int n_keep = std::max(1, static_cast<int>(std::ceil(scored.size() / static_cast<double>(factor_))));
        candidates.clear();
        for (int i = 0; i < n_keep; ++i) {
            candidates.push_back(scored[i].first);
        }

        resources = std::min(n_samples, resources * factor_);
        if (aggressive_elimination_ && candidates.size() <= 1) {
            break;
        }
    }

    scored = evaluate_candidates(candidates, X, y, n_samples);
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    best_params_ = scored.front().first;
    best_score_ = scored.front().second;

    estimator_.set_params(best_params_);
    estimator_.fit(X, y);
    fitted_ = true;
    return estimator_;
}

VectorXd HalvingGridSearchCV::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("HalvingGridSearchCV must be fitted before predict");
    }
    if (auto* predictor = dynamic_cast<const Predictor*>(&estimator_)) {
        return predictor->predict(X);
    }
    throw std::runtime_error("Estimator must be a Predictor to make predictions");
}

Params HalvingGridSearchCV::best_params() const {
    return best_params_;
}

double HalvingGridSearchCV::best_score() const {
    return best_score_;
}

Params HalvingGridSearchCV::get_params() const {
    return {{"scoring", scoring_},
            {"factor", std::to_string(factor_)},
            {"min_resources", std::to_string(min_resources_)},
            {"aggressive_elimination", aggressive_elimination_ ? "1" : "0"},
            {"n_jobs", std::to_string(n_jobs_)},
            {"verbose", verbose_ ? "1" : "0"}};
}

Estimator& HalvingGridSearchCV::set_params(const Params& params) {
    if (params.count("scoring")) scoring_ = params.at("scoring");
    if (params.count("factor")) factor_ = std::stoi(params.at("factor"));
    if (params.count("min_resources")) min_resources_ = std::stoi(params.at("min_resources"));
    if (params.count("aggressive_elimination")) aggressive_elimination_ = (params.at("aggressive_elimination") == "1");
    if (params.count("n_jobs")) n_jobs_ = std::stoi(params.at("n_jobs"));
    if (params.count("verbose")) verbose_ = (params.at("verbose") == "1");
    return *this;
}

int HalvingGridSearchCV::get_n_splits() const {
    return cv_.get_n_splits();
}

std::vector<std::pair<Params, double>> HalvingGridSearchCV::evaluate_candidates(
    const std::vector<Params>& candidates, const MatrixXd& X, const VectorXd& y, int n_resources) {

    int n_samples = X.rows();
    int resources = std::min(n_resources, n_samples);
    int min_cv = cv_.get_n_splits();
    if (resources < min_cv) {
        resources = std::min(n_samples, min_cv);
    }
    if (resources < 1) {
        throw std::runtime_error("Insufficient resources for evaluation");
    }

    MatrixXd X_sub = X.topRows(resources);
    VectorXd y_sub = y.head(resources);

    std::vector<std::pair<Params, double>> results;
    results.reserve(candidates.size());

    for (const auto& params : candidates) {
        estimator_.set_params(params);
        VectorXd scores = cross_val_score(estimator_, X_sub, y_sub, cv_, scoring_);
        results.emplace_back(params, scores.mean());
    }

    return results;
}

// HalvingRandomSearchCV implementation
HalvingRandomSearchCV::HalvingRandomSearchCV(Estimator& estimator, const std::map<std::string, std::vector<std::string>>& param_distributions,
                                             BaseCrossValidator& cv, const std::string& scoring,
                                             int n_candidates, int factor, int min_resources,
                                             bool aggressive_elimination, int random_state,
                                             int n_jobs, bool verbose)
    : estimator_(estimator), param_distributions_(param_distributions), cv_(cv), scoring_(scoring),
      n_candidates_(n_candidates), factor_(factor), min_resources_(min_resources),
      aggressive_elimination_(aggressive_elimination), random_state_(random_state),
      n_jobs_(n_jobs), verbose_(verbose), best_score_(-std::numeric_limits<double>::infinity()), fitted_(false) {
    if (param_distributions_.empty()) {
        throw std::invalid_argument("param_distributions cannot be empty");
    }
    if (n_candidates_ < 1) {
        throw std::invalid_argument("n_candidates must be at least 1");
    }
    if (factor_ < 2) {
        throw std::invalid_argument("factor must be at least 2");
    }
    if (min_resources_ < 1) {
        throw std::invalid_argument("min_resources must be at least 1");
    }
}

Estimator& HalvingRandomSearchCV::fit(const MatrixXd& X, const VectorXd& y) {
    int n_samples = X.rows();
    int resources = std::min(min_resources_, n_samples);

    ParameterSampler sampler(param_distributions_, n_candidates_, random_state_);
    std::vector<Params> candidates = sampler.samples();
    std::vector<std::pair<Params, double>> scored;

    while (candidates.size() > 1 && resources <= n_samples) {
        scored = evaluate_candidates(candidates, X, y, resources);

        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        int n_keep = std::max(1, static_cast<int>(std::ceil(scored.size() / static_cast<double>(factor_))));
        candidates.clear();
        for (int i = 0; i < n_keep; ++i) {
            candidates.push_back(scored[i].first);
        }

        resources = std::min(n_samples, resources * factor_);
        if (aggressive_elimination_ && candidates.size() <= 1) {
            break;
        }
    }

    scored = evaluate_candidates(candidates, X, y, n_samples);
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    best_params_ = scored.front().first;
    best_score_ = scored.front().second;

    estimator_.set_params(best_params_);
    estimator_.fit(X, y);
    fitted_ = true;
    return estimator_;
}

VectorXd HalvingRandomSearchCV::predict(const MatrixXd& X) const {
    if (!fitted_) {
        throw std::runtime_error("HalvingRandomSearchCV must be fitted before predict");
    }
    if (auto* predictor = dynamic_cast<const Predictor*>(&estimator_)) {
        return predictor->predict(X);
    }
    throw std::runtime_error("Estimator must be a Predictor to make predictions");
}

Params HalvingRandomSearchCV::best_params() const {
    return best_params_;
}

double HalvingRandomSearchCV::best_score() const {
    return best_score_;
}

Params HalvingRandomSearchCV::get_params() const {
    return {{"scoring", scoring_},
            {"n_candidates", std::to_string(n_candidates_)},
            {"factor", std::to_string(factor_)},
            {"min_resources", std::to_string(min_resources_)},
            {"aggressive_elimination", aggressive_elimination_ ? "1" : "0"},
            {"random_state", std::to_string(random_state_)},
            {"n_jobs", std::to_string(n_jobs_)},
            {"verbose", verbose_ ? "1" : "0"}};
}

Estimator& HalvingRandomSearchCV::set_params(const Params& params) {
    if (params.count("scoring")) scoring_ = params.at("scoring");
    if (params.count("n_candidates")) n_candidates_ = std::stoi(params.at("n_candidates"));
    if (params.count("factor")) factor_ = std::stoi(params.at("factor"));
    if (params.count("min_resources")) min_resources_ = std::stoi(params.at("min_resources"));
    if (params.count("aggressive_elimination")) aggressive_elimination_ = (params.at("aggressive_elimination") == "1");
    if (params.count("random_state")) random_state_ = std::stoi(params.at("random_state"));
    if (params.count("n_jobs")) n_jobs_ = std::stoi(params.at("n_jobs"));
    if (params.count("verbose")) verbose_ = (params.at("verbose") == "1");
    return *this;
}

int HalvingRandomSearchCV::get_n_splits() const {
    return cv_.get_n_splits();
}

std::vector<std::pair<Params, double>> HalvingRandomSearchCV::evaluate_candidates(
    const std::vector<Params>& candidates, const MatrixXd& X, const VectorXd& y, int n_resources) {

    int n_samples = X.rows();
    int resources = std::min(n_resources, n_samples);
    int min_cv = cv_.get_n_splits();
    if (resources < min_cv) {
        resources = std::min(n_samples, min_cv);
    }
    if (resources < 1) {
        throw std::runtime_error("Insufficient resources for evaluation");
    }

    MatrixXd X_sub = X.topRows(resources);
    VectorXd y_sub = y.head(resources);

    std::vector<std::pair<Params, double>> results;
    results.reserve(candidates.size());

    for (const auto& params : candidates) {
        estimator_.set_params(params);
        VectorXd scores = cross_val_score(estimator_, X_sub, y_sub, cv_, scoring_);
        results.emplace_back(params, scores.mean());
    }

    return results;
}

} // namespace model_selection
} // namespace ingenuityml

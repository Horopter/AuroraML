#pragma once

#include <cstdint>
#include <random>
#include <limits>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>
#include <stdexcept>

#include "base.hpp"

namespace ingenuityml {
namespace random {

/**
 * PCG64 Random Number Generator
 * High-quality, fast random number generator
 */
class PCG64 {
private:
    uint64_t state_;
    uint64_t inc_;
    uint64_t seed_value_;
    bool has_spare_;
    double spare_;

public:
    using result_type = uint32_t;

    PCG64(uint64_t seed_val = 0) : state_(0), inc_(0), seed_value_(seed_val), has_spare_(false), spare_(0.0) {
        seed(seed_val);
    }
    
    void seed(uint64_t seed_val) {
        seed_value_ = seed_val;
        state_ = 0;
        inc_ = (seed_val << 1) | 1;
        (*this)();
        state_ += seed_val;
        (*this)();
    }
    
    static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    result_type operator()() {
        uint64_t oldstate = state_;
        state_ = oldstate * 6364136223846793005ULL + inc_;
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
        return static_cast<result_type>((xorshifted >> rot) | (xorshifted << ((-rot) & 31)));
    }
    
    double uniform(double low = 0.0, double high = 1.0) {
        if (high < low) {
            throw std::invalid_argument("high must be >= low");
        }
        if (high == low) {
            return low;
        }
        double u = static_cast<double>((*this)()) / (static_cast<double>(max()) + 1.0);
        return low + (high - low) * u;
    }
    
    double normal(double mean = 0.0, double std = 1.0) {
        if (std <= 0.0) {
            throw std::invalid_argument("std must be positive");
        }
        if (has_spare_) {
            has_spare_ = false;
            return spare_ * std + mean;
        }
        
        has_spare_ = true;
        double mag, u, v;
        do {
            u = uniform(-1.0, 1.0);
            v = uniform(-1.0, 1.0);
            mag = u * u + v * v;
        } while (mag >= 1.0 || mag == 0.0);
        
        mag = sqrt(-2.0 * log(mag) / mag);
        spare_ = v * mag;
        return u * mag * std + mean;
    }

    int randint(int low, int high) {
        if (high < low) {
            throw std::invalid_argument("high must be >= low");
        }
        if (high == low) {
            return low;
        }
        uint64_t range = static_cast<uint64_t>(static_cast<int64_t>(high) - static_cast<int64_t>(low) + 1);
        uint64_t r = static_cast<uint64_t>((*this)());
        return static_cast<int>(low + (r % range));
    }

    std::pair<uint64_t, uint64_t> get_state() const {
        return {state_, inc_};
    }

    void set_state(const std::pair<uint64_t, uint64_t>& state) {
        state_ = state.first;
        inc_ = state.second;
    }

    Params get_params() const {
        return {{"seed", std::to_string(seed_value_)}};
    }

    PCG64& set_params(const Params& params) {
        auto it = params.find("seed");
        if (it != params.end()) {
            seed(static_cast<uint64_t>(std::stoull(it->second)));
        }
        return *this;
    }
};

} // namespace random
} // namespace ingenuityml

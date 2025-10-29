#pragma once

#include <cstdint>
#include <random>

namespace auroraml {
namespace random {

/**
 * PCG64 Random Number Generator
 * High-quality, fast random number generator
 */
class PCG64 {
private:
    uint64_t state_;
    uint64_t inc_;

public:
    PCG64(uint64_t seed_val = 0) : state_(0), inc_(0) {
        seed(seed_val);
    }
    
    void seed(uint64_t seed_val) {
        state_ = 0;
        inc_ = (seed_val << 1) | 1;
        (*this)();
        state_ += seed_val;
        (*this)();
    }
    
    uint64_t operator()() {
        uint64_t oldstate = state_;
        state_ = oldstate * 6364136223846793005ULL + inc_;
        uint64_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint64_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    
    double uniform() {
        return static_cast<double>((*this)()) / static_cast<double>(UINT64_MAX);
    }
    
    double normal() {
        static bool has_spare = false;
        static double spare;
        
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        
        has_spare = true;
        static const double two_pi = 6.283185307179586476925286766559;
        double mag, u, v;
        do {
            u = uniform() * 2.0 - 1.0;
            v = uniform() * 2.0 - 1.0;
            mag = u * u + v * v;
        } while (mag >= 1.0 || mag == 0.0);
        
        mag = sqrt(-2.0 * log(mag) / mag);
        spare = v * mag;
        return u * mag;
    }
};

} // namespace random
} // namespace cxml

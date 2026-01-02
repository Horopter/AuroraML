#include <gtest/gtest.h>
#include "auroraml/random.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

namespace auroraml {
namespace test {

class RandomTest : public ::testing::Test {
protected:
    void SetUp() override {
        seed = 42;
    }
    
    uint64_t seed;
};

// Positive test cases
TEST_F(RandomTest, PCG64Uniform) {
    random::PCG64 rng(seed);
    
    std::vector<double> uniform_nums;
    for (int i = 0; i < 1000; ++i) {
        uniform_nums.push_back(rng.uniform());
    }
    
    EXPECT_EQ(uniform_nums.size(), 1000);
    for (double val : uniform_nums) {
        EXPECT_GE(val, 0.0);
        EXPECT_LE(val, 1.0);
    }
}

TEST_F(RandomTest, PCG64Normal) {
    random::PCG64 rng(seed);
    
    std::vector<double> normal_nums;
    for (int i = 0; i < 1000; ++i) {
        normal_nums.push_back(rng.normal());
    }
    
    EXPECT_EQ(normal_nums.size(), 1000);
    for (double val : normal_nums) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST_F(RandomTest, PCG64NormalWithParameters) {
    random::PCG64 rng(seed);
    
    double mean = 5.0;
    double std = 2.0;
    
    std::vector<double> normal_nums;
    for (int i = 0; i < 1000; ++i) {
        // Generate standard normal and scale/shift
        double z = rng.normal();
        normal_nums.push_back(mean + std * z);
    }
    
    EXPECT_EQ(normal_nums.size(), 1000);
    
    // Test mean
    double sum = 0.0;
    for (double val : normal_nums) {
        sum += val;
    }
    double computed_mean = sum / normal_nums.size();
    EXPECT_NEAR(computed_mean, mean, 0.5);
}

TEST_F(RandomTest, PCG64Reproducibility) {
    random::PCG64 rng1(seed);
    random::PCG64 rng2(seed);
    
    std::vector<double> values1, values2;
    for (int i = 0; i < 100; ++i) {
        values1.push_back(rng1.uniform());
        values2.push_back(rng2.uniform());
    }
    
    // With same seed, should produce same values
    for (size_t i = 0; i < values1.size(); ++i) {
        EXPECT_NEAR(values1[i], values2[i], 1e-10);
    }
}

TEST_F(RandomTest, PCG64Seed) {
    random::PCG64 rng;
    rng.seed(seed);
    
    double val1 = rng.uniform();
    
    rng.seed(seed);
    double val2 = rng.uniform();
    
    EXPECT_NEAR(val1, val2, 1e-10);
}

// Negative test cases
TEST_F(RandomTest, PCG64UniformDistribution) {
    random::PCG64 rng(seed);
    
    std::vector<double> uniform_nums;
    for (int i = 0; i < 10000; ++i) {
        uniform_nums.push_back(rng.uniform());
    }
    
    // Mean should be close to 0.5
    double sum = 0.0;
    for (double val : uniform_nums) {
        sum += val;
    }
    double mean_val = sum / uniform_nums.size();
    EXPECT_NEAR(mean_val, 0.5, 0.1);
}

TEST_F(RandomTest, PCG64NormalDistribution) {
    random::PCG64 rng(seed);
    
    std::vector<double> normal_nums;
    for (int i = 0; i < 10000; ++i) {
        normal_nums.push_back(rng.normal());
    }
    
    // Mean should be close to 0
    double sum = 0.0;
    for (double val : normal_nums) {
        sum += val;
    }
    double mean_val = sum / normal_nums.size();
    EXPECT_NEAR(mean_val, 0.0, 0.1);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

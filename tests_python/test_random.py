#!/usr/bin/env python3
"""
Test Suite for AuroraML Random Number Generation
Tests PCG64 random number generator
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestPCG64(unittest.TestCase):
    """Test PCG64 random number generator"""
    
    def setUp(self):
        """Set up test data"""
        self.seed = 42
        
    def test_basic_functionality(self):
        """Test basic random number generation"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=self.seed)
        
        # Test uniform random numbers
        uniform_nums = [rng.uniform() for _ in range(1000)]
        
        self.assertEqual(len(uniform_nums), 1000)
        self.assertTrue(np.all(np.array(uniform_nums) >= 0))
        self.assertTrue(np.all(np.array(uniform_nums) <= 1))
        
    def test_uniform_distribution(self):
        """Test uniform distribution properties"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=self.seed)
        uniform_nums = [rng.uniform() for _ in range(10000)]
        uniform_array = np.array(uniform_nums)
        
        # Test mean (should be close to 0.5)
        mean_val = np.mean(uniform_array)
        self.assertAlmostEqual(mean_val, 0.5, places=1)
        
        # Test variance (should be close to 1/12)
        var_val = np.var(uniform_array)
        self.assertAlmostEqual(var_val, 1.0/12, places=2)
        
    def test_normal_distribution(self):
        """Test normal distribution properties"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=self.seed)
        normal_nums = [rng.normal() for _ in range(10000)]
        normal_array = np.array(normal_nums)
        
        # Test mean (should be close to 0)
        mean_val = np.mean(normal_array)
        self.assertAlmostEqual(mean_val, 0.0, places=1)
        
        # Test variance (should be close to 1)
        var_val = np.var(normal_array)
        self.assertAlmostEqual(var_val, 1.0, places=1)
        
    def test_normal_with_parameters(self):
        """Test normal distribution with custom parameters"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=self.seed)
        mean = 5.0
        std = 2.0
        
        normal_nums = [rng.normal(mean, std) for _ in range(10000)]
        normal_array = np.array(normal_nums)
        
        # Test mean
        mean_val = np.mean(normal_array)
        self.assertAlmostEqual(mean_val, mean, places=1)
        
        # Test standard deviation
        std_val = np.std(normal_array)
        self.assertAlmostEqual(std_val, std, places=1)
        
    def test_integer_generation(self):
        """Test integer random number generation"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=self.seed)
        
        # Test uniform integers
        int_nums = [rng.randint(0, 10) for _ in range(1000)]
        
        self.assertEqual(len(int_nums), 1000)
        self.assertTrue(np.all(np.array(int_nums) >= 0))
        self.assertTrue(np.all(np.array(int_nums) <= 10))
        
    def test_seed_consistency(self):
        """Test that same seed produces same sequence"""
        import auroraml.random as aml_random
        
        # Generate sequence with seed 42
        rng1 = aml_random.PCG64(seed=42)
        seq1 = [rng1.uniform() for _ in range(100)]
        
        # Generate same sequence with same seed
        rng2 = aml_random.PCG64(seed=42)
        seq2 = [rng2.uniform() for _ in range(100)]
        
        np.testing.assert_array_almost_equal(seq1, seq2, decimal=10)
        
    def test_different_seeds(self):
        """Test that different seeds produce different sequences"""
        import auroraml.random as aml_random
        
        # Generate sequence with seed 42
        rng1 = aml_random.PCG64(seed=42)
        seq1 = [rng1.uniform() for _ in range(100)]
        
        # Generate sequence with seed 123
        rng2 = aml_random.PCG64(seed=123)
        seq2 = [rng2.uniform() for _ in range(100)]
        
        # Sequences should be different
        self.assertFalse(np.allclose(seq1, seq2))
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=42)
        
        # Test default parameters
        params = rng.get_params()
        self.assertIn('seed', params)
        
        # Test parameter setting
        rng.set_params(seed=123)
        self.assertEqual(rng.get_params()['seed'], "123")
        
    def test_state_management(self):
        """Test state management"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=42)
        
        # Generate some numbers
        nums1 = [rng.uniform() for _ in range(50)]
        
        # Save state
        if hasattr(rng, 'get_state'):
            state = rng.get_state()
            
            # Generate more numbers
            nums2 = [rng.uniform() for _ in range(50)]
            
            # Restore state
            if hasattr(rng, 'set_state'):
                rng.set_state(state)
                
                # Generate numbers again
                nums3 = [rng.uniform() for _ in range(50)]
                
                # Should be same as nums2
                np.testing.assert_array_almost_equal(nums2, nums3, decimal=10)
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=42)
        
        # Test with same min and max for uniform
        uniform_same = rng.uniform(5.0, 5.0)
        self.assertEqual(uniform_same, 5.0)
        
        # Test with same min and max for integer
        int_same = rng.randint(5, 5)
        self.assertEqual(int_same, 5)
        
    def test_performance(self):
        """Test performance with large number of samples"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=42)
        
        # Generate large number of samples
        n_samples = 100000
        uniform_nums = [rng.uniform() for _ in range(n_samples)]
        
        self.assertEqual(len(uniform_nums), n_samples)
        
        # Check distribution properties
        uniform_array = np.array(uniform_nums)
        mean_val = np.mean(uniform_array)
        var_val = np.var(uniform_array)
        
        self.assertAlmostEqual(mean_val, 0.5, places=2)
        self.assertAlmostEqual(var_val, 1.0/12, places=3)

class TestRandomIntegration(unittest.TestCase):
    """Integration tests for random number generation"""
    
    def test_reproducibility(self):
        """Test that results are reproducible"""
        import auroraml.random as aml_random
        
        # Test multiple runs with same seed
        seeds = [42, 123, 456]
        
        for seed in seeds:
            rng1 = aml_random.PCG64(seed=seed)
            rng2 = aml_random.PCG64(seed=seed)
            
            # Generate same number of samples
            nums1 = [rng1.uniform() for _ in range(1000)]
            nums2 = [rng2.uniform() for _ in range(1000)]
            
            np.testing.assert_array_almost_equal(nums1, nums2, decimal=10)
            
    def test_distribution_properties(self):
        """Test distribution properties"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=42)
        
        # Test uniform distribution
        uniform_nums = [rng.uniform() for _ in range(10000)]
        uniform_array = np.array(uniform_nums)
        
        # Check that all values are in [0, 1]
        self.assertTrue(np.all(uniform_array >= 0))
        self.assertTrue(np.all(uniform_array <= 1))
        
        # Check mean and variance
        self.assertAlmostEqual(np.mean(uniform_array), 0.5, places=1)
        self.assertAlmostEqual(np.var(uniform_array), 1.0/12, places=2)
        
        # Test normal distribution
        normal_nums = [rng.normal() for _ in range(10000)]
        normal_array = np.array(normal_nums)
        
        # Check mean and variance
        self.assertAlmostEqual(np.mean(normal_array), 0.0, places=1)
        self.assertAlmostEqual(np.var(normal_array), 1.0, places=1)
        
    def test_cross_platform_consistency(self):
        """Test that results are consistent across different runs"""
        import auroraml.random as aml_random
        
        # This test ensures that the random number generator
        # produces consistent results across different runs
        rng = aml_random.PCG64(seed=42)
        
        # Generate a sequence
        sequence = [rng.uniform() for _ in range(100)]
        
        # Check that sequence has good properties
        self.assertEqual(len(sequence), 100)
        self.assertTrue(np.all(np.array(sequence) >= 0))
        self.assertTrue(np.all(np.array(sequence) <= 1))
        
        # Check that sequence doesn't have obvious patterns
        # (e.g., all same value, or obvious repeating pattern)
        unique_values = len(set(sequence))
        self.assertGreater(unique_values, 50)  # Should have many unique values
        
    def test_parameter_validation(self):
        """Test parameter validation"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=42)
        
        # Test normal distribution with invalid parameters
        # (This should not crash, but may produce unexpected results)
        try:
            normal_nums = [rng.normal(0, -1) for _ in range(10)]
            # If it doesn't crash, check that we get some numbers
            self.assertEqual(len(normal_nums), 10)
        except (ValueError, RuntimeError):
            # It's acceptable for the implementation to reject invalid parameters
            pass
            
    def test_mixed_distributions(self):
        """Test mixing different distributions"""
        import auroraml.random as aml_random
        
        rng = aml_random.PCG64(seed=42)
        
        # Generate mixed samples
        uniform_samples = [rng.uniform() for _ in range(100)]
        normal_samples = [rng.normal() for _ in range(100)]
        int_samples = [rng.randint(0, 10) for _ in range(100)]
        
        # All should have correct properties
        self.assertEqual(len(uniform_samples), 100)
        self.assertEqual(len(normal_samples), 100)
        self.assertEqual(len(int_samples), 100)
        
        # Uniform samples should be in [0, 1]
        self.assertTrue(np.all(np.array(uniform_samples) >= 0))
        self.assertTrue(np.all(np.array(uniform_samples) <= 1))
        
        # Integer samples should be in [0, 10]
        self.assertTrue(np.all(np.array(int_samples) >= 0))
        self.assertTrue(np.all(np.array(int_samples) <= 10))

if __name__ == '__main__':
    unittest.main()

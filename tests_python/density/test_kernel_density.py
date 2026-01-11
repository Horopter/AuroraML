#!/usr/bin/env python3
"""
Kernel density tests.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))


class TestKernelDensity(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(30, 2).astype(np.float64)

    def test_kernel_density_score_samples(self):
        import ingenuityml.density as ing_density

        model = ing_density.KernelDensity(bandwidth=0.5)
        model.fit(self.X, np.zeros(self.X.shape[0]))
        scores = model.score_samples(self.X)
        self.assertEqual(scores.shape[0], self.X.shape[0])
        self.assertFalse(np.any(np.isnan(scores)))


if __name__ == '__main__':
    unittest.main()

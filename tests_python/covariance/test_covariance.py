"""Tests for covariance estimators"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestCovariance(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(120, 3).astype(np.float64)
        self.X[0] += 5.0
        self.X[1] -= 4.0

    def test_empirical_covariance(self):
        model = ingenuityml.covariance.EmpiricalCovariance()
        model.fit(self.X, None)
        cov = model.covariance()
        self.assertEqual(cov.shape, (3, 3))
        dists = model.mahalanobis(self.X)
        self.assertEqual(dists.shape[0], self.X.shape[0])

    def test_shrunk_covariance(self):
        model = ingenuityml.covariance.ShrunkCovariance(shrinkage=0.2)
        model.fit(self.X, None)
        self.assertEqual(model.covariance().shape, (3, 3))

    def test_ledoit_wolf(self):
        model = ingenuityml.covariance.LedoitWolf()
        model.fit(self.X, None)
        self.assertGreaterEqual(model.shrinkage(), 0.0)
        self.assertLessEqual(model.shrinkage(), 1.0)

    def test_oas(self):
        model = ingenuityml.covariance.OAS()
        model.fit(self.X, None)
        self.assertGreaterEqual(model.shrinkage(), 0.0)
        self.assertLessEqual(model.shrinkage(), 1.0)

    def test_mincovdet(self):
        model = ingenuityml.covariance.MinCovDet(support_fraction=0.75)
        model.fit(self.X, None)
        support = model.support()
        self.assertEqual(support.shape[0], self.X.shape[0])

    def test_elliptic_envelope(self):
        model = ingenuityml.covariance.EllipticEnvelope(contamination=0.1, support_fraction=0.75, random_state=42)
        model.fit(self.X, None)
        labels = model.predict(self.X)
        self.assertEqual(labels.shape[0], self.X.shape[0])
        scores = model.decision_function(self.X)
        self.assertEqual(scores.shape[0], self.X.shape[0])

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)

    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)

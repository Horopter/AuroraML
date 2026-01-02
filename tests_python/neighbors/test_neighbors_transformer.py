#!/usr/bin/env python3
"""
Tests for KNeighborsTransformer and RadiusNeighborsTransformer
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))


class TestNeighborsTransformer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(60, 3).astype(np.float64)
        self.X_test = np.random.randn(10, 3).astype(np.float64)
        self.y_dummy = np.zeros(self.X.shape[0]).astype(np.float64)

    def test_kneighbors_transformer_distance(self):
        import auroraml.neighbors as aml_neighbors

        transformer = aml_neighbors.KNeighborsTransformer(n_neighbors=4, mode="distance")
        transformer.fit(self.X, self.y_dummy)
        graph = transformer.transform(self.X_test)

        self.assertEqual(graph.shape, (self.X_test.shape[0], self.X.shape[0]))
        self.assertTrue(np.all(graph >= 0))

    def test_kneighbors_transformer_connectivity(self):
        import auroraml.neighbors as aml_neighbors

        transformer = aml_neighbors.KNeighborsTransformer(n_neighbors=3, mode="connectivity")
        transformer.fit(self.X, self.y_dummy)
        graph = transformer.transform(self.X_test)

        self.assertEqual(graph.shape, (self.X_test.shape[0], self.X.shape[0]))
        self.assertTrue(np.all(np.isin(graph, [0.0, 1.0])))

    def test_radius_neighbors_transformer_connectivity(self):
        import auroraml.neighbors as aml_neighbors

        transformer = aml_neighbors.RadiusNeighborsTransformer(radius=0.8, mode="connectivity")
        transformer.fit(self.X, self.y_dummy)
        graph = transformer.transform(self.X_test)

        self.assertEqual(graph.shape, (self.X_test.shape[0], self.X.shape[0]))
        self.assertTrue(np.all(np.isin(graph, [0.0, 1.0])))

    def test_not_fitted(self):
        import auroraml.neighbors as aml_neighbors

        knn = aml_neighbors.KNeighborsTransformer(n_neighbors=3, mode="distance")
        with self.assertRaises((RuntimeError, ValueError)):
            knn.transform(self.X_test)

        radius = aml_neighbors.RadiusNeighborsTransformer(radius=1.0, mode="distance")
        with self.assertRaises((RuntimeError, ValueError)):
            radius.transform(self.X_test)

    def test_wrong_feature_count(self):
        import auroraml.neighbors as aml_neighbors

        knn = aml_neighbors.KNeighborsTransformer(n_neighbors=3, mode="distance")
        knn.fit(self.X, self.y_dummy)
        X_wrong = np.random.randn(10, 5).astype(np.float64)
        with self.assertRaises((RuntimeError, ValueError)):
            knn.transform(X_wrong)


if __name__ == '__main__':
    unittest.main()

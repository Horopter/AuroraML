"""Tests for Inspection module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestInspection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (np.random.randn(100) > 0).astype(np.float64)
        self.y_class = (np.random.randn(100) > 0).astype(np.int32)

    def test_permutation_importance_classification(self):
        """Test PermutationImportance for classification"""
        clf = ingenuityml.tree.DecisionTreeClassifier(max_depth=3)
        clf.fit(self.X, self.y_class)
        
        perm_imp = ingenuityml.inspection.PermutationImportance(
            estimator=clf, scoring="accuracy", n_repeats=3
        )
        perm_imp.fit(self.X, self.y_class)
        
        importances = perm_imp.feature_importances()
        self.assertEqual(len(importances), self.X.shape[1])
        self.assertTrue(all(imp >= 0 for imp in importances))

    def test_permutation_importance_regression(self):
        """Test PermutationImportance for regression"""
        reg = ingenuityml.linear_model.LinearRegression()
        reg.fit(self.X, self.y)
        
        perm_imp = ingenuityml.inspection.PermutationImportance(
            estimator=reg, scoring="r2", n_repeats=3
        )
        perm_imp.fit(self.X, self.y)
        
        importances = perm_imp.feature_importances()
        self.assertEqual(len(importances), self.X.shape[1])

    def test_partial_dependence(self):
        """Test PartialDependence"""
        reg = ingenuityml.linear_model.LinearRegression()
        reg.fit(self.X, self.y)
        
        pd = ingenuityml.inspection.PartialDependence(
            estimator=reg, features=[0, 1]
        )
        pd.compute(self.X)
        
        grid = pd.grid()
        self.assertEqual(grid.shape[1], 2)  # Two features
        
        pd_values = pd.partial_dependence()
        self.assertGreater(len(pd_values), 0)

if __name__ == '__main__':
    # Shuffle tests within this file
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    test_methods = [test for test in suite]
    random.seed(42)  # Reproducible shuffle
    random.shuffle(test_methods)
    
    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)


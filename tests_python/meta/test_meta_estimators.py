"""Tests for meta estimators"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestMetaEstimators(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(80, 4).astype(np.float64)

        y_class = np.zeros(self.X.shape[0], dtype=np.float64)
        for i in range(self.X.shape[0]):
            cls = 0
            if self.X[i, 0] > 0:
                cls += 1
            if self.X[i, 1] > 0:
                cls += 1
            y_class[i] = cls
        self.y_class = y_class

        self.Y_multi = np.zeros((self.X.shape[0], 2), dtype=np.float64)
        self.Y_multi[:, 0] = (self.X[:, 0] > 0).astype(np.float64)
        self.Y_multi[:, 1] = (self.X[:, 1] > 0).astype(np.float64)

        self.Y_reg = np.zeros((self.X.shape[0], 2), dtype=np.float64)
        self.Y_reg[:, 0] = self.X.sum(axis=1)
        self.Y_reg[:, 1] = self.X.mean(axis=1)

    def test_one_vs_rest(self):
        model = ingenuityml.meta.OneVsRestClassifier(lambda: ingenuityml.naive_bayes.GaussianNB())
        model.fit(self.X, self.y_class)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape[0], self.X.shape[0])

    def test_one_vs_one(self):
        model = ingenuityml.meta.OneVsOneClassifier(lambda: ingenuityml.naive_bayes.GaussianNB())
        model.fit(self.X, self.y_class)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape[0], self.X.shape[0])

    def test_output_code(self):
        model = ingenuityml.meta.OutputCodeClassifier(lambda: ingenuityml.naive_bayes.GaussianNB(), code_size=5, random_state=42)
        model.fit(self.X, self.y_class)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape[0], self.X.shape[0])

    def test_multioutput_classifier(self):
        model = ingenuityml.meta.MultiOutputClassifier(lambda: ingenuityml.naive_bayes.GaussianNB())
        model.fit(self.X, self.Y_multi)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (self.X.shape[0], 2))

    def test_classifier_chain(self):
        model = ingenuityml.meta.ClassifierChain(lambda: ingenuityml.naive_bayes.GaussianNB())
        model.fit(self.X, self.Y_multi)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (self.X.shape[0], 2))

    def test_multioutput_regressor(self):
        model = ingenuityml.meta.MultiOutputRegressor(lambda: ingenuityml.linear_model.LinearRegression())
        model.fit(self.X, self.Y_reg)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (self.X.shape[0], 2))

    def test_regressor_chain(self):
        model = ingenuityml.meta.RegressorChain(lambda: ingenuityml.linear_model.LinearRegression())
        model.fit(self.X, self.Y_reg)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (self.X.shape[0], 2))

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)

    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)

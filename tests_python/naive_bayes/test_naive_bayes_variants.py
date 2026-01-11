"""Tests for Naive Bayes variants"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestNaiveBayesVariants(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (np.random.randn(100) > 0).astype(np.float64)
        self.y_class = (np.random.randn(100) > 0).astype(np.int32)
        # For multinomial, use positive counts
        self.X_multinomial = np.abs(np.random.randint(0, 10, (100, 4))).astype(np.float64)
        # For categorical, use small integer categories
        self.X_categorical = np.random.randint(0, 4, (100, 4)).astype(np.float64)

    def test_multinomial_nb(self):
        """Test MultinomialNB"""
        nb = ingenuityml.naive_bayes.MultinomialNB(alpha=1.0)
        nb.fit(self.X_multinomial, self.y_class)
        
        predictions = nb.predict(self.X_multinomial)
        self.assertEqual(predictions.shape[0], self.X_multinomial.shape[0])
        
        proba = nb.predict_proba(self.X_multinomial)
        self.assertEqual(proba.shape[0], self.X_multinomial.shape[0])
        
        classes = nb.classes()
        self.assertGreater(len(classes), 0)

    def test_bernoulli_nb(self):
        """Test BernoulliNB"""
        # Binarize data
        X_binary = (self.X > 0).astype(np.float64)
        
        nb = ingenuityml.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0)
        nb.fit(X_binary, self.y_class)
        
        predictions = nb.predict(X_binary)
        self.assertEqual(predictions.shape[0], X_binary.shape[0])
        
        proba = nb.predict_proba(X_binary)
        self.assertEqual(proba.shape[0], X_binary.shape[0])

    def test_complement_nb(self):
        """Test ComplementNB"""
        nb = ingenuityml.naive_bayes.ComplementNB(alpha=1.0)
        nb.fit(self.X_multinomial, self.y_class)
        
        predictions = nb.predict(self.X_multinomial)
        self.assertEqual(predictions.shape[0], self.X_multinomial.shape[0])
        
        proba = nb.predict_proba(self.X_multinomial)
        self.assertEqual(proba.shape[0], self.X_multinomial.shape[0])

    def test_categorical_nb(self):
        """Test CategoricalNB"""
        nb = ingenuityml.naive_bayes.CategoricalNB(alpha=1.0)
        nb.fit(self.X_categorical, self.y_class)

        predictions = nb.predict(self.X_categorical)
        self.assertEqual(predictions.shape[0], self.X_categorical.shape[0])

        proba = nb.predict_proba(self.X_categorical)
        self.assertEqual(proba.shape[0], self.X_categorical.shape[0])
        self.assertEqual(proba.shape[1], len(nb.classes()))

        n_categories = nb.n_categories()
        self.assertEqual(len(n_categories), self.X_categorical.shape[1])

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

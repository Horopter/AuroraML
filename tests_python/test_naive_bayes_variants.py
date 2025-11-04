"""Tests for Naive Bayes variants"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml

class TestNaiveBayesVariants(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (np.random.randn(100) > 0).astype(np.float64)
        self.y_class = (np.random.randn(100) > 0).astype(np.int32)
        # For multinomial, use positive counts
        self.X_multinomial = np.abs(np.random.randint(0, 10, (100, 4))).astype(np.float64)

    def test_multinomial_nb(self):
        """Test MultinomialNB"""
        nb = auroraml.naive_bayes.MultinomialNB(alpha=1.0)
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
        
        nb = auroraml.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0)
        nb.fit(X_binary, self.y_class)
        
        predictions = nb.predict(X_binary)
        self.assertEqual(predictions.shape[0], X_binary.shape[0])
        
        proba = nb.predict_proba(X_binary)
        self.assertEqual(proba.shape[0], X_binary.shape[0])

    def test_complement_nb(self):
        """Test ComplementNB"""
        nb = auroraml.naive_bayes.ComplementNB(alpha=1.0)
        nb.fit(self.X_multinomial, self.y_class)
        
        predictions = nb.predict(self.X_multinomial)
        self.assertEqual(predictions.shape[0], self.X_multinomial.shape[0])
        
        proba = nb.predict_proba(self.X_multinomial)
        self.assertEqual(proba.shape[0], self.X_multinomial.shape[0])

if __name__ == '__main__':
    unittest.main()


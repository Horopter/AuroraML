"""Tests for Ensemble Wrappers (Bagging, Voting, Stacking)"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml

class TestEnsembleWrappers(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randn(100).astype(np.float64)
        self.y_class = (np.random.randn(100) > 0).astype(np.int32)

    def test_bagging_classifier(self):
        """Test BaggingClassifier"""
        base_clf = auroraml.tree.DecisionTreeClassifier(max_depth=3)
        bagging = auroraml.ensemble.BaggingClassifier(
            base_estimator=base_clf, n_estimators=5, random_state=42
        )
        bagging.fit(self.X, self.y_class)
        
        predictions = bagging.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = bagging.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])
        
        classes = bagging.classes()
        self.assertGreater(len(classes), 0)

    def test_bagging_regressor(self):
        """Test BaggingRegressor"""
        base_reg = auroraml.tree.DecisionTreeRegressor(max_depth=3)
        bagging = auroraml.ensemble.BaggingRegressor(
            base_estimator=base_reg, n_estimators=5, random_state=42
        )
        bagging.fit(self.X, self.y)
        
        predictions = bagging.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

    def test_voting_classifier(self):
        """Test VotingClassifier"""
        clf1 = auroraml.tree.DecisionTreeClassifier(max_depth=2)
        clf2 = auroraml.neighbors.KNeighborsClassifier(n_neighbors=5)
        clf3 = auroraml.linear_model.LogisticRegression()
        
        estimators = [
            ("dt", clf1),
            ("knn", clf2),
            ("lr", clf3)
        ]
        voting = auroraml.ensemble.VotingClassifier(estimators, voting="hard")
        voting.fit(self.X, self.y_class)
        
        predictions = voting.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = voting.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])
        
        classes = voting.classes()
        self.assertGreater(len(classes), 0)

    def test_voting_regressor(self):
        """Test VotingRegressor"""
        reg1 = auroraml.tree.DecisionTreeRegressor(max_depth=2)
        reg2 = auroraml.linear_model.LinearRegression()
        reg3 = auroraml.neighbors.KNeighborsRegressor(n_neighbors=5)
        
        estimators = [
            ("dt", reg1),
            ("lr", reg2),
            ("knn", reg3)
        ]
        voting = auroraml.ensemble.VotingRegressor(estimators)
        voting.fit(self.X, self.y)
        
        predictions = voting.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

    def test_stacking_classifier(self):
        """Test StackingClassifier"""
        base_clf1 = auroraml.tree.DecisionTreeClassifier(max_depth=2)
        base_clf2 = auroraml.neighbors.KNeighborsClassifier(n_neighbors=5)
        
        base_estimators = [
            ("dt", base_clf1),
            ("knn", base_clf2)
        ]
        meta_clf = auroraml.linear_model.LogisticRegression()
        
        stacking = auroraml.ensemble.StackingClassifier(
            base_estimators=base_estimators, meta_classifier=meta_clf
        )
        stacking.fit(self.X, self.y_class)
        
        predictions = stacking.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = stacking.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])

    def test_stacking_regressor(self):
        """Test StackingRegressor"""
        base_reg1 = auroraml.tree.DecisionTreeRegressor(max_depth=2)
        base_reg2 = auroraml.linear_model.LinearRegression()
        
        base_estimators = [
            ("dt", base_reg1),
            ("lr", base_reg2)
        ]
        meta_reg = auroraml.linear_model.LinearRegression()
        
        stacking = auroraml.ensemble.StackingRegressor(
            base_estimators=base_estimators, meta_regressor=meta_reg
        )
        stacking.fit(self.X, self.y)
        
        predictions = stacking.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

if __name__ == '__main__':
    unittest.main()


#!/usr/bin/env python3
import random
"""
Test Suite for AuroraML Model Selection
Tests train_test_split, KFold, StratifiedKFold, GroupKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestTrainTestSplit(unittest.TestCase):
    """Test train_test_split function"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)
        
    def test_basic_functionality(self):
        """Test basic train_test_split functionality"""
        import auroraml.model_selection as aml_ms
        
        X_train, X_test, y_train, y_test = aml_ms.train_test_split(
            self.X, self.y, test_size=0.25, random_state=42
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 75)
        self.assertEqual(X_test.shape[0], 25)
        self.assertEqual(y_train.shape[0], 75)
        self.assertEqual(y_test.shape[0], 25)
        
        # Check that all samples are accounted for
        self.assertEqual(X_train.shape[0] + X_test.shape[0], len(self.X))
        self.assertEqual(y_train.shape[0] + y_test.shape[0], len(self.y))
        
    def test_different_test_sizes(self):
        """Test with different test sizes"""
        import auroraml.model_selection as aml_ms
        
        test_sizes = [0.1, 0.2, 0.3, 0.5]
        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = aml_ms.train_test_split(
                self.X, self.y, test_size=test_size, random_state=42
            )
            
            expected_test_size = int(len(self.X) * test_size)
            self.assertEqual(X_test.shape[0], expected_test_size)
            
    def test_random_state(self):
        """Test that random_state produces consistent results"""
        import auroraml.model_selection as aml_ms
        
        # Same random state should produce same split
        X_train1, X_test1, y_train1, y_test1 = aml_ms.train_test_split(
            self.X, self.y, test_size=0.25, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = aml_ms.train_test_split(
            self.X, self.y, test_size=0.25, random_state=42
        )
        
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.model_selection as aml_ms
        
        # Test with single sample
        X_single = self.X[:1]
        y_single = self.y[:1]
        
        with self.assertRaises(ValueError):
            aml_ms.train_test_split(X_single, y_single, test_size=0.5)

class TestKFold(unittest.TestCase):
    """Test KFold cross-validation"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)
        
    def test_basic_functionality(self):
        """Test basic KFold functionality"""
        import auroraml.model_selection as aml_ms
        
        kfold = aml_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kfold.split(self.X))
        
        self.assertEqual(len(splits), 5)
        
        for train_idx, val_idx in splits:
            self.assertEqual(len(train_idx), 80)  # 80% for training
            self.assertEqual(len(val_idx), 20)    # 20% for validation
            
            # Check that indices are unique
            all_indices = np.concatenate([train_idx, val_idx])
            self.assertEqual(len(np.unique(all_indices)), len(all_indices))
            
    def test_different_n_splits(self):
        """Test with different n_splits values"""
        import auroraml.model_selection as aml_ms
        
        n_splits_values = [3, 5, 10]
        for n_splits in n_splits_values:
            kfold = aml_ms.KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kfold.split(self.X))
            
            self.assertEqual(len(splits), n_splits)
            
    def test_no_shuffle(self):
        """Test KFold without shuffle"""
        import auroraml.model_selection as aml_ms
        
        kfold = aml_ms.KFold(n_splits=5, shuffle=False)
        splits = list(kfold.split(self.X))
        
        self.assertEqual(len(splits), 5)
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.model_selection as aml_ms
        
        kfold = aml_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Test default parameters
        params = kfold.get_params()
        self.assertIn('n_splits', params)
        self.assertIn('shuffle', params)
        self.assertIn('random_state', params)
        
        # Test parameter setting
        kfold.set_params({"n_splits": "10"})
        self.assertEqual(kfold.get_params()['n_splits'], "10")

class TestStratifiedKFold(unittest.TestCase):
    """Test StratifiedKFold cross-validation"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)
        
    def test_basic_functionality(self):
        """Test basic StratifiedKFold functionality"""
        import auroraml.model_selection as aml_ms
        
        skfold = aml_ms.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skfold.split(self.X, self.y))
        
        self.assertEqual(len(splits), 5)
        
        for train_idx, val_idx in splits:
            self.assertEqual(len(train_idx), 80)
            self.assertEqual(len(val_idx), 20)
            
    def test_stratification(self):
        """Test that stratification maintains class distribution"""
        import auroraml.model_selection as aml_ms
        
        # Create imbalanced dataset
        y_imbalanced = np.concatenate([np.zeros(80), np.ones(20)])
        X_imbalanced = np.random.randn(100, 4).astype(np.float64)
        
        skfold = aml_ms.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skfold.split(X_imbalanced, y_imbalanced))
        
        for train_idx, val_idx in splits:
            # Check that both splits maintain similar class distribution
            train_class_ratio = np.mean(y_imbalanced[train_idx])
            val_class_ratio = np.mean(y_imbalanced[val_idx])
            
            # Ratios should be similar (within 10%)
            self.assertLess(abs(train_class_ratio - val_class_ratio), 0.1)

class TestGroupKFold(unittest.TestCase):
    """Test GroupKFold cross-validation"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)
        self.groups = np.random.randint(0, 10, 100)  # 10 groups
        
    def test_basic_functionality(self):
        """Test basic GroupKFold functionality"""
        import auroraml.model_selection as aml_ms
        
        gkfold = aml_ms.GroupKFold(n_splits=5)
        splits = list(gkfold.split(self.X, self.y, groups=self.groups))
        
        self.assertEqual(len(splits), 5)
        
        for train_idx, val_idx in splits:
            # Check that no group appears in both train and validation
            train_groups = set(self.groups[train_idx])
            val_groups = set(self.groups[val_idx])
            
            self.assertEqual(len(train_groups.intersection(val_groups)), 0)

class TestCrossValScore(unittest.TestCase):
    """Test cross_val_score function"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)
        
    def test_basic_functionality(self):
        """Test basic cross_val_score functionality"""
        import auroraml.model_selection as aml_ms
        import auroraml.linear_model as aml_lm
        import auroraml.metrics as aml_metrics
        
        model = aml_lm.LinearRegression()
        kfold = aml_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = aml_ms.cross_val_score(model, self.X, self.y, cv=kfold, scoring='accuracy')
        
        self.assertEqual(len(scores), 5)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
        
    def test_different_scoring(self):
        """Test with different scoring metrics"""
        import auroraml.model_selection as aml_ms
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        kfold = aml_ms.KFold(n_splits=3, shuffle=True, random_state=42)
        
        # Test with different scoring methods
        scoring_methods = ['accuracy', 'precision', 'recall', 'f1']
        
        for scoring in scoring_methods:
            scores = aml_ms.cross_val_score(model, self.X, self.y, cv=kfold, scoring=scoring)
            self.assertEqual(len(scores), 3)
            self.assertTrue(np.all(scores >= 0))
            self.assertTrue(np.all(scores <= 1))

class TestGridSearchCV(unittest.TestCase):
    """Test GridSearchCV"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)
        
    def test_basic_functionality(self):
        """Test basic GridSearchCV functionality"""
        import auroraml.model_selection as aml_ms
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.Ridge()
        param_grid = [{'alpha': '0.1'}, {'alpha': '1.0'}, {'alpha': '10.0'}]
        kfold = aml_ms.KFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = aml_ms.GridSearchCV(
            model, param_grid, cv=kfold, scoring='accuracy'
        )
        
        grid_search.fit(self.X, self.y)
        
        # Check that best parameters are found
        best_params = grid_search.best_params()
        self.assertIn('alpha', best_params)
        self.assertIn(best_params['alpha'], ['0.1', '1.0', '10.0'])
        
        # Check that best score is reasonable
        self.assertGreaterEqual(grid_search.best_score(), 0.0)
        self.assertLessEqual(grid_search.best_score(), 1.0)
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.model_selection as aml_ms
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.Ridge()
        param_grid = [{'alpha': '0.1'}, {'alpha': '1.0'}]
        kfold = aml_ms.KFold(n_splits=3)
        
        grid_search = aml_ms.GridSearchCV(model, param_grid, cv=kfold)
        
        # Test default parameters
        params = grid_search.get_params()
        self.assertIn('scoring', params)
        
        # Test parameter setting
        grid_search.set_params({"scoring": "precision"})
        self.assertEqual(grid_search.get_params()['scoring'], "precision")

class TestRandomizedSearchCV(unittest.TestCase):
    """Test RandomizedSearchCV"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)
        
    def test_basic_functionality(self):
        """Test basic RandomizedSearchCV functionality"""
        import auroraml.model_selection as aml_ms
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.Ridge()
        param_distributions = [{'alpha': '0.1'}, {'alpha': '1.0'}, {'alpha': '10.0'}]
        kfold = aml_ms.KFold(n_splits=3, shuffle=True, random_state=42)
        
        random_search = aml_ms.RandomizedSearchCV(
            model, param_distributions, n_iter=3, cv=kfold, scoring='accuracy'
        )
        
        random_search.fit(self.X, self.y)
        
        # Check that best parameters are found
        best_params = random_search.best_params()
        self.assertIn('alpha', best_params)
        self.assertIn(best_params['alpha'], ['0.1', '1.0', '10.0'])
        
        # Check that best score is reasonable
        self.assertGreaterEqual(random_search.best_score(), 0.0)
        self.assertLessEqual(random_search.best_score(), 1.0)

class TestParameterGrid(unittest.TestCase):
    """Test ParameterGrid"""

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        param_grid = {'alpha': [0.1, 1.0], 'fit_intercept': [0, 1]}
        grid = aml_ms.ParameterGrid(param_grid)

        self.assertEqual(grid.size(), 4)
        combos = grid.grid()
        self.assertEqual(len(combos), 4)
        for combo in combos:
            self.assertIn('alpha', combo)
            self.assertIn('fit_intercept', combo)

class TestParameterSampler(unittest.TestCase):
    """Test ParameterSampler"""

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        param_dist = {'alpha': [0.1, 1.0, 10.0], 'fit_intercept': [0, 1]}
        sampler = aml_ms.ParameterSampler(param_dist, n_iter=5, random_state=42)

        self.assertEqual(sampler.size(), 5)
        samples = sampler.samples()
        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertIn('alpha', sample)
            self.assertIn('fit_intercept', sample)

class TestRepeatedKFold(unittest.TestCase):
    """Test RepeatedKFold cross-validation"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(60, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 60).astype(np.int32)

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        rkfold = aml_ms.RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
        splits = list(rkfold.split(self.X, self.y))

        self.assertEqual(len(splits), 6)

class TestRepeatedStratifiedKFold(unittest.TestCase):
    """Test RepeatedStratifiedKFold cross-validation"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(60, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 60).astype(np.int32)

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        rskfold = aml_ms.RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)
        splits = list(rskfold.split(self.X, self.y))

        self.assertEqual(len(splits), 6)

class TestShuffleSplit(unittest.TestCase):
    """Test ShuffleSplit cross-validation"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        ss = aml_ms.ShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
        splits = list(ss.split(self.X))

        self.assertEqual(len(splits), 4)
        for train_idx, test_idx in splits:
            self.assertEqual(len(test_idx), 20)
            self.assertEqual(len(train_idx), 80)

class TestStratifiedShuffleSplit(unittest.TestCase):
    """Test StratifiedShuffleSplit cross-validation"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        sss = aml_ms.StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
        splits = list(sss.split(self.X, self.y))

        self.assertEqual(len(splits), 4)
        for train_idx, test_idx in splits:
            self.assertEqual(len(test_idx), 20)
            self.assertEqual(len(train_idx), 80)

class TestGroupShuffleSplit(unittest.TestCase):
    """Test GroupShuffleSplit cross-validation"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)
        self.groups = np.random.randint(0, 10, 100)

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        gss = aml_ms.GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        splits = list(gss.split(self.X, self.y, groups=self.groups))

        self.assertEqual(len(splits), 3)
        for train_idx, test_idx in splits:
            train_groups = set(self.groups[train_idx])
            test_groups = set(self.groups[test_idx])
            self.assertEqual(len(train_groups.intersection(test_groups)), 0)

class TestPredefinedSplit(unittest.TestCase):
    """Test PredefinedSplit cross-validation"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        test_fold = [-1] * 100
        for i in range(40):
            test_fold[i] = 0 if i < 20 else 1

        ps = aml_ms.PredefinedSplit(test_fold)
        self.assertEqual(ps.get_n_splits(), 2)
        splits = list(ps.split(self.X))

        self.assertEqual(len(splits), 2)
        for train_idx, test_idx in splits:
            self.assertEqual(len(test_idx), 20)
            self.assertEqual(len(train_idx), 80)

class TestLeaveOneOut(unittest.TestCase):
    """Test LeaveOneOut cross-validation"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(6, 3).astype(np.float64)

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        loo = aml_ms.LeaveOneOut()
        splits = list(loo.split(self.X))

        self.assertEqual(len(splits), 6)
        for train_idx, test_idx in splits:
            self.assertEqual(len(test_idx), 1)
            self.assertEqual(len(train_idx), 5)

class TestLeavePOut(unittest.TestCase):
    """Test LeavePOut cross-validation"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(6, 3).astype(np.float64)

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms

        lpo = aml_ms.LeavePOut(p=2)
        splits = list(lpo.split(self.X))

        self.assertEqual(len(splits), 15)
        for train_idx, test_idx in splits:
            self.assertEqual(len(test_idx), 2)
            self.assertEqual(len(train_idx), 4)

class TestHalvingRandomSearchCV(unittest.TestCase):
    """Test HalvingRandomSearchCV"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(40, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)

    def test_basic_functionality(self):
        import auroraml.model_selection as aml_ms
        import auroraml.linear_model as aml_lm

        model = aml_lm.LogisticRegression()
        param_distributions = {'C': [0.1, 1.0]}
        kfold = aml_ms.KFold(n_splits=3, shuffle=True, random_state=42)

        halving = aml_ms.HalvingRandomSearchCV(
            model, param_distributions, cv=kfold, scoring='accuracy',
            n_candidates=2, factor=2, min_resources=20, random_state=42
        )
        halving.fit(self.X, self.y)

        self.assertTrue(halving.is_fitted())
        best_params = halving.best_params()
        self.assertIn('C', best_params)
        self.assertGreaterEqual(halving.best_score(), 0.0)
        self.assertLessEqual(halving.best_score(), 1.0)

class TestModelSelectionIntegration(unittest.TestCase):
    """Integration tests for model selection"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 2, 100).astype(np.int32)
        
    def test_cross_validation_workflow(self):
        """Test complete cross-validation workflow"""
        import auroraml.model_selection as aml_ms
        import auroraml.linear_model as aml_lm
        import auroraml.metrics as aml_metrics
        
        # Split data
        X_train, X_test, y_train, y_test = aml_ms.train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Cross-validation
        model = aml_lm.LinearRegression()
        kfold = aml_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        scores = aml_ms.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        
        # Final evaluation
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        final_accuracy = aml_metrics.accuracy_score(y_test, y_pred)
        
        # All should be reasonable
        self.assertGreater(np.mean(scores), 0.0)
        self.assertGreater(final_accuracy, 0.0)
        
    def test_hyperparameter_tuning_workflow(self):
        """Test complete hyperparameter tuning workflow"""
        import auroraml.model_selection as aml_ms
        import auroraml.linear_model as aml_lm
        
        # Grid search
        model = aml_lm.Ridge()
        param_grid = [{'alpha': '0.1'}, {'alpha': '1.0'}, {'alpha': '10.0'}]
        kfold = aml_ms.KFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = aml_ms.GridSearchCV(
            model, param_grid, cv=kfold, scoring='accuracy'
        )
        
        grid_search.fit(self.X, self.y)
        
        # Best model should be usable
        predictions = grid_search.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))

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

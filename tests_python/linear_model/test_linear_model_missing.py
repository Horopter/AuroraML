#!/usr/bin/env python3
"""
Tests for newly implemented linear models.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))


class TestLinearModelMissing(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(80, 6).astype(np.float64)
        coef = np.random.randn(6)
        self.y_reg = self.X @ coef + np.random.randn(80) * 0.05
        self.y_cls = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 6).astype(np.float64)

        eta = self.X[:, :3] @ np.random.randn(3)
        self.X_glm = self.X[:, :3]
        self.y_pos = np.exp(eta).astype(np.float64)

    def test_lars_family(self):
        import auroraml.linear_model as lm

        model = lm.Lars(n_nonzero_coefs=3)
        model.fit(self.X, self.y_reg)
        preds = model.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        model_cv = lm.LarsCV(cv=3)
        model_cv.fit(self.X, self.y_reg)
        self.assertGreater(model_cv.best_n_nonzero_coefs(), 0)

        lasso = lm.LassoLars(alpha=0.1)
        lasso.fit(self.X, self.y_reg)
        preds = lasso.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        lasso_cv = lm.LassoLarsCV(alphas=[0.05, 0.1, 0.5], cv=3)
        lasso_cv.fit(self.X, self.y_reg)
        self.assertGreater(lasso_cv.best_alpha(), 0.0)

        lasso_ic = lm.LassoLarsIC(alphas=[0.05, 0.1, 0.5], criterion="aic")
        lasso_ic.fit(self.X, self.y_reg)
        self.assertGreater(lasso_ic.best_alpha(), 0.0)

        omp = lm.OrthogonalMatchingPursuit(n_nonzero_coefs=3)
        omp.fit(self.X, self.y_reg)
        preds = omp.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        omp_cv = lm.OrthogonalMatchingPursuitCV(cv=3)
        omp_cv.fit(self.X, self.y_reg)
        self.assertGreater(omp_cv.best_n_nonzero_coefs(), 0)

    def test_robust_regressors(self):
        import auroraml.linear_model as lm

        ransac = lm.RANSACRegressor(max_trials=30, random_state=42)
        ransac.fit(self.X, self.y_reg)
        preds = ransac.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        theilsen = lm.TheilSenRegressor(n_subsamples=50, random_state=42)
        theilsen.fit(self.X, self.y_reg)
        preds = theilsen.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        quantile = lm.QuantileRegressor(quantile=0.5, max_iter=200, learning_rate=0.05)
        quantile.fit(self.X, self.y_reg)
        preds = quantile.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

    def test_online_models(self):
        import auroraml.linear_model as lm

        sgd_reg = lm.SGDRegressor(max_iter=200, random_state=42)
        sgd_reg.fit(self.X, self.y_reg)
        preds = sgd_reg.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        sgd_cls = lm.SGDClassifier(max_iter=200, random_state=42)
        sgd_cls.fit(self.X, self.y_cls)
        preds = sgd_cls.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        pa_reg = lm.PassiveAggressiveRegressor(max_iter=200, random_state=42)
        pa_reg.fit(self.X, self.y_reg)
        preds = pa_reg.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        pa_cls = lm.PassiveAggressiveClassifier(max_iter=200, random_state=42)
        pa_cls.fit(self.X, self.y_cls)
        preds = pa_cls.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        perceptron = lm.Perceptron(max_iter=200, random_state=42)
        perceptron.fit(self.X, self.y_cls)
        preds = perceptron.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

    def test_glm_regressors(self):
        import auroraml.linear_model as lm

        poisson = lm.PoissonRegressor(max_iter=200, learning_rate=0.01)
        poisson.fit(self.X_glm, self.y_pos)
        preds = poisson.predict(self.X_glm[:10])
        self.assertEqual(preds.shape[0], 10)

        gamma = lm.GammaRegressor(max_iter=200, learning_rate=0.01)
        gamma.fit(self.X_glm, self.y_pos + 0.5)
        preds = gamma.predict(self.X_glm[:10])
        self.assertEqual(preds.shape[0], 10)

        tweedie = lm.TweedieRegressor(power=1.5, max_iter=200, learning_rate=0.01)
        tweedie.fit(self.X_glm, self.y_pos)
        preds = tweedie.predict(self.X_glm[:10])
        self.assertEqual(preds.shape[0], 10)

    def test_multitask_models(self):
        import auroraml.linear_model as lm

        mt_lasso = lm.MultiTaskLasso(alpha=0.1, max_iter=200)
        mt_lasso.fit(self.X, self.y_reg)
        preds = mt_lasso.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        mt_lasso_cv = lm.MultiTaskLassoCV(alphas=[0.05, 0.1], cv=3)
        mt_lasso_cv.fit(self.X, self.y_reg)
        self.assertGreater(mt_lasso_cv.best_alpha(), 0.0)

        mt_enet = lm.MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=200)
        mt_enet.fit(self.X, self.y_reg)
        preds = mt_enet.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

        mt_enet_cv = lm.MultiTaskElasticNetCV(alphas=[0.05, 0.1], l1_ratios=[0.2, 0.8], cv=3)
        mt_enet_cv.fit(self.X, self.y_reg)
        self.assertGreater(mt_enet_cv.best_alpha(), 0.0)


if __name__ == '__main__':
    unittest.main()

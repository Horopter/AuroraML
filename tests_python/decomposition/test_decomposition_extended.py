#!/usr/bin/env python3
"""
Extended decomposition tests (IncrementalPCA, SparsePCA, NMF, DictionaryLearning, LDA, etc.)
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))


class TestDecompositionExtended(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(60, 6).astype(np.float64)
        self.X_nonneg = np.abs(self.X)
        self.X_counts = np.zeros((25, 10), dtype=np.float64)
        for i in range(self.X_counts.shape[0]):
            for j in range(self.X_counts.shape[1]):
                self.X_counts[i, j] = (i + j) % 3
        self.y_dummy = np.zeros(self.X.shape[0]).astype(np.float64)

    def test_incremental_pca(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.IncrementalPCA(n_components=2)
        model.fit(self.X, self.y_dummy)
        Xt = model.transform(self.X)
        self.assertEqual(Xt.shape, (self.X.shape[0], 2))

    def test_incremental_pca_partial_fit(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.IncrementalPCA(n_components=2)
        model.partial_fit(self.X[:20], self.y_dummy[:20])
        model.partial_fit(self.X[20:], self.y_dummy[20:])
        Xt = model.transform(self.X)
        self.assertEqual(Xt.shape, (self.X.shape[0], 2))

    def test_sparse_pca(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.SparsePCA(n_components=2, alpha=0.1)
        model.fit(self.X, self.y_dummy)
        Xt = model.transform(self.X)
        self.assertEqual(Xt.shape, (self.X.shape[0], 2))

    def test_minibatch_sparse_pca(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.MiniBatchSparsePCA(n_components=2, alpha=0.1, batch_size=20)
        model.fit(self.X, self.y_dummy)
        Xt = model.transform(self.X)
        self.assertEqual(Xt.shape, (self.X.shape[0], 2))

    def test_nmf(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.NMF(n_components=3, max_iter=100)
        model.fit(self.X_nonneg, self.y_dummy)
        W = model.transform(self.X_nonneg)
        self.assertEqual(W.shape, (self.X_nonneg.shape[0], 3))
        self.assertTrue(np.all(W >= 0))

    def test_minibatch_nmf(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.MiniBatchNMF(n_components=3, max_iter=100, batch_size=20)
        model.fit(self.X_nonneg, self.y_dummy)
        W = model.transform(self.X_nonneg)
        self.assertEqual(W.shape, (self.X_nonneg.shape[0], 3))
        self.assertTrue(np.all(W >= 0))

    def test_dictionary_learning(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.DictionaryLearning(n_components=3, alpha=0.1, max_iter=50)
        model.fit(self.X, self.y_dummy)
        codes = model.transform(self.X)
        self.assertEqual(codes.shape, (self.X.shape[0], 3))

    def test_minibatch_dictionary_learning(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.MiniBatchDictionaryLearning(n_components=3, alpha=0.1, max_iter=50, batch_size=20)
        model.fit(self.X, self.y_dummy)
        codes = model.transform(self.X)
        self.assertEqual(codes.shape, (self.X.shape[0], 3))

    def test_latent_dirichlet_allocation(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.LatentDirichletAllocation(n_components=3, max_iter=5)
        model.fit(self.X_counts, np.zeros(self.X_counts.shape[0]))
        topics = model.components()
        self.assertEqual(topics.shape, (3, self.X_counts.shape[1]))
        doc_topic = model.transform(self.X_counts)
        self.assertEqual(doc_topic.shape, (self.X_counts.shape[0], 3))

    def test_kernel_pca(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.KernelPCA(n_components=2, kernel="rbf", gamma=1.0)
        model.fit(self.X, self.y_dummy)
        Xt = model.transform(self.X)
        self.assertEqual(Xt.shape, (self.X.shape[0], 2))

    def test_fastica(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.FastICA(n_components=2)
        model.fit(self.X, self.y_dummy)
        Xt = model.transform(self.X)
        self.assertEqual(Xt.shape, (self.X.shape[0], 2))

    def test_factor_analysis(self):
        import auroraml.decomposition as aml_decomp

        model = aml_decomp.FactorAnalysis(n_components=2)
        model.fit(self.X, self.y_dummy)
        Xt = model.transform(self.X)
        self.assertEqual(Xt.shape, (self.X.shape[0], 2))


if __name__ == '__main__':
    unittest.main()

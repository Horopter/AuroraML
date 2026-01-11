#include <gtest/gtest.h>
#include "ingenuityml/pca.hpp"
#include "ingenuityml/decomposition_extended.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class DecompositionExtendedTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 50;
        n_features = 6;

        X = MatrixXd::Random(n_samples, n_features);
        X_nonneg = X.cwiseAbs();
        X_counts = MatrixXd::Zero(20, 10);
        for (int i = 0; i < X_counts.rows(); ++i) {
            for (int j = 0; j < X_counts.cols(); ++j) {
                X_counts(i, j) = (i + j) % 3;
            }
        }
        y_dummy = VectorXd::Zero(n_samples);
    }

    int n_samples, n_features;
    MatrixXd X;
    MatrixXd X_nonneg;
    MatrixXd X_counts;
    VectorXd y_dummy;
};

TEST_F(DecompositionExtendedTest, IncrementalPCAFitTransform) {
    decomposition::IncrementalPCA ipca(2);
    ipca.fit(X, y_dummy);
    MatrixXd Xt = ipca.transform(X);
    EXPECT_EQ(Xt.rows(), X.rows());
    EXPECT_EQ(Xt.cols(), 2);
}

TEST_F(DecompositionExtendedTest, IncrementalPCAPartialFit) {
    decomposition::IncrementalPCA ipca(2);
    ipca.partial_fit(X.topRows(20));
    ipca.partial_fit(X.bottomRows(30));
    MatrixXd Xt = ipca.transform(X);
    EXPECT_EQ(Xt.rows(), X.rows());
    EXPECT_EQ(Xt.cols(), 2);
}

TEST_F(DecompositionExtendedTest, SparsePCAFit) {
    decomposition::SparsePCA spca(2, 0.1);
    spca.fit(X, y_dummy);
    MatrixXd comps = spca.components();
    EXPECT_EQ(comps.rows(), 2);
    EXPECT_EQ(comps.cols(), X.cols());
}

TEST_F(DecompositionExtendedTest, MiniBatchSparsePCAFit) {
    decomposition::MiniBatchSparsePCA mspca(2, 0.1, 100, 20);
    mspca.fit(X, y_dummy);
    MatrixXd comps = mspca.components();
    EXPECT_EQ(comps.rows(), 2);
    EXPECT_EQ(comps.cols(), X.cols());
}

TEST_F(DecompositionExtendedTest, NMFTransform) {
    decomposition::NMF nmf(3, 100);
    nmf.fit(X_nonneg, y_dummy);
    MatrixXd W = nmf.transform(X_nonneg);
    EXPECT_EQ(W.rows(), X_nonneg.rows());
    EXPECT_EQ(W.cols(), 3);
    EXPECT_TRUE((W.array() >= 0.0).all());
}

TEST_F(DecompositionExtendedTest, MiniBatchNMFTransform) {
    decomposition::MiniBatchNMF nmf(3, 100, 20);
    nmf.fit(X_nonneg, y_dummy);
    MatrixXd W = nmf.transform(X_nonneg);
    EXPECT_EQ(W.rows(), X_nonneg.rows());
    EXPECT_EQ(W.cols(), 3);
    EXPECT_TRUE((W.array() >= 0.0).all());
}

TEST_F(DecompositionExtendedTest, DictionaryLearningTransform) {
    decomposition::DictionaryLearning dl(3, 0.1, 50);
    dl.fit(X, y_dummy);
    MatrixXd codes = dl.transform(X);
    EXPECT_EQ(codes.rows(), X.rows());
    EXPECT_EQ(codes.cols(), 3);
}

TEST_F(DecompositionExtendedTest, MiniBatchDictionaryLearningTransform) {
    decomposition::MiniBatchDictionaryLearning dl(3, 0.1, 50, 20);
    dl.fit(X, y_dummy);
    MatrixXd codes = dl.transform(X);
    EXPECT_EQ(codes.rows(), X.rows());
    EXPECT_EQ(codes.cols(), 3);
}

TEST_F(DecompositionExtendedTest, LatentDirichletAllocationFit) {
    decomposition::LatentDirichletAllocation lda(3, 5);
    lda.fit(X_counts, VectorXd());
    MatrixXd topics = lda.components();
    EXPECT_EQ(topics.rows(), 3);
    EXPECT_EQ(topics.cols(), X_counts.cols());
    MatrixXd doc_topic = lda.transform(X_counts);
    EXPECT_EQ(doc_topic.rows(), X_counts.rows());
    EXPECT_EQ(doc_topic.cols(), 3);
}

TEST_F(DecompositionExtendedTest, KernelPCAFit) {
    decomposition::KernelPCA kpca(2, "rbf", 1.0, 3.0, 1.0);
    kpca.fit(X, y_dummy);
    MatrixXd Xt = kpca.transform(X);
    EXPECT_EQ(Xt.rows(), X.rows());
    EXPECT_EQ(Xt.cols(), 2);
}

TEST_F(DecompositionExtendedTest, FastICAFit) {
    decomposition::FastICA ica(2);
    ica.fit(X, y_dummy);
    MatrixXd Xt = ica.transform(X);
    EXPECT_EQ(Xt.rows(), X.rows());
    EXPECT_EQ(Xt.cols(), 2);
}

TEST_F(DecompositionExtendedTest, FactorAnalysisFit) {
    decomposition::FactorAnalysis fa(2);
    fa.fit(X, y_dummy);
    MatrixXd Xt = fa.transform(X);
    EXPECT_EQ(Xt.rows(), X.rows());
    EXPECT_EQ(Xt.cols(), 2);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

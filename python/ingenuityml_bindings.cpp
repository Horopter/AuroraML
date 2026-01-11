#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <map>
#include <limits>

#include "ingenuityml/base.hpp"
#include "ingenuityml/random.hpp"
#include "ingenuityml/linear_model.hpp"
#include "ingenuityml/neighbors.hpp"
#include "ingenuityml/tree.hpp"
#include "ingenuityml/metrics.hpp"
#include "ingenuityml/preprocessing.hpp"
#include "ingenuityml/model_selection.hpp"
#include "ingenuityml/naive_bayes.hpp"
#include "ingenuityml/kmeans.hpp"
#include "ingenuityml/pca.hpp"
#include "ingenuityml/dbscan.hpp"
#include "ingenuityml/truncated_svd.hpp"
#include "ingenuityml/decomposition_extended.hpp"
#include "ingenuityml/cross_decomposition.hpp"
#include "ingenuityml/lda.hpp"
#include "ingenuityml/agglomerative.hpp"
#include "ingenuityml/svm.hpp"
#include "ingenuityml/random_forest.hpp"
#include "ingenuityml/extra_trees.hpp"
#include "ingenuityml/random_trees_embedding.hpp"
#include "ingenuityml/gradient_boosting.hpp"
#include "ingenuityml/adaboost.hpp"
#include "ingenuityml/xgboost.hpp"
#include "ingenuityml/catboost.hpp"
#include "ingenuityml/pipeline.hpp"
#include "ingenuityml/compose.hpp"
#include "ingenuityml/neural_network.hpp"
#include "ingenuityml/feature_selection.hpp"
#include "ingenuityml/impute.hpp"
#include "ingenuityml/utils.hpp"
#include "ingenuityml/inspection.hpp"
#include "ingenuityml/ensemble_wrappers.hpp"
#include "ingenuityml/dummy.hpp"
#include "ingenuityml/covariance.hpp"
#include "ingenuityml/meta_estimators.hpp"
#include "ingenuityml/calibration.hpp"
#include "ingenuityml/isotonic.hpp"
#include "ingenuityml/discriminant_analysis.hpp"
#include "ingenuityml/naive_bayes_variants.hpp"
#include "ingenuityml/extratree.hpp"
#include "ingenuityml/outlier_detection.hpp"
#include "ingenuityml/mixture.hpp"
#include "ingenuityml/gaussian_process.hpp"
#include "ingenuityml/semi_supervised.hpp"
#include "ingenuityml/preprocessing_extended.hpp"
#include "ingenuityml/density_estimation.hpp"
#include "ingenuityml/random_projection.hpp"
#include "ingenuityml/cluster_extended.hpp"
#include "ingenuityml/manifold.hpp"

namespace py = pybind11;

static ingenuityml::Params params_from_kwargs(const py::kwargs& kwargs) {
    ingenuityml::Params params;
    for (auto item : kwargs) {
        params[py::cast<std::string>(item.first)] = py::str(item.second);
    }
    return params;
}

// Helper function to convert numpy arrays to Eigen matrices
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> numpy_to_eigen(py::array_t<T> arr) {
    py::buffer_info buf_info = arr.request();
    
    if (buf_info.ndim == 1) {
        // Convert 1D array to column vector
        return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            static_cast<T*>(buf_info.ptr), buf_info.shape[0]).transpose();
    } else if (buf_info.ndim == 2) {
        // Convert 2D array to matrix
        return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
            static_cast<T*>(buf_info.ptr), buf_info.shape[0], buf_info.shape[1]);
    } else {
        throw std::runtime_error("Number of dimensions must be 1 or 2");
    }
}

// Helper function to convert Eigen matrices to numpy arrays
template<typename T>
py::array_t<T> eigen_to_numpy(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    // Create numpy array with proper shape (rows, cols)
    py::array_t<T> result({mat.rows(), mat.cols()});
    py::buffer_info buf_info = result.request();
    T* ptr = static_cast<T*>(buf_info.ptr);
    
    // Copy data row by row to ensure correct layout
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            ptr[i * mat.cols() + j] = mat(i, j);
        }
    }
    
    return result;
}

PYBIND11_MODULE(ingenuityml, m) {
    m.doc() = "IngenuityML: High-performance C++ machine learning library";
    
    // Bind base classes
    py::class_<ingenuityml::Estimator>(m, "Estimator");
    py::class_<ingenuityml::Predictor>(m, "Predictor");
    py::class_<ingenuityml::Classifier, ingenuityml::Predictor>(m, "Classifier");
    py::class_<ingenuityml::Regressor, ingenuityml::Predictor>(m, "Regressor");
    py::class_<ingenuityml::Transformer>(m, "Transformer");
    
    // Bind parameter types
    py::class_<ingenuityml::Params>(m, "Params")
        .def(py::init<>())
        .def("__getitem__", [](const ingenuityml::Params& p, const std::string& key) {
            auto it = p.find(key);
            if (it != p.end()) return it->second;
            throw std::runtime_error("Key not found");
        })
        .def("__setitem__", [](ingenuityml::Params& p, const std::string& key, const std::string& value) {
            p[key] = value;
        });
    
    // Random module
    py::module_ random_module = m.def_submodule("random", "Random number generation");
    
    py::class_<ingenuityml::random::PCG64>(random_module, "PCG64")
        .def(py::init<uint64_t>(), py::arg("seed") = 0)
        .def("seed", &ingenuityml::random::PCG64::seed)
        .def("uniform", &ingenuityml::random::PCG64::uniform, py::arg("low") = 0.0, py::arg("high") = 1.0)
        .def("normal", &ingenuityml::random::PCG64::normal, py::arg("mean") = 0.0, py::arg("std") = 1.0)
        .def("randint", &ingenuityml::random::PCG64::randint, py::arg("low"), py::arg("high"))
        .def("get_state", &ingenuityml::random::PCG64::get_state)
        .def("set_state", &ingenuityml::random::PCG64::set_state)
        .def("get_params", &ingenuityml::random::PCG64::get_params)
        .def("set_params", &ingenuityml::random::PCG64::set_params)
        .def("set_params", [](ingenuityml::random::PCG64& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        });
    
    // Linear models module
    py::module_ linear_module = m.def_submodule("linear_model", "Linear models");
    
    py::class_<ingenuityml::linear_model::LinearRegression, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "LinearRegression")
        .def(py::init<bool, bool, int>(), py::arg("fit_intercept") = true, py::arg("copy_X") = true, py::arg("n_jobs") = 1)
        .def("fit", &ingenuityml::linear_model::LinearRegression::fit)
        .def("predict", &ingenuityml::linear_model::LinearRegression::predict)
        .def("get_params", &ingenuityml::linear_model::LinearRegression::get_params)
        .def("set_params", &ingenuityml::linear_model::LinearRegression::set_params)
        .def("is_fitted", &ingenuityml::linear_model::LinearRegression::is_fitted)
        .def("coef", &ingenuityml::linear_model::LinearRegression::coef)
        .def("intercept", &ingenuityml::linear_model::LinearRegression::intercept)
        .def("save_to_file", &ingenuityml::linear_model::LinearRegression::save_to_file)
        .def("load_from_file", &ingenuityml::linear_model::LinearRegression::load_from_file);
    
    py::class_<ingenuityml::linear_model::Ridge, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "Ridge")
        .def(py::init<double, bool, bool, int>(), py::arg("alpha") = 1.0, py::arg("fit_intercept") = true, py::arg("copy_X") = true, py::arg("n_jobs") = 1)
        .def("fit", &ingenuityml::linear_model::Ridge::fit)
        .def("predict", &ingenuityml::linear_model::Ridge::predict)
        .def("get_params", &ingenuityml::linear_model::Ridge::get_params)
        .def("set_params", &ingenuityml::linear_model::Ridge::set_params)
        .def("is_fitted", &ingenuityml::linear_model::Ridge::is_fitted)
        .def("coef", &ingenuityml::linear_model::Ridge::coef)
        .def("intercept", &ingenuityml::linear_model::Ridge::intercept)
        .def("save_to_file", &ingenuityml::linear_model::Ridge::save_to_file)
        .def("load_from_file", &ingenuityml::linear_model::Ridge::load_from_file);
    
    py::class_<ingenuityml::linear_model::Lasso, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "Lasso")
        .def(py::init<double, bool, bool, int>(), py::arg("alpha") = 1.0, py::arg("fit_intercept") = true, py::arg("copy_X") = true, py::arg("n_jobs") = 1)
        .def("fit", &ingenuityml::linear_model::Lasso::fit)
        .def("predict", &ingenuityml::linear_model::Lasso::predict)
        .def("get_params", &ingenuityml::linear_model::Lasso::get_params)
        .def("set_params", &ingenuityml::linear_model::Lasso::set_params)
        .def("is_fitted", &ingenuityml::linear_model::Lasso::is_fitted)
        .def("coef", &ingenuityml::linear_model::Lasso::coef)
        .def("intercept", &ingenuityml::linear_model::Lasso::intercept);
    
    py::class_<ingenuityml::linear_model::ElasticNet, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "ElasticNet")
        .def(py::init<double, double, bool, bool, int, int, double>(),
             py::arg("alpha") = 1.0, py::arg("l1_ratio") = 0.5, py::arg("fit_intercept") = true,
             py::arg("copy_X") = true, py::arg("n_jobs") = 1, py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::ElasticNet::fit)
        .def("predict", &ingenuityml::linear_model::ElasticNet::predict)
        .def("get_params", &ingenuityml::linear_model::ElasticNet::get_params)
        .def("set_params", &ingenuityml::linear_model::ElasticNet::set_params)
        .def("is_fitted", &ingenuityml::linear_model::ElasticNet::is_fitted)
        .def("coef", &ingenuityml::linear_model::ElasticNet::coef)
        .def("intercept", &ingenuityml::linear_model::ElasticNet::intercept);
    
    py::class_<ingenuityml::linear_model::RidgeCV, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "RidgeCV")
        .def(py::init<const std::vector<double>&, bool, int>(),
             py::arg("alphas") = std::vector<double>{0.1, 1.0, 10.0},
             py::arg("fit_intercept") = true, py::arg("cv") = 5)
        .def("fit", &ingenuityml::linear_model::RidgeCV::fit)
        .def("predict", &ingenuityml::linear_model::RidgeCV::predict)
        .def("get_params", &ingenuityml::linear_model::RidgeCV::get_params)
        .def("set_params", &ingenuityml::linear_model::RidgeCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::RidgeCV::is_fitted)
        .def("best_alpha", &ingenuityml::linear_model::RidgeCV::best_alpha)
        .def("coef", &ingenuityml::linear_model::RidgeCV::coef)
        .def("intercept", &ingenuityml::linear_model::RidgeCV::intercept);

    py::class_<ingenuityml::linear_model::LassoCV, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "LassoCV")
        .def(py::init<const std::vector<double>&, bool, int, int, double>(),
             py::arg("alphas") = std::vector<double>{0.1, 1.0, 10.0},
             py::arg("fit_intercept") = true, py::arg("cv") = 5,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::LassoCV::fit)
        .def("predict", &ingenuityml::linear_model::LassoCV::predict)
        .def("get_params", &ingenuityml::linear_model::LassoCV::get_params)
        .def("set_params", &ingenuityml::linear_model::LassoCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::LassoCV::is_fitted)
        .def("best_alpha", &ingenuityml::linear_model::LassoCV::best_alpha)
        .def("coef", &ingenuityml::linear_model::LassoCV::coef)
        .def("intercept", &ingenuityml::linear_model::LassoCV::intercept);

    py::class_<ingenuityml::linear_model::ElasticNetCV, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "ElasticNetCV")
        .def(py::init<const std::vector<double>&, const std::vector<double>&, bool, int, int, double>(),
             py::arg("alphas") = std::vector<double>{0.1, 1.0, 10.0},
             py::arg("l1_ratios") = std::vector<double>{0.1, 0.5, 0.9},
             py::arg("fit_intercept") = true, py::arg("cv") = 5,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::ElasticNetCV::fit)
        .def("predict", &ingenuityml::linear_model::ElasticNetCV::predict)
        .def("get_params", &ingenuityml::linear_model::ElasticNetCV::get_params)
        .def("set_params", &ingenuityml::linear_model::ElasticNetCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::ElasticNetCV::is_fitted)
        .def("best_alpha", &ingenuityml::linear_model::ElasticNetCV::best_alpha)
        .def("best_l1_ratio", &ingenuityml::linear_model::ElasticNetCV::best_l1_ratio)
        .def("coef", &ingenuityml::linear_model::ElasticNetCV::coef)
        .def("intercept", &ingenuityml::linear_model::ElasticNetCV::intercept);

    py::class_<ingenuityml::linear_model::BayesianRidge, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "BayesianRidge")
        .def(py::init<double, double, double, double, bool, int, double>(),
             py::arg("alpha_1") = 1e-6, py::arg("alpha_2") = 1e-6,
             py::arg("lambda_1") = 1e-6, py::arg("lambda_2") = 1e-6,
             py::arg("fit_intercept") = true, py::arg("max_iter") = 300, py::arg("tol") = 1e-3)
        .def("fit", &ingenuityml::linear_model::BayesianRidge::fit)
        .def("predict", &ingenuityml::linear_model::BayesianRidge::predict)
        .def("get_params", &ingenuityml::linear_model::BayesianRidge::get_params)
        .def("set_params", &ingenuityml::linear_model::BayesianRidge::set_params)
        .def("is_fitted", &ingenuityml::linear_model::BayesianRidge::is_fitted)
        .def("coef", &ingenuityml::linear_model::BayesianRidge::coef)
        .def("intercept", &ingenuityml::linear_model::BayesianRidge::intercept);

    py::class_<ingenuityml::linear_model::ARDRegression, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "ARDRegression")
        .def(py::init<double, double, double, double, bool, int, double>(),
             py::arg("alpha_1") = 1e-6, py::arg("alpha_2") = 1e-6,
             py::arg("lambda_1") = 1e-6, py::arg("lambda_2") = 1e-6,
             py::arg("fit_intercept") = true, py::arg("max_iter") = 300, py::arg("tol") = 1e-3)
        .def("fit", &ingenuityml::linear_model::ARDRegression::fit)
        .def("predict", &ingenuityml::linear_model::ARDRegression::predict)
        .def("get_params", &ingenuityml::linear_model::ARDRegression::get_params)
        .def("set_params", &ingenuityml::linear_model::ARDRegression::set_params)
        .def("is_fitted", &ingenuityml::linear_model::ARDRegression::is_fitted)
        .def("coef", &ingenuityml::linear_model::ARDRegression::coef)
        .def("intercept", &ingenuityml::linear_model::ARDRegression::intercept);

    py::class_<ingenuityml::linear_model::HuberRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "HuberRegressor")
        .def(py::init<double, double, bool, int, double>(),
             py::arg("epsilon") = 1.35, py::arg("alpha") = 0.0001,
             py::arg("fit_intercept") = true, py::arg("max_iter") = 100, py::arg("tol") = 1e-5)
        .def("fit", &ingenuityml::linear_model::HuberRegressor::fit)
        .def("predict", &ingenuityml::linear_model::HuberRegressor::predict)
        .def("get_params", &ingenuityml::linear_model::HuberRegressor::get_params)
        .def("set_params", &ingenuityml::linear_model::HuberRegressor::set_params)
        .def("is_fitted", &ingenuityml::linear_model::HuberRegressor::is_fitted)
        .def("coef", &ingenuityml::linear_model::HuberRegressor::coef)
        .def("intercept", &ingenuityml::linear_model::HuberRegressor::intercept);

    py::class_<ingenuityml::linear_model::LogisticRegression, ingenuityml::Estimator, ingenuityml::Classifier>(linear_module, "LogisticRegression")
        .def(py::init<double, bool, int, double, int>(),
             py::arg("C") = 1.0, py::arg("fit_intercept") = true, py::arg("max_iter") = 100,
             py::arg("tol") = 1e-4, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::linear_model::LogisticRegression::fit)
        .def("predict", &ingenuityml::linear_model::LogisticRegression::predict_classes)
        .def("predict_proba", &ingenuityml::linear_model::LogisticRegression::predict_proba)
        .def("decision_function", &ingenuityml::linear_model::LogisticRegression::decision_function)
        .def("get_params", &ingenuityml::linear_model::LogisticRegression::get_params)
        .def("set_params", &ingenuityml::linear_model::LogisticRegression::set_params)
        .def("is_fitted", &ingenuityml::linear_model::LogisticRegression::is_fitted)
        .def("coef", &ingenuityml::linear_model::LogisticRegression::coef)
        .def("intercept", &ingenuityml::linear_model::LogisticRegression::intercept)
        .def("classes", &ingenuityml::linear_model::LogisticRegression::classes)
        .def("n_classes", &ingenuityml::linear_model::LogisticRegression::n_classes);

    py::class_<ingenuityml::linear_model::Lars, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "Lars")
        .def(py::init<int, bool, int, double>(),
             py::arg("n_nonzero_coefs") = 0, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 500, py::arg("eps") = 1e-3)
        .def("fit", &ingenuityml::linear_model::Lars::fit)
        .def("predict", &ingenuityml::linear_model::Lars::predict)
        .def("get_params", &ingenuityml::linear_model::Lars::get_params)
        .def("set_params", &ingenuityml::linear_model::Lars::set_params)
        .def("is_fitted", &ingenuityml::linear_model::Lars::is_fitted)
        .def("coef", &ingenuityml::linear_model::Lars::coef)
        .def("intercept", &ingenuityml::linear_model::Lars::intercept);

    py::class_<ingenuityml::linear_model::LarsCV, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "LarsCV")
        .def(py::init<int, bool, int, double>(),
             py::arg("cv") = 5, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 500, py::arg("eps") = 1e-3)
        .def("fit", &ingenuityml::linear_model::LarsCV::fit)
        .def("predict", &ingenuityml::linear_model::LarsCV::predict)
        .def("get_params", &ingenuityml::linear_model::LarsCV::get_params)
        .def("set_params", &ingenuityml::linear_model::LarsCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::LarsCV::is_fitted)
        .def("best_n_nonzero_coefs", &ingenuityml::linear_model::LarsCV::best_n_nonzero_coefs)
        .def("coef", &ingenuityml::linear_model::LarsCV::coef)
        .def("intercept", &ingenuityml::linear_model::LarsCV::intercept);

    py::class_<ingenuityml::linear_model::LassoLars, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "LassoLars")
        .def(py::init<double, bool, int, double>(),
             py::arg("alpha") = 1.0, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::LassoLars::fit)
        .def("predict", &ingenuityml::linear_model::LassoLars::predict)
        .def("get_params", &ingenuityml::linear_model::LassoLars::get_params)
        .def("set_params", &ingenuityml::linear_model::LassoLars::set_params)
        .def("is_fitted", &ingenuityml::linear_model::LassoLars::is_fitted)
        .def("coef", &ingenuityml::linear_model::LassoLars::coef)
        .def("intercept", &ingenuityml::linear_model::LassoLars::intercept);

    py::class_<ingenuityml::linear_model::LassoLarsCV, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "LassoLarsCV")
        .def(py::init<const std::vector<double>&, int, bool, int, double>(),
             py::arg("alphas") = std::vector<double>{0.1, 1.0, 10.0}, py::arg("cv") = 5,
             py::arg("fit_intercept") = true, py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::LassoLarsCV::fit)
        .def("predict", &ingenuityml::linear_model::LassoLarsCV::predict)
        .def("get_params", &ingenuityml::linear_model::LassoLarsCV::get_params)
        .def("set_params", &ingenuityml::linear_model::LassoLarsCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::LassoLarsCV::is_fitted)
        .def("best_alpha", &ingenuityml::linear_model::LassoLarsCV::best_alpha)
        .def("coef", &ingenuityml::linear_model::LassoLarsCV::coef)
        .def("intercept", &ingenuityml::linear_model::LassoLarsCV::intercept);

    py::class_<ingenuityml::linear_model::LassoLarsIC, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "LassoLarsIC")
        .def(py::init<const std::vector<double>&, const std::string&, bool, int, double>(),
             py::arg("alphas") = std::vector<double>{0.1, 1.0, 10.0}, py::arg("criterion") = "aic",
             py::arg("fit_intercept") = true, py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::LassoLarsIC::fit)
        .def("predict", &ingenuityml::linear_model::LassoLarsIC::predict)
        .def("get_params", &ingenuityml::linear_model::LassoLarsIC::get_params)
        .def("set_params", &ingenuityml::linear_model::LassoLarsIC::set_params)
        .def("is_fitted", &ingenuityml::linear_model::LassoLarsIC::is_fitted)
        .def("best_alpha", &ingenuityml::linear_model::LassoLarsIC::best_alpha)
        .def("coef", &ingenuityml::linear_model::LassoLarsIC::coef)
        .def("intercept", &ingenuityml::linear_model::LassoLarsIC::intercept);

    py::class_<ingenuityml::linear_model::OrthogonalMatchingPursuit, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "OrthogonalMatchingPursuit")
        .def(py::init<int, bool, int, double>(),
             py::arg("n_nonzero_coefs") = 0, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::OrthogonalMatchingPursuit::fit)
        .def("predict", &ingenuityml::linear_model::OrthogonalMatchingPursuit::predict)
        .def("get_params", &ingenuityml::linear_model::OrthogonalMatchingPursuit::get_params)
        .def("set_params", &ingenuityml::linear_model::OrthogonalMatchingPursuit::set_params)
        .def("is_fitted", &ingenuityml::linear_model::OrthogonalMatchingPursuit::is_fitted)
        .def("coef", &ingenuityml::linear_model::OrthogonalMatchingPursuit::coef)
        .def("intercept", &ingenuityml::linear_model::OrthogonalMatchingPursuit::intercept);

    py::class_<ingenuityml::linear_model::OrthogonalMatchingPursuitCV, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "OrthogonalMatchingPursuitCV")
        .def(py::init<int, bool, int, double>(),
             py::arg("cv") = 5, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::OrthogonalMatchingPursuitCV::fit)
        .def("predict", &ingenuityml::linear_model::OrthogonalMatchingPursuitCV::predict)
        .def("get_params", &ingenuityml::linear_model::OrthogonalMatchingPursuitCV::get_params)
        .def("set_params", &ingenuityml::linear_model::OrthogonalMatchingPursuitCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::OrthogonalMatchingPursuitCV::is_fitted)
        .def("best_n_nonzero_coefs", &ingenuityml::linear_model::OrthogonalMatchingPursuitCV::best_n_nonzero_coefs)
        .def("coef", &ingenuityml::linear_model::OrthogonalMatchingPursuitCV::coef)
        .def("intercept", &ingenuityml::linear_model::OrthogonalMatchingPursuitCV::intercept);

    py::class_<ingenuityml::linear_model::RANSACRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "RANSACRegressor")
        .def(py::init<int, int, double, int, bool>(),
             py::arg("max_trials") = 100, py::arg("min_samples") = -1,
             py::arg("residual_threshold") = -1.0, py::arg("random_state") = -1,
             py::arg("fit_intercept") = true)
        .def("fit", &ingenuityml::linear_model::RANSACRegressor::fit)
        .def("predict", &ingenuityml::linear_model::RANSACRegressor::predict)
        .def("get_params", &ingenuityml::linear_model::RANSACRegressor::get_params)
        .def("set_params", &ingenuityml::linear_model::RANSACRegressor::set_params)
        .def("is_fitted", &ingenuityml::linear_model::RANSACRegressor::is_fitted)
        .def("coef", &ingenuityml::linear_model::RANSACRegressor::coef)
        .def("intercept", &ingenuityml::linear_model::RANSACRegressor::intercept);

    py::class_<ingenuityml::linear_model::TheilSenRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "TheilSenRegressor")
        .def(py::init<int, int, bool>(),
             py::arg("n_subsamples") = 100, py::arg("random_state") = -1, py::arg("fit_intercept") = true)
        .def("fit", &ingenuityml::linear_model::TheilSenRegressor::fit)
        .def("predict", &ingenuityml::linear_model::TheilSenRegressor::predict)
        .def("get_params", &ingenuityml::linear_model::TheilSenRegressor::get_params)
        .def("set_params", &ingenuityml::linear_model::TheilSenRegressor::set_params)
        .def("is_fitted", &ingenuityml::linear_model::TheilSenRegressor::is_fitted)
        .def("coef", &ingenuityml::linear_model::TheilSenRegressor::coef)
        .def("intercept", &ingenuityml::linear_model::TheilSenRegressor::intercept);

    py::class_<ingenuityml::linear_model::SGDRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "SGDRegressor")
        .def(py::init<const std::string&, const std::string&, double, double, bool, int, double, const std::string&, double, double, bool, int, double>(),
             py::arg("loss") = "squared_loss", py::arg("penalty") = "l2", py::arg("alpha") = 0.0001,
             py::arg("l1_ratio") = 0.15, py::arg("fit_intercept") = true, py::arg("max_iter") = 1000,
             py::arg("tol") = 1e-3, py::arg("learning_rate") = "invscaling", py::arg("eta0") = 0.01,
             py::arg("power_t") = 0.5, py::arg("shuffle") = true, py::arg("random_state") = -1,
             py::arg("epsilon") = 0.1)
        .def("fit", &ingenuityml::linear_model::SGDRegressor::fit)
        .def("predict", &ingenuityml::linear_model::SGDRegressor::predict)
        .def("get_params", &ingenuityml::linear_model::SGDRegressor::get_params)
        .def("set_params", &ingenuityml::linear_model::SGDRegressor::set_params)
        .def("is_fitted", &ingenuityml::linear_model::SGDRegressor::is_fitted)
        .def("coef", &ingenuityml::linear_model::SGDRegressor::coef)
        .def("intercept", &ingenuityml::linear_model::SGDRegressor::intercept);

    py::class_<ingenuityml::linear_model::SGDClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(linear_module, "SGDClassifier")
        .def(py::init<const std::string&, const std::string&, double, double, bool, int, double, const std::string&, double, double, bool, int>(),
             py::arg("loss") = "hinge", py::arg("penalty") = "l2", py::arg("alpha") = 0.0001,
             py::arg("l1_ratio") = 0.15, py::arg("fit_intercept") = true, py::arg("max_iter") = 1000,
             py::arg("tol") = 1e-3, py::arg("learning_rate") = "invscaling", py::arg("eta0") = 0.01,
             py::arg("power_t") = 0.5, py::arg("shuffle") = true, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::linear_model::SGDClassifier::fit)
        .def("predict", &ingenuityml::linear_model::SGDClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::linear_model::SGDClassifier::predict_proba)
        .def("decision_function", &ingenuityml::linear_model::SGDClassifier::decision_function)
        .def("get_params", &ingenuityml::linear_model::SGDClassifier::get_params)
        .def("set_params", &ingenuityml::linear_model::SGDClassifier::set_params)
        .def("is_fitted", &ingenuityml::linear_model::SGDClassifier::is_fitted)
        .def("coef", &ingenuityml::linear_model::SGDClassifier::coef)
        .def("intercept", &ingenuityml::linear_model::SGDClassifier::intercept)
        .def("classes", &ingenuityml::linear_model::SGDClassifier::classes)
        .def("n_classes", &ingenuityml::linear_model::SGDClassifier::n_classes);

    py::class_<ingenuityml::linear_model::PassiveAggressiveRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "PassiveAggressiveRegressor")
        .def(py::init<double, double, bool, int, bool, int, const std::string&>(),
             py::arg("C") = 1.0, py::arg("epsilon") = 0.1, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("shuffle") = true, py::arg("random_state") = -1,
             py::arg("loss") = "epsilon_insensitive")
        .def("fit", &ingenuityml::linear_model::PassiveAggressiveRegressor::fit)
        .def("predict", &ingenuityml::linear_model::PassiveAggressiveRegressor::predict)
        .def("get_params", &ingenuityml::linear_model::PassiveAggressiveRegressor::get_params)
        .def("set_params", &ingenuityml::linear_model::PassiveAggressiveRegressor::set_params)
        .def("is_fitted", &ingenuityml::linear_model::PassiveAggressiveRegressor::is_fitted)
        .def("coef", &ingenuityml::linear_model::PassiveAggressiveRegressor::coef)
        .def("intercept", &ingenuityml::linear_model::PassiveAggressiveRegressor::intercept);

    py::class_<ingenuityml::linear_model::PassiveAggressiveClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(linear_module, "PassiveAggressiveClassifier")
        .def(py::init<double, bool, int, bool, int>(),
             py::arg("C") = 1.0, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("shuffle") = true, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::linear_model::PassiveAggressiveClassifier::fit)
        .def("predict", &ingenuityml::linear_model::PassiveAggressiveClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::linear_model::PassiveAggressiveClassifier::predict_proba)
        .def("decision_function", &ingenuityml::linear_model::PassiveAggressiveClassifier::decision_function)
        .def("get_params", &ingenuityml::linear_model::PassiveAggressiveClassifier::get_params)
        .def("set_params", &ingenuityml::linear_model::PassiveAggressiveClassifier::set_params)
        .def("is_fitted", &ingenuityml::linear_model::PassiveAggressiveClassifier::is_fitted)
        .def("coef", &ingenuityml::linear_model::PassiveAggressiveClassifier::coef)
        .def("intercept", &ingenuityml::linear_model::PassiveAggressiveClassifier::intercept)
        .def("classes", &ingenuityml::linear_model::PassiveAggressiveClassifier::classes)
        .def("n_classes", &ingenuityml::linear_model::PassiveAggressiveClassifier::n_classes);

    py::class_<ingenuityml::linear_model::Perceptron, ingenuityml::Estimator, ingenuityml::Classifier>(linear_module, "Perceptron")
        .def(py::init<bool, int, double, bool, int>(),
             py::arg("fit_intercept") = true, py::arg("max_iter") = 1000, py::arg("tol") = 1e-3,
             py::arg("shuffle") = true, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::linear_model::Perceptron::fit)
        .def("predict", &ingenuityml::linear_model::Perceptron::predict_classes)
        .def("predict_proba", &ingenuityml::linear_model::Perceptron::predict_proba)
        .def("decision_function", &ingenuityml::linear_model::Perceptron::decision_function)
        .def("get_params", &ingenuityml::linear_model::Perceptron::get_params)
        .def("set_params", &ingenuityml::linear_model::Perceptron::set_params)
        .def("is_fitted", &ingenuityml::linear_model::Perceptron::is_fitted)
        .def("coef", &ingenuityml::linear_model::Perceptron::coef)
        .def("intercept", &ingenuityml::linear_model::Perceptron::intercept)
        .def("classes", &ingenuityml::linear_model::Perceptron::classes)
        .def("n_classes", &ingenuityml::linear_model::Perceptron::n_classes);

    py::class_<ingenuityml::linear_model::LogisticRegressionCV, ingenuityml::Estimator, ingenuityml::Classifier>(linear_module, "LogisticRegressionCV")
        .def(py::init<const std::vector<double>&, int, const std::string&, bool, int, double, int>(),
             py::arg("Cs") = std::vector<double>{0.1, 1.0, 10.0}, py::arg("cv") = 5,
             py::arg("scoring") = "accuracy", py::arg("fit_intercept") = true,
             py::arg("max_iter") = 100, py::arg("tol") = 1e-4, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::linear_model::LogisticRegressionCV::fit)
        .def("predict", &ingenuityml::linear_model::LogisticRegressionCV::predict_classes)
        .def("predict_proba", &ingenuityml::linear_model::LogisticRegressionCV::predict_proba)
        .def("decision_function", &ingenuityml::linear_model::LogisticRegressionCV::decision_function)
        .def("get_params", &ingenuityml::linear_model::LogisticRegressionCV::get_params)
        .def("set_params", &ingenuityml::linear_model::LogisticRegressionCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::LogisticRegressionCV::is_fitted)
        .def("best_C", &ingenuityml::linear_model::LogisticRegressionCV::best_C)
        .def("coef", &ingenuityml::linear_model::LogisticRegressionCV::coef)
        .def("intercept", &ingenuityml::linear_model::LogisticRegressionCV::intercept)
        .def("classes", &ingenuityml::linear_model::LogisticRegressionCV::classes)
        .def("n_classes", &ingenuityml::linear_model::LogisticRegressionCV::n_classes);

    py::class_<ingenuityml::linear_model::RidgeClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(linear_module, "RidgeClassifier")
        .def(py::init<double, bool>(), py::arg("alpha") = 1.0, py::arg("fit_intercept") = true)
        .def("fit", &ingenuityml::linear_model::RidgeClassifier::fit)
        .def("predict", &ingenuityml::linear_model::RidgeClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::linear_model::RidgeClassifier::predict_proba)
        .def("decision_function", &ingenuityml::linear_model::RidgeClassifier::decision_function)
        .def("get_params", &ingenuityml::linear_model::RidgeClassifier::get_params)
        .def("set_params", &ingenuityml::linear_model::RidgeClassifier::set_params)
        .def("is_fitted", &ingenuityml::linear_model::RidgeClassifier::is_fitted)
        .def("coef", &ingenuityml::linear_model::RidgeClassifier::coef)
        .def("intercept", &ingenuityml::linear_model::RidgeClassifier::intercept)
        .def("classes", &ingenuityml::linear_model::RidgeClassifier::classes)
        .def("n_classes", &ingenuityml::linear_model::RidgeClassifier::n_classes);

    py::class_<ingenuityml::linear_model::RidgeClassifierCV, ingenuityml::Estimator, ingenuityml::Classifier>(linear_module, "RidgeClassifierCV")
        .def(py::init<const std::vector<double>&, int, const std::string&, bool>(),
             py::arg("alphas") = std::vector<double>{0.1, 1.0, 10.0}, py::arg("cv") = 5,
             py::arg("scoring") = "accuracy", py::arg("fit_intercept") = true)
        .def("fit", &ingenuityml::linear_model::RidgeClassifierCV::fit)
        .def("predict", &ingenuityml::linear_model::RidgeClassifierCV::predict_classes)
        .def("predict_proba", &ingenuityml::linear_model::RidgeClassifierCV::predict_proba)
        .def("decision_function", &ingenuityml::linear_model::RidgeClassifierCV::decision_function)
        .def("get_params", &ingenuityml::linear_model::RidgeClassifierCV::get_params)
        .def("set_params", &ingenuityml::linear_model::RidgeClassifierCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::RidgeClassifierCV::is_fitted)
        .def("best_alpha", &ingenuityml::linear_model::RidgeClassifierCV::best_alpha)
        .def("coef", &ingenuityml::linear_model::RidgeClassifierCV::coef)
        .def("intercept", &ingenuityml::linear_model::RidgeClassifierCV::intercept)
        .def("classes", &ingenuityml::linear_model::RidgeClassifierCV::classes)
        .def("n_classes", &ingenuityml::linear_model::RidgeClassifierCV::n_classes);

    py::class_<ingenuityml::linear_model::QuantileRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "QuantileRegressor")
        .def(py::init<double, double, bool, int, double, double>(),
             py::arg("quantile") = 0.5, py::arg("alpha") = 0.0, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4, py::arg("learning_rate") = 0.01)
        .def("fit", &ingenuityml::linear_model::QuantileRegressor::fit)
        .def("predict", &ingenuityml::linear_model::QuantileRegressor::predict)
        .def("get_params", &ingenuityml::linear_model::QuantileRegressor::get_params)
        .def("set_params", &ingenuityml::linear_model::QuantileRegressor::set_params)
        .def("is_fitted", &ingenuityml::linear_model::QuantileRegressor::is_fitted)
        .def("coef", &ingenuityml::linear_model::QuantileRegressor::coef)
        .def("intercept", &ingenuityml::linear_model::QuantileRegressor::intercept);

    py::class_<ingenuityml::linear_model::PoissonRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "PoissonRegressor")
        .def(py::init<double, bool, int, double, double>(),
             py::arg("alpha") = 0.0, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4, py::arg("learning_rate") = 0.01)
        .def("fit", &ingenuityml::linear_model::PoissonRegressor::fit)
        .def("predict", &ingenuityml::linear_model::PoissonRegressor::predict)
        .def("get_params", &ingenuityml::linear_model::PoissonRegressor::get_params)
        .def("set_params", &ingenuityml::linear_model::PoissonRegressor::set_params)
        .def("is_fitted", &ingenuityml::linear_model::PoissonRegressor::is_fitted)
        .def("coef", &ingenuityml::linear_model::PoissonRegressor::coef)
        .def("intercept", &ingenuityml::linear_model::PoissonRegressor::intercept);

    py::class_<ingenuityml::linear_model::GammaRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "GammaRegressor")
        .def(py::init<double, bool, int, double, double>(),
             py::arg("alpha") = 0.0, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4, py::arg("learning_rate") = 0.01)
        .def("fit", &ingenuityml::linear_model::GammaRegressor::fit)
        .def("predict", &ingenuityml::linear_model::GammaRegressor::predict)
        .def("get_params", &ingenuityml::linear_model::GammaRegressor::get_params)
        .def("set_params", &ingenuityml::linear_model::GammaRegressor::set_params)
        .def("is_fitted", &ingenuityml::linear_model::GammaRegressor::is_fitted)
        .def("coef", &ingenuityml::linear_model::GammaRegressor::coef)
        .def("intercept", &ingenuityml::linear_model::GammaRegressor::intercept);

    py::class_<ingenuityml::linear_model::TweedieRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "TweedieRegressor")
        .def(py::init<double, double, bool, int, double, double>(),
             py::arg("power") = 1.5, py::arg("alpha") = 0.0, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4, py::arg("learning_rate") = 0.01)
        .def("fit", &ingenuityml::linear_model::TweedieRegressor::fit)
        .def("predict", &ingenuityml::linear_model::TweedieRegressor::predict)
        .def("get_params", &ingenuityml::linear_model::TweedieRegressor::get_params)
        .def("set_params", &ingenuityml::linear_model::TweedieRegressor::set_params)
        .def("is_fitted", &ingenuityml::linear_model::TweedieRegressor::is_fitted)
        .def("coef", &ingenuityml::linear_model::TweedieRegressor::coef)
        .def("intercept", &ingenuityml::linear_model::TweedieRegressor::intercept);

    py::class_<ingenuityml::linear_model::MultiTaskLasso, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "MultiTaskLasso")
        .def(py::init<double, bool, int, double>(),
             py::arg("alpha") = 1.0, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::MultiTaskLasso::fit)
        .def("predict", &ingenuityml::linear_model::MultiTaskLasso::predict)
        .def("get_params", &ingenuityml::linear_model::MultiTaskLasso::get_params)
        .def("set_params", &ingenuityml::linear_model::MultiTaskLasso::set_params)
        .def("is_fitted", &ingenuityml::linear_model::MultiTaskLasso::is_fitted)
        .def("coef", &ingenuityml::linear_model::MultiTaskLasso::coef)
        .def("intercept", &ingenuityml::linear_model::MultiTaskLasso::intercept);

    py::class_<ingenuityml::linear_model::MultiTaskLassoCV, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "MultiTaskLassoCV")
        .def(py::init<const std::vector<double>&, int, bool, int, double>(),
             py::arg("alphas") = std::vector<double>{0.1, 1.0, 10.0}, py::arg("cv") = 5,
             py::arg("fit_intercept") = true, py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::MultiTaskLassoCV::fit)
        .def("predict", &ingenuityml::linear_model::MultiTaskLassoCV::predict)
        .def("get_params", &ingenuityml::linear_model::MultiTaskLassoCV::get_params)
        .def("set_params", &ingenuityml::linear_model::MultiTaskLassoCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::MultiTaskLassoCV::is_fitted)
        .def("best_alpha", &ingenuityml::linear_model::MultiTaskLassoCV::best_alpha)
        .def("coef", &ingenuityml::linear_model::MultiTaskLassoCV::coef)
        .def("intercept", &ingenuityml::linear_model::MultiTaskLassoCV::intercept);

    py::class_<ingenuityml::linear_model::MultiTaskElasticNet, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "MultiTaskElasticNet")
        .def(py::init<double, double, bool, int, double>(),
             py::arg("alpha") = 1.0, py::arg("l1_ratio") = 0.5, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::MultiTaskElasticNet::fit)
        .def("predict", &ingenuityml::linear_model::MultiTaskElasticNet::predict)
        .def("get_params", &ingenuityml::linear_model::MultiTaskElasticNet::get_params)
        .def("set_params", &ingenuityml::linear_model::MultiTaskElasticNet::set_params)
        .def("is_fitted", &ingenuityml::linear_model::MultiTaskElasticNet::is_fitted)
        .def("coef", &ingenuityml::linear_model::MultiTaskElasticNet::coef)
        .def("intercept", &ingenuityml::linear_model::MultiTaskElasticNet::intercept);

    py::class_<ingenuityml::linear_model::MultiTaskElasticNetCV, ingenuityml::Estimator, ingenuityml::Regressor>(linear_module, "MultiTaskElasticNetCV")
        .def(py::init<const std::vector<double>&, const std::vector<double>&, int, bool, int, double>(),
             py::arg("alphas") = std::vector<double>{0.1, 1.0, 10.0},
             py::arg("l1_ratios") = std::vector<double>{0.1, 0.5, 0.9},
             py::arg("cv") = 5, py::arg("fit_intercept") = true,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::linear_model::MultiTaskElasticNetCV::fit)
        .def("predict", &ingenuityml::linear_model::MultiTaskElasticNetCV::predict)
        .def("get_params", &ingenuityml::linear_model::MultiTaskElasticNetCV::get_params)
        .def("set_params", &ingenuityml::linear_model::MultiTaskElasticNetCV::set_params)
        .def("is_fitted", &ingenuityml::linear_model::MultiTaskElasticNetCV::is_fitted)
        .def("best_alpha", &ingenuityml::linear_model::MultiTaskElasticNetCV::best_alpha)
        .def("best_l1_ratio", &ingenuityml::linear_model::MultiTaskElasticNetCV::best_l1_ratio)
        .def("coef", &ingenuityml::linear_model::MultiTaskElasticNetCV::coef)
        .def("intercept", &ingenuityml::linear_model::MultiTaskElasticNetCV::intercept);
    
    // Neighbors module
    py::module_ neighbors_module = m.def_submodule("neighbors", "Nearest neighbors");
    
    py::class_<ingenuityml::neighbors::KNeighborsClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(neighbors_module, "KNeighborsClassifier")
        .def(py::init<int, std::string, std::string, std::string, double, int>(),
             py::arg("n_neighbors") = 5, py::arg("weights") = "uniform", py::arg("algorithm") = "auto",
             py::arg("metric") = "euclidean", py::arg("p") = 2, py::arg("n_jobs") = 1)
        .def("fit", &ingenuityml::neighbors::KNeighborsClassifier::fit)
        .def("predict", &ingenuityml::neighbors::KNeighborsClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::neighbors::KNeighborsClassifier::predict_proba)
        .def("decision_function", &ingenuityml::neighbors::KNeighborsClassifier::decision_function)
        .def("get_params", &ingenuityml::neighbors::KNeighborsClassifier::get_params)
        .def("set_params", &ingenuityml::neighbors::KNeighborsClassifier::set_params)
        .def("is_fitted", &ingenuityml::neighbors::KNeighborsClassifier::is_fitted)
        .def("save", &ingenuityml::neighbors::KNeighborsClassifier::save)
        .def("load", &ingenuityml::neighbors::KNeighborsClassifier::load);
    
    py::class_<ingenuityml::neighbors::KNeighborsRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(neighbors_module, "KNeighborsRegressor")
        .def(py::init<int, std::string, std::string, std::string, double, int>(),
             py::arg("n_neighbors") = 5, py::arg("weights") = "uniform", py::arg("algorithm") = "auto",
             py::arg("metric") = "euclidean", py::arg("p") = 2, py::arg("n_jobs") = 1)
        .def("fit", &ingenuityml::neighbors::KNeighborsRegressor::fit)
        .def("predict", &ingenuityml::neighbors::KNeighborsRegressor::predict)
        .def("get_params", &ingenuityml::neighbors::KNeighborsRegressor::get_params)
        .def("set_params", &ingenuityml::neighbors::KNeighborsRegressor::set_params)
        .def("is_fitted", &ingenuityml::neighbors::KNeighborsRegressor::is_fitted)
        .def("save", &ingenuityml::neighbors::KNeighborsRegressor::save)
        .def("load", &ingenuityml::neighbors::KNeighborsRegressor::load);

    auto nearest_neighbors_fit = [](ingenuityml::neighbors::NearestNeighbors& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::neighbors::NearestNeighbors& {
        ingenuityml::VectorXd y_dummy = ingenuityml::VectorXd::Zero(X.rows());
        if (!y_py.is_none()) {
            y_dummy = y_py.cast<ingenuityml::VectorXd>();
        }
        self.fit(X, y_dummy);
        return self;
    };

    py::class_<ingenuityml::neighbors::NearestNeighbors, ingenuityml::Estimator>(neighbors_module, "NearestNeighbors")
        .def(py::init<int, double, std::string, std::string, double, int>(),
             py::arg("n_neighbors") = 5, py::arg("radius") = 1.0, py::arg("algorithm") = "auto",
             py::arg("metric") = "euclidean", py::arg("p") = 2.0, py::arg("n_jobs") = 1)
        .def("fit", nearest_neighbors_fit, py::arg("X"), py::arg("y") = py::none())
        .def("kneighbors", &ingenuityml::neighbors::NearestNeighbors::kneighbors,
             py::arg("X"), py::arg("n_neighbors") = -1)
        .def("radius_neighbors", &ingenuityml::neighbors::NearestNeighbors::radius_neighbors,
             py::arg("X"), py::arg("radius") = -1.0)
        .def("get_params", &ingenuityml::neighbors::NearestNeighbors::get_params)
        .def("set_params", &ingenuityml::neighbors::NearestNeighbors::set_params)
        .def("is_fitted", &ingenuityml::neighbors::NearestNeighbors::is_fitted);

    py::class_<ingenuityml::neighbors::RadiusNeighborsClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(neighbors_module, "RadiusNeighborsClassifier")
        .def(py::init<double, std::string, std::string, std::string, double, int>(),
             py::arg("radius") = 1.0, py::arg("weights") = "uniform", py::arg("algorithm") = "auto",
             py::arg("metric") = "euclidean", py::arg("p") = 2, py::arg("n_jobs") = 1)
        .def("fit", &ingenuityml::neighbors::RadiusNeighborsClassifier::fit)
        .def("predict", &ingenuityml::neighbors::RadiusNeighborsClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::neighbors::RadiusNeighborsClassifier::predict_proba)
        .def("decision_function", &ingenuityml::neighbors::RadiusNeighborsClassifier::decision_function)
        .def("get_params", &ingenuityml::neighbors::RadiusNeighborsClassifier::get_params)
        .def("set_params", &ingenuityml::neighbors::RadiusNeighborsClassifier::set_params)
        .def("is_fitted", &ingenuityml::neighbors::RadiusNeighborsClassifier::is_fitted);

    py::class_<ingenuityml::neighbors::RadiusNeighborsRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(neighbors_module, "RadiusNeighborsRegressor")
        .def(py::init<double, std::string, std::string, std::string, double, int>(),
             py::arg("radius") = 1.0, py::arg("weights") = "uniform", py::arg("algorithm") = "auto",
             py::arg("metric") = "euclidean", py::arg("p") = 2, py::arg("n_jobs") = 1)
        .def("fit", &ingenuityml::neighbors::RadiusNeighborsRegressor::fit)
        .def("predict", &ingenuityml::neighbors::RadiusNeighborsRegressor::predict)
        .def("get_params", &ingenuityml::neighbors::RadiusNeighborsRegressor::get_params)
        .def("set_params", &ingenuityml::neighbors::RadiusNeighborsRegressor::set_params)
        .def("is_fitted", &ingenuityml::neighbors::RadiusNeighborsRegressor::is_fitted);

    auto knn_transformer_fit = [](ingenuityml::neighbors::KNeighborsTransformer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::neighbors::KNeighborsTransformer& {
        ingenuityml::VectorXd y_dummy = ingenuityml::VectorXd::Zero(X.rows());
        if (!y_py.is_none()) {
            y_dummy = y_py.cast<ingenuityml::VectorXd>();
        }
        self.fit(X, y_dummy);
        return self;
    };
    auto knn_transformer_fit_transform = [](ingenuityml::neighbors::KNeighborsTransformer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) {
        ingenuityml::VectorXd y_dummy = ingenuityml::VectorXd::Zero(X.rows());
        if (!y_py.is_none()) {
            y_dummy = y_py.cast<ingenuityml::VectorXd>();
        }
        return self.fit_transform(X, y_dummy);
    };

    py::class_<ingenuityml::neighbors::KNeighborsTransformer, ingenuityml::Estimator, ingenuityml::Transformer>(neighbors_module, "KNeighborsTransformer")
        .def(py::init<int, std::string, std::string, double, int>(),
             py::arg("n_neighbors") = 5, py::arg("mode") = "distance",
             py::arg("metric") = "euclidean", py::arg("p") = 2.0, py::arg("n_jobs") = 1)
        .def("fit", knn_transformer_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &ingenuityml::neighbors::KNeighborsTransformer::transform)
        .def("inverse_transform", &ingenuityml::neighbors::KNeighborsTransformer::inverse_transform)
        .def("fit_transform", knn_transformer_fit_transform, py::arg("X"), py::arg("y") = py::none())
        .def("get_params", &ingenuityml::neighbors::KNeighborsTransformer::get_params)
        .def("set_params", &ingenuityml::neighbors::KNeighborsTransformer::set_params)
        .def("is_fitted", &ingenuityml::neighbors::KNeighborsTransformer::is_fitted);

    auto radius_transformer_fit = [](ingenuityml::neighbors::RadiusNeighborsTransformer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::neighbors::RadiusNeighborsTransformer& {
        ingenuityml::VectorXd y_dummy = ingenuityml::VectorXd::Zero(X.rows());
        if (!y_py.is_none()) {
            y_dummy = y_py.cast<ingenuityml::VectorXd>();
        }
        self.fit(X, y_dummy);
        return self;
    };
    auto radius_transformer_fit_transform = [](ingenuityml::neighbors::RadiusNeighborsTransformer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) {
        ingenuityml::VectorXd y_dummy = ingenuityml::VectorXd::Zero(X.rows());
        if (!y_py.is_none()) {
            y_dummy = y_py.cast<ingenuityml::VectorXd>();
        }
        return self.fit_transform(X, y_dummy);
    };

    py::class_<ingenuityml::neighbors::RadiusNeighborsTransformer, ingenuityml::Estimator, ingenuityml::Transformer>(neighbors_module, "RadiusNeighborsTransformer")
        .def(py::init<double, std::string, std::string, double, int>(),
             py::arg("radius") = 1.0, py::arg("mode") = "distance",
             py::arg("metric") = "euclidean", py::arg("p") = 2.0, py::arg("n_jobs") = 1)
        .def("fit", radius_transformer_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &ingenuityml::neighbors::RadiusNeighborsTransformer::transform)
        .def("inverse_transform", &ingenuityml::neighbors::RadiusNeighborsTransformer::inverse_transform)
        .def("fit_transform", radius_transformer_fit_transform, py::arg("X"), py::arg("y") = py::none())
        .def("get_params", &ingenuityml::neighbors::RadiusNeighborsTransformer::get_params)
        .def("set_params", &ingenuityml::neighbors::RadiusNeighborsTransformer::set_params)
        .def("is_fitted", &ingenuityml::neighbors::RadiusNeighborsTransformer::is_fitted);

    py::class_<ingenuityml::neighbors::NearestCentroid, ingenuityml::Estimator, ingenuityml::Classifier>(m, "NearestCentroid")
        .def(py::init<std::string, double>(), py::arg("metric") = "euclidean", py::arg("p") = 2.0)
        .def("fit", &ingenuityml::neighbors::NearestCentroid::fit)
        .def("predict", &ingenuityml::neighbors::NearestCentroid::predict_classes)
        .def("predict_proba", &ingenuityml::neighbors::NearestCentroid::predict_proba)
        .def("decision_function", &ingenuityml::neighbors::NearestCentroid::decision_function)
        .def("get_params", &ingenuityml::neighbors::NearestCentroid::get_params)
        .def("set_params", &ingenuityml::neighbors::NearestCentroid::set_params)
        .def("is_fitted", &ingenuityml::neighbors::NearestCentroid::is_fitted);
    
    // Tree module
    py::module_ tree_module = m.def_submodule("tree", "Decision trees");
    
    py::class_<ingenuityml::tree::DecisionTreeClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(tree_module, "DecisionTreeClassifier")
        .def(py::init<std::string, int, int, int, double>(),
             py::arg("criterion") = "gini", py::arg("max_depth") = -1, py::arg("min_samples_split") = 2,
             py::arg("min_samples_leaf") = 1, py::arg("min_impurity_decrease") = 0.0)
        .def("fit", &ingenuityml::tree::DecisionTreeClassifier::fit)
        .def("predict", &ingenuityml::tree::DecisionTreeClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::tree::DecisionTreeClassifier::predict_proba)
        .def("decision_function", &ingenuityml::tree::DecisionTreeClassifier::decision_function)
        .def("get_params", &ingenuityml::tree::DecisionTreeClassifier::get_params)
        .def("set_params", &ingenuityml::tree::DecisionTreeClassifier::set_params)
        .def("is_fitted", &ingenuityml::tree::DecisionTreeClassifier::is_fitted)
        .def("save", &ingenuityml::tree::DecisionTreeClassifier::save)
        .def("load", &ingenuityml::tree::DecisionTreeClassifier::load);
    
    py::class_<ingenuityml::tree::DecisionTreeRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(tree_module, "DecisionTreeRegressor")
        .def(py::init<std::string, int, int, int, double>(),
             py::arg("criterion") = "mse", py::arg("max_depth") = -1, py::arg("min_samples_split") = 2,
             py::arg("min_samples_leaf") = 1, py::arg("min_impurity_decrease") = 0.0)
        .def("fit", &ingenuityml::tree::DecisionTreeRegressor::fit)
        .def("predict", &ingenuityml::tree::DecisionTreeRegressor::predict)
        .def("get_params", &ingenuityml::tree::DecisionTreeRegressor::get_params)
        .def("set_params", &ingenuityml::tree::DecisionTreeRegressor::set_params)
        .def("is_fitted", &ingenuityml::tree::DecisionTreeRegressor::is_fitted)
        .def("save", &ingenuityml::tree::DecisionTreeRegressor::save)
        .def("load", &ingenuityml::tree::DecisionTreeRegressor::load);
    
    // Metrics module
    py::module_ metrics_module = m.def_submodule("metrics", "Evaluation metrics");
    
    // Classification metrics
    metrics_module.def("accuracy_score", &ingenuityml::metrics::accuracy_score);
    metrics_module.def("balanced_accuracy_score", &ingenuityml::metrics::balanced_accuracy_score);
    metrics_module.def("top_k_accuracy_score", &ingenuityml::metrics::top_k_accuracy_score, py::arg("y_true"), py::arg("y_score"), py::arg("k") = 5);
    metrics_module.def("roc_auc_score", 
        [](const ingenuityml::VectorXi& y_true, const ingenuityml::VectorXd& y_score) {
            return ingenuityml::metrics::roc_auc_score(y_true, y_score);
        }, py::arg("y_true"), py::arg("y_score"));
    metrics_module.def("roc_auc_score_multiclass", &ingenuityml::metrics::roc_auc_score_multiclass, py::arg("y_true"), py::arg("y_score"), py::arg("average") = "macro");
    metrics_module.def("average_precision_score", &ingenuityml::metrics::average_precision_score);
    metrics_module.def("log_loss", &ingenuityml::metrics::log_loss);
    metrics_module.def("hinge_loss", &ingenuityml::metrics::hinge_loss);
    metrics_module.def("cohen_kappa_score", &ingenuityml::metrics::cohen_kappa_score);
    metrics_module.def("matthews_corrcoef", &ingenuityml::metrics::matthews_corrcoef);
    metrics_module.def("hamming_loss", &ingenuityml::metrics::hamming_loss);
    metrics_module.def("jaccard_score", &ingenuityml::metrics::jaccard_score, py::arg("y_true"), py::arg("y_pred"), py::arg("average") = "macro");
    metrics_module.def("zero_one_loss", &ingenuityml::metrics::zero_one_loss);
    metrics_module.def("brier_score_loss", &ingenuityml::metrics::brier_score_loss);
    metrics_module.def("precision_score", &ingenuityml::metrics::precision_score, py::arg("y_true"), py::arg("y_pred"), py::arg("average") = "macro");
    metrics_module.def("recall_score", &ingenuityml::metrics::recall_score, py::arg("y_true"), py::arg("y_pred"), py::arg("average") = "macro");
    metrics_module.def("f1_score", &ingenuityml::metrics::f1_score, py::arg("y_true"), py::arg("y_pred"), py::arg("average") = "macro");
    metrics_module.def("confusion_matrix", &ingenuityml::metrics::confusion_matrix);
    metrics_module.def("classification_report", &ingenuityml::metrics::classification_report);
    
    // Regression metrics
    metrics_module.def("mean_squared_error", &ingenuityml::metrics::mean_squared_error);
    metrics_module.def("root_mean_squared_error", &ingenuityml::metrics::root_mean_squared_error);
    metrics_module.def("mean_absolute_error", &ingenuityml::metrics::mean_absolute_error);
    metrics_module.def("median_absolute_error", &ingenuityml::metrics::median_absolute_error);
    metrics_module.def("max_error", &ingenuityml::metrics::max_error);
    metrics_module.def("mean_poisson_deviance", &ingenuityml::metrics::mean_poisson_deviance);
    metrics_module.def("mean_gamma_deviance", &ingenuityml::metrics::mean_gamma_deviance);
    metrics_module.def("mean_tweedie_deviance", &ingenuityml::metrics::mean_tweedie_deviance, py::arg("y_true"), py::arg("y_pred"), py::arg("power") = 0.0);
    metrics_module.def("d2_tweedie_score", &ingenuityml::metrics::d2_tweedie_score, py::arg("y_true"), py::arg("y_pred"), py::arg("power") = 0.0);
    metrics_module.def("d2_pinball_score", &ingenuityml::metrics::d2_pinball_score, py::arg("y_true"), py::arg("y_pred"), py::arg("alpha") = 0.5);
    metrics_module.def("d2_absolute_error_score", &ingenuityml::metrics::d2_absolute_error_score);
    metrics_module.def("r2_score", &ingenuityml::metrics::r2_score);
    metrics_module.def("explained_variance_score", &ingenuityml::metrics::explained_variance_score);
    metrics_module.def("mean_absolute_percentage_error", &ingenuityml::metrics::mean_absolute_percentage_error);
    
    // Clustering metrics
    metrics_module.def("silhouette_score", &ingenuityml::metrics::silhouette_score);
    metrics_module.def("silhouette_samples", &ingenuityml::metrics::silhouette_samples);
    metrics_module.def("calinski_harabasz_score", &ingenuityml::metrics::calinski_harabasz_score);
    metrics_module.def("davies_bouldin_score", &ingenuityml::metrics::davies_bouldin_score);
    
    // Clustering comparison metrics
    metrics_module.def("adjusted_rand_score", &ingenuityml::metrics::adjusted_rand_score);
    metrics_module.def("adjusted_mutual_info_score", &ingenuityml::metrics::adjusted_mutual_info_score);
    metrics_module.def("normalized_mutual_info_score", &ingenuityml::metrics::normalized_mutual_info_score);
    metrics_module.def("homogeneity_score", &ingenuityml::metrics::homogeneity_score);
    metrics_module.def("completeness_score", &ingenuityml::metrics::completeness_score);
    metrics_module.def("v_measure_score", &ingenuityml::metrics::v_measure_score);
    metrics_module.def("fowlkes_mallows_score", &ingenuityml::metrics::fowlkes_mallows_score);
    
    // Preprocessing module
    py::module_ preprocessing_module = m.def_submodule("preprocessing", "Data preprocessing");
    
    py::class_<ingenuityml::preprocessing::StandardScaler, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "StandardScaler")
        .def(py::init<bool, bool>(), py::arg("with_mean") = true, py::arg("with_std") = true)
        .def("fit", &ingenuityml::preprocessing::StandardScaler::fit)
        .def("transform", &ingenuityml::preprocessing::StandardScaler::transform)
        .def("transform", [](ingenuityml::preprocessing::StandardScaler& self, const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y) {
            (void)y;
            return self.transform(X);
        })
        .def("inverse_transform", &ingenuityml::preprocessing::StandardScaler::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::StandardScaler::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::StandardScaler::get_params)
        .def("set_params", &ingenuityml::preprocessing::StandardScaler::set_params)
        .def("set_params", [](ingenuityml::preprocessing::StandardScaler& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::preprocessing::StandardScaler::is_fitted)
        .def("mean", &ingenuityml::preprocessing::StandardScaler::mean)
        .def("scale", &ingenuityml::preprocessing::StandardScaler::scale)
        .def_property_readonly("mean_", &ingenuityml::preprocessing::StandardScaler::mean)
        .def_property_readonly("scale_", &ingenuityml::preprocessing::StandardScaler::scale)
        .def("save", &ingenuityml::preprocessing::StandardScaler::save)
        .def("load", &ingenuityml::preprocessing::StandardScaler::load);
    
    py::class_<ingenuityml::preprocessing::MinMaxScaler, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "MinMaxScaler")
        .def(py::init<double, double>(), py::arg("feature_range_min") = 0.0, py::arg("feature_range_max") = 1.0)
        .def("fit", &ingenuityml::preprocessing::MinMaxScaler::fit)
        .def("transform", &ingenuityml::preprocessing::MinMaxScaler::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::MinMaxScaler::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::MinMaxScaler::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::MinMaxScaler::get_params)
        .def("set_params", &ingenuityml::preprocessing::MinMaxScaler::set_params)
        .def("set_params", [](ingenuityml::preprocessing::MinMaxScaler& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::preprocessing::MinMaxScaler::is_fitted)
        .def("data_min", &ingenuityml::preprocessing::MinMaxScaler::data_min)
        .def("data_max", &ingenuityml::preprocessing::MinMaxScaler::data_max)
        .def("scale", &ingenuityml::preprocessing::MinMaxScaler::scale)
        .def("min", &ingenuityml::preprocessing::MinMaxScaler::min);

    py::class_<ingenuityml::preprocessing::RobustScaler, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "RobustScaler")
        .def(py::init<bool, bool>(), py::arg("with_centering") = true, py::arg("with_scaling") = true)
        .def("fit", &ingenuityml::preprocessing::RobustScaler::fit)
        .def("transform", &ingenuityml::preprocessing::RobustScaler::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::RobustScaler::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::RobustScaler::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::RobustScaler::get_params)
        .def("set_params", &ingenuityml::preprocessing::RobustScaler::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::RobustScaler::is_fitted);
    
    py::class_<ingenuityml::preprocessing::LabelEncoder, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "LabelEncoder")
        .def(py::init<>())
        .def("fit", &ingenuityml::preprocessing::LabelEncoder::fit)
        .def("transform", static_cast<ingenuityml::MatrixXd (ingenuityml::preprocessing::LabelEncoder::*)(const ingenuityml::MatrixXd&) const>(&ingenuityml::preprocessing::LabelEncoder::transform))
        .def("inverse_transform", static_cast<ingenuityml::MatrixXd (ingenuityml::preprocessing::LabelEncoder::*)(const ingenuityml::MatrixXd&) const>(&ingenuityml::preprocessing::LabelEncoder::inverse_transform))
        .def("fit_transform", &ingenuityml::preprocessing::LabelEncoder::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::LabelEncoder::get_params)
        .def("set_params", &ingenuityml::preprocessing::LabelEncoder::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::LabelEncoder::is_fitted)
        .def("n_classes", &ingenuityml::preprocessing::LabelEncoder::n_classes);

    py::class_<ingenuityml::preprocessing::OneHotEncoder, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "OneHotEncoder")
        .def(py::init<>())
        .def("fit", &ingenuityml::preprocessing::OneHotEncoder::fit)
        .def("transform", &ingenuityml::preprocessing::OneHotEncoder::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::OneHotEncoder::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::OneHotEncoder::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::OneHotEncoder::get_params)
        .def("set_params", &ingenuityml::preprocessing::OneHotEncoder::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::OneHotEncoder::is_fitted);

    py::class_<ingenuityml::preprocessing::OrdinalEncoder, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "OrdinalEncoder")
        .def(py::init<>())
        .def("fit", &ingenuityml::preprocessing::OrdinalEncoder::fit)
        .def("transform", &ingenuityml::preprocessing::OrdinalEncoder::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::OrdinalEncoder::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::OrdinalEncoder::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::OrdinalEncoder::get_params)
        .def("set_params", &ingenuityml::preprocessing::OrdinalEncoder::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::OrdinalEncoder::is_fitted)
        .def("categories", &ingenuityml::preprocessing::OrdinalEncoder::categories);
    
    py::class_<ingenuityml::preprocessing::Normalizer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "Normalizer")
        .def(py::init<const std::string&>(), py::arg("norm") = "l2")
        .def("fit", &ingenuityml::preprocessing::Normalizer::fit)
        .def("transform", &ingenuityml::preprocessing::Normalizer::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::Normalizer::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::Normalizer::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::Normalizer::get_params)
        .def("set_params", &ingenuityml::preprocessing::Normalizer::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::Normalizer::is_fitted);
    
    py::class_<ingenuityml::preprocessing::PolynomialFeatures, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "PolynomialFeatures")
        .def(py::init<int, bool, bool>(), py::arg("degree") = 2, py::arg("interaction_only") = false, py::arg("include_bias") = true)
        .def("fit", &ingenuityml::preprocessing::PolynomialFeatures::fit)
        .def("transform", &ingenuityml::preprocessing::PolynomialFeatures::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::PolynomialFeatures::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::PolynomialFeatures::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::PolynomialFeatures::get_params)
        .def("set_params", &ingenuityml::preprocessing::PolynomialFeatures::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::PolynomialFeatures::is_fitted)
        .def("n_input_features", &ingenuityml::preprocessing::PolynomialFeatures::n_input_features)
        .def("n_output_features", &ingenuityml::preprocessing::PolynomialFeatures::n_output_features);
    
    py::class_<ingenuityml::preprocessing::SimpleImputer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "SimpleImputer")
        .def(py::init<const std::string&, double>(), py::arg("strategy") = "mean", py::arg("fill_value") = 0.0)
        .def("fit", &ingenuityml::preprocessing::SimpleImputer::fit)
        .def("transform", &ingenuityml::preprocessing::SimpleImputer::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::SimpleImputer::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::SimpleImputer::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::SimpleImputer::get_params)
        .def("set_params", &ingenuityml::preprocessing::SimpleImputer::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::SimpleImputer::is_fitted)
        .def("statistics", &ingenuityml::preprocessing::SimpleImputer::statistics);
    
    // Model selection module
    py::module_ model_selection_module = m.def_submodule("model_selection", "Model selection utilities");
    
    // Base cross validator class
    py::class_<ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "BaseCrossValidator")
        .def("split", &ingenuityml::model_selection::BaseCrossValidator::split)
        .def("get_n_splits", &ingenuityml::model_selection::BaseCrossValidator::get_n_splits);

    auto param_map_from_py = [](const py::dict& param_dict) {
        std::map<std::string, std::vector<std::string>> params;
        for (auto item : param_dict) {
            std::string key = py::cast<std::string>(item.first);
            py::iterable values = py::cast<py::iterable>(item.second);
            std::vector<std::string> vals;
            for (auto v : values) {
                vals.push_back(py::str(v));
            }
            params[key] = vals;
        }
        return params;
    };

    py::class_<ingenuityml::model_selection::ParameterGrid>(model_selection_module, "ParameterGrid")
        .def(py::init([&](const py::dict& param_grid) {
            return new ingenuityml::model_selection::ParameterGrid(param_map_from_py(param_grid));
        }), py::arg("param_grid"))
        .def("grid", &ingenuityml::model_selection::ParameterGrid::grid)
        .def("size", &ingenuityml::model_selection::ParameterGrid::size);

    py::class_<ingenuityml::model_selection::ParameterSampler>(model_selection_module, "ParameterSampler")
        .def(py::init([&](const py::dict& param_distributions, int n_iter, int random_state) {
            return new ingenuityml::model_selection::ParameterSampler(param_map_from_py(param_distributions), n_iter, random_state);
        }), py::arg("param_distributions"), py::arg("n_iter") = 10, py::arg("random_state") = -1)
        .def("samples", &ingenuityml::model_selection::ParameterSampler::samples)
        .def("size", &ingenuityml::model_selection::ParameterSampler::size);
    
    model_selection_module.def("train_test_split", 
        [](const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y, double test_size, double train_size, 
           int random_state, bool shuffle, const ingenuityml::VectorXd& stratify) {
            return ingenuityml::model_selection::train_test_split(X, y, test_size, train_size, random_state, shuffle, stratify);
        },
        py::arg("X"), py::arg("y"), py::arg("test_size") = 0.25, py::arg("train_size") = -1,
        py::arg("random_state") = -1, py::arg("shuffle") = true, py::arg("stratify") = ingenuityml::VectorXd());
    
    py::class_<ingenuityml::model_selection::KFold, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "KFold")
        .def(py::init<int, bool, int>(), py::arg("n_splits") = 5, py::arg("shuffle") = false, py::arg("random_state") = -1)
        .def("split", [](const ingenuityml::model_selection::KFold& self, const ingenuityml::MatrixXd& X) {
            return self.split(X);
        })
        .def("split", [](const ingenuityml::model_selection::KFold& self, const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y) {
            return self.split(X, y);
        })
        .def("get_n_splits", &ingenuityml::model_selection::KFold::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::KFold::get_params)
        .def("set_params", &ingenuityml::model_selection::KFold::set_params);

    py::class_<ingenuityml::model_selection::StratifiedKFold, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "StratifiedKFold")
        .def(py::init<int, bool, int>(), py::arg("n_splits") = 5, py::arg("shuffle") = false, py::arg("random_state") = -1)
        .def("split", [](const ingenuityml::model_selection::StratifiedKFold& self, const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y) {
            return self.split(X, y);
        })
        .def("get_n_splits", &ingenuityml::model_selection::StratifiedKFold::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::StratifiedKFold::get_params)
        .def("set_params", &ingenuityml::model_selection::StratifiedKFold::set_params);

    py::class_<ingenuityml::model_selection::GroupKFold>(model_selection_module, "GroupKFold")
        .def(py::init<int>(), py::arg("n_splits") = 5)
        .def("split", [](const ingenuityml::model_selection::GroupKFold& self, const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y, const ingenuityml::VectorXd& groups) {
            return self.split(X, y, groups);
        })
        .def("get_n_splits", &ingenuityml::model_selection::GroupKFold::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::GroupKFold::get_params)
        .def("set_params", &ingenuityml::model_selection::GroupKFold::set_params);

    py::class_<ingenuityml::model_selection::RepeatedKFold, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "RepeatedKFold")
        .def(py::init<int, int, int>(), py::arg("n_splits") = 5, py::arg("n_repeats") = 10, py::arg("random_state") = -1)
        .def("split", [](const ingenuityml::model_selection::RepeatedKFold& self, const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y) {
            return self.split(X, y);
        })
        .def("get_n_splits", &ingenuityml::model_selection::RepeatedKFold::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::RepeatedKFold::get_params)
        .def("set_params", &ingenuityml::model_selection::RepeatedKFold::set_params);

    py::class_<ingenuityml::model_selection::RepeatedStratifiedKFold, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "RepeatedStratifiedKFold")
        .def(py::init<int, int, int>(), py::arg("n_splits") = 5, py::arg("n_repeats") = 10, py::arg("random_state") = -1)
        .def("split", [](const ingenuityml::model_selection::RepeatedStratifiedKFold& self, const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y) {
            return self.split(X, y);
        })
        .def("get_n_splits", &ingenuityml::model_selection::RepeatedStratifiedKFold::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::RepeatedStratifiedKFold::get_params)
        .def("set_params", &ingenuityml::model_selection::RepeatedStratifiedKFold::set_params);

    py::class_<ingenuityml::model_selection::ShuffleSplit, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "ShuffleSplit")
        .def(py::init<int, double, double, int>(), py::arg("n_splits") = 10, py::arg("test_size") = 0.1, py::arg("train_size") = -1.0, py::arg("random_state") = -1)
        .def("split", [](const ingenuityml::model_selection::ShuffleSplit& self, const ingenuityml::MatrixXd& X) {
            return self.split(X);
        })
        .def("get_n_splits", &ingenuityml::model_selection::ShuffleSplit::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::ShuffleSplit::get_params)
        .def("set_params", &ingenuityml::model_selection::ShuffleSplit::set_params);

    py::class_<ingenuityml::model_selection::StratifiedShuffleSplit, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "StratifiedShuffleSplit")
        .def(py::init<int, double, double, int>(), py::arg("n_splits") = 10, py::arg("test_size") = 0.1, py::arg("train_size") = -1.0, py::arg("random_state") = -1)
        .def("split", [](const ingenuityml::model_selection::StratifiedShuffleSplit& self, const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y) {
            return self.split(X, y);
        })
        .def("get_n_splits", &ingenuityml::model_selection::StratifiedShuffleSplit::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::StratifiedShuffleSplit::get_params)
        .def("set_params", &ingenuityml::model_selection::StratifiedShuffleSplit::set_params);

    py::class_<ingenuityml::model_selection::GroupShuffleSplit>(model_selection_module, "GroupShuffleSplit")
        .def(py::init<int, double, double, int>(), py::arg("n_splits") = 5, py::arg("test_size") = 0.2, py::arg("train_size") = -1.0, py::arg("random_state") = -1)
        .def("split", [](const ingenuityml::model_selection::GroupShuffleSplit& self, const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y, const ingenuityml::VectorXd& groups) {
            return self.split(X, y, groups);
        })
        .def("get_n_splits", &ingenuityml::model_selection::GroupShuffleSplit::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::GroupShuffleSplit::get_params)
        .def("set_params", &ingenuityml::model_selection::GroupShuffleSplit::set_params);

    py::class_<ingenuityml::model_selection::PredefinedSplit, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "PredefinedSplit")
        .def(py::init<const std::vector<int>&>(), py::arg("test_fold"))
        .def("split", [](const ingenuityml::model_selection::PredefinedSplit& self, const ingenuityml::MatrixXd& X) {
            return self.split(X);
        })
        .def("get_n_splits", &ingenuityml::model_selection::PredefinedSplit::get_n_splits)
        .def("test_fold", &ingenuityml::model_selection::PredefinedSplit::test_fold);

    py::class_<ingenuityml::model_selection::LeaveOneOut, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "LeaveOneOut")
        .def(py::init<>())
        .def("split", [](const ingenuityml::model_selection::LeaveOneOut& self, const ingenuityml::MatrixXd& X) {
            return self.split(X);
        })
        .def("get_n_splits", &ingenuityml::model_selection::LeaveOneOut::get_n_splits);

    py::class_<ingenuityml::model_selection::LeavePOut, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "LeavePOut")
        .def(py::init<int>(), py::arg("p") = 2)
        .def("split", [](const ingenuityml::model_selection::LeavePOut& self, const ingenuityml::MatrixXd& X) {
            return self.split(X);
        })
        .def("get_n_splits", &ingenuityml::model_selection::LeavePOut::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::LeavePOut::get_params)
        .def("set_params", &ingenuityml::model_selection::LeavePOut::set_params);
    
    // Helper function for cross_val_score
    auto cross_val_score_wrapper = [](py::object estimator, py::array_t<double> X_array, py::array_t<double> y_array, 
                                      py::object cv, const std::string& scoring) {
        // Convert numpy arrays to Eigen types
        py::buffer_info X_buf = X_array.request();
        py::buffer_info y_buf = y_array.request();
        
        if (X_buf.ndim != 2) {
            throw std::runtime_error("X must be a 2D array");
        }
        if (y_buf.ndim != 1) {
            throw std::runtime_error("y must be a 1D array");
        }
        
        ingenuityml::MatrixXd X = Eigen::Map<ingenuityml::MatrixXd>(static_cast<double*>(X_buf.ptr), X_buf.shape[0], X_buf.shape[1]);
        ingenuityml::VectorXd y = Eigen::Map<ingenuityml::VectorXd>(static_cast<double*>(y_buf.ptr), y_buf.shape[0]);
        
        // Cast estimator to the correct type
        ingenuityml::Estimator& est = estimator.cast<ingenuityml::Estimator&>();
        const ingenuityml::model_selection::BaseCrossValidator& cv_obj = cv.cast<const ingenuityml::model_selection::BaseCrossValidator&>();
        
        // Call the C++ function
        ingenuityml::VectorXd scores = ingenuityml::model_selection::cross_val_score(est, X, y, cv_obj, scoring);
        
        // Convert back to numpy array
        return py::array_t<double>(scores.size(), scores.data());
    };
    
    model_selection_module.def("cross_val_score", cross_val_score_wrapper,
        py::arg("estimator"), py::arg("X"), py::arg("y"), py::arg("cv"), py::arg("scoring") = "accuracy");

    model_selection_module.attr("LogisticRegressionCV") = linear_module.attr("LogisticRegressionCV");
    
    // Naive Bayes submodule
    py::module_ nb_module = m.def_submodule("naive_bayes", "Naive Bayes algorithms");
    py::class_<ingenuityml::naive_bayes::GaussianNB, ingenuityml::Classifier, ingenuityml::Estimator>(nb_module, "GaussianNB")
        .def(py::init<double>(), py::arg("var_smoothing") = 1e-9)
        .def("fit", &ingenuityml::naive_bayes::GaussianNB::fit)
        .def("predict", &ingenuityml::naive_bayes::GaussianNB::predict_classes)
        .def("predict_classes", &ingenuityml::naive_bayes::GaussianNB::predict_classes)
        .def("predict_proba", &ingenuityml::naive_bayes::GaussianNB::predict_proba)
        .def("decision_function", &ingenuityml::naive_bayes::GaussianNB::decision_function)
        .def("get_params", &ingenuityml::naive_bayes::GaussianNB::get_params)
        .def("set_params", &ingenuityml::naive_bayes::GaussianNB::set_params)
        .def("is_fitted", &ingenuityml::naive_bayes::GaussianNB::is_fitted)
        .def("save", &ingenuityml::naive_bayes::GaussianNB::save)
        .def("load", &ingenuityml::naive_bayes::GaussianNB::load);

    // Clustering
    py::module_ cluster_module = m.def_submodule("cluster", "Clustering algorithms");
    py::class_<ingenuityml::cluster::KMeans, ingenuityml::Estimator, ingenuityml::Transformer>(cluster_module, "KMeans")
        .def(py::init<int,int,double,const std::string&,int>(), py::arg("n_clusters")=8, py::arg("max_iter")=300, py::arg("tol")=1e-4, py::arg("init")="k-means++", py::arg("random_state")=-1)
        .def("fit", &ingenuityml::cluster::KMeans::fit)
        .def("transform", &ingenuityml::cluster::KMeans::transform)
        .def("inverse_transform", &ingenuityml::cluster::KMeans::inverse_transform)
        .def("fit_transform", &ingenuityml::cluster::KMeans::fit_transform)
        .def("predict_labels", &ingenuityml::cluster::KMeans::predict_labels)
        .def("cluster_centers", &ingenuityml::cluster::KMeans::cluster_centers)
        .def("inertia", &ingenuityml::cluster::KMeans::inertia)
        .def("get_params", &ingenuityml::cluster::KMeans::get_params)
        .def("set_params", &ingenuityml::cluster::KMeans::set_params)
        .def("is_fitted", &ingenuityml::cluster::KMeans::is_fitted);

    py::class_<ingenuityml::cluster::DBSCAN, ingenuityml::Estimator>(cluster_module, "DBSCAN")
        .def(py::init<double,int>(), py::arg("eps")=0.5, py::arg("min_samples")=5)
        .def("fit", &ingenuityml::cluster::DBSCAN::fit)
        .def("fit_predict", &ingenuityml::cluster::DBSCAN::fit_predict)
        .def("get_params", &ingenuityml::cluster::DBSCAN::get_params)
        .def("set_params", &ingenuityml::cluster::DBSCAN::set_params)
        .def("is_fitted", &ingenuityml::cluster::DBSCAN::is_fitted)
        .def("labels", &ingenuityml::cluster::DBSCAN::labels);

    py::class_<ingenuityml::cluster::AgglomerativeClustering, ingenuityml::Estimator>(cluster_module, "AgglomerativeClustering")
        .def(py::init<int,const std::string&,const std::string&>(), py::arg("n_clusters")=2, py::arg("linkage")="single", py::arg("affinity")="euclidean")
        .def("fit", &ingenuityml::cluster::AgglomerativeClustering::fit)
        .def("get_params", &ingenuityml::cluster::AgglomerativeClustering::get_params)
        .def("set_params", &ingenuityml::cluster::AgglomerativeClustering::set_params)
        .def("is_fitted", &ingenuityml::cluster::AgglomerativeClustering::is_fitted)
        .def("labels", &ingenuityml::cluster::AgglomerativeClustering::labels);

    // Decomposition
    py::module_ decomp_module = m.def_submodule("decomposition", "Decomposition algorithms");
    py::class_<ingenuityml::decomposition::PCA, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "PCA")
        .def(py::init<int,bool>(), py::arg("n_components")=-1, py::arg("whiten")=false)
        .def("fit", &ingenuityml::decomposition::PCA::fit)
        .def("transform", &ingenuityml::decomposition::PCA::transform)
        .def("inverse_transform", &ingenuityml::decomposition::PCA::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::PCA::fit_transform)
        .def("get_params", &ingenuityml::decomposition::PCA::get_params)
        .def("set_params", &ingenuityml::decomposition::PCA::set_params)
        .def("set_params", [](ingenuityml::decomposition::PCA& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::decomposition::PCA::is_fitted);

    py::class_<ingenuityml::decomposition::KernelPCA, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "KernelPCA")
        .def(py::init<int, std::string, double, double, double>(),
             py::arg("n_components") = -1, py::arg("kernel") = "rbf",
             py::arg("gamma") = 1.0, py::arg("degree") = 3.0, py::arg("coef0") = 1.0)
        .def("fit", &ingenuityml::decomposition::KernelPCA::fit)
        .def("transform", &ingenuityml::decomposition::KernelPCA::transform)
        .def("inverse_transform", &ingenuityml::decomposition::KernelPCA::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::KernelPCA::fit_transform)
        .def("get_params", &ingenuityml::decomposition::KernelPCA::get_params)
        .def("set_params", &ingenuityml::decomposition::KernelPCA::set_params)
        .def("is_fitted", &ingenuityml::decomposition::KernelPCA::is_fitted);

    py::class_<ingenuityml::decomposition::TSNE, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "TSNE")
        .def(py::init<int, double, double, double, int, int>(),
             py::arg("n_components") = 2, py::arg("perplexity") = 30.0,
             py::arg("early_exaggeration") = 12.0, py::arg("learning_rate") = 200.0,
             py::arg("max_iter") = 1000, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::decomposition::TSNE::fit)
        .def("transform", &ingenuityml::decomposition::TSNE::transform)
        .def("inverse_transform", &ingenuityml::decomposition::TSNE::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::TSNE::fit_transform)
        .def("get_params", &ingenuityml::decomposition::TSNE::get_params)
        .def("set_params", &ingenuityml::decomposition::TSNE::set_params)
        .def("is_fitted", &ingenuityml::decomposition::TSNE::is_fitted)
        .def("embedding", &ingenuityml::decomposition::TSNE::embedding);

    py::class_<ingenuityml::decomposition::IncrementalPCA, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "IncrementalPCA")
        .def(py::init<int, bool, int>(), py::arg("n_components") = -1, py::arg("whiten") = false, py::arg("batch_size") = 0)
        .def("fit", &ingenuityml::decomposition::IncrementalPCA::fit)
        .def("partial_fit", &ingenuityml::decomposition::IncrementalPCA::partial_fit)
        .def("transform", &ingenuityml::decomposition::IncrementalPCA::transform)
        .def("inverse_transform", &ingenuityml::decomposition::IncrementalPCA::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::IncrementalPCA::fit_transform)
        .def("get_params", &ingenuityml::decomposition::IncrementalPCA::get_params)
        .def("set_params", &ingenuityml::decomposition::IncrementalPCA::set_params)
        .def("is_fitted", &ingenuityml::decomposition::IncrementalPCA::is_fitted)
        .def("components", &ingenuityml::decomposition::IncrementalPCA::components)
        .def("explained_variance", &ingenuityml::decomposition::IncrementalPCA::explained_variance)
        .def("explained_variance_ratio", &ingenuityml::decomposition::IncrementalPCA::explained_variance_ratio)
        .def("mean", &ingenuityml::decomposition::IncrementalPCA::mean);

    py::class_<ingenuityml::decomposition::SparsePCA, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "SparsePCA")
        .def(py::init<int, double, int, double>(),
             py::arg("n_components") = -1, py::arg("alpha") = 1.0, py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &ingenuityml::decomposition::SparsePCA::fit)
        .def("transform", &ingenuityml::decomposition::SparsePCA::transform)
        .def("inverse_transform", &ingenuityml::decomposition::SparsePCA::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::SparsePCA::fit_transform)
        .def("get_params", &ingenuityml::decomposition::SparsePCA::get_params)
        .def("set_params", &ingenuityml::decomposition::SparsePCA::set_params)
        .def("is_fitted", &ingenuityml::decomposition::SparsePCA::is_fitted)
        .def("components", &ingenuityml::decomposition::SparsePCA::components);

    py::class_<ingenuityml::decomposition::MiniBatchSparsePCA, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "MiniBatchSparsePCA")
        .def(py::init<int, double, int, int, double, int>(),
             py::arg("n_components") = -1, py::arg("alpha") = 1.0, py::arg("max_iter") = 1000,
             py::arg("batch_size") = 100, py::arg("tol") = 1e-4, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::decomposition::MiniBatchSparsePCA::fit)
        .def("transform", &ingenuityml::decomposition::MiniBatchSparsePCA::transform)
        .def("inverse_transform", &ingenuityml::decomposition::MiniBatchSparsePCA::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::MiniBatchSparsePCA::fit_transform)
        .def("get_params", &ingenuityml::decomposition::MiniBatchSparsePCA::get_params)
        .def("set_params", &ingenuityml::decomposition::MiniBatchSparsePCA::set_params)
        .def("is_fitted", &ingenuityml::decomposition::MiniBatchSparsePCA::is_fitted)
        .def("components", &ingenuityml::decomposition::MiniBatchSparsePCA::components);

    py::class_<ingenuityml::decomposition::FastICA, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "FastICA")
        .def(py::init<int, std::string, std::string, int, double, int>(),
             py::arg("n_components") = -1, py::arg("algorithm") = "parallel", py::arg("fun") = "logcosh",
             py::arg("max_iter") = 200, py::arg("tol") = 1e-4, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::decomposition::FastICA::fit)
        .def("transform", &ingenuityml::decomposition::FastICA::transform)
        .def("inverse_transform", &ingenuityml::decomposition::FastICA::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::FastICA::fit_transform)
        .def("get_params", &ingenuityml::decomposition::FastICA::get_params)
        .def("set_params", &ingenuityml::decomposition::FastICA::set_params)
        .def("is_fitted", &ingenuityml::decomposition::FastICA::is_fitted)
        .def("components", &ingenuityml::decomposition::FastICA::components)
        .def("mixing", &ingenuityml::decomposition::FastICA::mixing);

    py::class_<ingenuityml::decomposition::FactorAnalysis, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "FactorAnalysis")
        .def(py::init<int, double, int, int>(),
             py::arg("n_components") = -1, py::arg("tol") = 1e-2, py::arg("max_iter") = 1000, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::decomposition::FactorAnalysis::fit)
        .def("transform", &ingenuityml::decomposition::FactorAnalysis::transform)
        .def("inverse_transform", &ingenuityml::decomposition::FactorAnalysis::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::FactorAnalysis::fit_transform)
        .def("get_params", &ingenuityml::decomposition::FactorAnalysis::get_params)
        .def("set_params", &ingenuityml::decomposition::FactorAnalysis::set_params)
        .def("is_fitted", &ingenuityml::decomposition::FactorAnalysis::is_fitted)
        .def("components", &ingenuityml::decomposition::FactorAnalysis::components)
        .def("noise_variance", &ingenuityml::decomposition::FactorAnalysis::noise_variance)
        .def("loglike", &ingenuityml::decomposition::FactorAnalysis::loglike);

    py::class_<ingenuityml::decomposition::TruncatedSVD, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "TruncatedSVD")
        .def(py::init<int>(), py::arg("n_components"))
        .def("fit", &ingenuityml::decomposition::TruncatedSVD::fit)
        .def("transform", &ingenuityml::decomposition::TruncatedSVD::transform)
        .def("inverse_transform", &ingenuityml::decomposition::TruncatedSVD::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::TruncatedSVD::fit_transform)
        .def("get_params", &ingenuityml::decomposition::TruncatedSVD::get_params)
        .def("set_params", &ingenuityml::decomposition::TruncatedSVD::set_params)
        .def("set_params", [](ingenuityml::decomposition::TruncatedSVD& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::decomposition::TruncatedSVD::is_fitted)
        .def("components", &ingenuityml::decomposition::TruncatedSVD::components)
        .def("singular_values", &ingenuityml::decomposition::TruncatedSVD::singular_values)
        .def("explained_variance", &ingenuityml::decomposition::TruncatedSVD::explained_variance);

    py::class_<ingenuityml::decomposition::NMF, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "NMF")
        .def(py::init<int, int, double, double, int>(),
             py::arg("n_components") = 2, py::arg("max_iter") = 200, py::arg("tol") = 1e-4,
             py::arg("alpha") = 0.0, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::decomposition::NMF::fit)
        .def("transform", &ingenuityml::decomposition::NMF::transform)
        .def("inverse_transform", &ingenuityml::decomposition::NMF::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::NMF::fit_transform)
        .def("get_params", &ingenuityml::decomposition::NMF::get_params)
        .def("set_params", &ingenuityml::decomposition::NMF::set_params)
        .def("is_fitted", &ingenuityml::decomposition::NMF::is_fitted)
        .def("components", &ingenuityml::decomposition::NMF::components);

    py::class_<ingenuityml::decomposition::MiniBatchNMF, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "MiniBatchNMF")
        .def(py::init<int, int, int, double, double, int>(),
             py::arg("n_components") = 2, py::arg("max_iter") = 200, py::arg("batch_size") = 100,
             py::arg("tol") = 1e-4, py::arg("alpha") = 0.0, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::decomposition::MiniBatchNMF::fit)
        .def("transform", &ingenuityml::decomposition::MiniBatchNMF::transform)
        .def("inverse_transform", &ingenuityml::decomposition::MiniBatchNMF::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::MiniBatchNMF::fit_transform)
        .def("get_params", &ingenuityml::decomposition::MiniBatchNMF::get_params)
        .def("set_params", &ingenuityml::decomposition::MiniBatchNMF::set_params)
        .def("is_fitted", &ingenuityml::decomposition::MiniBatchNMF::is_fitted)
        .def("components", &ingenuityml::decomposition::MiniBatchNMF::components);

    py::class_<ingenuityml::decomposition::DictionaryLearning, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "DictionaryLearning")
        .def(py::init<int, double, int, double, int>(),
             py::arg("n_components") = 2, py::arg("alpha") = 1.0, py::arg("max_iter") = 100,
             py::arg("tol") = 1e-4, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::decomposition::DictionaryLearning::fit)
        .def("transform", &ingenuityml::decomposition::DictionaryLearning::transform)
        .def("inverse_transform", &ingenuityml::decomposition::DictionaryLearning::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::DictionaryLearning::fit_transform)
        .def("get_params", &ingenuityml::decomposition::DictionaryLearning::get_params)
        .def("set_params", &ingenuityml::decomposition::DictionaryLearning::set_params)
        .def("is_fitted", &ingenuityml::decomposition::DictionaryLearning::is_fitted)
        .def("components", &ingenuityml::decomposition::DictionaryLearning::components)
        .def("codes", &ingenuityml::decomposition::DictionaryLearning::codes);

    py::class_<ingenuityml::decomposition::MiniBatchDictionaryLearning, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "MiniBatchDictionaryLearning")
        .def(py::init<int, double, int, int, double, int>(),
             py::arg("n_components") = 2, py::arg("alpha") = 1.0, py::arg("max_iter") = 100,
             py::arg("batch_size") = 100, py::arg("tol") = 1e-4, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::decomposition::MiniBatchDictionaryLearning::fit)
        .def("transform", &ingenuityml::decomposition::MiniBatchDictionaryLearning::transform)
        .def("inverse_transform", &ingenuityml::decomposition::MiniBatchDictionaryLearning::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::MiniBatchDictionaryLearning::fit_transform)
        .def("get_params", &ingenuityml::decomposition::MiniBatchDictionaryLearning::get_params)
        .def("set_params", &ingenuityml::decomposition::MiniBatchDictionaryLearning::set_params)
        .def("is_fitted", &ingenuityml::decomposition::MiniBatchDictionaryLearning::is_fitted)
        .def("components", &ingenuityml::decomposition::MiniBatchDictionaryLearning::components);

    py::class_<ingenuityml::decomposition::LDA, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "LDA")
        .def(py::init<int>(), py::arg("n_components") = -1)
        .def("fit", &ingenuityml::decomposition::LDA::fit)
        .def("transform", &ingenuityml::decomposition::LDA::transform)
        .def("inverse_transform", &ingenuityml::decomposition::LDA::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::LDA::fit_transform)
        .def("get_params", &ingenuityml::decomposition::LDA::get_params)
        .def("set_params", &ingenuityml::decomposition::LDA::set_params)
        .def("set_params", [](ingenuityml::decomposition::LDA& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::decomposition::LDA::is_fitted)
        .def("components", &ingenuityml::decomposition::LDA::components)
        .def("explained_variance", &ingenuityml::decomposition::LDA::explained_variance)
        .def("explained_variance_ratio", &ingenuityml::decomposition::LDA::explained_variance_ratio)
        .def("mean", &ingenuityml::decomposition::LDA::mean)
        .def("class_means", &ingenuityml::decomposition::LDA::class_means)
        .def("classes", &ingenuityml::decomposition::LDA::classes);

    py::class_<ingenuityml::decomposition::LatentDirichletAllocation, ingenuityml::Estimator, ingenuityml::Transformer>(decomp_module, "LatentDirichletAllocation")
        .def(py::init<int, int, double, double, int>(),
             py::arg("n_components") = 10, py::arg("max_iter") = 10,
             py::arg("doc_topic_prior") = 0.1, py::arg("topic_word_prior") = 0.01,
             py::arg("random_state") = -1)
        .def("fit", &ingenuityml::decomposition::LatentDirichletAllocation::fit)
        .def("transform", &ingenuityml::decomposition::LatentDirichletAllocation::transform)
        .def("inverse_transform", &ingenuityml::decomposition::LatentDirichletAllocation::inverse_transform)
        .def("fit_transform", &ingenuityml::decomposition::LatentDirichletAllocation::fit_transform)
        .def("get_params", &ingenuityml::decomposition::LatentDirichletAllocation::get_params)
        .def("set_params", &ingenuityml::decomposition::LatentDirichletAllocation::set_params)
        .def("is_fitted", &ingenuityml::decomposition::LatentDirichletAllocation::is_fitted)
        .def("components", &ingenuityml::decomposition::LatentDirichletAllocation::components)
        .def("doc_topic", &ingenuityml::decomposition::LatentDirichletAllocation::doc_topic);

    // Cross-Decomposition
    py::module_ cross_module = m.def_submodule("cross_decomposition", "Cross-decomposition algorithms");
    py::class_<ingenuityml::cross_decomposition::PLSCanonical, ingenuityml::Estimator, ingenuityml::Transformer>(cross_module, "PLSCanonical")
        .def(py::init<int, bool, int, double>(),
             py::arg("n_components") = 2, py::arg("scale") = true, py::arg("max_iter") = 500, py::arg("tol") = 1e-6)
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::cross_decomposition::PLSCanonical::fit))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::cross_decomposition::PLSCanonical::fit))
        .def("transform", &ingenuityml::cross_decomposition::PLSCanonical::transform)
        .def("transform_y", &ingenuityml::cross_decomposition::PLSCanonical::transform_y)
        .def("fit_transform", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::cross_decomposition::PLSCanonical::fit_transform))
        .def("fit_transform", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::cross_decomposition::PLSCanonical::fit_transform))
        .def("get_params", &ingenuityml::cross_decomposition::PLSCanonical::get_params)
        .def("set_params", &ingenuityml::cross_decomposition::PLSCanonical::set_params)
        .def("is_fitted", &ingenuityml::cross_decomposition::PLSCanonical::is_fitted)
        .def("x_weights", &ingenuityml::cross_decomposition::PLSCanonical::x_weights)
        .def("y_weights", &ingenuityml::cross_decomposition::PLSCanonical::y_weights)
        .def("x_loadings", &ingenuityml::cross_decomposition::PLSCanonical::x_loadings)
        .def("y_loadings", &ingenuityml::cross_decomposition::PLSCanonical::y_loadings)
        .def("x_scores", &ingenuityml::cross_decomposition::PLSCanonical::x_scores)
        .def("y_scores", &ingenuityml::cross_decomposition::PLSCanonical::y_scores);

    py::class_<ingenuityml::cross_decomposition::PLSRegression, ingenuityml::Estimator, ingenuityml::Regressor, ingenuityml::Transformer>(cross_module, "PLSRegression")
        .def(py::init<int, bool, int, double>(),
             py::arg("n_components") = 2, py::arg("scale") = true, py::arg("max_iter") = 500, py::arg("tol") = 1e-6)
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::cross_decomposition::PLSRegression::fit))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::cross_decomposition::PLSRegression::fit))
        .def("predict", [](const ingenuityml::cross_decomposition::PLSRegression& self, const ingenuityml::MatrixXd& X) {
            return self.predict_multi(X);
        })
        .def("predict_multi", &ingenuityml::cross_decomposition::PLSRegression::predict_multi)
        .def("transform", &ingenuityml::cross_decomposition::PLSRegression::transform)
        .def("transform_y", &ingenuityml::cross_decomposition::PLSRegression::transform_y)
        .def("fit_transform", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::cross_decomposition::PLSRegression::fit_transform))
        .def("fit_transform", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::cross_decomposition::PLSRegression::fit_transform))
        .def("get_params", &ingenuityml::cross_decomposition::PLSRegression::get_params)
        .def("set_params", &ingenuityml::cross_decomposition::PLSRegression::set_params)
        .def("is_fitted", &ingenuityml::cross_decomposition::PLSRegression::is_fitted)
        .def("x_weights", &ingenuityml::cross_decomposition::PLSRegression::x_weights)
        .def("y_weights", &ingenuityml::cross_decomposition::PLSRegression::y_weights)
        .def("x_loadings", &ingenuityml::cross_decomposition::PLSRegression::x_loadings)
        .def("y_loadings", &ingenuityml::cross_decomposition::PLSRegression::y_loadings)
        .def("x_scores", &ingenuityml::cross_decomposition::PLSRegression::x_scores)
        .def("y_scores", &ingenuityml::cross_decomposition::PLSRegression::y_scores)
        .def("coef", &ingenuityml::cross_decomposition::PLSRegression::coef)
        .def("intercept", &ingenuityml::cross_decomposition::PLSRegression::intercept);

    py::class_<ingenuityml::cross_decomposition::CCA, ingenuityml::Estimator, ingenuityml::Transformer>(cross_module, "CCA")
        .def(py::init<int, bool, int, double>(),
             py::arg("n_components") = 2, py::arg("scale") = true, py::arg("max_iter") = 500, py::arg("tol") = 1e-6)
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::cross_decomposition::CCA::fit))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::cross_decomposition::CCA::fit))
        .def("transform", &ingenuityml::cross_decomposition::CCA::transform)
        .def("transform_y", &ingenuityml::cross_decomposition::CCA::transform_y)
        .def("fit_transform", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::cross_decomposition::CCA::fit_transform))
        .def("fit_transform", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::cross_decomposition::CCA::fit_transform))
        .def("get_params", &ingenuityml::cross_decomposition::CCA::get_params)
        .def("set_params", &ingenuityml::cross_decomposition::CCA::set_params)
        .def("is_fitted", &ingenuityml::cross_decomposition::CCA::is_fitted)
        .def("x_weights", &ingenuityml::cross_decomposition::CCA::x_weights)
        .def("y_weights", &ingenuityml::cross_decomposition::CCA::y_weights)
        .def("x_scores", &ingenuityml::cross_decomposition::CCA::x_scores)
        .def("y_scores", &ingenuityml::cross_decomposition::CCA::y_scores)
        .def("correlations", &ingenuityml::cross_decomposition::CCA::correlations);

    py::class_<ingenuityml::cross_decomposition::PLSSVD, ingenuityml::Estimator, ingenuityml::Transformer>(cross_module, "PLSSVD")
        .def(py::init<int, bool>(), py::arg("n_components") = 2, py::arg("scale") = true)
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::cross_decomposition::PLSSVD::fit))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::cross_decomposition::PLSSVD::fit))
        .def("transform", &ingenuityml::cross_decomposition::PLSSVD::transform)
        .def("transform_y", &ingenuityml::cross_decomposition::PLSSVD::transform_y)
        .def("fit_transform", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::cross_decomposition::PLSSVD::fit_transform))
        .def("fit_transform", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::cross_decomposition::PLSSVD::fit_transform))
        .def("get_params", &ingenuityml::cross_decomposition::PLSSVD::get_params)
        .def("set_params", &ingenuityml::cross_decomposition::PLSSVD::set_params)
        .def("is_fitted", &ingenuityml::cross_decomposition::PLSSVD::is_fitted)
        .def("x_weights", &ingenuityml::cross_decomposition::PLSSVD::x_weights)
        .def("y_weights", &ingenuityml::cross_decomposition::PLSSVD::y_weights)
        .def("x_scores", &ingenuityml::cross_decomposition::PLSSVD::x_scores)
        .def("y_scores", &ingenuityml::cross_decomposition::PLSSVD::y_scores);

    // Random Projection
    py::module_ random_projection_module = m.def_submodule("random_projection", "Random projection algorithms");
    py::class_<ingenuityml::random_projection::GaussianRandomProjection, ingenuityml::Estimator, ingenuityml::Transformer>(random_projection_module, "GaussianRandomProjection")
        .def(py::init<int, int>(), py::arg("n_components") = -1, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::random_projection::GaussianRandomProjection::fit)
        .def("transform", &ingenuityml::random_projection::GaussianRandomProjection::transform)
        .def("inverse_transform", &ingenuityml::random_projection::GaussianRandomProjection::inverse_transform)
        .def("fit_transform", &ingenuityml::random_projection::GaussianRandomProjection::fit_transform)
        .def("get_params", &ingenuityml::random_projection::GaussianRandomProjection::get_params)
        .def("set_params", &ingenuityml::random_projection::GaussianRandomProjection::set_params)
        .def("is_fitted", &ingenuityml::random_projection::GaussianRandomProjection::is_fitted)
        .def("components", &ingenuityml::random_projection::GaussianRandomProjection::components);

    py::class_<ingenuityml::random_projection::SparseRandomProjection, ingenuityml::Estimator, ingenuityml::Transformer>(random_projection_module, "SparseRandomProjection")
        .def(py::init<int, double, int>(), py::arg("n_components") = -1, py::arg("density") = -1.0, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::random_projection::SparseRandomProjection::fit)
        .def("transform", &ingenuityml::random_projection::SparseRandomProjection::transform)
        .def("inverse_transform", &ingenuityml::random_projection::SparseRandomProjection::inverse_transform)
        .def("fit_transform", &ingenuityml::random_projection::SparseRandomProjection::fit_transform)
        .def("get_params", &ingenuityml::random_projection::SparseRandomProjection::get_params)
        .def("set_params", &ingenuityml::random_projection::SparseRandomProjection::set_params)
        .def("is_fitted", &ingenuityml::random_projection::SparseRandomProjection::is_fitted)
        .def("components", &ingenuityml::random_projection::SparseRandomProjection::components);

    // Gaussian Processes
    py::module_ gaussian_process_module = m.def_submodule("gaussian_process", "Gaussian process models");
    py::class_<ingenuityml::gaussian_process::GaussianProcessRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(gaussian_process_module, "GaussianProcessRegressor")
        .def(py::init<double, double, bool>(), py::arg("length_scale") = 1.0, py::arg("alpha") = 1e-10, py::arg("normalize_y") = true)
        .def("fit", &ingenuityml::gaussian_process::GaussianProcessRegressor::fit)
        .def("predict", &ingenuityml::gaussian_process::GaussianProcessRegressor::predict)
        .def("get_params", &ingenuityml::gaussian_process::GaussianProcessRegressor::get_params)
        .def("set_params", &ingenuityml::gaussian_process::GaussianProcessRegressor::set_params)
        .def("is_fitted", &ingenuityml::gaussian_process::GaussianProcessRegressor::is_fitted);

    py::class_<ingenuityml::gaussian_process::GaussianProcessClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(gaussian_process_module, "GaussianProcessClassifier")
        .def(py::init<double, double>(), py::arg("length_scale") = 1.0, py::arg("alpha") = 1e-10)
        .def("fit", &ingenuityml::gaussian_process::GaussianProcessClassifier::fit)
        .def("predict", &ingenuityml::gaussian_process::GaussianProcessClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::gaussian_process::GaussianProcessClassifier::predict_proba)
        .def("decision_function", &ingenuityml::gaussian_process::GaussianProcessClassifier::decision_function)
        .def("get_params", &ingenuityml::gaussian_process::GaussianProcessClassifier::get_params)
        .def("set_params", &ingenuityml::gaussian_process::GaussianProcessClassifier::set_params)
        .def("is_fitted", &ingenuityml::gaussian_process::GaussianProcessClassifier::is_fitted)
        .def("classes", &ingenuityml::gaussian_process::GaussianProcessClassifier::classes);

    // Density Estimation
    py::module_ density_module = m.def_submodule("density", "Density estimation");
    py::class_<ingenuityml::density::KernelDensity, ingenuityml::Estimator>(density_module, "KernelDensity")
        .def(py::init<double, const std::string&>(), py::arg("bandwidth") = 1.0, py::arg("kernel") = "gaussian")
        .def("fit", &ingenuityml::density::KernelDensity::fit)
        .def("score_samples", &ingenuityml::density::KernelDensity::score_samples)
        .def("score", &ingenuityml::density::KernelDensity::score)
        .def("get_params", &ingenuityml::density::KernelDensity::get_params)
        .def("set_params", &ingenuityml::density::KernelDensity::set_params)
        .def("is_fitted", &ingenuityml::density::KernelDensity::is_fitted);

    // Covariance estimation
    py::module_ covariance_module = m.def_submodule("covariance", "Covariance estimation");

    auto empirical_cov_fit = [](ingenuityml::covariance::EmpiricalCovariance& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::covariance::EmpiricalCovariance& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto shrunk_cov_fit = [](ingenuityml::covariance::ShrunkCovariance& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::covariance::ShrunkCovariance& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto ledoit_cov_fit = [](ingenuityml::covariance::LedoitWolf& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::covariance::LedoitWolf& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto oas_cov_fit = [](ingenuityml::covariance::OAS& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::covariance::OAS& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto mcd_fit = [](ingenuityml::covariance::MinCovDet& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::covariance::MinCovDet& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto envelope_fit = [](ingenuityml::covariance::EllipticEnvelope& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::covariance::EllipticEnvelope& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };

    py::class_<ingenuityml::covariance::EmpiricalCovariance, ingenuityml::Estimator>(covariance_module, "EmpiricalCovariance")
        .def(py::init<bool>(), py::arg("assume_centered") = false)
        .def("fit", empirical_cov_fit, py::arg("X"), py::arg("y") = py::none())
        .def("mahalanobis", &ingenuityml::covariance::EmpiricalCovariance::mahalanobis)
        .def("score_samples", &ingenuityml::covariance::EmpiricalCovariance::score_samples)
        .def("covariance", &ingenuityml::covariance::EmpiricalCovariance::covariance)
        .def("location", &ingenuityml::covariance::EmpiricalCovariance::location)
        .def("precision", &ingenuityml::covariance::EmpiricalCovariance::precision)
        .def("get_params", &ingenuityml::covariance::EmpiricalCovariance::get_params)
        .def("set_params", &ingenuityml::covariance::EmpiricalCovariance::set_params)
        .def("is_fitted", &ingenuityml::covariance::EmpiricalCovariance::is_fitted);

    py::class_<ingenuityml::covariance::ShrunkCovariance, ingenuityml::Estimator>(covariance_module, "ShrunkCovariance")
        .def(py::init<double, bool>(), py::arg("shrinkage") = 0.1, py::arg("assume_centered") = false)
        .def("fit", shrunk_cov_fit, py::arg("X"), py::arg("y") = py::none())
        .def("mahalanobis", &ingenuityml::covariance::ShrunkCovariance::mahalanobis)
        .def("score_samples", &ingenuityml::covariance::ShrunkCovariance::score_samples)
        .def("covariance", &ingenuityml::covariance::ShrunkCovariance::covariance)
        .def("location", &ingenuityml::covariance::ShrunkCovariance::location)
        .def("precision", &ingenuityml::covariance::ShrunkCovariance::precision)
        .def("shrinkage", &ingenuityml::covariance::ShrunkCovariance::shrinkage)
        .def("get_params", &ingenuityml::covariance::ShrunkCovariance::get_params)
        .def("set_params", &ingenuityml::covariance::ShrunkCovariance::set_params)
        .def("is_fitted", &ingenuityml::covariance::ShrunkCovariance::is_fitted);

    py::class_<ingenuityml::covariance::LedoitWolf, ingenuityml::Estimator>(covariance_module, "LedoitWolf")
        .def(py::init<bool>(), py::arg("assume_centered") = false)
        .def("fit", ledoit_cov_fit, py::arg("X"), py::arg("y") = py::none())
        .def("mahalanobis", &ingenuityml::covariance::LedoitWolf::mahalanobis)
        .def("score_samples", &ingenuityml::covariance::LedoitWolf::score_samples)
        .def("covariance", &ingenuityml::covariance::LedoitWolf::covariance)
        .def("location", &ingenuityml::covariance::LedoitWolf::location)
        .def("precision", &ingenuityml::covariance::LedoitWolf::precision)
        .def("shrinkage", &ingenuityml::covariance::LedoitWolf::shrinkage)
        .def("get_params", &ingenuityml::covariance::LedoitWolf::get_params)
        .def("set_params", &ingenuityml::covariance::LedoitWolf::set_params)
        .def("is_fitted", &ingenuityml::covariance::LedoitWolf::is_fitted);

    py::class_<ingenuityml::covariance::OAS, ingenuityml::Estimator>(covariance_module, "OAS")
        .def(py::init<bool>(), py::arg("assume_centered") = false)
        .def("fit", oas_cov_fit, py::arg("X"), py::arg("y") = py::none())
        .def("mahalanobis", &ingenuityml::covariance::OAS::mahalanobis)
        .def("score_samples", &ingenuityml::covariance::OAS::score_samples)
        .def("covariance", &ingenuityml::covariance::OAS::covariance)
        .def("location", &ingenuityml::covariance::OAS::location)
        .def("precision", &ingenuityml::covariance::OAS::precision)
        .def("shrinkage", &ingenuityml::covariance::OAS::shrinkage)
        .def("get_params", &ingenuityml::covariance::OAS::get_params)
        .def("set_params", &ingenuityml::covariance::OAS::set_params)
        .def("is_fitted", &ingenuityml::covariance::OAS::is_fitted);

    py::class_<ingenuityml::covariance::MinCovDet, ingenuityml::Estimator>(covariance_module, "MinCovDet")
        .def(py::init<double, bool, int, double, int>(),
             py::arg("support_fraction") = 0.75, py::arg("assume_centered") = false,
             py::arg("max_iter") = 100, py::arg("tol") = 1e-3, py::arg("random_state") = -1)
        .def("fit", mcd_fit, py::arg("X"), py::arg("y") = py::none())
        .def("mahalanobis", &ingenuityml::covariance::MinCovDet::mahalanobis)
        .def("score_samples", &ingenuityml::covariance::MinCovDet::score_samples)
        .def("covariance", &ingenuityml::covariance::MinCovDet::covariance)
        .def("location", &ingenuityml::covariance::MinCovDet::location)
        .def("precision", &ingenuityml::covariance::MinCovDet::precision)
        .def("support", &ingenuityml::covariance::MinCovDet::support)
        .def("get_params", &ingenuityml::covariance::MinCovDet::get_params)
        .def("set_params", &ingenuityml::covariance::MinCovDet::set_params)
        .def("is_fitted", &ingenuityml::covariance::MinCovDet::is_fitted);

    py::class_<ingenuityml::covariance::EllipticEnvelope, ingenuityml::Estimator>(covariance_module, "EllipticEnvelope")
        .def(py::init<double, double, int, double, int>(),
             py::arg("contamination") = 0.1, py::arg("support_fraction") = 0.75,
             py::arg("max_iter") = 100, py::arg("tol") = 1e-3, py::arg("random_state") = -1)
        .def("fit", envelope_fit, py::arg("X"), py::arg("y") = py::none())
        .def("predict", &ingenuityml::covariance::EllipticEnvelope::predict)
        .def("decision_function", &ingenuityml::covariance::EllipticEnvelope::decision_function)
        .def("score_samples", &ingenuityml::covariance::EllipticEnvelope::score_samples)
        .def("covariance", &ingenuityml::covariance::EllipticEnvelope::covariance)
        .def("location", &ingenuityml::covariance::EllipticEnvelope::location)
        .def("precision", &ingenuityml::covariance::EllipticEnvelope::precision)
        .def("threshold", &ingenuityml::covariance::EllipticEnvelope::threshold)
        .def("get_params", &ingenuityml::covariance::EllipticEnvelope::get_params)
        .def("set_params", &ingenuityml::covariance::EllipticEnvelope::set_params)
        .def("is_fitted", &ingenuityml::covariance::EllipticEnvelope::is_fitted);

    // SVM
    py::module_ svm_module = m.def_submodule("svm", "Support Vector Machines");
    py::class_<ingenuityml::svm::LinearSVC, ingenuityml::Estimator, ingenuityml::Classifier>(svm_module, "LinearSVC")
        .def(py::init<double,int,double,int>(), py::arg("C")=1.0, py::arg("max_iter")=1000, py::arg("lr")=0.01, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::svm::LinearSVC::fit)
        .def("predict", &ingenuityml::svm::LinearSVC::predict_classes)
        .def("predict_proba", &ingenuityml::svm::LinearSVC::predict_proba)
        .def("decision_function", &ingenuityml::svm::LinearSVC::decision_function)
        .def("get_params", &ingenuityml::svm::LinearSVC::get_params)
        .def("set_params", &ingenuityml::svm::LinearSVC::set_params)
        .def("is_fitted", &ingenuityml::svm::LinearSVC::is_fitted)
        .def("save", &ingenuityml::svm::LinearSVC::save)
        .def("load", &ingenuityml::svm::LinearSVC::load);

    py::class_<ingenuityml::svm::SVR, ingenuityml::Estimator, ingenuityml::Regressor>(svm_module, "SVR")
        .def(py::init<double,double,int,double,int,const std::string&,double,double,double>(), 
             py::arg("C")=1.0, py::arg("epsilon")=0.1, py::arg("max_iter")=1000, 
             py::arg("lr")=0.01, py::arg("random_state")=-1,
             py::arg("kernel")="linear", py::arg("gamma")=1.0, py::arg("degree")=3.0, py::arg("coef0")=0.0)
        .def("fit", &ingenuityml::svm::SVR::fit)
        .def("predict", &ingenuityml::svm::SVR::predict)
        .def("get_params", &ingenuityml::svm::SVR::get_params)
        .def("set_params", &ingenuityml::svm::SVR::set_params)
        .def("is_fitted", &ingenuityml::svm::SVR::is_fitted)
        .def("coef", &ingenuityml::svm::SVR::coef)
        .def("intercept", &ingenuityml::svm::SVR::intercept);

    py::class_<ingenuityml::svm::LinearSVR, ingenuityml::Estimator, ingenuityml::Regressor>(svm_module, "LinearSVR")
        .def(py::init<double,double,int,double,int>(), 
             py::arg("C")=1.0, py::arg("epsilon")=0.1, py::arg("max_iter")=1000, 
             py::arg("lr")=0.01, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::svm::LinearSVR::fit)
        .def("predict", &ingenuityml::svm::LinearSVR::predict)
        .def("get_params", &ingenuityml::svm::LinearSVR::get_params)
        .def("set_params", &ingenuityml::svm::LinearSVR::set_params)
        .def("is_fitted", &ingenuityml::svm::LinearSVR::is_fitted)
        .def("coef", &ingenuityml::svm::LinearSVR::coef)
        .def("intercept", &ingenuityml::svm::LinearSVR::intercept);

    py::class_<ingenuityml::svm::NuSVC, ingenuityml::Estimator, ingenuityml::Classifier>(svm_module, "NuSVC")
        .def(py::init<double,int,double,int>(), 
             py::arg("nu")=0.5, py::arg("max_iter")=1000, 
             py::arg("lr")=0.01, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::svm::NuSVC::fit)
        .def("predict", &ingenuityml::svm::NuSVC::predict_classes)
        .def("predict_proba", &ingenuityml::svm::NuSVC::predict_proba)
        .def("decision_function", &ingenuityml::svm::NuSVC::decision_function)
        .def("get_params", &ingenuityml::svm::NuSVC::get_params)
        .def("set_params", &ingenuityml::svm::NuSVC::set_params)
        .def("is_fitted", &ingenuityml::svm::NuSVC::is_fitted)
        .def("classes", &ingenuityml::svm::NuSVC::classes);

    py::class_<ingenuityml::svm::NuSVR, ingenuityml::Estimator, ingenuityml::Regressor>(svm_module, "NuSVR")
        .def(py::init<double,double,int,double,int>(), 
             py::arg("nu")=0.5, py::arg("C")=1.0, py::arg("max_iter")=1000, 
             py::arg("lr")=0.01, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::svm::NuSVR::fit)
        .def("predict", &ingenuityml::svm::NuSVR::predict)
        .def("get_params", &ingenuityml::svm::NuSVR::get_params)
        .def("set_params", &ingenuityml::svm::NuSVR::set_params)
        .def("is_fitted", &ingenuityml::svm::NuSVR::is_fitted)
        .def("coef", &ingenuityml::svm::NuSVR::coef)
        .def("intercept", &ingenuityml::svm::NuSVR::intercept);

    py::class_<ingenuityml::svm::OneClassSVM, ingenuityml::Estimator>(svm_module, "OneClassSVM")
        .def(py::init<double,double,const std::string&,int,double,int>(), 
             py::arg("nu")=0.5, py::arg("gamma")=1.0, py::arg("kernel")="rbf", 
             py::arg("max_iter")=1000, py::arg("lr")=0.01, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::svm::OneClassSVM::fit, py::arg("X"), py::arg("y")=ingenuityml::VectorXd())
        .def("predict", &ingenuityml::svm::OneClassSVM::predict)
        .def("decision_function", &ingenuityml::svm::OneClassSVM::decision_function)
        .def("score_samples", &ingenuityml::svm::OneClassSVM::score_samples)
        .def("get_params", &ingenuityml::svm::OneClassSVM::get_params)
        .def("set_params", &ingenuityml::svm::OneClassSVM::set_params)
        .def("is_fitted", &ingenuityml::svm::OneClassSVM::is_fitted)
        .def("get_threshold", &ingenuityml::svm::OneClassSVM::get_threshold);

    py::class_<ingenuityml::svm::SVC, ingenuityml::Estimator, ingenuityml::Classifier>(svm_module, "SVC")
        .def(py::init<const std::string&, double, int, double, int, double, double, double>(),
             py::arg("kernel") = "rbf", py::arg("C") = 1.0, py::arg("max_iter") = 1000,
             py::arg("lr") = 0.01, py::arg("random_state") = -1,
             py::arg("gamma") = 1.0, py::arg("degree") = 3.0, py::arg("coef0") = 0.0)
        .def("fit", &ingenuityml::svm::SVC::fit)
        .def("predict", &ingenuityml::svm::SVC::predict_classes)
        .def("predict_proba", &ingenuityml::svm::SVC::predict_proba)
        .def("decision_function", &ingenuityml::svm::SVC::decision_function)
        .def("get_params", &ingenuityml::svm::SVC::get_params)
        .def("set_params", &ingenuityml::svm::SVC::set_params)
        .def("is_fitted", &ingenuityml::svm::SVC::is_fitted)
        .def("classes", &ingenuityml::svm::SVC::classes);

    // RandomForest
    py::module_ rf_module = m.def_submodule("ensemble", "Ensemble methods");
    py::class_<ingenuityml::ensemble::RandomForestClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(rf_module, "RandomForestClassifier")
        .def(py::init<int,int,int,int>(), py::arg("n_estimators")=100, py::arg("max_depth")=-1, py::arg("max_features")=-1, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::ensemble::RandomForestClassifier::fit)
        .def("predict", &ingenuityml::ensemble::RandomForestClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::RandomForestClassifier::predict_proba)
        .def("decision_function", &ingenuityml::ensemble::RandomForestClassifier::decision_function)
        .def("get_params", &ingenuityml::ensemble::RandomForestClassifier::get_params)
        .def("set_params", &ingenuityml::ensemble::RandomForestClassifier::set_params)
        .def("is_fitted", &ingenuityml::ensemble::RandomForestClassifier::is_fitted)
        .def("save", &ingenuityml::ensemble::RandomForestClassifier::save)
        .def("load", &ingenuityml::ensemble::RandomForestClassifier::load);

    py::class_<ingenuityml::ensemble::RandomForestRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(rf_module, "RandomForestRegressor")
        .def(py::init<int,int,int,int>(), py::arg("n_estimators")=100, py::arg("max_depth")=-1, py::arg("max_features")=-1, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::ensemble::RandomForestRegressor::fit)
        .def("predict", &ingenuityml::ensemble::RandomForestRegressor::predict)
        .def("get_params", &ingenuityml::ensemble::RandomForestRegressor::get_params)
        .def("set_params", &ingenuityml::ensemble::RandomForestRegressor::set_params)
        .def("is_fitted", &ingenuityml::ensemble::RandomForestRegressor::is_fitted)
        .def("save", &ingenuityml::ensemble::RandomForestRegressor::save)
        .def("load", &ingenuityml::ensemble::RandomForestRegressor::load);

    py::class_<ingenuityml::ensemble::ExtraTreesClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(rf_module, "ExtraTreesClassifier")
        .def(py::init<int,int,int,int,int,bool,int>(),
             py::arg("n_estimators")=100, py::arg("max_depth")=-1, py::arg("min_samples_split")=2,
             py::arg("min_samples_leaf")=1, py::arg("max_features")=-1,
             py::arg("bootstrap")=false, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::ensemble::ExtraTreesClassifier::fit)
        .def("predict", &ingenuityml::ensemble::ExtraTreesClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::ExtraTreesClassifier::predict_proba)
        .def("decision_function", &ingenuityml::ensemble::ExtraTreesClassifier::decision_function)
        .def("get_params", &ingenuityml::ensemble::ExtraTreesClassifier::get_params)
        .def("set_params", &ingenuityml::ensemble::ExtraTreesClassifier::set_params)
        .def("is_fitted", &ingenuityml::ensemble::ExtraTreesClassifier::is_fitted);

    py::class_<ingenuityml::ensemble::ExtraTreesRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(rf_module, "ExtraTreesRegressor")
        .def(py::init<int,int,int,int,int,bool,int>(),
             py::arg("n_estimators")=100, py::arg("max_depth")=-1, py::arg("min_samples_split")=2,
             py::arg("min_samples_leaf")=1, py::arg("max_features")=-1,
             py::arg("bootstrap")=false, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::ensemble::ExtraTreesRegressor::fit)
        .def("predict", &ingenuityml::ensemble::ExtraTreesRegressor::predict)
        .def("get_params", &ingenuityml::ensemble::ExtraTreesRegressor::get_params)
        .def("set_params", &ingenuityml::ensemble::ExtraTreesRegressor::set_params)
        .def("is_fitted", &ingenuityml::ensemble::ExtraTreesRegressor::is_fitted);

    py::class_<ingenuityml::ensemble::RandomTreesEmbedding, ingenuityml::Estimator, ingenuityml::Transformer>(rf_module, "RandomTreesEmbedding")
        .def(py::init<int,int,int,int,int,bool,int>(),
             py::arg("n_estimators")=100, py::arg("max_depth")=-1, py::arg("min_samples_split")=2,
             py::arg("min_samples_leaf")=1, py::arg("max_features")=-1,
             py::arg("bootstrap")=false, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::ensemble::RandomTreesEmbedding::fit)
        .def("transform", &ingenuityml::ensemble::RandomTreesEmbedding::transform)
        .def("inverse_transform", &ingenuityml::ensemble::RandomTreesEmbedding::inverse_transform)
        .def("fit_transform", &ingenuityml::ensemble::RandomTreesEmbedding::fit_transform)
        .def("get_params", &ingenuityml::ensemble::RandomTreesEmbedding::get_params)
        .def("set_params", &ingenuityml::ensemble::RandomTreesEmbedding::set_params)
        .def("is_fitted", &ingenuityml::ensemble::RandomTreesEmbedding::is_fitted);

    py::class_<ingenuityml::ensemble::DummyClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(rf_module, "DummyClassifier")
        .def(py::init<const std::string&>(), py::arg("strategy") = "most_frequent")
        .def("fit", &ingenuityml::ensemble::DummyClassifier::fit)
        .def("predict", &ingenuityml::ensemble::DummyClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::DummyClassifier::predict_proba)
        .def("decision_function", &ingenuityml::ensemble::DummyClassifier::decision_function)
        .def("get_params", &ingenuityml::ensemble::DummyClassifier::get_params)
        .def("set_params", &ingenuityml::ensemble::DummyClassifier::set_params)
        .def("set_params", [](ingenuityml::ensemble::DummyClassifier& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::ensemble::DummyClassifier::is_fitted)
        .def("classes", &ingenuityml::ensemble::DummyClassifier::classes);

    py::class_<ingenuityml::ensemble::DummyRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(rf_module, "DummyRegressor")
        .def(py::init<const std::string&, double, double>(),
             py::arg("strategy") = "mean", py::arg("quantile") = 0.5, py::arg("constant") = 0.0)
        .def("fit", &ingenuityml::ensemble::DummyRegressor::fit)
        .def("predict", &ingenuityml::ensemble::DummyRegressor::predict)
        .def("get_params", &ingenuityml::ensemble::DummyRegressor::get_params)
        .def("set_params", &ingenuityml::ensemble::DummyRegressor::set_params)
        .def("set_params", [](ingenuityml::ensemble::DummyRegressor& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::ensemble::DummyRegressor::is_fitted)
        .def("statistic", &ingenuityml::ensemble::DummyRegressor::statistic);

    // Gradient Boosting
    py::module_ gb_module = m.def_submodule("gradient_boosting", "Gradient Boosting algorithms");
    py::class_<ingenuityml::ensemble::GradientBoostingClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(gb_module, "GradientBoostingClassifier")
        .def(py::init<int,double,int,int,int,double,int>(),
             py::arg("n_estimators")=100, py::arg("learning_rate")=0.1, py::arg("max_depth")=3,
             py::arg("min_samples_split")=2, py::arg("min_samples_leaf")=1, 
             py::arg("min_impurity_decrease")=0.0, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::ensemble::GradientBoostingClassifier::fit)
        .def("predict_classes", &ingenuityml::ensemble::GradientBoostingClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::GradientBoostingClassifier::predict_proba)
        .def("predict", &ingenuityml::ensemble::GradientBoostingClassifier::predict_classes)
        .def("get_params", &ingenuityml::ensemble::GradientBoostingClassifier::get_params)
        .def("set_params", &ingenuityml::ensemble::GradientBoostingClassifier::set_params)
        .def("is_fitted", &ingenuityml::ensemble::GradientBoostingClassifier::is_fitted)
        .def("n_estimators", &ingenuityml::ensemble::GradientBoostingClassifier::n_estimators)
        .def("learning_rate", &ingenuityml::ensemble::GradientBoostingClassifier::learning_rate)
        .def("classes", &ingenuityml::ensemble::GradientBoostingClassifier::classes)
        .def("save", &ingenuityml::ensemble::GradientBoostingClassifier::save)
        .def("load", &ingenuityml::ensemble::GradientBoostingClassifier::load);

    py::class_<ingenuityml::ensemble::HistGradientBoostingClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(gb_module, "HistGradientBoostingClassifier")
        .def(py::init<int,double,int,int,int,double,int>(),
             py::arg("max_iter")=100, py::arg("learning_rate")=0.1, py::arg("max_depth")=3,
             py::arg("min_samples_split")=2, py::arg("min_samples_leaf")=1,
             py::arg("min_impurity_decrease")=0.0, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::ensemble::HistGradientBoostingClassifier::fit)
        .def("predict_classes", &ingenuityml::ensemble::HistGradientBoostingClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::HistGradientBoostingClassifier::predict_proba)
        .def("predict", &ingenuityml::ensemble::HistGradientBoostingClassifier::predict_classes)
        .def("get_params", &ingenuityml::ensemble::HistGradientBoostingClassifier::get_params)
        .def("set_params", &ingenuityml::ensemble::HistGradientBoostingClassifier::set_params)
        .def("is_fitted", &ingenuityml::ensemble::HistGradientBoostingClassifier::is_fitted)
        .def("max_iter", &ingenuityml::ensemble::HistGradientBoostingClassifier::max_iter)
        .def("learning_rate", &ingenuityml::ensemble::HistGradientBoostingClassifier::learning_rate)
        .def("classes", &ingenuityml::ensemble::HistGradientBoostingClassifier::classes)
        .def("save", &ingenuityml::ensemble::HistGradientBoostingClassifier::save)
        .def("load", &ingenuityml::ensemble::HistGradientBoostingClassifier::load);

    py::class_<ingenuityml::ensemble::HistGradientBoostingRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(gb_module, "HistGradientBoostingRegressor")
        .def(py::init<int,double,int,int,int,double,int>(),
             py::arg("max_iter")=100, py::arg("learning_rate")=0.1, py::arg("max_depth")=3,
             py::arg("min_samples_split")=2, py::arg("min_samples_leaf")=1,
             py::arg("min_impurity_decrease")=0.0, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::ensemble::HistGradientBoostingRegressor::fit)
        .def("predict", &ingenuityml::ensemble::HistGradientBoostingRegressor::predict)
        .def("get_params", &ingenuityml::ensemble::HistGradientBoostingRegressor::get_params)
        .def("set_params", &ingenuityml::ensemble::HistGradientBoostingRegressor::set_params)
        .def("is_fitted", &ingenuityml::ensemble::HistGradientBoostingRegressor::is_fitted)
        .def("max_iter", &ingenuityml::ensemble::HistGradientBoostingRegressor::max_iter)
        .def("learning_rate", &ingenuityml::ensemble::HistGradientBoostingRegressor::learning_rate)
        .def("save", &ingenuityml::ensemble::HistGradientBoostingRegressor::save)
        .def("load", &ingenuityml::ensemble::HistGradientBoostingRegressor::load);

    py::class_<ingenuityml::ensemble::GradientBoostingRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(gb_module, "GradientBoostingRegressor")
        .def(py::init<int,double,int,int,int,double,int>(),
             py::arg("n_estimators")=100, py::arg("learning_rate")=0.1, py::arg("max_depth")=3,
             py::arg("min_samples_split")=2, py::arg("min_samples_leaf")=1, 
             py::arg("min_impurity_decrease")=0.0, py::arg("random_state")=-1)
        .def("fit", &ingenuityml::ensemble::GradientBoostingRegressor::fit)
        .def("predict", &ingenuityml::ensemble::GradientBoostingRegressor::predict)
        .def("get_params", &ingenuityml::ensemble::GradientBoostingRegressor::get_params)
        .def("set_params", &ingenuityml::ensemble::GradientBoostingRegressor::set_params)
        .def("is_fitted", &ingenuityml::ensemble::GradientBoostingRegressor::is_fitted)
        .def("n_estimators", &ingenuityml::ensemble::GradientBoostingRegressor::n_estimators)
        .def("learning_rate", &ingenuityml::ensemble::GradientBoostingRegressor::learning_rate)
        .def("save", &ingenuityml::ensemble::GradientBoostingRegressor::save)
        .def("load", &ingenuityml::ensemble::GradientBoostingRegressor::load);

    // AdaBoost
    py::module_ adaboost_module = m.def_submodule("adaboost", "AdaBoost algorithms");
    py::class_<ingenuityml::ensemble::AdaBoostClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(adaboost_module, "AdaBoostClassifier")
        .def(py::init<int, double, int>(),
             py::arg("n_estimators") = 50, py::arg("learning_rate") = 1.0, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::ensemble::AdaBoostClassifier::fit)
        .def("predict", &ingenuityml::ensemble::AdaBoostClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::AdaBoostClassifier::predict_proba)
        .def("decision_function", &ingenuityml::ensemble::AdaBoostClassifier::decision_function)
        .def("get_params", &ingenuityml::ensemble::AdaBoostClassifier::get_params)
        .def("set_params", &ingenuityml::ensemble::AdaBoostClassifier::set_params)
        .def("set_params", [](ingenuityml::ensemble::AdaBoostClassifier& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::ensemble::AdaBoostClassifier::is_fitted)
        .def("classes", &ingenuityml::ensemble::AdaBoostClassifier::classes);

    py::class_<ingenuityml::ensemble::AdaBoostRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(adaboost_module, "AdaBoostRegressor")
        .def(py::init<int, double, std::string, int>(),
             py::arg("n_estimators") = 50, py::arg("learning_rate") = 1.0,
             py::arg("loss") = "linear", py::arg("random_state") = -1)
        .def("fit", &ingenuityml::ensemble::AdaBoostRegressor::fit)
        .def("predict", &ingenuityml::ensemble::AdaBoostRegressor::predict)
        .def("get_params", &ingenuityml::ensemble::AdaBoostRegressor::get_params)
        .def("set_params", &ingenuityml::ensemble::AdaBoostRegressor::set_params)
        .def("set_params", [](ingenuityml::ensemble::AdaBoostRegressor& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::ensemble::AdaBoostRegressor::is_fitted);

    // XGBoost
    py::module_ xgb_module = m.def_submodule("xgboost", "XGBoost algorithms");
    py::class_<ingenuityml::ensemble::XGBClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(xgb_module, "XGBClassifier")
        .def(py::init<int, double, int, double, double, double, int, double, double, int>(),
             py::arg("n_estimators") = 100, py::arg("learning_rate") = 0.1,
             py::arg("max_depth") = 6, py::arg("gamma") = 0.0, py::arg("reg_alpha") = 0.0,
             py::arg("reg_lambda") = 1.0, py::arg("min_child_weight") = 1,
             py::arg("subsample") = 1.0, py::arg("colsample_bytree") = 1.0,
             py::arg("random_state") = -1)
        .def("fit", &ingenuityml::ensemble::XGBClassifier::fit)
        .def("predict", &ingenuityml::ensemble::XGBClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::XGBClassifier::predict_proba)
        .def("decision_function", &ingenuityml::ensemble::XGBClassifier::decision_function)
        .def("get_params", &ingenuityml::ensemble::XGBClassifier::get_params)
        .def("set_params", &ingenuityml::ensemble::XGBClassifier::set_params)
        .def("set_params", [](ingenuityml::ensemble::XGBClassifier& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::ensemble::XGBClassifier::is_fitted)
        .def("classes", &ingenuityml::ensemble::XGBClassifier::classes);

    py::class_<ingenuityml::ensemble::XGBRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(xgb_module, "XGBRegressor")
        .def(py::init<int, double, int, double, double, double, int, double, double, int>(),
             py::arg("n_estimators") = 100, py::arg("learning_rate") = 0.1,
             py::arg("max_depth") = 6, py::arg("gamma") = 0.0, py::arg("reg_alpha") = 0.0,
             py::arg("reg_lambda") = 1.0, py::arg("min_child_weight") = 1,
             py::arg("subsample") = 1.0, py::arg("colsample_bytree") = 1.0,
             py::arg("random_state") = -1)
        .def("fit", &ingenuityml::ensemble::XGBRegressor::fit)
        .def("predict", &ingenuityml::ensemble::XGBRegressor::predict)
        .def("get_params", &ingenuityml::ensemble::XGBRegressor::get_params)
        .def("set_params", &ingenuityml::ensemble::XGBRegressor::set_params)
        .def("set_params", [](ingenuityml::ensemble::XGBRegressor& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::ensemble::XGBRegressor::is_fitted);

    // CatBoost
    py::module_ catboost_module = m.def_submodule("catboost", "CatBoost algorithms");
    py::class_<ingenuityml::ensemble::CatBoostClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(catboost_module, "CatBoostClassifier")
        .def(py::init<int, double, int, double, double, double, int>(),
             py::arg("n_estimators") = 100, py::arg("learning_rate") = 0.03,
             py::arg("max_depth") = 6, py::arg("l2_leaf_reg") = 3.0,
             py::arg("border_count") = 32.0, py::arg("bagging_temperature") = 1.0,
             py::arg("random_state") = -1)
        .def("fit", &ingenuityml::ensemble::CatBoostClassifier::fit)
        .def("predict", &ingenuityml::ensemble::CatBoostClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::CatBoostClassifier::predict_proba)
        .def("decision_function", &ingenuityml::ensemble::CatBoostClassifier::decision_function)
        .def("get_params", &ingenuityml::ensemble::CatBoostClassifier::get_params)
        .def("set_params", &ingenuityml::ensemble::CatBoostClassifier::set_params)
        .def("set_params", [](ingenuityml::ensemble::CatBoostClassifier& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::ensemble::CatBoostClassifier::is_fitted)
        .def("classes", &ingenuityml::ensemble::CatBoostClassifier::classes);

    py::class_<ingenuityml::ensemble::CatBoostRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(catboost_module, "CatBoostRegressor")
        .def(py::init<int, double, int, double, double, double, int>(),
             py::arg("n_estimators") = 100, py::arg("learning_rate") = 0.03,
             py::arg("max_depth") = 6, py::arg("l2_leaf_reg") = 3.0,
             py::arg("border_count") = 32.0, py::arg("bagging_temperature") = 1.0,
             py::arg("random_state") = -1)
        .def("fit", &ingenuityml::ensemble::CatBoostRegressor::fit)
        .def("predict", &ingenuityml::ensemble::CatBoostRegressor::predict)
        .def("get_params", &ingenuityml::ensemble::CatBoostRegressor::get_params)
        .def("set_params", &ingenuityml::ensemble::CatBoostRegressor::set_params)
        .def("set_params", [](ingenuityml::ensemble::CatBoostRegressor& self, py::kwargs kwargs) -> void {
            self.set_params(params_from_kwargs(kwargs));
        })
        .def("is_fitted", &ingenuityml::ensemble::CatBoostRegressor::is_fitted);

    py::class_<ingenuityml::model_selection::GridSearchCV>(model_selection_module, "GridSearchCV")
        .def(py::init<ingenuityml::Estimator&, const std::vector<ingenuityml::Params>&, const ingenuityml::model_selection::BaseCrossValidator&, std::string, int, bool>(),
             py::arg("estimator"), py::arg("param_grid"), py::arg("cv"), py::arg("scoring") = "accuracy",
             py::arg("n_jobs") = 1, py::arg("verbose") = false)
        .def("fit", &ingenuityml::model_selection::GridSearchCV::fit)
        .def("predict", &ingenuityml::model_selection::GridSearchCV::predict)
        .def("best_params", &ingenuityml::model_selection::GridSearchCV::best_params)
        .def("best_score", &ingenuityml::model_selection::GridSearchCV::best_score)
        .def("get_params", &ingenuityml::model_selection::GridSearchCV::get_params)
        .def("set_params", &ingenuityml::model_selection::GridSearchCV::set_params)
        .def("get_n_splits", &ingenuityml::model_selection::GridSearchCV::get_n_splits);
    
    py::class_<ingenuityml::model_selection::RandomizedSearchCV>(model_selection_module, "RandomizedSearchCV")
        .def(py::init<ingenuityml::Estimator&, const std::vector<ingenuityml::Params>&, const ingenuityml::model_selection::BaseCrossValidator&, std::string, int, int, bool>(),
             py::arg("estimator"), py::arg("param_distributions"), py::arg("cv"), py::arg("scoring") = "accuracy",
             py::arg("n_iter") = 10, py::arg("n_jobs") = 1, py::arg("verbose") = false)
        .def("fit", &ingenuityml::model_selection::RandomizedSearchCV::fit)
        .def("predict", &ingenuityml::model_selection::RandomizedSearchCV::predict)
        .def("best_params", &ingenuityml::model_selection::RandomizedSearchCV::best_params)
        .def("best_score", &ingenuityml::model_selection::RandomizedSearchCV::best_score)
        .def("get_params", &ingenuityml::model_selection::RandomizedSearchCV::get_params)
        .def("set_params", &ingenuityml::model_selection::RandomizedSearchCV::set_params)
        .def("get_n_splits", &ingenuityml::model_selection::RandomizedSearchCV::get_n_splits);

    py::class_<ingenuityml::model_selection::HalvingGridSearchCV>(model_selection_module, "HalvingGridSearchCV")
        .def(py::init<ingenuityml::Estimator&, const std::vector<ingenuityml::Params>&, const ingenuityml::model_selection::BaseCrossValidator&, std::string, int, int, bool, int, bool>(),
             py::arg("estimator"), py::arg("param_grid"), py::arg("cv"), py::arg("scoring") = "accuracy",
             py::arg("factor") = 3, py::arg("min_resources") = 1, py::arg("aggressive_elimination") = false,
             py::arg("n_jobs") = 1, py::arg("verbose") = false)
        .def("fit", &ingenuityml::model_selection::HalvingGridSearchCV::fit)
        .def("predict", &ingenuityml::model_selection::HalvingGridSearchCV::predict)
        .def("best_params", &ingenuityml::model_selection::HalvingGridSearchCV::best_params)
        .def("best_score", &ingenuityml::model_selection::HalvingGridSearchCV::best_score)
        .def("get_params", &ingenuityml::model_selection::HalvingGridSearchCV::get_params)
        .def("set_params", &ingenuityml::model_selection::HalvingGridSearchCV::set_params)
        .def("is_fitted", &ingenuityml::model_selection::HalvingGridSearchCV::is_fitted)
        .def("get_n_splits", &ingenuityml::model_selection::HalvingGridSearchCV::get_n_splits);

    py::class_<ingenuityml::model_selection::HalvingRandomSearchCV>(model_selection_module, "HalvingRandomSearchCV")
        .def(py::init([&](ingenuityml::Estimator& estimator, const py::dict& param_distributions,
                          ingenuityml::model_selection::BaseCrossValidator& cv, const std::string& scoring,
                          int n_candidates, int factor, int min_resources, bool aggressive_elimination,
                          int random_state, int n_jobs, bool verbose) {
            return new ingenuityml::model_selection::HalvingRandomSearchCV(
                estimator, param_map_from_py(param_distributions), cv, scoring, n_candidates, factor,
                min_resources, aggressive_elimination, random_state, n_jobs, verbose);
        }),
             py::arg("estimator"), py::arg("param_distributions"), py::arg("cv"), py::arg("scoring") = "accuracy",
             py::arg("n_candidates") = 10, py::arg("factor") = 3, py::arg("min_resources") = 1,
             py::arg("aggressive_elimination") = false, py::arg("random_state") = -1,
             py::arg("n_jobs") = 1, py::arg("verbose") = false)
        .def("fit", &ingenuityml::model_selection::HalvingRandomSearchCV::fit)
        .def("predict", &ingenuityml::model_selection::HalvingRandomSearchCV::predict)
        .def("best_params", &ingenuityml::model_selection::HalvingRandomSearchCV::best_params)
        .def("best_score", &ingenuityml::model_selection::HalvingRandomSearchCV::best_score)
        .def("get_params", &ingenuityml::model_selection::HalvingRandomSearchCV::get_params)
        .def("set_params", &ingenuityml::model_selection::HalvingRandomSearchCV::set_params)
        .def("is_fitted", &ingenuityml::model_selection::HalvingRandomSearchCV::is_fitted)
        .def("get_n_splits", &ingenuityml::model_selection::HalvingRandomSearchCV::get_n_splits);

    py::class_<ingenuityml::model_selection::TimeSeriesSplit, ingenuityml::model_selection::BaseCrossValidator>(model_selection_module, "TimeSeriesSplit")
        .def(py::init<int, int, int, int>(), py::arg("n_splits") = 5, py::arg("max_train_size") = -1, py::arg("test_size") = -1, py::arg("gap") = 0)
        .def("split", [](const ingenuityml::model_selection::TimeSeriesSplit& self, const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y) {
            auto splits = self.split(X, y);
            py::list result;
            for (const auto& split : splits) {
                py::tuple split_tuple = py::make_tuple(
                    py::array_t<int>(split.first.size(), split.first.data()),
                    py::array_t<int>(split.second.size(), split.second.data())
                );
                result.append(split_tuple);
            }
            return result;
        }, py::arg("X"), py::arg("y") = ingenuityml::VectorXd())
        .def("get_n_splits", &ingenuityml::model_selection::TimeSeriesSplit::get_n_splits)
        .def("get_params", &ingenuityml::model_selection::TimeSeriesSplit::get_params)
        .def("set_params", &ingenuityml::model_selection::TimeSeriesSplit::set_params);
    
    // Pipeline module
    py::module_ pipeline_module = m.def_submodule("pipeline", "Pipeline and FeatureUnion utilities");
    
    // Helper lambda to create Pipeline from Python list of (name, estimator) tuples
    auto pipeline_init = [](py::list steps_py) {
        std::vector<std::pair<std::string, std::shared_ptr<ingenuityml::Estimator>>> steps;
        for (auto item : steps_py) {
            py::tuple step_tuple = py::cast<py::tuple>(item);
            if (step_tuple.size() != 2) {
                throw std::runtime_error("Each step must be a (name, estimator) tuple");
            }
            std::string name = py::cast<std::string>(step_tuple[0]);
            py::object estimator_obj = step_tuple[1];
            // Extract raw pointer and wrap in shared_ptr with no-op deleter (Python manages lifetime)
            ingenuityml::Estimator* estimator_ptr = estimator_obj.cast<ingenuityml::Estimator*>();
            std::shared_ptr<ingenuityml::Estimator> estimator(estimator_ptr, [](ingenuityml::Estimator*) {});
            steps.push_back({name, estimator});
        }
        return new ingenuityml::pipeline::Pipeline(steps);
    };
    
    py::class_<ingenuityml::pipeline::Pipeline>(pipeline_module, "Pipeline")
        .def(py::init(pipeline_init), py::arg("steps"))
        .def("fit", &ingenuityml::pipeline::Pipeline::fit, py::arg("X"), py::arg("y"))
        .def("transform", &ingenuityml::pipeline::Pipeline::transform, py::arg("X"))
        .def("predict", &ingenuityml::pipeline::Pipeline::predict, py::arg("X"))
        .def("predict_classes", &ingenuityml::pipeline::Pipeline::predict_classes, py::arg("X"))
        .def("predict_proba", &ingenuityml::pipeline::Pipeline::predict_proba, py::arg("X"))
        .def("fit_transform", &ingenuityml::pipeline::Pipeline::fit_transform, py::arg("X"), py::arg("y"))
        .def("get_params", &ingenuityml::pipeline::Pipeline::get_params)
        .def("set_params", &ingenuityml::pipeline::Pipeline::set_params)
        .def("is_fitted", &ingenuityml::pipeline::Pipeline::is_fitted)
        .def("get_step", &ingenuityml::pipeline::Pipeline::get_step, py::arg("name"))
        .def("get_step_names", &ingenuityml::pipeline::Pipeline::get_step_names);
    
    // Helper lambda to create FeatureUnion from Python list of (name, transformer) tuples
    auto featureunion_init = [](py::list transformers_py) {
        std::vector<std::pair<std::string, std::shared_ptr<ingenuityml::Transformer>>> transformers;
        for (auto item : transformers_py) {
            py::tuple transformer_tuple = py::cast<py::tuple>(item);
            if (transformer_tuple.size() != 2) {
                throw std::runtime_error("Each transformer must be a (name, transformer) tuple");
            }
            std::string name = py::cast<std::string>(transformer_tuple[0]);
            py::object transformer_obj = transformer_tuple[1];
            // Extract raw pointer and wrap in shared_ptr with no-op deleter (Python manages lifetime)
            ingenuityml::Transformer* transformer_ptr = transformer_obj.cast<ingenuityml::Transformer*>();
            std::shared_ptr<ingenuityml::Transformer> transformer(transformer_ptr, [](ingenuityml::Transformer*) {});
            transformers.push_back({name, transformer});
        }
        return new ingenuityml::pipeline::FeatureUnion(transformers);
    };
    
    py::class_<ingenuityml::pipeline::FeatureUnion>(pipeline_module, "FeatureUnion")
        .def(py::init(featureunion_init), py::arg("transformers"))
        .def("fit", &ingenuityml::pipeline::FeatureUnion::fit, py::arg("X"), py::arg("y"))
        .def("transform", &ingenuityml::pipeline::FeatureUnion::transform, py::arg("X"))
        .def("fit_transform", &ingenuityml::pipeline::FeatureUnion::fit_transform, py::arg("X"), py::arg("y"))
        .def("get_params", &ingenuityml::pipeline::FeatureUnion::get_params)
        .def("set_params", &ingenuityml::pipeline::FeatureUnion::set_params)
        .def("is_fitted", &ingenuityml::pipeline::FeatureUnion::is_fitted)
        .def("get_transformer", &ingenuityml::pipeline::FeatureUnion::get_transformer, py::arg("name"))
        .def("get_transformer_names", &ingenuityml::pipeline::FeatureUnion::get_transformer_names);
    
    // Compose module
    py::module_ compose_module = m.def_submodule("compose", "Composition utilities");
    
    // Helper lambda to create ColumnTransformer from Python list
    auto columntransformer_init = [](py::list transformers_py, const std::string& remainder = "drop", double sparse_threshold = 0.3) {
        std::vector<std::tuple<std::string, std::shared_ptr<ingenuityml::Transformer>, std::vector<int>>> transformers;
        for (auto item : transformers_py) {
            py::tuple transformer_tuple = py::cast<py::tuple>(item);
            if (transformer_tuple.size() != 3) {
                throw std::runtime_error("Each transformer must be a (name, transformer, column_indices) tuple");
            }
            std::string name = py::cast<std::string>(transformer_tuple[0]);
            py::object transformer_obj = transformer_tuple[1];
            ingenuityml::Transformer* transformer_ptr = transformer_obj.cast<ingenuityml::Transformer*>();
            std::shared_ptr<ingenuityml::Transformer> transformer(transformer_ptr, [](ingenuityml::Transformer*) {});
            py::list columns_py = py::cast<py::list>(transformer_tuple[2]);
            std::vector<int> column_indices;
            for (auto col : columns_py) {
                column_indices.push_back(py::cast<int>(col));
            }
            transformers.push_back(std::make_tuple(name, transformer, column_indices));
        }
        return new ingenuityml::compose::ColumnTransformer(transformers, remainder, sparse_threshold);
    };
    
    py::class_<ingenuityml::compose::ColumnTransformer>(compose_module, "ColumnTransformer")
        .def(py::init(columntransformer_init), 
             py::arg("transformers"), py::arg("remainder") = "drop", py::arg("sparse_threshold") = 0.3)
        .def("fit", &ingenuityml::compose::ColumnTransformer::fit, py::arg("X"), py::arg("y"))
        .def("transform", &ingenuityml::compose::ColumnTransformer::transform, py::arg("X"))
        .def("fit_transform", &ingenuityml::compose::ColumnTransformer::fit_transform, py::arg("X"), py::arg("y"))
        .def("get_params", &ingenuityml::compose::ColumnTransformer::get_params)
        .def("set_params", &ingenuityml::compose::ColumnTransformer::set_params)
        .def("is_fitted", &ingenuityml::compose::ColumnTransformer::is_fitted)
        .def("get_transformer", &ingenuityml::compose::ColumnTransformer::get_transformer, py::arg("name"))
        .def("get_transformer_names", &ingenuityml::compose::ColumnTransformer::get_transformer_names);
    
    // Helper lambda to create TransformedTargetRegressor
    auto transformedtargetregressor_init = [](py::object regressor_py, py::object transformer_py = py::none()) {
        ingenuityml::Regressor* regressor_ptr = regressor_py.cast<ingenuityml::Regressor*>();
        std::shared_ptr<ingenuityml::Regressor> regressor(regressor_ptr, [](ingenuityml::Regressor*) {});
        std::shared_ptr<ingenuityml::Transformer> transformer = nullptr;
        if (!transformer_py.is_none()) {
            ingenuityml::Transformer* transformer_ptr = transformer_py.cast<ingenuityml::Transformer*>();
            transformer = std::shared_ptr<ingenuityml::Transformer>(transformer_ptr, [](ingenuityml::Transformer*) {});
        }
        return new ingenuityml::compose::TransformedTargetRegressor(regressor, transformer);
    };
    
    py::class_<ingenuityml::compose::TransformedTargetRegressor>(compose_module, "TransformedTargetRegressor")
        .def(py::init(transformedtargetregressor_init), 
             py::arg("regressor"), py::arg("transformer") = py::none())
        .def("fit", &ingenuityml::compose::TransformedTargetRegressor::fit, py::arg("X"), py::arg("y"))
        .def("predict", &ingenuityml::compose::TransformedTargetRegressor::predict, py::arg("X"))
        .def("get_params", &ingenuityml::compose::TransformedTargetRegressor::get_params)
        .def("set_params", &ingenuityml::compose::TransformedTargetRegressor::set_params)
        .def("is_fitted", &ingenuityml::compose::TransformedTargetRegressor::is_fitted)
        .def("regressor", &ingenuityml::compose::TransformedTargetRegressor::regressor)
        .def("transformer", &ingenuityml::compose::TransformedTargetRegressor::transformer);
    
    // Feature selection module
    py::module_ feature_selection_module = m.def_submodule("feature_selection", "Feature selection utilities");
    
    py::class_<ingenuityml::feature_selection::VarianceThreshold, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "VarianceThreshold")
        .def(py::init<double>(), py::arg("threshold") = 0.0)
        .def("fit", &ingenuityml::feature_selection::VarianceThreshold::fit)
        .def("transform", &ingenuityml::feature_selection::VarianceThreshold::transform)
        .def("fit_transform", &ingenuityml::feature_selection::VarianceThreshold::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::VarianceThreshold::get_params)
        .def("set_params", &ingenuityml::feature_selection::VarianceThreshold::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::VarianceThreshold::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::VarianceThreshold::get_support);
    
    auto score_func_from_py = [](py::object score_func_py) {
        return [score_func_py](const ingenuityml::VectorXd& X_feature, const ingenuityml::VectorXd& y) {
            py::array_t<double> X_arr = py::cast(X_feature);
            py::array_t<double> y_arr = py::cast(y);
            py::object result = score_func_py(X_arr, y_arr);
            return py::cast<double>(result);
        };
    };

    // Helper lambda to create SelectKBest with scoring function
    auto selectkbest_init = [&](py::object score_func_py, int k = 10) {
        return new ingenuityml::feature_selection::SelectKBest(score_func_from_py(score_func_py), k);
    };
    
    py::class_<ingenuityml::feature_selection::SelectKBest, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "SelectKBest")
        .def(py::init(selectkbest_init), py::arg("score_func"), py::arg("k") = 10)
        .def("fit", &ingenuityml::feature_selection::SelectKBest::fit)
        .def("transform", &ingenuityml::feature_selection::SelectKBest::transform)
        .def("fit_transform", &ingenuityml::feature_selection::SelectKBest::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::SelectKBest::get_params)
        .def("set_params", &ingenuityml::feature_selection::SelectKBest::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::SelectKBest::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::SelectKBest::get_support)
        .def("scores", &ingenuityml::feature_selection::SelectKBest::scores);
    
    // Helper lambda to create SelectPercentile with scoring function
    auto selectpercentile_init = [&](py::object score_func_py, int percentile = 10) {
        return new ingenuityml::feature_selection::SelectPercentile(score_func_from_py(score_func_py), percentile);
    };
    
    py::class_<ingenuityml::feature_selection::SelectPercentile, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "SelectPercentile")
        .def(py::init(selectpercentile_init), py::arg("score_func"), py::arg("percentile") = 10)
        .def("fit", &ingenuityml::feature_selection::SelectPercentile::fit)
        .def("transform", &ingenuityml::feature_selection::SelectPercentile::transform)
        .def("fit_transform", &ingenuityml::feature_selection::SelectPercentile::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::SelectPercentile::get_params)
        .def("set_params", &ingenuityml::feature_selection::SelectPercentile::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::SelectPercentile::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::SelectPercentile::get_support)
        .def("scores", &ingenuityml::feature_selection::SelectPercentile::scores);

    auto selectfpr_init = [&](py::object score_func_py, double alpha = 0.05) {
        return new ingenuityml::feature_selection::SelectFpr(score_func_from_py(score_func_py), alpha);
    };
    auto selectfdr_init = [&](py::object score_func_py, double alpha = 0.05) {
        return new ingenuityml::feature_selection::SelectFdr(score_func_from_py(score_func_py), alpha);
    };
    auto selectfwe_init = [&](py::object score_func_py, double alpha = 0.05) {
        return new ingenuityml::feature_selection::SelectFwe(score_func_from_py(score_func_py), alpha);
    };
    auto generic_univariate_init = [&](py::object score_func_py, const std::string& mode, double param) {
        return new ingenuityml::feature_selection::GenericUnivariateSelect(score_func_from_py(score_func_py), mode, param);
    };

    py::class_<ingenuityml::feature_selection::SelectFpr, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "SelectFpr")
        .def(py::init(selectfpr_init), py::arg("score_func"), py::arg("alpha") = 0.05)
        .def("fit", &ingenuityml::feature_selection::SelectFpr::fit)
        .def("transform", &ingenuityml::feature_selection::SelectFpr::transform)
        .def("fit_transform", &ingenuityml::feature_selection::SelectFpr::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::SelectFpr::get_params)
        .def("set_params", &ingenuityml::feature_selection::SelectFpr::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::SelectFpr::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::SelectFpr::get_support)
        .def("scores", &ingenuityml::feature_selection::SelectFpr::scores);

    py::class_<ingenuityml::feature_selection::SelectFdr, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "SelectFdr")
        .def(py::init(selectfdr_init), py::arg("score_func"), py::arg("alpha") = 0.05)
        .def("fit", &ingenuityml::feature_selection::SelectFdr::fit)
        .def("transform", &ingenuityml::feature_selection::SelectFdr::transform)
        .def("fit_transform", &ingenuityml::feature_selection::SelectFdr::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::SelectFdr::get_params)
        .def("set_params", &ingenuityml::feature_selection::SelectFdr::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::SelectFdr::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::SelectFdr::get_support)
        .def("scores", &ingenuityml::feature_selection::SelectFdr::scores);

    py::class_<ingenuityml::feature_selection::SelectFwe, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "SelectFwe")
        .def(py::init(selectfwe_init), py::arg("score_func"), py::arg("alpha") = 0.05)
        .def("fit", &ingenuityml::feature_selection::SelectFwe::fit)
        .def("transform", &ingenuityml::feature_selection::SelectFwe::transform)
        .def("fit_transform", &ingenuityml::feature_selection::SelectFwe::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::SelectFwe::get_params)
        .def("set_params", &ingenuityml::feature_selection::SelectFwe::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::SelectFwe::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::SelectFwe::get_support)
        .def("scores", &ingenuityml::feature_selection::SelectFwe::scores);

    py::class_<ingenuityml::feature_selection::GenericUnivariateSelect, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "GenericUnivariateSelect")
        .def(py::init(generic_univariate_init), py::arg("score_func"), py::arg("mode") = "percentile", py::arg("param") = 10.0)
        .def("fit", &ingenuityml::feature_selection::GenericUnivariateSelect::fit)
        .def("transform", &ingenuityml::feature_selection::GenericUnivariateSelect::transform)
        .def("fit_transform", &ingenuityml::feature_selection::GenericUnivariateSelect::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::GenericUnivariateSelect::get_params)
        .def("set_params", &ingenuityml::feature_selection::GenericUnivariateSelect::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::GenericUnivariateSelect::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::GenericUnivariateSelect::get_support)
        .def("scores", &ingenuityml::feature_selection::GenericUnivariateSelect::scores);

    py::class_<ingenuityml::feature_selection::SelectFromModel, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "SelectFromModel")
        .def(py::init<ingenuityml::Estimator&, double, int>(), py::arg("estimator"), py::arg("threshold") = 0.0, py::arg("max_features") = -1)
        .def("fit", &ingenuityml::feature_selection::SelectFromModel::fit)
        .def("transform", &ingenuityml::feature_selection::SelectFromModel::transform)
        .def("fit_transform", &ingenuityml::feature_selection::SelectFromModel::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::SelectFromModel::get_params)
        .def("set_params", &ingenuityml::feature_selection::SelectFromModel::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::SelectFromModel::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::SelectFromModel::get_support)
        .def("importances", &ingenuityml::feature_selection::SelectFromModel::importances);

    py::class_<ingenuityml::feature_selection::RFE, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "RFE")
        .def(py::init<ingenuityml::Estimator&, int, int>(), py::arg("estimator"), py::arg("n_features_to_select") = -1, py::arg("step") = 1)
        .def("fit", &ingenuityml::feature_selection::RFE::fit)
        .def("transform", &ingenuityml::feature_selection::RFE::transform)
        .def("fit_transform", &ingenuityml::feature_selection::RFE::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::RFE::get_params)
        .def("set_params", &ingenuityml::feature_selection::RFE::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::RFE::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::RFE::get_support);

    py::class_<ingenuityml::feature_selection::RFECV, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "RFECV")
        .def(py::init<ingenuityml::Estimator&, ingenuityml::model_selection::BaseCrossValidator&, int, std::string, int>(),
             py::arg("estimator"), py::arg("cv"), py::arg("step") = 1, py::arg("scoring") = "accuracy",
             py::arg("min_features_to_select") = 1)
        .def("fit", &ingenuityml::feature_selection::RFECV::fit)
        .def("transform", &ingenuityml::feature_selection::RFECV::transform)
        .def("fit_transform", &ingenuityml::feature_selection::RFECV::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::RFECV::get_params)
        .def("set_params", &ingenuityml::feature_selection::RFECV::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::RFECV::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::RFECV::get_support);

    py::class_<ingenuityml::feature_selection::SequentialFeatureSelector, ingenuityml::Estimator, ingenuityml::Transformer>(feature_selection_module, "SequentialFeatureSelector")
        .def(py::init<ingenuityml::Estimator&, ingenuityml::model_selection::BaseCrossValidator&, int, std::string, std::string>(),
             py::arg("estimator"), py::arg("cv"), py::arg("n_features_to_select") = -1,
             py::arg("direction") = "forward", py::arg("scoring") = "accuracy")
        .def("fit", &ingenuityml::feature_selection::SequentialFeatureSelector::fit)
        .def("transform", &ingenuityml::feature_selection::SequentialFeatureSelector::transform)
        .def("fit_transform", &ingenuityml::feature_selection::SequentialFeatureSelector::fit_transform)
        .def("get_params", &ingenuityml::feature_selection::SequentialFeatureSelector::get_params)
        .def("set_params", &ingenuityml::feature_selection::SequentialFeatureSelector::set_params)
        .def("is_fitted", &ingenuityml::feature_selection::SequentialFeatureSelector::is_fitted)
        .def("get_support", &ingenuityml::feature_selection::SequentialFeatureSelector::get_support);
    
    // Scoring functions
    py::module_ scores_module = feature_selection_module.def_submodule("scores", "Scoring functions");
    scores_module.def("f_classif", &ingenuityml::feature_selection::scores::f_classif);
    scores_module.def("f_regression", &ingenuityml::feature_selection::scores::f_regression);
    scores_module.def("mutual_info_classif", &ingenuityml::feature_selection::scores::mutual_info_classif);
    scores_module.def("mutual_info_regression", &ingenuityml::feature_selection::scores::mutual_info_regression);
    scores_module.def("chi2", 
        [](const ingenuityml::VectorXd& X_feature, const ingenuityml::VectorXi& y) {
            return ingenuityml::feature_selection::scores::chi2(X_feature, y);
        });
    
    // Impute module
    py::module_ impute_module = m.def_submodule("impute", "Imputation utilities");
    
    // Helper lambdas for optional y parameter in imputers
    auto knn_imputer_fit = [](ingenuityml::impute::KNNImputer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::impute::KNNImputer& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto knn_imputer_fit_transform = [](ingenuityml::impute::KNNImputer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        return self.fit_transform(X, y);
    };
    auto iterative_imputer_fit = [](ingenuityml::impute::IterativeImputer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::impute::IterativeImputer& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto iterative_imputer_fit_transform = [](ingenuityml::impute::IterativeImputer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        return self.fit_transform(X, y);
    };
    
    py::class_<ingenuityml::impute::KNNImputer, ingenuityml::Estimator, ingenuityml::Transformer>(impute_module, "KNNImputer")
        .def(py::init<int, const std::string&>(), py::arg("n_neighbors") = 5, py::arg("metric") = "euclidean")
        .def("fit", knn_imputer_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &ingenuityml::impute::KNNImputer::transform)
        .def("fit_transform", knn_imputer_fit_transform, py::arg("X"), py::arg("y") = py::none())
        .def("get_params", &ingenuityml::impute::KNNImputer::get_params)
        .def("set_params", &ingenuityml::impute::KNNImputer::set_params)
        .def("is_fitted", &ingenuityml::impute::KNNImputer::is_fitted);
    
    py::class_<ingenuityml::impute::IterativeImputer, ingenuityml::Estimator, ingenuityml::Transformer>(impute_module, "IterativeImputer")
        .def(py::init<int, double, int>(), py::arg("max_iter") = 10, py::arg("tol") = 1e-3, py::arg("random_state") = -1)
        .def("fit", iterative_imputer_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &ingenuityml::impute::IterativeImputer::transform)
        .def("fit_transform", iterative_imputer_fit_transform, py::arg("X"), py::arg("y") = py::none())
        .def("get_params", &ingenuityml::impute::IterativeImputer::get_params)
        .def("set_params", &ingenuityml::impute::IterativeImputer::set_params)
        .def("is_fitted", &ingenuityml::impute::IterativeImputer::is_fitted);

    auto missing_indicator_fit = [](ingenuityml::impute::MissingIndicator& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::impute::MissingIndicator& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto missing_indicator_fit_transform = [](ingenuityml::impute::MissingIndicator& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        return self.fit_transform(X, y);
    };

    py::class_<ingenuityml::impute::MissingIndicator, ingenuityml::Estimator, ingenuityml::Transformer>(impute_module, "MissingIndicator")
        .def(py::init<const std::string&>(), py::arg("features") = "missing-only")
        .def("fit", missing_indicator_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &ingenuityml::impute::MissingIndicator::transform)
        .def("fit_transform", missing_indicator_fit_transform, py::arg("X"), py::arg("y") = py::none())
        .def("get_params", &ingenuityml::impute::MissingIndicator::get_params)
        .def("set_params", &ingenuityml::impute::MissingIndicator::set_params)
        .def("is_fitted", &ingenuityml::impute::MissingIndicator::is_fitted)
        .def("features", &ingenuityml::impute::MissingIndicator::features);
    
    // Utils module
    py::module_ utils_module = m.def_submodule("utils", "Utility functions");
    
    // Multiclass utilities
    py::module_ multiclass_module = utils_module.def_submodule("multiclass", "Multiclass utilities");
    multiclass_module.def("is_multiclass", &ingenuityml::utils::multiclass::is_multiclass);
    multiclass_module.def("unique_labels", &ingenuityml::utils::multiclass::unique_labels);
    multiclass_module.def("type_of_target", &ingenuityml::utils::multiclass::type_of_target);
    
    // Resample utilities
    py::module_ resample_module = utils_module.def_submodule("resample", "Resampling utilities");
    resample_module.def("resample", 
        [](const ingenuityml::MatrixXd& X, const ingenuityml::VectorXd& y, int n_samples, int random_state) {
            auto result = ingenuityml::utils::resample::resample(X, y, n_samples, random_state);
            return py::make_tuple(result.first, result.second);
        }, py::arg("X"), py::arg("y"), py::arg("n_samples") = -1, py::arg("random_state") = -1);
    resample_module.def("shuffle", &ingenuityml::utils::resample::shuffle, 
        py::arg("X"), py::arg("y"), py::arg("random_state") = -1);
    resample_module.def("train_test_split_stratified", 
        [](const ingenuityml::MatrixXd& X, const ingenuityml::VectorXi& y, double test_size, int random_state) {
            auto result = ingenuityml::utils::resample::train_test_split_stratified(X, y, test_size, random_state);
            return py::make_tuple(result.first.first, result.first.second, result.second.first, result.second.second);
        }, py::arg("X"), py::arg("y"), py::arg("test_size") = 0.25, py::arg("random_state") = -1);
    
    // Validation utilities
    py::module_ validation_module = utils_module.def_submodule("validation", "Validation utilities");
    validation_module.def("check_finite", &ingenuityml::utils::validation::check_finite);
    validation_module.def("check_has_nan", &ingenuityml::utils::validation::check_has_nan);
    validation_module.def("check_has_inf", &ingenuityml::utils::validation::check_has_inf);
    
    // Class weight utilities
    py::module_ class_weight_module = utils_module.def_submodule("class_weight", "Class weight utilities");
    class_weight_module.def("compute_class_weight", &ingenuityml::utils::class_weight::compute_class_weight);
    class_weight_module.def("compute_sample_weight", &ingenuityml::utils::class_weight::compute_sample_weight);
    
    // Array utilities
    py::module_ array_module = utils_module.def_submodule("array", "Array utilities");
    array_module.def("issparse", &ingenuityml::utils::array::issparse);
    array_module.def("shape", 
        [](const ingenuityml::MatrixXd& X) {
            auto shape = ingenuityml::utils::array::shape(X);
            return py::make_tuple(shape.first, shape.second);
        });
    
    // Inspection module
    py::module_ inspection_module = m.def_submodule("inspection", "Model inspection utilities");
    
    auto permutationimportance_init = [](py::object estimator_py, const std::string& scoring = "accuracy", int n_repeats = 5, int random_state = -1) {
        ingenuityml::Estimator* estimator_ptr = estimator_py.cast<ingenuityml::Estimator*>();
        std::shared_ptr<ingenuityml::Estimator> estimator(estimator_ptr, [](ingenuityml::Estimator*) {});
        return new ingenuityml::inspection::PermutationImportance(estimator, scoring, n_repeats, random_state);
    };
    
    py::class_<ingenuityml::inspection::PermutationImportance>(inspection_module, "PermutationImportance")
        .def(py::init(permutationimportance_init),
             py::arg("estimator"), py::arg("scoring") = "accuracy", py::arg("n_repeats") = 5, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::inspection::PermutationImportance::fit)
        .def("feature_importances", &ingenuityml::inspection::PermutationImportance::feature_importances);
    
    auto partialdependence_init = [](py::object estimator_py, const std::vector<int>& features) {
        ingenuityml::Predictor* estimator_ptr = estimator_py.cast<ingenuityml::Predictor*>();
        std::shared_ptr<ingenuityml::Predictor> estimator(estimator_ptr, [](ingenuityml::Predictor*) {});
        return new ingenuityml::inspection::PartialDependence(estimator, features);
    };
    
    py::class_<ingenuityml::inspection::PartialDependence>(inspection_module, "PartialDependence")
        .def(py::init(partialdependence_init),
             py::arg("estimator"), py::arg("features"))
        .def("compute", &ingenuityml::inspection::PartialDependence::compute)
        .def("grid", &ingenuityml::inspection::PartialDependence::grid)
        .def("partial_dependence", &ingenuityml::inspection::PartialDependence::partial_dependence);
    
    // Additional ensemble methods
    auto baggingclassifier_init = [](py::object base_estimator_py, int n_estimators = 10, int max_samples = -1, int max_features = -1, int random_state = -1) {
        ingenuityml::Classifier* base_estimator_ptr = base_estimator_py.cast<ingenuityml::Classifier*>();
        std::shared_ptr<ingenuityml::Classifier> base_estimator(base_estimator_ptr, [](ingenuityml::Classifier*) {});
        return new ingenuityml::ensemble::BaggingClassifier(base_estimator, n_estimators, max_samples, max_features, random_state);
    };
    
    py::class_<ingenuityml::ensemble::BaggingClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(rf_module, "BaggingClassifier")
        .def(py::init(baggingclassifier_init), py::arg("base_estimator"), py::arg("n_estimators") = 10, 
             py::arg("max_samples") = -1, py::arg("max_features") = -1, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::ensemble::BaggingClassifier::fit)
        .def("predict", &ingenuityml::ensemble::BaggingClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::BaggingClassifier::predict_proba)
        .def("classes", &ingenuityml::ensemble::BaggingClassifier::classes);
    
    auto baggingregressor_init = [](py::object base_estimator_py, int n_estimators = 10, int max_samples = -1, int max_features = -1, int random_state = -1) {
        ingenuityml::Regressor* base_estimator_ptr = base_estimator_py.cast<ingenuityml::Regressor*>();
        std::shared_ptr<ingenuityml::Regressor> base_estimator(base_estimator_ptr, [](ingenuityml::Regressor*) {});
        return new ingenuityml::ensemble::BaggingRegressor(base_estimator, n_estimators, max_samples, max_features, random_state);
    };
    
    py::class_<ingenuityml::ensemble::BaggingRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(rf_module, "BaggingRegressor")
        .def(py::init(baggingregressor_init), py::arg("base_estimator"), py::arg("n_estimators") = 10, 
             py::arg("max_samples") = -1, py::arg("max_features") = -1, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::ensemble::BaggingRegressor::fit)
        .def("predict", &ingenuityml::ensemble::BaggingRegressor::predict);
    
    auto votingclassifier_init = [](py::list estimators_py, const std::string& voting = "hard") {
        std::vector<std::pair<std::string, std::shared_ptr<ingenuityml::Classifier>>> estimators;
        for (auto item : estimators_py) {
            py::tuple est_tuple = py::cast<py::tuple>(item);
            std::string name = py::cast<std::string>(est_tuple[0]);
            py::object estimator_obj = est_tuple[1];
            ingenuityml::Classifier* estimator_ptr = estimator_obj.cast<ingenuityml::Classifier*>();
            std::shared_ptr<ingenuityml::Classifier> estimator(estimator_ptr, [](ingenuityml::Classifier*) {});
            estimators.push_back({name, estimator});
        }
        return new ingenuityml::ensemble::VotingClassifier(estimators, voting);
    };
    
    py::class_<ingenuityml::ensemble::VotingClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(rf_module, "VotingClassifier")
        .def(py::init(votingclassifier_init), py::arg("estimators"), py::arg("voting") = "hard")
        .def("fit", &ingenuityml::ensemble::VotingClassifier::fit)
        .def("predict", &ingenuityml::ensemble::VotingClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::VotingClassifier::predict_proba)
        .def("classes", &ingenuityml::ensemble::VotingClassifier::classes);
    
    auto votingregressor_init = [](py::list estimators_py) {
        std::vector<std::pair<std::string, std::shared_ptr<ingenuityml::Regressor>>> estimators;
        for (auto item : estimators_py) {
            py::tuple est_tuple = py::cast<py::tuple>(item);
            std::string name = py::cast<std::string>(est_tuple[0]);
            py::object estimator_obj = est_tuple[1];
            ingenuityml::Regressor* estimator_ptr = estimator_obj.cast<ingenuityml::Regressor*>();
            std::shared_ptr<ingenuityml::Regressor> estimator(estimator_ptr, [](ingenuityml::Regressor*) {});
            estimators.push_back({name, estimator});
        }
        return new ingenuityml::ensemble::VotingRegressor(estimators);
    };
    
    py::class_<ingenuityml::ensemble::VotingRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(rf_module, "VotingRegressor")
        .def(py::init(votingregressor_init), py::arg("estimators"))
        .def("fit", &ingenuityml::ensemble::VotingRegressor::fit)
        .def("predict", &ingenuityml::ensemble::VotingRegressor::predict);
    
    auto stackingclassifier_init = [](py::list base_estimators_py, py::object meta_classifier_py) {
        std::vector<std::pair<std::string, std::shared_ptr<ingenuityml::Classifier>>> base_estimators;
        for (auto item : base_estimators_py) {
            py::tuple est_tuple = py::cast<py::tuple>(item);
            std::string name = py::cast<std::string>(est_tuple[0]);
            py::object estimator_obj = est_tuple[1];
            ingenuityml::Classifier* estimator_ptr = estimator_obj.cast<ingenuityml::Classifier*>();
            std::shared_ptr<ingenuityml::Classifier> estimator(estimator_ptr, [](ingenuityml::Classifier*) {});
            base_estimators.push_back({name, estimator});
        }
        ingenuityml::Classifier* meta_classifier_ptr = meta_classifier_py.cast<ingenuityml::Classifier*>();
        std::shared_ptr<ingenuityml::Classifier> meta_classifier(meta_classifier_ptr, [](ingenuityml::Classifier*) {});
        return new ingenuityml::ensemble::StackingClassifier(base_estimators, meta_classifier);
    };
    
    py::class_<ingenuityml::ensemble::StackingClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(rf_module, "StackingClassifier")
        .def(py::init(stackingclassifier_init), py::arg("base_estimators"), py::arg("meta_classifier"))
        .def("fit", &ingenuityml::ensemble::StackingClassifier::fit)
        .def("predict", &ingenuityml::ensemble::StackingClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::ensemble::StackingClassifier::predict_proba)
        .def("classes", &ingenuityml::ensemble::StackingClassifier::classes);
    
    auto stackingregressor_init = [](py::list base_estimators_py, py::object meta_regressor_py) {
        std::vector<std::pair<std::string, std::shared_ptr<ingenuityml::Regressor>>> base_estimators;
        for (auto item : base_estimators_py) {
            py::tuple est_tuple = py::cast<py::tuple>(item);
            std::string name = py::cast<std::string>(est_tuple[0]);
            py::object estimator_obj = est_tuple[1];
            ingenuityml::Regressor* estimator_ptr = estimator_obj.cast<ingenuityml::Regressor*>();
            std::shared_ptr<ingenuityml::Regressor> estimator(estimator_ptr, [](ingenuityml::Regressor*) {});
            base_estimators.push_back({name, estimator});
        }
        ingenuityml::Regressor* meta_regressor_ptr = meta_regressor_py.cast<ingenuityml::Regressor*>();
        std::shared_ptr<ingenuityml::Regressor> meta_regressor(meta_regressor_ptr, [](ingenuityml::Regressor*) {});
        return new ingenuityml::ensemble::StackingRegressor(base_estimators, meta_regressor);
    };
    
    py::class_<ingenuityml::ensemble::StackingRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(rf_module, "StackingRegressor")
        .def(py::init(stackingregressor_init), py::arg("base_estimators"), py::arg("meta_regressor"))
        .def("fit", &ingenuityml::ensemble::StackingRegressor::fit)
        .def("predict", &ingenuityml::ensemble::StackingRegressor::predict);
    
    // Calibration module
    py::module_ calibration_module = m.def_submodule("calibration", "Probability calibration");
    
    auto calibratedclassifiercv_init = [](py::object base_estimator_py, const std::string& method = "sigmoid", int cv = 3) {
        ingenuityml::Classifier* base_estimator_ptr = base_estimator_py.cast<ingenuityml::Classifier*>();
        std::shared_ptr<ingenuityml::Classifier> base_estimator(base_estimator_ptr, [](ingenuityml::Classifier*) {});
        return new ingenuityml::calibration::CalibratedClassifierCV(base_estimator, method, cv);
    };
    
    py::class_<ingenuityml::calibration::CalibratedClassifierCV, ingenuityml::Estimator, ingenuityml::Classifier>(calibration_module, "CalibratedClassifierCV")
        .def(py::init(calibratedclassifiercv_init), py::arg("base_estimator"), py::arg("method") = "sigmoid", py::arg("cv") = 3)
        .def("fit", &ingenuityml::calibration::CalibratedClassifierCV::fit)
        .def("predict", &ingenuityml::calibration::CalibratedClassifierCV::predict_classes)
        .def("predict_proba", &ingenuityml::calibration::CalibratedClassifierCV::predict_proba)
        .def("classes", &ingenuityml::calibration::CalibratedClassifierCV::classes);
    
    // Isotonic module
    py::module_ isotonic_module = m.def_submodule("isotonic", "Isotonic regression");
    
    py::class_<ingenuityml::isotonic::IsotonicRegression, ingenuityml::Estimator, ingenuityml::Regressor>(isotonic_module, "IsotonicRegression")
        .def(py::init<bool>(), py::arg("increasing") = true)
        .def("fit", &ingenuityml::isotonic::IsotonicRegression::fit)
        .def("predict", &ingenuityml::isotonic::IsotonicRegression::predict)
        .def("transform", &ingenuityml::isotonic::IsotonicRegression::transform);
    
    // Discriminant analysis module
    py::module_ discriminant_module = m.def_submodule("discriminant_analysis", "Discriminant analysis");
    
    py::class_<ingenuityml::discriminant_analysis::LinearDiscriminantAnalysis, ingenuityml::Estimator, ingenuityml::Classifier>(discriminant_module, "LinearDiscriminantAnalysis")
        .def(py::init<double>(), py::arg("regularization") = 0.0)
        .def("fit", &ingenuityml::discriminant_analysis::LinearDiscriminantAnalysis::fit)
        .def("predict", &ingenuityml::discriminant_analysis::LinearDiscriminantAnalysis::predict_classes)
        .def("predict_proba", &ingenuityml::discriminant_analysis::LinearDiscriminantAnalysis::predict_proba)
        .def("decision_function", &ingenuityml::discriminant_analysis::LinearDiscriminantAnalysis::decision_function)
        .def("get_params", &ingenuityml::discriminant_analysis::LinearDiscriminantAnalysis::get_params)
        .def("set_params", &ingenuityml::discriminant_analysis::LinearDiscriminantAnalysis::set_params)
        .def("is_fitted", &ingenuityml::discriminant_analysis::LinearDiscriminantAnalysis::is_fitted)
        .def("classes", &ingenuityml::discriminant_analysis::LinearDiscriminantAnalysis::classes);

    py::class_<ingenuityml::discriminant_analysis::QuadraticDiscriminantAnalysis, ingenuityml::Estimator, ingenuityml::Classifier>(discriminant_module, "QuadraticDiscriminantAnalysis")
        .def(py::init<double>(), py::arg("regularization") = 0.0)
        .def("fit", &ingenuityml::discriminant_analysis::QuadraticDiscriminantAnalysis::fit)
        .def("predict", &ingenuityml::discriminant_analysis::QuadraticDiscriminantAnalysis::predict_classes)
        .def("predict_proba", &ingenuityml::discriminant_analysis::QuadraticDiscriminantAnalysis::predict_proba)
        .def("classes", &ingenuityml::discriminant_analysis::QuadraticDiscriminantAnalysis::classes);
    
    // Additional Naive Bayes variants
    py::class_<ingenuityml::naive_bayes::MultinomialNB, ingenuityml::Estimator, ingenuityml::Classifier>(nb_module, "MultinomialNB")
        .def(py::init<double, bool>(), py::arg("alpha") = 1.0, py::arg("fit_prior") = true)
        .def("fit", &ingenuityml::naive_bayes::MultinomialNB::fit)
        .def("predict", &ingenuityml::naive_bayes::MultinomialNB::predict_classes)
        .def("predict_proba", &ingenuityml::naive_bayes::MultinomialNB::predict_proba)
        .def("classes", &ingenuityml::naive_bayes::MultinomialNB::classes);
    
    py::class_<ingenuityml::naive_bayes::BernoulliNB, ingenuityml::Estimator, ingenuityml::Classifier>(nb_module, "BernoulliNB")
        .def(py::init<double, double, bool>(), py::arg("alpha") = 1.0, py::arg("binarize") = 0.0, py::arg("fit_prior") = true)
        .def("fit", &ingenuityml::naive_bayes::BernoulliNB::fit)
        .def("predict", &ingenuityml::naive_bayes::BernoulliNB::predict_classes)
        .def("predict_proba", &ingenuityml::naive_bayes::BernoulliNB::predict_proba)
        .def("classes", &ingenuityml::naive_bayes::BernoulliNB::classes);
    
    py::class_<ingenuityml::naive_bayes::ComplementNB, ingenuityml::Estimator, ingenuityml::Classifier>(nb_module, "ComplementNB")
        .def(py::init<double, bool>(), py::arg("alpha") = 1.0, py::arg("fit_prior") = true)
        .def("fit", &ingenuityml::naive_bayes::ComplementNB::fit)
        .def("predict", &ingenuityml::naive_bayes::ComplementNB::predict_classes)
        .def("predict_proba", &ingenuityml::naive_bayes::ComplementNB::predict_proba)
        .def("classes", &ingenuityml::naive_bayes::ComplementNB::classes);

    py::class_<ingenuityml::naive_bayes::CategoricalNB, ingenuityml::Estimator, ingenuityml::Classifier>(nb_module, "CategoricalNB")
        .def(py::init<double, bool>(), py::arg("alpha") = 1.0, py::arg("fit_prior") = true)
        .def("fit", &ingenuityml::naive_bayes::CategoricalNB::fit)
        .def("predict", &ingenuityml::naive_bayes::CategoricalNB::predict_classes)
        .def("predict_proba", &ingenuityml::naive_bayes::CategoricalNB::predict_proba)
        .def("decision_function", &ingenuityml::naive_bayes::CategoricalNB::decision_function)
        .def("get_params", &ingenuityml::naive_bayes::CategoricalNB::get_params)
        .def("set_params", &ingenuityml::naive_bayes::CategoricalNB::set_params)
        .def("is_fitted", &ingenuityml::naive_bayes::CategoricalNB::is_fitted)
        .def("classes", &ingenuityml::naive_bayes::CategoricalNB::classes)
        .def("n_categories", &ingenuityml::naive_bayes::CategoricalNB::n_categories);
    
    // ExtraTree variants
    py::class_<ingenuityml::tree::ExtraTreeClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(tree_module, "ExtraTreeClassifier")
        .def(py::init<int, int, int, int, int>(), 
             py::arg("max_depth") = -1, py::arg("min_samples_split") = 2, 
             py::arg("min_samples_leaf") = 1, py::arg("max_features") = -1, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::tree::ExtraTreeClassifier::fit)
        .def("predict", &ingenuityml::tree::ExtraTreeClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::tree::ExtraTreeClassifier::predict_proba)
        .def("classes", &ingenuityml::tree::ExtraTreeClassifier::classes);
    
    py::class_<ingenuityml::tree::ExtraTreeRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(tree_module, "ExtraTreeRegressor")
        .def(py::init<int, int, int, int, int>(), 
             py::arg("max_depth") = -1, py::arg("min_samples_split") = 2, 
             py::arg("min_samples_leaf") = 1, py::arg("max_features") = -1, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::tree::ExtraTreeRegressor::fit)
        .def("predict", &ingenuityml::tree::ExtraTreeRegressor::predict);
    
    // Outlier detection module
    py::module_ outlier_module = m.def_submodule("outlier_detection", "Outlier detection");
    
    // Helper lambda for optional y parameter in IsolationForest
    auto isolation_forest_fit = [](ingenuityml::outlier_detection::IsolationForest& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::Estimator& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        return self.fit(X, y);
    };
    
    py::class_<ingenuityml::outlier_detection::IsolationForest, ingenuityml::Estimator>(outlier_module, "IsolationForest")
        .def(py::init<int, int, double, int>(), 
             py::arg("n_estimators") = 100, py::arg("max_samples") = -1, 
             py::arg("contamination") = 0.1, py::arg("random_state") = -1)
        .def("fit", isolation_forest_fit, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("predict", &ingenuityml::outlier_detection::IsolationForest::predict)
        .def("decision_function", &ingenuityml::outlier_detection::IsolationForest::decision_function)
        .def("fit_predict", &ingenuityml::outlier_detection::IsolationForest::fit_predict);
    
    // Helper lambda for optional y parameter in LocalOutlierFactor
    auto lof_fit = [](ingenuityml::outlier_detection::LocalOutlierFactor& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::Estimator& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        return self.fit(X, y);
    };
    
    py::class_<ingenuityml::outlier_detection::LocalOutlierFactor, ingenuityml::Estimator>(outlier_module, "LocalOutlierFactor")
        .def(py::init<int, const std::string&, double>(), 
             py::arg("n_neighbors") = 20, py::arg("metric") = "euclidean", py::arg("contamination") = 0.1)
        .def("fit", lof_fit, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("predict", &ingenuityml::outlier_detection::LocalOutlierFactor::predict)
        .def("decision_function", &ingenuityml::outlier_detection::LocalOutlierFactor::decision_function)
        .def("fit_predict", &ingenuityml::outlier_detection::LocalOutlierFactor::fit_predict);

    outlier_module.attr("EllipticEnvelope") = covariance_module.attr("EllipticEnvelope");

    // Meta-estimators
    py::module_ meta_module = m.def_submodule("meta", "Meta-estimators");

    auto classifier_factory_from_py = [](py::function factory) {
        return [factory]() -> std::shared_ptr<ingenuityml::Classifier> {
            py::gil_scoped_acquire gil;
            py::object obj = factory();
            auto* ptr = obj.cast<ingenuityml::Classifier*>();
            return std::shared_ptr<ingenuityml::Classifier>(ptr, [obj](ingenuityml::Classifier*) mutable {
                py::gil_scoped_acquire gil;
                obj = py::object();
            });
        };
    };

    auto regressor_factory_from_py = [](py::function factory) {
        return [factory]() -> std::shared_ptr<ingenuityml::Regressor> {
            py::gil_scoped_acquire gil;
            py::object obj = factory();
            auto* ptr = obj.cast<ingenuityml::Regressor*>();
            return std::shared_ptr<ingenuityml::Regressor>(ptr, [obj](ingenuityml::Regressor*) mutable {
                py::gil_scoped_acquire gil;
                obj = py::object();
            });
        };
    };

    py::class_<ingenuityml::meta::OneVsRestClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(meta_module, "OneVsRestClassifier")
        .def(py::init([classifier_factory_from_py](py::function factory, int n_jobs) {
            return new ingenuityml::meta::OneVsRestClassifier(classifier_factory_from_py(factory), n_jobs);
        }), py::arg("estimator_factory"), py::arg("n_jobs") = 1)
        .def("fit", &ingenuityml::meta::OneVsRestClassifier::fit)
        .def("predict", &ingenuityml::meta::OneVsRestClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::meta::OneVsRestClassifier::predict_proba)
        .def("decision_function", &ingenuityml::meta::OneVsRestClassifier::decision_function)
        .def("get_params", &ingenuityml::meta::OneVsRestClassifier::get_params)
        .def("set_params", &ingenuityml::meta::OneVsRestClassifier::set_params)
        .def("classes", &ingenuityml::meta::OneVsRestClassifier::classes)
        .def("is_fitted", &ingenuityml::meta::OneVsRestClassifier::is_fitted);

    py::class_<ingenuityml::meta::OneVsOneClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(meta_module, "OneVsOneClassifier")
        .def(py::init([classifier_factory_from_py](py::function factory, int n_jobs) {
            return new ingenuityml::meta::OneVsOneClassifier(classifier_factory_from_py(factory), n_jobs);
        }), py::arg("estimator_factory"), py::arg("n_jobs") = 1)
        .def("fit", &ingenuityml::meta::OneVsOneClassifier::fit)
        .def("predict", &ingenuityml::meta::OneVsOneClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::meta::OneVsOneClassifier::predict_proba)
        .def("decision_function", &ingenuityml::meta::OneVsOneClassifier::decision_function)
        .def("get_params", &ingenuityml::meta::OneVsOneClassifier::get_params)
        .def("set_params", &ingenuityml::meta::OneVsOneClassifier::set_params)
        .def("classes", &ingenuityml::meta::OneVsOneClassifier::classes)
        .def("is_fitted", &ingenuityml::meta::OneVsOneClassifier::is_fitted);

    py::class_<ingenuityml::meta::OutputCodeClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(meta_module, "OutputCodeClassifier")
        .def(py::init([classifier_factory_from_py](py::function factory, int code_size, int random_state) {
            return new ingenuityml::meta::OutputCodeClassifier(classifier_factory_from_py(factory), code_size, random_state);
        }), py::arg("estimator_factory"), py::arg("code_size") = 0, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::meta::OutputCodeClassifier::fit)
        .def("predict", &ingenuityml::meta::OutputCodeClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::meta::OutputCodeClassifier::predict_proba)
        .def("decision_function", &ingenuityml::meta::OutputCodeClassifier::decision_function)
        .def("get_params", &ingenuityml::meta::OutputCodeClassifier::get_params)
        .def("set_params", &ingenuityml::meta::OutputCodeClassifier::set_params)
        .def("classes", &ingenuityml::meta::OutputCodeClassifier::classes)
        .def("code_book", &ingenuityml::meta::OutputCodeClassifier::code_book)
        .def("is_fitted", &ingenuityml::meta::OutputCodeClassifier::is_fitted);

    py::class_<ingenuityml::meta::MultiOutputClassifier, ingenuityml::Estimator>(meta_module, "MultiOutputClassifier")
        .def(py::init([classifier_factory_from_py](py::function factory) {
            return new ingenuityml::meta::MultiOutputClassifier(classifier_factory_from_py(factory));
        }), py::arg("estimator_factory"))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::meta::MultiOutputClassifier::fit))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::meta::MultiOutputClassifier::fit))
        .def("predict", &ingenuityml::meta::MultiOutputClassifier::predict)
        .def("predict_proba", &ingenuityml::meta::MultiOutputClassifier::predict_proba)
        .def("get_params", &ingenuityml::meta::MultiOutputClassifier::get_params)
        .def("set_params", &ingenuityml::meta::MultiOutputClassifier::set_params)
        .def("n_outputs", &ingenuityml::meta::MultiOutputClassifier::n_outputs)
        .def("is_fitted", &ingenuityml::meta::MultiOutputClassifier::is_fitted);

    py::class_<ingenuityml::meta::MultiOutputRegressor, ingenuityml::Estimator>(meta_module, "MultiOutputRegressor")
        .def(py::init([regressor_factory_from_py](py::function factory) {
            return new ingenuityml::meta::MultiOutputRegressor(regressor_factory_from_py(factory));
        }), py::arg("estimator_factory"))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::meta::MultiOutputRegressor::fit))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::meta::MultiOutputRegressor::fit))
        .def("predict", &ingenuityml::meta::MultiOutputRegressor::predict)
        .def("get_params", &ingenuityml::meta::MultiOutputRegressor::get_params)
        .def("set_params", &ingenuityml::meta::MultiOutputRegressor::set_params)
        .def("n_outputs", &ingenuityml::meta::MultiOutputRegressor::n_outputs)
        .def("is_fitted", &ingenuityml::meta::MultiOutputRegressor::is_fitted);

    py::class_<ingenuityml::meta::ClassifierChain, ingenuityml::Estimator>(meta_module, "ClassifierChain")
        .def(py::init([classifier_factory_from_py](py::function factory, const std::vector<int>& order) {
            return new ingenuityml::meta::ClassifierChain(classifier_factory_from_py(factory), order);
        }), py::arg("estimator_factory"), py::arg("order") = std::vector<int>{})
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::meta::ClassifierChain::fit))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::meta::ClassifierChain::fit))
        .def("predict", &ingenuityml::meta::ClassifierChain::predict)
        .def("predict_proba", &ingenuityml::meta::ClassifierChain::predict_proba)
        .def("get_params", &ingenuityml::meta::ClassifierChain::get_params)
        .def("set_params", &ingenuityml::meta::ClassifierChain::set_params)
        .def("order", &ingenuityml::meta::ClassifierChain::order)
        .def("is_fitted", &ingenuityml::meta::ClassifierChain::is_fitted);

    py::class_<ingenuityml::meta::RegressorChain, ingenuityml::Estimator>(meta_module, "RegressorChain")
        .def(py::init([regressor_factory_from_py](py::function factory, const std::vector<int>& order) {
            return new ingenuityml::meta::RegressorChain(regressor_factory_from_py(factory), order);
        }), py::arg("estimator_factory"), py::arg("order") = std::vector<int>{})
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::VectorXd&>(&ingenuityml::meta::RegressorChain::fit))
        .def("fit", py::overload_cast<const ingenuityml::MatrixXd&, const ingenuityml::MatrixXd&>(&ingenuityml::meta::RegressorChain::fit))
        .def("predict", &ingenuityml::meta::RegressorChain::predict)
        .def("get_params", &ingenuityml::meta::RegressorChain::get_params)
        .def("set_params", &ingenuityml::meta::RegressorChain::set_params)
        .def("order", &ingenuityml::meta::RegressorChain::order)
        .def("is_fitted", &ingenuityml::meta::RegressorChain::is_fitted);

    // Mixture module
    py::module_ mixture_module = m.def_submodule("mixture", "Mixture models");
    
    // Helper lambda for optional y parameter in GaussianMixture
    auto gaussian_mixture_fit = [](ingenuityml::mixture::GaussianMixture& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::mixture::GaussianMixture& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    
    py::class_<ingenuityml::mixture::GaussianMixture, ingenuityml::Estimator>(mixture_module, "GaussianMixture")
        .def(py::init<int, int, double, int>(), 
             py::arg("n_components") = 1, py::arg("max_iter") = 100, 
             py::arg("tol") = 1e-3, py::arg("random_state") = -1)
        .def("fit", gaussian_mixture_fit, py::arg("X"), py::arg("y") = py::none())
        .def("predict", &ingenuityml::mixture::GaussianMixture::predict)
        .def("predict_proba", &ingenuityml::mixture::GaussianMixture::predict_proba)
        .def("score_samples", &ingenuityml::mixture::GaussianMixture::score_samples)
        .def("means", &ingenuityml::mixture::GaussianMixture::means)
        .def("covariances", &ingenuityml::mixture::GaussianMixture::covariances)
        .def("weights", &ingenuityml::mixture::GaussianMixture::weights);

    py::class_<ingenuityml::mixture::BayesianGaussianMixture, ingenuityml::Estimator>(mixture_module, "BayesianGaussianMixture")
        .def(py::init<int, int, double, double, int>(),
             py::arg("n_components") = 1, py::arg("max_iter") = 100,
             py::arg("tol") = 1e-3, py::arg("weight_concentration_prior") = 1.0,
             py::arg("random_state") = -1)
        .def("fit", [](ingenuityml::mixture::BayesianGaussianMixture& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::mixture::BayesianGaussianMixture& {
            ingenuityml::VectorXd y_vec = y_py.is_none() ? ingenuityml::VectorXd() : y_py.cast<ingenuityml::VectorXd>();
            return static_cast<ingenuityml::mixture::BayesianGaussianMixture&>(self.fit(X, y_vec));
        }, py::arg("X"), py::arg("y") = py::none())
        .def("predict", &ingenuityml::mixture::BayesianGaussianMixture::predict)
        .def("predict_proba", &ingenuityml::mixture::BayesianGaussianMixture::predict_proba)
        .def("score_samples", &ingenuityml::mixture::BayesianGaussianMixture::score_samples)
        .def("means", &ingenuityml::mixture::BayesianGaussianMixture::means)
        .def("covariances", &ingenuityml::mixture::BayesianGaussianMixture::covariances)
        .def("weights", &ingenuityml::mixture::BayesianGaussianMixture::weights)
        .def("get_params", &ingenuityml::mixture::BayesianGaussianMixture::get_params)
        .def("set_params", &ingenuityml::mixture::BayesianGaussianMixture::set_params)
        .def("is_fitted", &ingenuityml::mixture::BayesianGaussianMixture::is_fitted);

    // Manifold Learning
    py::module_ manifold_module = m.def_submodule("manifold", "Manifold learning algorithms");
    py::class_<ingenuityml::manifold::MDS, ingenuityml::Estimator, ingenuityml::Transformer>(manifold_module, "MDS")
        .def(py::init<int>(), py::arg("n_components") = 2)
        .def("fit", &ingenuityml::manifold::MDS::fit)
        .def("transform", &ingenuityml::manifold::MDS::transform)
        .def("inverse_transform", &ingenuityml::manifold::MDS::inverse_transform)
        .def("fit_transform", &ingenuityml::manifold::MDS::fit_transform)
        .def("get_params", &ingenuityml::manifold::MDS::get_params)
        .def("set_params", &ingenuityml::manifold::MDS::set_params)
        .def("is_fitted", &ingenuityml::manifold::MDS::is_fitted)
        .def("embedding", &ingenuityml::manifold::MDS::embedding);

    py::class_<ingenuityml::manifold::Isomap, ingenuityml::Estimator, ingenuityml::Transformer>(manifold_module, "Isomap")
        .def(py::init<int, int>(), py::arg("n_components") = 2, py::arg("n_neighbors") = 5)
        .def("fit", &ingenuityml::manifold::Isomap::fit)
        .def("transform", &ingenuityml::manifold::Isomap::transform)
        .def("inverse_transform", &ingenuityml::manifold::Isomap::inverse_transform)
        .def("fit_transform", &ingenuityml::manifold::Isomap::fit_transform)
        .def("get_params", &ingenuityml::manifold::Isomap::get_params)
        .def("set_params", &ingenuityml::manifold::Isomap::set_params)
        .def("is_fitted", &ingenuityml::manifold::Isomap::is_fitted)
        .def("embedding", &ingenuityml::manifold::Isomap::embedding);

    py::class_<ingenuityml::manifold::LocallyLinearEmbedding, ingenuityml::Estimator, ingenuityml::Transformer>(manifold_module, "LocallyLinearEmbedding")
        .def(py::init<int, int>(), py::arg("n_components") = 2, py::arg("n_neighbors") = 5)
        .def("fit", &ingenuityml::manifold::LocallyLinearEmbedding::fit)
        .def("transform", &ingenuityml::manifold::LocallyLinearEmbedding::transform)
        .def("inverse_transform", &ingenuityml::manifold::LocallyLinearEmbedding::inverse_transform)
        .def("fit_transform", &ingenuityml::manifold::LocallyLinearEmbedding::fit_transform)
        .def("get_params", &ingenuityml::manifold::LocallyLinearEmbedding::get_params)
        .def("set_params", &ingenuityml::manifold::LocallyLinearEmbedding::set_params)
        .def("is_fitted", &ingenuityml::manifold::LocallyLinearEmbedding::is_fitted)
        .def("embedding", &ingenuityml::manifold::LocallyLinearEmbedding::embedding);

    py::class_<ingenuityml::manifold::SpectralEmbedding, ingenuityml::Estimator, ingenuityml::Transformer>(manifold_module, "SpectralEmbedding")
        .def(py::init<int, int>(), py::arg("n_components") = 2, py::arg("n_neighbors") = 5)
        .def("fit", &ingenuityml::manifold::SpectralEmbedding::fit)
        .def("transform", &ingenuityml::manifold::SpectralEmbedding::transform)
        .def("inverse_transform", &ingenuityml::manifold::SpectralEmbedding::inverse_transform)
        .def("fit_transform", &ingenuityml::manifold::SpectralEmbedding::fit_transform)
        .def("get_params", &ingenuityml::manifold::SpectralEmbedding::get_params)
        .def("set_params", &ingenuityml::manifold::SpectralEmbedding::set_params)
        .def("is_fitted", &ingenuityml::manifold::SpectralEmbedding::is_fitted)
        .def("embedding", &ingenuityml::manifold::SpectralEmbedding::embedding);
    
    // Semi-supervised module
    py::module_ semi_supervised_module = m.def_submodule("semi_supervised", "Semi-supervised learning");
    
    py::class_<ingenuityml::semi_supervised::LabelPropagation, ingenuityml::Estimator, ingenuityml::Classifier>(semi_supervised_module, "LabelPropagation")
        .def(py::init<double, int, double, const std::string&>(), 
             py::arg("gamma") = 20.0, py::arg("max_iter") = 30, 
             py::arg("tol") = 1e-3, py::arg("kernel") = "rbf")
        .def("fit", &ingenuityml::semi_supervised::LabelPropagation::fit)
        .def("predict", &ingenuityml::semi_supervised::LabelPropagation::predict_classes)
        .def("predict_proba", &ingenuityml::semi_supervised::LabelPropagation::predict_proba)
        .def("classes", &ingenuityml::semi_supervised::LabelPropagation::classes);
    
    py::class_<ingenuityml::semi_supervised::LabelSpreading, ingenuityml::Estimator, ingenuityml::Classifier>(semi_supervised_module, "LabelSpreading")
        .def(py::init<double, double, int, double, const std::string&>(), 
             py::arg("alpha") = 0.2, py::arg("gamma") = 20.0, py::arg("max_iter") = 30, 
             py::arg("tol") = 1e-3, py::arg("kernel") = "rbf")
        .def("fit", &ingenuityml::semi_supervised::LabelSpreading::fit)
        .def("predict", &ingenuityml::semi_supervised::LabelSpreading::predict_classes)
        .def("predict_proba", &ingenuityml::semi_supervised::LabelSpreading::predict_proba)
        .def("classes", &ingenuityml::semi_supervised::LabelSpreading::classes);

    py::class_<ingenuityml::semi_supervised::SelfTrainingClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(semi_supervised_module, "SelfTrainingClassifier")
        .def(py::init<int, double, int>(),
             py::arg("n_neighbors") = 5, py::arg("threshold") = 0.75, py::arg("max_iter") = 10)
        .def("fit", &ingenuityml::semi_supervised::SelfTrainingClassifier::fit)
        .def("predict", &ingenuityml::semi_supervised::SelfTrainingClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::semi_supervised::SelfTrainingClassifier::predict_proba)
        .def("decision_function", &ingenuityml::semi_supervised::SelfTrainingClassifier::decision_function)
        .def("get_params", &ingenuityml::semi_supervised::SelfTrainingClassifier::get_params)
        .def("set_params", &ingenuityml::semi_supervised::SelfTrainingClassifier::set_params)
        .def("is_fitted", &ingenuityml::semi_supervised::SelfTrainingClassifier::is_fitted)
        .def("classes", &ingenuityml::semi_supervised::SelfTrainingClassifier::classes);
    
    // Additional preprocessing utilities
    // Helper lambda for optional y parameter in MaxAbsScaler
    auto maxabs_scaler_fit = [](ingenuityml::preprocessing::MaxAbsScaler& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::preprocessing::MaxAbsScaler& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto maxabs_scaler_fit_transform = [](ingenuityml::preprocessing::MaxAbsScaler& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        return self.fit_transform(X, y);
    };
    
    py::class_<ingenuityml::preprocessing::MaxAbsScaler, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "MaxAbsScaler")
        .def(py::init<>())
        .def("fit", maxabs_scaler_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &ingenuityml::preprocessing::MaxAbsScaler::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::MaxAbsScaler::inverse_transform)
        .def("fit_transform", maxabs_scaler_fit_transform, py::arg("X"), py::arg("y") = py::none())
        .def("max_abs", &ingenuityml::preprocessing::MaxAbsScaler::max_abs);
    
    // Helper lambda for optional y parameter in Binarizer
    auto binarizer_fit = [](ingenuityml::preprocessing::Binarizer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::preprocessing::Binarizer& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto binarizer_fit_transform = [](ingenuityml::preprocessing::Binarizer& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        return self.fit_transform(X, y);
    };
    
    py::class_<ingenuityml::preprocessing::Binarizer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "Binarizer")
        .def(py::init<double>(), py::arg("threshold") = 0.0)
        .def("fit", binarizer_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &ingenuityml::preprocessing::Binarizer::transform)
        .def("fit_transform", binarizer_fit_transform, py::arg("X"), py::arg("y") = py::none());

    py::class_<ingenuityml::preprocessing::LabelBinarizer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "LabelBinarizer")
        .def(py::init<int, int>(), py::arg("neg_label") = 0, py::arg("pos_label") = 1)
        .def("fit", &ingenuityml::preprocessing::LabelBinarizer::fit)
        .def("transform", &ingenuityml::preprocessing::LabelBinarizer::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::LabelBinarizer::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::LabelBinarizer::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::LabelBinarizer::get_params)
        .def("set_params", &ingenuityml::preprocessing::LabelBinarizer::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::LabelBinarizer::is_fitted)
        .def("classes", &ingenuityml::preprocessing::LabelBinarizer::classes);

    py::class_<ingenuityml::preprocessing::MultiLabelBinarizer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "MultiLabelBinarizer")
        .def(py::init<>())
        .def("fit", &ingenuityml::preprocessing::MultiLabelBinarizer::fit)
        .def("transform", &ingenuityml::preprocessing::MultiLabelBinarizer::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::MultiLabelBinarizer::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::MultiLabelBinarizer::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::MultiLabelBinarizer::get_params)
        .def("set_params", &ingenuityml::preprocessing::MultiLabelBinarizer::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::MultiLabelBinarizer::is_fitted)
        .def("classes", &ingenuityml::preprocessing::MultiLabelBinarizer::classes);

    py::class_<ingenuityml::preprocessing::KBinsDiscretizer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "KBinsDiscretizer")
        .def(py::init<int, const std::string&, const std::string&>(),
             py::arg("n_bins") = 5, py::arg("encode") = "onehot", py::arg("strategy") = "uniform")
        .def("fit", &ingenuityml::preprocessing::KBinsDiscretizer::fit)
        .def("transform", &ingenuityml::preprocessing::KBinsDiscretizer::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::KBinsDiscretizer::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::KBinsDiscretizer::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::KBinsDiscretizer::get_params)
        .def("set_params", &ingenuityml::preprocessing::KBinsDiscretizer::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::KBinsDiscretizer::is_fitted)
        .def("output_dim", &ingenuityml::preprocessing::KBinsDiscretizer::output_dim);

    py::class_<ingenuityml::preprocessing::QuantileTransformer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "QuantileTransformer")
        .def(py::init<int, const std::string&>(), py::arg("n_quantiles") = 1000, py::arg("output_distribution") = "uniform")
        .def("fit", &ingenuityml::preprocessing::QuantileTransformer::fit)
        .def("transform", &ingenuityml::preprocessing::QuantileTransformer::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::QuantileTransformer::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::QuantileTransformer::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::QuantileTransformer::get_params)
        .def("set_params", &ingenuityml::preprocessing::QuantileTransformer::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::QuantileTransformer::is_fitted);

    py::class_<ingenuityml::preprocessing::PowerTransformer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "PowerTransformer")
        .def(py::init<const std::string&, bool>(), py::arg("method") = "yeo-johnson", py::arg("standardize") = true)
        .def("fit", &ingenuityml::preprocessing::PowerTransformer::fit)
        .def("transform", &ingenuityml::preprocessing::PowerTransformer::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::PowerTransformer::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::PowerTransformer::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::PowerTransformer::get_params)
        .def("set_params", &ingenuityml::preprocessing::PowerTransformer::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::PowerTransformer::is_fitted)
        .def("lambdas", &ingenuityml::preprocessing::PowerTransformer::lambdas);

    py::class_<ingenuityml::preprocessing::FunctionTransformer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "FunctionTransformer")
        .def(py::init<const std::string&, const std::string&, bool>(),
             py::arg("func") = "identity", py::arg("inverse_func") = "identity", py::arg("validate") = true)
        .def("fit", &ingenuityml::preprocessing::FunctionTransformer::fit)
        .def("transform", &ingenuityml::preprocessing::FunctionTransformer::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::FunctionTransformer::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::FunctionTransformer::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::FunctionTransformer::get_params)
        .def("set_params", &ingenuityml::preprocessing::FunctionTransformer::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::FunctionTransformer::is_fitted);

    py::class_<ingenuityml::preprocessing::SplineTransformer, ingenuityml::Estimator, ingenuityml::Transformer>(preprocessing_module, "SplineTransformer")
        .def(py::init<int, int, bool>(), py::arg("n_knots") = 5, py::arg("degree") = 3, py::arg("include_bias") = true)
        .def("fit", &ingenuityml::preprocessing::SplineTransformer::fit)
        .def("transform", &ingenuityml::preprocessing::SplineTransformer::transform)
        .def("inverse_transform", &ingenuityml::preprocessing::SplineTransformer::inverse_transform)
        .def("fit_transform", &ingenuityml::preprocessing::SplineTransformer::fit_transform)
        .def("get_params", &ingenuityml::preprocessing::SplineTransformer::get_params)
        .def("set_params", &ingenuityml::preprocessing::SplineTransformer::set_params)
        .def("is_fitted", &ingenuityml::preprocessing::SplineTransformer::is_fitted)
        .def("output_dim", &ingenuityml::preprocessing::SplineTransformer::output_dim);
    
    // Additional clustering methods
    py::class_<ingenuityml::cluster::SpectralClustering, ingenuityml::Estimator>(cluster_module, "SpectralClustering")
        .def(py::init<int, const std::string&, double, int, int>(), 
             py::arg("n_clusters") = 8, py::arg("affinity") = "rbf", 
             py::arg("gamma") = 1.0, py::arg("n_neighbors") = 10, py::arg("random_state") = -1)
        .def("fit", &ingenuityml::cluster::SpectralClustering::fit)
        .def("fit_predict", &ingenuityml::cluster::SpectralClustering::fit_predict)
        .def("labels", &ingenuityml::cluster::SpectralClustering::labels);
    
    // Helper lambda for optional y parameter in MiniBatchKMeans
    auto minibatch_kmeans_fit = [](ingenuityml::cluster::MiniBatchKMeans& self, const ingenuityml::MatrixXd& X, py::object y_py = py::none()) -> ingenuityml::Estimator& {
        ingenuityml::VectorXd y = y_py.is_none() ? ingenuityml::VectorXd::Zero(X.rows()) : py::cast<ingenuityml::VectorXd>(y_py);
        return self.fit(X, y);
    };
    
    py::class_<ingenuityml::cluster::MiniBatchKMeans, ingenuityml::Estimator>(cluster_module, "MiniBatchKMeans")
        .def(py::init<int, int, double, int, int>(), 
             py::arg("n_clusters") = 8, py::arg("max_iter") = 100, 
             py::arg("tol") = 1e-4, py::arg("batch_size") = 100, py::arg("random_state") = -1)
        .def("fit", minibatch_kmeans_fit, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("fit_predict", &ingenuityml::cluster::MiniBatchKMeans::fit_predict)
        .def("predict", &ingenuityml::cluster::MiniBatchKMeans::predict)
        .def("cluster_centers", &ingenuityml::cluster::MiniBatchKMeans::cluster_centers)
        .def("labels", &ingenuityml::cluster::MiniBatchKMeans::labels);

    py::class_<ingenuityml::cluster::MeanShift, ingenuityml::Estimator>(cluster_module, "MeanShift")
        .def(py::init<double, const std::vector<ingenuityml::VectorXd>&, bool, int, bool, int>(), 
             py::arg("bandwidth") = -1.0, py::arg("seeds") = std::vector<ingenuityml::VectorXd>(), 
             py::arg("bin_seeding") = false, py::arg("min_bin_freq") = 1, 
             py::arg("cluster_all") = true, py::arg("max_iter") = 300)
        .def("fit", [](ingenuityml::cluster::MeanShift& self, const ingenuityml::MatrixXd& X, py::object y) -> ingenuityml::cluster::MeanShift& {
            ingenuityml::VectorXd y_vec = y.is_none() ? ingenuityml::VectorXd() : y.cast<ingenuityml::VectorXd>();
            return static_cast<ingenuityml::cluster::MeanShift&>(self.fit(X, y_vec));
        }, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("fit_predict", &ingenuityml::cluster::MeanShift::fit_predict)
        .def("predict", &ingenuityml::cluster::MeanShift::predict)
        .def("cluster_centers", &ingenuityml::cluster::MeanShift::cluster_centers)
        .def("labels", &ingenuityml::cluster::MeanShift::labels)
        .def("get_params", &ingenuityml::cluster::MeanShift::get_params)
        .def("set_params", &ingenuityml::cluster::MeanShift::set_params)
        .def("is_fitted", &ingenuityml::cluster::MeanShift::is_fitted);

    py::class_<ingenuityml::cluster::OPTICS, ingenuityml::Estimator>(cluster_module, "OPTICS")
        .def(py::init<int, double, const std::string&, double>(), 
             py::arg("min_samples") = 5, py::arg("max_eps") = std::numeric_limits<double>::infinity(), 
             py::arg("metric") = "euclidean", py::arg("eps") = -1.0)
        .def("fit", [](ingenuityml::cluster::OPTICS& self, const ingenuityml::MatrixXd& X, py::object y) -> ingenuityml::cluster::OPTICS& {
            ingenuityml::VectorXd y_vec = y.is_none() ? ingenuityml::VectorXd() : y.cast<ingenuityml::VectorXd>();
            return static_cast<ingenuityml::cluster::OPTICS&>(self.fit(X, y_vec));
        }, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("fit_predict", &ingenuityml::cluster::OPTICS::fit_predict)
        .def("predict", &ingenuityml::cluster::OPTICS::predict)
        .def("labels", &ingenuityml::cluster::OPTICS::labels)
        .def("reachability", &ingenuityml::cluster::OPTICS::reachability)
        .def("ordering", &ingenuityml::cluster::OPTICS::ordering)
        .def("core_distances", &ingenuityml::cluster::OPTICS::core_distances)
        .def("get_params", &ingenuityml::cluster::OPTICS::get_params)
        .def("set_params", &ingenuityml::cluster::OPTICS::set_params)
        .def("is_fitted", &ingenuityml::cluster::OPTICS::is_fitted);

    py::class_<ingenuityml::cluster::Birch, ingenuityml::Estimator>(cluster_module, "Birch")
        .def(py::init<int, double, int>(), 
             py::arg("n_clusters") = 8, py::arg("threshold") = 0.5, py::arg("branching_factor") = 50)
        .def("fit", [](ingenuityml::cluster::Birch& self, const ingenuityml::MatrixXd& X, py::object y) -> ingenuityml::cluster::Birch& {
            ingenuityml::VectorXd y_vec = y.is_none() ? ingenuityml::VectorXd() : y.cast<ingenuityml::VectorXd>();
            return static_cast<ingenuityml::cluster::Birch&>(self.fit(X, y_vec));
        }, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("fit_predict", &ingenuityml::cluster::Birch::fit_predict)
        .def("predict", &ingenuityml::cluster::Birch::predict)
        .def("labels", &ingenuityml::cluster::Birch::labels)
        .def("subcluster_centers", &ingenuityml::cluster::Birch::subcluster_centers)
        .def("get_params", &ingenuityml::cluster::Birch::get_params)
        .def("set_params", &ingenuityml::cluster::Birch::set_params)
        .def("is_fitted", &ingenuityml::cluster::Birch::is_fitted);

    py::class_<ingenuityml::cluster::BisectingKMeans, ingenuityml::Estimator>(cluster_module, "BisectingKMeans")
        .def(py::init<int, int, double, const std::string&, int>(),
             py::arg("n_clusters") = 8, py::arg("max_iter") = 300, py::arg("tol") = 1e-4,
             py::arg("init") = "k-means++", py::arg("random_state") = -1)
        .def("fit", [](ingenuityml::cluster::BisectingKMeans& self, const ingenuityml::MatrixXd& X, py::object y) -> ingenuityml::cluster::BisectingKMeans& {
            ingenuityml::VectorXd y_vec = y.is_none() ? ingenuityml::VectorXd() : y.cast<ingenuityml::VectorXd>();
            return static_cast<ingenuityml::cluster::BisectingKMeans&>(self.fit(X, y_vec));
        }, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("fit_predict", &ingenuityml::cluster::BisectingKMeans::fit_predict)
        .def("predict", &ingenuityml::cluster::BisectingKMeans::predict)
        .def("cluster_centers", &ingenuityml::cluster::BisectingKMeans::cluster_centers)
        .def("labels", &ingenuityml::cluster::BisectingKMeans::labels)
        .def("get_params", &ingenuityml::cluster::BisectingKMeans::get_params)
        .def("set_params", &ingenuityml::cluster::BisectingKMeans::set_params)
        .def("is_fitted", &ingenuityml::cluster::BisectingKMeans::is_fitted);

    py::class_<ingenuityml::cluster::AffinityPropagation, ingenuityml::Estimator>(cluster_module, "AffinityPropagation")
        .def(py::init<double, int, int, double, int>(),
             py::arg("damping") = 0.5, py::arg("max_iter") = 200, py::arg("convergence_iter") = 15,
             py::arg("preference") = std::numeric_limits<double>::quiet_NaN(), py::arg("random_state") = -1)
        .def("fit", [](ingenuityml::cluster::AffinityPropagation& self, const ingenuityml::MatrixXd& X, py::object y) -> ingenuityml::cluster::AffinityPropagation& {
            ingenuityml::VectorXd y_vec = y.is_none() ? ingenuityml::VectorXd() : y.cast<ingenuityml::VectorXd>();
            return static_cast<ingenuityml::cluster::AffinityPropagation&>(self.fit(X, y_vec));
        }, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("fit_predict", &ingenuityml::cluster::AffinityPropagation::fit_predict)
        .def("predict", &ingenuityml::cluster::AffinityPropagation::predict)
        .def("labels", &ingenuityml::cluster::AffinityPropagation::labels)
        .def("cluster_centers", &ingenuityml::cluster::AffinityPropagation::cluster_centers)
        .def("exemplar_indices", &ingenuityml::cluster::AffinityPropagation::exemplar_indices)
        .def("get_params", &ingenuityml::cluster::AffinityPropagation::get_params)
        .def("set_params", &ingenuityml::cluster::AffinityPropagation::set_params)
        .def("is_fitted", &ingenuityml::cluster::AffinityPropagation::is_fitted);

    py::class_<ingenuityml::cluster::FeatureAgglomeration, ingenuityml::Estimator, ingenuityml::Transformer>(cluster_module, "FeatureAgglomeration")
        .def(py::init<int, const std::string&, const std::string&>(),
             py::arg("n_clusters") = 2, py::arg("linkage") = "single", py::arg("affinity") = "euclidean")
        .def("fit", [](ingenuityml::cluster::FeatureAgglomeration& self, const ingenuityml::MatrixXd& X, py::object y) -> ingenuityml::cluster::FeatureAgglomeration& {
            ingenuityml::VectorXd y_vec = y.is_none() ? ingenuityml::VectorXd() : y.cast<ingenuityml::VectorXd>();
            return static_cast<ingenuityml::cluster::FeatureAgglomeration&>(self.fit(X, y_vec));
        }, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("transform", &ingenuityml::cluster::FeatureAgglomeration::transform)
        .def("inverse_transform", &ingenuityml::cluster::FeatureAgglomeration::inverse_transform)
        .def("fit_transform", &ingenuityml::cluster::FeatureAgglomeration::fit_transform)
        .def("labels", &ingenuityml::cluster::FeatureAgglomeration::labels)
        .def("get_params", &ingenuityml::cluster::FeatureAgglomeration::get_params)
        .def("set_params", &ingenuityml::cluster::FeatureAgglomeration::set_params)
        .def("is_fitted", &ingenuityml::cluster::FeatureAgglomeration::is_fitted);

    py::class_<ingenuityml::cluster::SpectralBiclustering, ingenuityml::Estimator>(cluster_module, "SpectralBiclustering")
        .def(py::init<int, int>(), py::arg("n_clusters") = 3, py::arg("random_state") = -1)
        .def("fit", [](ingenuityml::cluster::SpectralBiclustering& self, const ingenuityml::MatrixXd& X, py::object y) -> ingenuityml::cluster::SpectralBiclustering& {
            ingenuityml::VectorXd y_vec = y.is_none() ? ingenuityml::VectorXd() : y.cast<ingenuityml::VectorXd>();
            return static_cast<ingenuityml::cluster::SpectralBiclustering&>(self.fit(X, y_vec));
        }, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("row_labels", &ingenuityml::cluster::SpectralBiclustering::row_labels)
        .def("column_labels", &ingenuityml::cluster::SpectralBiclustering::column_labels)
        .def("get_params", &ingenuityml::cluster::SpectralBiclustering::get_params)
        .def("set_params", &ingenuityml::cluster::SpectralBiclustering::set_params)
        .def("is_fitted", &ingenuityml::cluster::SpectralBiclustering::is_fitted);

    py::class_<ingenuityml::cluster::SpectralCoclustering, ingenuityml::Estimator>(cluster_module, "SpectralCoclustering")
        .def(py::init<int, int>(), py::arg("n_clusters") = 3, py::arg("random_state") = -1)
        .def("fit", [](ingenuityml::cluster::SpectralCoclustering& self, const ingenuityml::MatrixXd& X, py::object y) -> ingenuityml::cluster::SpectralCoclustering& {
            ingenuityml::VectorXd y_vec = y.is_none() ? ingenuityml::VectorXd() : y.cast<ingenuityml::VectorXd>();
            return static_cast<ingenuityml::cluster::SpectralCoclustering&>(self.fit(X, y_vec));
        }, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("row_labels", &ingenuityml::cluster::SpectralCoclustering::row_labels)
        .def("column_labels", &ingenuityml::cluster::SpectralCoclustering::column_labels)
        .def("get_params", &ingenuityml::cluster::SpectralCoclustering::get_params)
        .def("set_params", &ingenuityml::cluster::SpectralCoclustering::set_params)
        .def("is_fitted", &ingenuityml::cluster::SpectralCoclustering::is_fitted);

    // Neural Network
    py::module_ neural_network_module = m.def_submodule("neural_network", "Neural Network algorithms");
    
    py::enum_<ingenuityml::neural_network::ActivationFunction>(neural_network_module, "ActivationFunction")
        .value("RELU", ingenuityml::neural_network::ActivationFunction::RELU)
        .value("TANH", ingenuityml::neural_network::ActivationFunction::TANH)
        .value("LOGISTIC", ingenuityml::neural_network::ActivationFunction::LOGISTIC)
        .value("IDENTITY", ingenuityml::neural_network::ActivationFunction::IDENTITY);
        
    py::enum_<ingenuityml::neural_network::Solver>(neural_network_module, "Solver")
        .value("LBFGS", ingenuityml::neural_network::Solver::LBFGS)
        .value("SGD", ingenuityml::neural_network::Solver::SGD)
        .value("ADAM", ingenuityml::neural_network::Solver::ADAM);

    py::class_<ingenuityml::neural_network::MLPClassifier, ingenuityml::Estimator, ingenuityml::Classifier>(neural_network_module, "MLPClassifier")
        .def(py::init<const std::vector<int>&, ingenuityml::neural_network::ActivationFunction, ingenuityml::neural_network::Solver,
                      double, int, double, int, int, double, bool, bool, double, bool, bool, double, double, double, double, int>(),
             py::arg("hidden_layer_sizes") = std::vector<int>{100},
             py::arg("activation") = ingenuityml::neural_network::ActivationFunction::RELU,
             py::arg("solver") = ingenuityml::neural_network::Solver::ADAM,
             py::arg("alpha") = 0.0001,
             py::arg("batch_size") = 200,
             py::arg("learning_rate") = 0.001,
             py::arg("max_iter") = 200,
             py::arg("random_state") = -1,
             py::arg("tol") = 1e-4,
             py::arg("verbose") = false,
             py::arg("warm_start") = false,
             py::arg("momentum") = 0.9,
             py::arg("nesterovs_momentum") = true,
             py::arg("early_stopping") = false,
             py::arg("validation_fraction") = 0.1,
             py::arg("beta_1") = 0.9,
             py::arg("beta_2") = 0.999,
             py::arg("epsilon") = 1e-8,
             py::arg("n_iter_no_change") = 10)
        .def("fit", &ingenuityml::neural_network::MLPClassifier::fit)
        .def("predict", &ingenuityml::neural_network::MLPClassifier::predict_classes)
        .def("predict_proba", &ingenuityml::neural_network::MLPClassifier::predict_proba)
        .def("decision_function", &ingenuityml::neural_network::MLPClassifier::decision_function)
        .def("alpha", &ingenuityml::neural_network::MLPClassifier::alpha)
        .def("learning_rate", &ingenuityml::neural_network::MLPClassifier::learning_rate)
        .def("max_iter", &ingenuityml::neural_network::MLPClassifier::max_iter)
        .def("hidden_layer_sizes", &ingenuityml::neural_network::MLPClassifier::hidden_layer_sizes)
        .def("loss_curve", &ingenuityml::neural_network::MLPClassifier::loss_curve)
        .def("n_iter", &ingenuityml::neural_network::MLPClassifier::n_iter)
        .def("classes", &ingenuityml::neural_network::MLPClassifier::classes)
        .def("n_classes", &ingenuityml::neural_network::MLPClassifier::n_classes)
        .def("get_params", &ingenuityml::neural_network::MLPClassifier::get_params)
        .def("set_params", &ingenuityml::neural_network::MLPClassifier::set_params)
        .def("is_fitted", &ingenuityml::neural_network::MLPClassifier::is_fitted);

    py::class_<ingenuityml::neural_network::MLPRegressor, ingenuityml::Estimator, ingenuityml::Regressor>(neural_network_module, "MLPRegressor")
        .def(py::init<const std::vector<int>&, ingenuityml::neural_network::ActivationFunction, ingenuityml::neural_network::Solver,
                      double, int, double, int, int, double, bool, bool, double, bool, bool, double, double, double, double, int>(),
             py::arg("hidden_layer_sizes") = std::vector<int>{100},
             py::arg("activation") = ingenuityml::neural_network::ActivationFunction::RELU,
             py::arg("solver") = ingenuityml::neural_network::Solver::ADAM,
             py::arg("alpha") = 0.0001,
             py::arg("batch_size") = 200,
             py::arg("learning_rate") = 0.001,
             py::arg("max_iter") = 200,
             py::arg("random_state") = -1,
             py::arg("tol") = 1e-4,
             py::arg("verbose") = false,
             py::arg("warm_start") = false,
             py::arg("momentum") = 0.9,
             py::arg("nesterovs_momentum") = true,
             py::arg("early_stopping") = false,
             py::arg("validation_fraction") = 0.1,
             py::arg("beta_1") = 0.9,
             py::arg("beta_2") = 0.999,
             py::arg("epsilon") = 1e-8,
             py::arg("n_iter_no_change") = 10)
        .def("fit", &ingenuityml::neural_network::MLPRegressor::fit)
        .def("predict", &ingenuityml::neural_network::MLPRegressor::predict)
        .def("alpha", &ingenuityml::neural_network::MLPRegressor::alpha)
        .def("learning_rate", &ingenuityml::neural_network::MLPRegressor::learning_rate)
        .def("max_iter", &ingenuityml::neural_network::MLPRegressor::max_iter)
        .def("hidden_layer_sizes", &ingenuityml::neural_network::MLPRegressor::hidden_layer_sizes)
        .def("loss_curve", &ingenuityml::neural_network::MLPRegressor::loss_curve)
        .def("n_iter", &ingenuityml::neural_network::MLPRegressor::n_iter)
        .def("get_params", &ingenuityml::neural_network::MLPRegressor::get_params)
        .def("set_params", &ingenuityml::neural_network::MLPRegressor::set_params)
        .def("is_fitted", &ingenuityml::neural_network::MLPRegressor::is_fitted);

    py::class_<ingenuityml::neural_network::BernoulliRBM, ingenuityml::Estimator, ingenuityml::Transformer>(neural_network_module, "BernoulliRBM")
        .def(py::init<int,double,int,int,int,bool>(),
             py::arg("n_components") = 256, py::arg("learning_rate") = 0.1,
             py::arg("batch_size") = 10, py::arg("n_iter") = 10,
             py::arg("random_state") = -1, py::arg("verbose") = false)
        .def("fit", &ingenuityml::neural_network::BernoulliRBM::fit)
        .def("transform", &ingenuityml::neural_network::BernoulliRBM::transform)
        .def("inverse_transform", &ingenuityml::neural_network::BernoulliRBM::inverse_transform)
        .def("fit_transform", &ingenuityml::neural_network::BernoulliRBM::fit_transform)
        .def("get_params", &ingenuityml::neural_network::BernoulliRBM::get_params)
        .def("set_params", &ingenuityml::neural_network::BernoulliRBM::set_params)
        .def("is_fitted", &ingenuityml::neural_network::BernoulliRBM::is_fitted)
        .def("components", &ingenuityml::neural_network::BernoulliRBM::components)
        .def("intercept_visible", &ingenuityml::neural_network::BernoulliRBM::intercept_visible)
        .def("intercept_hidden", &ingenuityml::neural_network::BernoulliRBM::intercept_hidden)
        .def("n_iter", &ingenuityml::neural_network::BernoulliRBM::n_iter);
}

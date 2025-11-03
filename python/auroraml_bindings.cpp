#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "auroraml/base.hpp"
#include "auroraml/random.hpp"
#include "auroraml/linear_model.hpp"
#include "auroraml/neighbors.hpp"
#include "auroraml/tree.hpp"
#include "auroraml/metrics.hpp"
#include "auroraml/preprocessing.hpp"
#include "auroraml/model_selection.hpp"
#include "auroraml/naive_bayes.hpp"
#include "auroraml/kmeans.hpp"
#include "auroraml/pca.hpp"
#include "auroraml/dbscan.hpp"
#include "auroraml/truncated_svd.hpp"
#include "auroraml/lda.hpp"
#include "auroraml/agglomerative.hpp"
#include "auroraml/svm.hpp"
#include "auroraml/random_forest.hpp"
#include "auroraml/gradient_boosting.hpp"
#include "auroraml/adaboost.hpp"
#include "auroraml/xgboost.hpp"
#include "auroraml/catboost.hpp"

namespace py = pybind11;

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

PYBIND11_MODULE(auroraml, m) {
    m.doc() = "AuroraML: High-performance C++ machine learning library";
    
    // Bind base classes
    py::class_<auroraml::Estimator>(m, "Estimator");
    py::class_<auroraml::Predictor>(m, "Predictor");
    py::class_<auroraml::Classifier, auroraml::Predictor>(m, "Classifier");
    py::class_<auroraml::Regressor, auroraml::Predictor>(m, "Regressor");
    py::class_<auroraml::Transformer>(m, "Transformer");
    
    // Bind parameter types
    py::class_<auroraml::Params>(m, "Params")
        .def(py::init<>())
        .def("__getitem__", [](const auroraml::Params& p, const std::string& key) {
            auto it = p.find(key);
            if (it != p.end()) return it->second;
            throw std::runtime_error("Key not found");
        })
        .def("__setitem__", [](auroraml::Params& p, const std::string& key, const std::string& value) {
            p[key] = value;
        });
    
    // Random module
    py::module_ random_module = m.def_submodule("random", "Random number generation");
    
    py::class_<auroraml::random::PCG64>(random_module, "PCG64")
        .def(py::init<uint64_t>(), py::arg("seed") = 0)
        .def("seed", &auroraml::random::PCG64::seed)
        .def("uniform", &auroraml::random::PCG64::uniform)
        .def("normal", &auroraml::random::PCG64::normal);
    
    // Linear models module
    py::module_ linear_module = m.def_submodule("linear_model", "Linear models");
    
    py::class_<auroraml::linear_model::LinearRegression, auroraml::Estimator, auroraml::Regressor>(linear_module, "LinearRegression")
        .def(py::init<bool, bool, int>(), py::arg("fit_intercept") = true, py::arg("copy_X") = true, py::arg("n_jobs") = 1)
        .def("fit", &auroraml::linear_model::LinearRegression::fit)
        .def("predict", &auroraml::linear_model::LinearRegression::predict)
        .def("get_params", &auroraml::linear_model::LinearRegression::get_params)
        .def("set_params", &auroraml::linear_model::LinearRegression::set_params)
        .def("is_fitted", &auroraml::linear_model::LinearRegression::is_fitted)
        .def("coef", &auroraml::linear_model::LinearRegression::coef)
        .def("intercept", &auroraml::linear_model::LinearRegression::intercept)
        .def("save_to_file", &auroraml::linear_model::LinearRegression::save_to_file)
        .def("load_from_file", &auroraml::linear_model::LinearRegression::load_from_file);
    
    py::class_<auroraml::linear_model::Ridge, auroraml::Estimator, auroraml::Regressor>(linear_module, "Ridge")
        .def(py::init<double, bool, bool, int>(), py::arg("alpha") = 1.0, py::arg("fit_intercept") = true, py::arg("copy_X") = true, py::arg("n_jobs") = 1)
        .def("fit", &auroraml::linear_model::Ridge::fit)
        .def("predict", &auroraml::linear_model::Ridge::predict)
        .def("get_params", &auroraml::linear_model::Ridge::get_params)
        .def("set_params", &auroraml::linear_model::Ridge::set_params)
        .def("is_fitted", &auroraml::linear_model::Ridge::is_fitted)
        .def("coef", &auroraml::linear_model::Ridge::coef)
        .def("intercept", &auroraml::linear_model::Ridge::intercept)
        .def("save_to_file", &auroraml::linear_model::Ridge::save_to_file)
        .def("load_from_file", &auroraml::linear_model::Ridge::load_from_file);
    
    py::class_<auroraml::linear_model::Lasso, auroraml::Estimator, auroraml::Regressor>(linear_module, "Lasso")
        .def(py::init<double, bool, bool, int>(), py::arg("alpha") = 1.0, py::arg("fit_intercept") = true, py::arg("copy_X") = true, py::arg("n_jobs") = 1)
        .def("fit", &auroraml::linear_model::Lasso::fit)
        .def("predict", &auroraml::linear_model::Lasso::predict)
        .def("get_params", &auroraml::linear_model::Lasso::get_params)
        .def("set_params", &auroraml::linear_model::Lasso::set_params)
        .def("is_fitted", &auroraml::linear_model::Lasso::is_fitted)
        .def("coef", &auroraml::linear_model::Lasso::coef)
        .def("intercept", &auroraml::linear_model::Lasso::intercept);
    
    py::class_<auroraml::linear_model::ElasticNet, auroraml::Estimator, auroraml::Regressor>(linear_module, "ElasticNet")
        .def(py::init<double, double, bool, bool, int, int, double>(),
             py::arg("alpha") = 1.0, py::arg("l1_ratio") = 0.5, py::arg("fit_intercept") = true,
             py::arg("copy_X") = true, py::arg("n_jobs") = 1, py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("fit", &auroraml::linear_model::ElasticNet::fit)
        .def("predict", &auroraml::linear_model::ElasticNet::predict)
        .def("get_params", &auroraml::linear_model::ElasticNet::get_params)
        .def("set_params", &auroraml::linear_model::ElasticNet::set_params)
        .def("is_fitted", &auroraml::linear_model::ElasticNet::is_fitted)
        .def("coef", &auroraml::linear_model::ElasticNet::coef)
        .def("intercept", &auroraml::linear_model::ElasticNet::intercept);
    
    py::class_<auroraml::linear_model::LogisticRegression, auroraml::Estimator, auroraml::Classifier>(linear_module, "LogisticRegression")
        .def(py::init<double, bool, int, double, int>(),
             py::arg("C") = 1.0, py::arg("fit_intercept") = true, py::arg("max_iter") = 100,
             py::arg("tol") = 1e-4, py::arg("random_state") = -1)
        .def("fit", &auroraml::linear_model::LogisticRegression::fit)
        .def("predict", &auroraml::linear_model::LogisticRegression::predict_classes)
        .def("predict_proba", &auroraml::linear_model::LogisticRegression::predict_proba)
        .def("decision_function", &auroraml::linear_model::LogisticRegression::decision_function)
        .def("get_params", &auroraml::linear_model::LogisticRegression::get_params)
        .def("set_params", &auroraml::linear_model::LogisticRegression::set_params)
        .def("is_fitted", &auroraml::linear_model::LogisticRegression::is_fitted)
        .def("coef", &auroraml::linear_model::LogisticRegression::coef)
        .def("intercept", &auroraml::linear_model::LogisticRegression::intercept)
        .def("classes", &auroraml::linear_model::LogisticRegression::classes)
        .def("n_classes", &auroraml::linear_model::LogisticRegression::n_classes);
    
    // Neighbors module
    py::module_ neighbors_module = m.def_submodule("neighbors", "Nearest neighbors");
    
    py::class_<auroraml::neighbors::KNeighborsClassifier, auroraml::Estimator, auroraml::Classifier>(neighbors_module, "KNeighborsClassifier")
        .def(py::init<int, std::string, std::string, std::string, double, int>(),
             py::arg("n_neighbors") = 5, py::arg("weights") = "uniform", py::arg("algorithm") = "auto",
             py::arg("metric") = "euclidean", py::arg("p") = 2, py::arg("n_jobs") = 1)
        .def("fit", &auroraml::neighbors::KNeighborsClassifier::fit)
        .def("predict", &auroraml::neighbors::KNeighborsClassifier::predict_classes)
        .def("predict_proba", &auroraml::neighbors::KNeighborsClassifier::predict_proba)
        .def("decision_function", &auroraml::neighbors::KNeighborsClassifier::decision_function)
        .def("get_params", &auroraml::neighbors::KNeighborsClassifier::get_params)
        .def("set_params", &auroraml::neighbors::KNeighborsClassifier::set_params)
        .def("is_fitted", &auroraml::neighbors::KNeighborsClassifier::is_fitted)
        .def("save", &auroraml::neighbors::KNeighborsClassifier::save)
        .def("load", &auroraml::neighbors::KNeighborsClassifier::load);
    
    py::class_<auroraml::neighbors::KNeighborsRegressor, auroraml::Estimator, auroraml::Regressor>(neighbors_module, "KNeighborsRegressor")
        .def(py::init<int, std::string, std::string, std::string, double, int>(),
             py::arg("n_neighbors") = 5, py::arg("weights") = "uniform", py::arg("algorithm") = "auto",
             py::arg("metric") = "euclidean", py::arg("p") = 2, py::arg("n_jobs") = 1)
        .def("fit", &auroraml::neighbors::KNeighborsRegressor::fit)
        .def("predict", &auroraml::neighbors::KNeighborsRegressor::predict)
        .def("get_params", &auroraml::neighbors::KNeighborsRegressor::get_params)
        .def("set_params", &auroraml::neighbors::KNeighborsRegressor::set_params)
        .def("is_fitted", &auroraml::neighbors::KNeighborsRegressor::is_fitted)
        .def("save", &auroraml::neighbors::KNeighborsRegressor::save)
        .def("load", &auroraml::neighbors::KNeighborsRegressor::load);
    
    // Tree module
    py::module_ tree_module = m.def_submodule("tree", "Decision trees");
    
    py::class_<auroraml::tree::DecisionTreeClassifier, auroraml::Estimator, auroraml::Classifier>(tree_module, "DecisionTreeClassifier")
        .def(py::init<std::string, int, int, int, double>(),
             py::arg("criterion") = "gini", py::arg("max_depth") = -1, py::arg("min_samples_split") = 2,
             py::arg("min_samples_leaf") = 1, py::arg("min_impurity_decrease") = 0.0)
        .def("fit", &auroraml::tree::DecisionTreeClassifier::fit)
        .def("predict", &auroraml::tree::DecisionTreeClassifier::predict_classes)
        .def("predict_proba", &auroraml::tree::DecisionTreeClassifier::predict_proba)
        .def("decision_function", &auroraml::tree::DecisionTreeClassifier::decision_function)
        .def("get_params", &auroraml::tree::DecisionTreeClassifier::get_params)
        .def("set_params", &auroraml::tree::DecisionTreeClassifier::set_params)
        .def("is_fitted", &auroraml::tree::DecisionTreeClassifier::is_fitted)
        .def("save", &auroraml::tree::DecisionTreeClassifier::save)
        .def("load", &auroraml::tree::DecisionTreeClassifier::load);
    
    py::class_<auroraml::tree::DecisionTreeRegressor, auroraml::Estimator, auroraml::Regressor>(tree_module, "DecisionTreeRegressor")
        .def(py::init<std::string, int, int, int, double>(),
             py::arg("criterion") = "mse", py::arg("max_depth") = -1, py::arg("min_samples_split") = 2,
             py::arg("min_samples_leaf") = 1, py::arg("min_impurity_decrease") = 0.0)
        .def("fit", &auroraml::tree::DecisionTreeRegressor::fit)
        .def("predict", &auroraml::tree::DecisionTreeRegressor::predict)
        .def("get_params", &auroraml::tree::DecisionTreeRegressor::get_params)
        .def("set_params", &auroraml::tree::DecisionTreeRegressor::set_params)
        .def("is_fitted", &auroraml::tree::DecisionTreeRegressor::is_fitted)
        .def("save", &auroraml::tree::DecisionTreeRegressor::save)
        .def("load", &auroraml::tree::DecisionTreeRegressor::load);
    
    // Metrics module
    py::module_ metrics_module = m.def_submodule("metrics", "Evaluation metrics");
    
    metrics_module.def("accuracy_score", &auroraml::metrics::accuracy_score);
    metrics_module.def("precision_score", &auroraml::metrics::precision_score);
    metrics_module.def("recall_score", &auroraml::metrics::recall_score);
    metrics_module.def("f1_score", &auroraml::metrics::f1_score);
    metrics_module.def("confusion_matrix", &auroraml::metrics::confusion_matrix);
    metrics_module.def("classification_report", &auroraml::metrics::classification_report);
    
    metrics_module.def("mean_squared_error", &auroraml::metrics::mean_squared_error);
    metrics_module.def("root_mean_squared_error", &auroraml::metrics::root_mean_squared_error);
    metrics_module.def("mean_absolute_error", &auroraml::metrics::mean_absolute_error);
    metrics_module.def("r2_score", &auroraml::metrics::r2_score);
    metrics_module.def("explained_variance_score", &auroraml::metrics::explained_variance_score);
    metrics_module.def("mean_absolute_percentage_error", &auroraml::metrics::mean_absolute_percentage_error);
    
    // Preprocessing module
    py::module_ preprocessing_module = m.def_submodule("preprocessing", "Data preprocessing");
    
    py::class_<auroraml::preprocessing::StandardScaler, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "StandardScaler")
        .def(py::init<bool, bool>(), py::arg("with_mean") = true, py::arg("with_std") = true)
        .def("fit", &auroraml::preprocessing::StandardScaler::fit)
        .def("transform", &auroraml::preprocessing::StandardScaler::transform)
        .def("inverse_transform", &auroraml::preprocessing::StandardScaler::inverse_transform)
        .def("fit_transform", &auroraml::preprocessing::StandardScaler::fit_transform)
        .def("get_params", &auroraml::preprocessing::StandardScaler::get_params)
        .def("set_params", &auroraml::preprocessing::StandardScaler::set_params)
        .def("is_fitted", &auroraml::preprocessing::StandardScaler::is_fitted)
        .def("mean", &auroraml::preprocessing::StandardScaler::mean)
        .def("scale", &auroraml::preprocessing::StandardScaler::scale);
    
    py::class_<auroraml::preprocessing::MinMaxScaler, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "MinMaxScaler")
        .def(py::init<double, double>(), py::arg("feature_range_min") = 0.0, py::arg("feature_range_max") = 1.0)
        .def("fit", &auroraml::preprocessing::MinMaxScaler::fit)
        .def("transform", &auroraml::preprocessing::MinMaxScaler::transform)
        .def("inverse_transform", &auroraml::preprocessing::MinMaxScaler::inverse_transform)
        .def("fit_transform", &auroraml::preprocessing::MinMaxScaler::fit_transform)
        .def("get_params", &auroraml::preprocessing::MinMaxScaler::get_params)
        .def("set_params", &auroraml::preprocessing::MinMaxScaler::set_params)
        .def("is_fitted", &auroraml::preprocessing::MinMaxScaler::is_fitted)
        .def("data_min", &auroraml::preprocessing::MinMaxScaler::data_min)
        .def("data_max", &auroraml::preprocessing::MinMaxScaler::data_max)
        .def("scale", &auroraml::preprocessing::MinMaxScaler::scale)
        .def("min", &auroraml::preprocessing::MinMaxScaler::min);

    py::class_<auroraml::preprocessing::RobustScaler, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "RobustScaler")
        .def(py::init<bool, bool>(), py::arg("with_centering") = true, py::arg("with_scaling") = true)
        .def("fit", &auroraml::preprocessing::RobustScaler::fit)
        .def("transform", &auroraml::preprocessing::RobustScaler::transform)
        .def("inverse_transform", &auroraml::preprocessing::RobustScaler::inverse_transform)
        .def("fit_transform", &auroraml::preprocessing::RobustScaler::fit_transform)
        .def("get_params", &auroraml::preprocessing::RobustScaler::get_params)
        .def("set_params", &auroraml::preprocessing::RobustScaler::set_params)
        .def("is_fitted", &auroraml::preprocessing::RobustScaler::is_fitted);
    
    py::class_<auroraml::preprocessing::LabelEncoder, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "LabelEncoder")
        .def(py::init<>())
        .def("fit", &auroraml::preprocessing::LabelEncoder::fit)
        .def("transform", static_cast<auroraml::MatrixXd (auroraml::preprocessing::LabelEncoder::*)(const auroraml::MatrixXd&) const>(&auroraml::preprocessing::LabelEncoder::transform))
        .def("inverse_transform", static_cast<auroraml::MatrixXd (auroraml::preprocessing::LabelEncoder::*)(const auroraml::MatrixXd&) const>(&auroraml::preprocessing::LabelEncoder::inverse_transform))
        .def("fit_transform", &auroraml::preprocessing::LabelEncoder::fit_transform)
        .def("get_params", &auroraml::preprocessing::LabelEncoder::get_params)
        .def("set_params", &auroraml::preprocessing::LabelEncoder::set_params)
        .def("is_fitted", &auroraml::preprocessing::LabelEncoder::is_fitted)
        .def("n_classes", &auroraml::preprocessing::LabelEncoder::n_classes);

    py::class_<auroraml::preprocessing::OneHotEncoder, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "OneHotEncoder")
        .def(py::init<>())
        .def("fit", &auroraml::preprocessing::OneHotEncoder::fit)
        .def("transform", &auroraml::preprocessing::OneHotEncoder::transform)
        .def("inverse_transform", &auroraml::preprocessing::OneHotEncoder::inverse_transform)
        .def("fit_transform", &auroraml::preprocessing::OneHotEncoder::fit_transform)
        .def("get_params", &auroraml::preprocessing::OneHotEncoder::get_params)
        .def("set_params", &auroraml::preprocessing::OneHotEncoder::set_params)
        .def("is_fitted", &auroraml::preprocessing::OneHotEncoder::is_fitted);

    py::class_<auroraml::preprocessing::OrdinalEncoder, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "OrdinalEncoder")
        .def(py::init<>())
        .def("fit", &auroraml::preprocessing::OrdinalEncoder::fit)
        .def("transform", &auroraml::preprocessing::OrdinalEncoder::transform)
        .def("inverse_transform", &auroraml::preprocessing::OrdinalEncoder::inverse_transform)
        .def("fit_transform", &auroraml::preprocessing::OrdinalEncoder::fit_transform)
        .def("get_params", &auroraml::preprocessing::OrdinalEncoder::get_params)
        .def("set_params", &auroraml::preprocessing::OrdinalEncoder::set_params)
        .def("is_fitted", &auroraml::preprocessing::OrdinalEncoder::is_fitted)
        .def("categories", &auroraml::preprocessing::OrdinalEncoder::categories);
    
    py::class_<auroraml::preprocessing::Normalizer, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "Normalizer")
        .def(py::init<const std::string&>(), py::arg("norm") = "l2")
        .def("fit", &auroraml::preprocessing::Normalizer::fit)
        .def("transform", &auroraml::preprocessing::Normalizer::transform)
        .def("inverse_transform", &auroraml::preprocessing::Normalizer::inverse_transform)
        .def("fit_transform", &auroraml::preprocessing::Normalizer::fit_transform)
        .def("get_params", &auroraml::preprocessing::Normalizer::get_params)
        .def("set_params", &auroraml::preprocessing::Normalizer::set_params)
        .def("is_fitted", &auroraml::preprocessing::Normalizer::is_fitted);
    
    py::class_<auroraml::preprocessing::PolynomialFeatures, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "PolynomialFeatures")
        .def(py::init<int, bool, bool>(), py::arg("degree") = 2, py::arg("interaction_only") = false, py::arg("include_bias") = true)
        .def("fit", &auroraml::preprocessing::PolynomialFeatures::fit)
        .def("transform", &auroraml::preprocessing::PolynomialFeatures::transform)
        .def("inverse_transform", &auroraml::preprocessing::PolynomialFeatures::inverse_transform)
        .def("fit_transform", &auroraml::preprocessing::PolynomialFeatures::fit_transform)
        .def("get_params", &auroraml::preprocessing::PolynomialFeatures::get_params)
        .def("set_params", &auroraml::preprocessing::PolynomialFeatures::set_params)
        .def("is_fitted", &auroraml::preprocessing::PolynomialFeatures::is_fitted)
        .def("n_input_features", &auroraml::preprocessing::PolynomialFeatures::n_input_features)
        .def("n_output_features", &auroraml::preprocessing::PolynomialFeatures::n_output_features);
    
    py::class_<auroraml::preprocessing::SimpleImputer, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "SimpleImputer")
        .def(py::init<const std::string&, double>(), py::arg("strategy") = "mean", py::arg("fill_value") = 0.0)
        .def("fit", &auroraml::preprocessing::SimpleImputer::fit)
        .def("transform", &auroraml::preprocessing::SimpleImputer::transform)
        .def("inverse_transform", &auroraml::preprocessing::SimpleImputer::inverse_transform)
        .def("fit_transform", &auroraml::preprocessing::SimpleImputer::fit_transform)
        .def("get_params", &auroraml::preprocessing::SimpleImputer::get_params)
        .def("set_params", &auroraml::preprocessing::SimpleImputer::set_params)
        .def("is_fitted", &auroraml::preprocessing::SimpleImputer::is_fitted)
        .def("statistics", &auroraml::preprocessing::SimpleImputer::statistics);
    
    // Model selection module
    py::module_ model_selection_module = m.def_submodule("model_selection", "Model selection utilities");
    
    // Base cross validator class
    py::class_<auroraml::model_selection::BaseCrossValidator>(model_selection_module, "BaseCrossValidator")
        .def("split", &auroraml::model_selection::BaseCrossValidator::split)
        .def("get_n_splits", &auroraml::model_selection::BaseCrossValidator::get_n_splits);
    
    model_selection_module.def("train_test_split", 
        [](const auroraml::MatrixXd& X, const auroraml::VectorXd& y, double test_size, double train_size, 
           int random_state, bool shuffle, const auroraml::VectorXd& stratify) {
            return auroraml::model_selection::train_test_split(X, y, test_size, train_size, random_state, shuffle, stratify);
        },
        py::arg("X"), py::arg("y"), py::arg("test_size") = 0.25, py::arg("train_size") = -1,
        py::arg("random_state") = -1, py::arg("shuffle") = true, py::arg("stratify") = auroraml::VectorXd());
    
    py::class_<auroraml::model_selection::KFold>(model_selection_module, "KFold", py::base<auroraml::model_selection::BaseCrossValidator>())
        .def(py::init<int, bool, int>(), py::arg("n_splits") = 5, py::arg("shuffle") = false, py::arg("random_state") = -1)
        .def("split", [](const auroraml::model_selection::KFold& self, const auroraml::MatrixXd& X) {
            return self.split(X);
        })
        .def("split", [](const auroraml::model_selection::KFold& self, const auroraml::MatrixXd& X, const auroraml::VectorXd& y) {
            return self.split(X, y);
        })
        .def("get_n_splits", &auroraml::model_selection::KFold::get_n_splits);

    py::class_<auroraml::model_selection::StratifiedKFold>(model_selection_module, "StratifiedKFold", py::base<auroraml::model_selection::BaseCrossValidator>())
        .def(py::init<int, bool, int>(), py::arg("n_splits") = 5, py::arg("shuffle") = false, py::arg("random_state") = -1)
        .def("split", [](const auroraml::model_selection::StratifiedKFold& self, const auroraml::MatrixXd& X, const auroraml::VectorXd& y) {
            return self.split(X, y);
        })
        .def("get_n_splits", &auroraml::model_selection::StratifiedKFold::get_n_splits);

    py::class_<auroraml::model_selection::GroupKFold>(model_selection_module, "GroupKFold")
        .def(py::init<int>(), py::arg("n_splits") = 5)
        .def("split", [](const auroraml::model_selection::GroupKFold& self, const auroraml::MatrixXd& X, const auroraml::VectorXd& y, const auroraml::VectorXd& groups) {
            return self.split(X, y, groups);
        })
        .def("get_n_splits", &auroraml::model_selection::GroupKFold::get_n_splits);
    
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
        
        auroraml::MatrixXd X = Eigen::Map<auroraml::MatrixXd>(static_cast<double*>(X_buf.ptr), X_buf.shape[0], X_buf.shape[1]);
        auroraml::VectorXd y = Eigen::Map<auroraml::VectorXd>(static_cast<double*>(y_buf.ptr), y_buf.shape[0]);
        
        // Cast estimator to the correct type
        auroraml::Estimator& est = estimator.cast<auroraml::Estimator&>();
        const auroraml::model_selection::BaseCrossValidator& cv_obj = cv.cast<const auroraml::model_selection::BaseCrossValidator&>();
        
        // Call the C++ function
        auroraml::VectorXd scores = auroraml::model_selection::cross_val_score(est, X, y, cv_obj, scoring);
        
        // Convert back to numpy array
        return py::array_t<double>(scores.size(), scores.data());
    };
    
    model_selection_module.def("cross_val_score", cross_val_score_wrapper,
        py::arg("estimator"), py::arg("X"), py::arg("y"), py::arg("cv"), py::arg("scoring") = "accuracy");
    
    // Naive Bayes submodule
    py::module_ nb_module = m.def_submodule("naive_bayes", "Naive Bayes algorithms");
    py::class_<auroraml::naive_bayes::GaussianNB, auroraml::Classifier, auroraml::Estimator>(nb_module, "GaussianNB")
        .def(py::init<double>(), py::arg("var_smoothing") = 1e-9)
        .def("fit", &auroraml::naive_bayes::GaussianNB::fit)
        .def("predict", &auroraml::naive_bayes::GaussianNB::predict_classes)
        .def("predict_classes", &auroraml::naive_bayes::GaussianNB::predict_classes)
        .def("predict_proba", &auroraml::naive_bayes::GaussianNB::predict_proba)
        .def("decision_function", &auroraml::naive_bayes::GaussianNB::decision_function)
        .def("get_params", &auroraml::naive_bayes::GaussianNB::get_params)
        .def("set_params", &auroraml::naive_bayes::GaussianNB::set_params)
        .def("is_fitted", &auroraml::naive_bayes::GaussianNB::is_fitted)
        .def("save", &auroraml::naive_bayes::GaussianNB::save)
        .def("load", &auroraml::naive_bayes::GaussianNB::load);

    // Clustering
    py::module_ cluster_module = m.def_submodule("cluster", "Clustering algorithms");
    py::class_<auroraml::cluster::KMeans, auroraml::Estimator, auroraml::Transformer>(cluster_module, "KMeans")
        .def(py::init<int,int,double,const std::string&,int>(), py::arg("n_clusters")=8, py::arg("max_iter")=300, py::arg("tol")=1e-4, py::arg("init")="k-means++", py::arg("random_state")=-1)
        .def("fit", &auroraml::cluster::KMeans::fit)
        .def("transform", &auroraml::cluster::KMeans::transform)
        .def("inverse_transform", &auroraml::cluster::KMeans::inverse_transform)
        .def("fit_transform", &auroraml::cluster::KMeans::fit_transform)
        .def("predict_labels", &auroraml::cluster::KMeans::predict_labels)
        .def("get_params", &auroraml::cluster::KMeans::get_params)
        .def("set_params", &auroraml::cluster::KMeans::set_params)
        .def("is_fitted", &auroraml::cluster::KMeans::is_fitted);

    py::class_<auroraml::cluster::DBSCAN, auroraml::Estimator>(cluster_module, "DBSCAN")
        .def(py::init<double,int>(), py::arg("eps")=0.5, py::arg("min_samples")=5)
        .def("fit", &auroraml::cluster::DBSCAN::fit)
        .def("get_params", &auroraml::cluster::DBSCAN::get_params)
        .def("set_params", &auroraml::cluster::DBSCAN::set_params)
        .def("is_fitted", &auroraml::cluster::DBSCAN::is_fitted)
        .def("labels", &auroraml::cluster::DBSCAN::labels);

    py::class_<auroraml::cluster::AgglomerativeClustering, auroraml::Estimator>(cluster_module, "AgglomerativeClustering")
        .def(py::init<int,const std::string&,const std::string&>(), py::arg("n_clusters")=2, py::arg("linkage")="single", py::arg("affinity")="euclidean")
        .def("fit", &auroraml::cluster::AgglomerativeClustering::fit)
        .def("get_params", &auroraml::cluster::AgglomerativeClustering::get_params)
        .def("set_params", &auroraml::cluster::AgglomerativeClustering::set_params)
        .def("is_fitted", &auroraml::cluster::AgglomerativeClustering::is_fitted)
        .def("labels", &auroraml::cluster::AgglomerativeClustering::labels);

    // Decomposition
    py::module_ decomp_module = m.def_submodule("decomposition", "Decomposition algorithms");
    py::class_<auroraml::decomposition::PCA, auroraml::Estimator, auroraml::Transformer>(decomp_module, "PCA")
        .def(py::init<int,bool>(), py::arg("n_components")=-1, py::arg("whiten")=false)
        .def("fit", &auroraml::decomposition::PCA::fit)
        .def("transform", &auroraml::decomposition::PCA::transform)
        .def("inverse_transform", &auroraml::decomposition::PCA::inverse_transform)
        .def("fit_transform", &auroraml::decomposition::PCA::fit_transform)
        .def("get_params", &auroraml::decomposition::PCA::get_params)
        .def("set_params", &auroraml::decomposition::PCA::set_params)
        .def("is_fitted", &auroraml::decomposition::PCA::is_fitted);

    py::class_<auroraml::decomposition::TruncatedSVD, auroraml::Estimator, auroraml::Transformer>(decomp_module, "TruncatedSVD")
        .def(py::init<int>(), py::arg("n_components"))
        .def("fit", &auroraml::decomposition::TruncatedSVD::fit)
        .def("transform", &auroraml::decomposition::TruncatedSVD::transform)
        .def("inverse_transform", &auroraml::decomposition::TruncatedSVD::inverse_transform)
        .def("fit_transform", &auroraml::decomposition::TruncatedSVD::fit_transform)
        .def("get_params", &auroraml::decomposition::TruncatedSVD::get_params)
        .def("set_params", &auroraml::decomposition::TruncatedSVD::set_params)
        .def("is_fitted", &auroraml::decomposition::TruncatedSVD::is_fitted)
        .def("components", &auroraml::decomposition::TruncatedSVD::components)
        .def("singular_values", &auroraml::decomposition::TruncatedSVD::singular_values)
        .def("explained_variance", &auroraml::decomposition::TruncatedSVD::explained_variance);

    py::class_<auroraml::decomposition::LDA, auroraml::Estimator, auroraml::Transformer>(decomp_module, "LDA")
        .def(py::init<int>(), py::arg("n_components") = -1)
        .def("fit", &auroraml::decomposition::LDA::fit)
        .def("transform", &auroraml::decomposition::LDA::transform)
        .def("inverse_transform", &auroraml::decomposition::LDA::inverse_transform)
        .def("fit_transform", &auroraml::decomposition::LDA::fit_transform)
        .def("get_params", &auroraml::decomposition::LDA::get_params)
        .def("set_params", &auroraml::decomposition::LDA::set_params)
        .def("is_fitted", &auroraml::decomposition::LDA::is_fitted)
        .def("components", &auroraml::decomposition::LDA::components)
        .def("explained_variance", &auroraml::decomposition::LDA::explained_variance)
        .def("explained_variance_ratio", &auroraml::decomposition::LDA::explained_variance_ratio)
        .def("mean", &auroraml::decomposition::LDA::mean)
        .def("class_means", &auroraml::decomposition::LDA::class_means)
        .def("classes", &auroraml::decomposition::LDA::classes);

    // SVM
    py::module_ svm_module = m.def_submodule("svm", "Support Vector Machines");
    py::class_<auroraml::svm::LinearSVC, auroraml::Estimator, auroraml::Classifier>(svm_module, "LinearSVC")
        .def(py::init<double,int,double,int>(), py::arg("C")=1.0, py::arg("max_iter")=1000, py::arg("lr")=0.01, py::arg("random_state")=-1)
        .def("fit", &auroraml::svm::LinearSVC::fit)
        .def("predict", &auroraml::svm::LinearSVC::predict_classes)
        .def("predict_proba", &auroraml::svm::LinearSVC::predict_proba)
        .def("decision_function", &auroraml::svm::LinearSVC::decision_function)
        .def("get_params", &auroraml::svm::LinearSVC::get_params)
        .def("set_params", &auroraml::svm::LinearSVC::set_params)
        .def("is_fitted", &auroraml::svm::LinearSVC::is_fitted)
        .def("save", &auroraml::svm::LinearSVC::save)
        .def("load", &auroraml::svm::LinearSVC::load);

    py::class_<auroraml::svm::SVR, auroraml::Estimator, auroraml::Regressor>(svm_module, "SVR")
        .def(py::init<double,double,int,double,int>(), 
             py::arg("C")=1.0, py::arg("epsilon")=0.1, py::arg("max_iter")=1000, 
             py::arg("lr")=0.01, py::arg("random_state")=-1)
        .def("fit", &auroraml::svm::SVR::fit)
        .def("predict", &auroraml::svm::SVR::predict)
        .def("get_params", &auroraml::svm::SVR::get_params)
        .def("set_params", &auroraml::svm::SVR::set_params)
        .def("is_fitted", &auroraml::svm::SVR::is_fitted)
        .def("coef", &auroraml::svm::SVR::coef)
        .def("intercept", &auroraml::svm::SVR::intercept);

    // RandomForest
    py::module_ rf_module = m.def_submodule("ensemble", "Ensemble methods");
    py::class_<auroraml::ensemble::RandomForestClassifier, auroraml::Estimator, auroraml::Classifier>(rf_module, "RandomForestClassifier")
        .def(py::init<int,int,int,int>(), py::arg("n_estimators")=100, py::arg("max_depth")=-1, py::arg("max_features")=-1, py::arg("random_state")=-1)
        .def("fit", &auroraml::ensemble::RandomForestClassifier::fit)
        .def("predict", &auroraml::ensemble::RandomForestClassifier::predict_classes)
        .def("predict_proba", &auroraml::ensemble::RandomForestClassifier::predict_proba)
        .def("decision_function", &auroraml::ensemble::RandomForestClassifier::decision_function)
        .def("get_params", &auroraml::ensemble::RandomForestClassifier::get_params)
        .def("set_params", &auroraml::ensemble::RandomForestClassifier::set_params)
        .def("is_fitted", &auroraml::ensemble::RandomForestClassifier::is_fitted)
        .def("save", &auroraml::ensemble::RandomForestClassifier::save)
        .def("load", &auroraml::ensemble::RandomForestClassifier::load);

    py::class_<auroraml::ensemble::RandomForestRegressor, auroraml::Estimator, auroraml::Regressor>(rf_module, "RandomForestRegressor")
        .def(py::init<int,int,int,int>(), py::arg("n_estimators")=100, py::arg("max_depth")=-1, py::arg("max_features")=-1, py::arg("random_state")=-1)
        .def("fit", &auroraml::ensemble::RandomForestRegressor::fit)
        .def("predict", &auroraml::ensemble::RandomForestRegressor::predict)
        .def("get_params", &auroraml::ensemble::RandomForestRegressor::get_params)
        .def("set_params", &auroraml::ensemble::RandomForestRegressor::set_params)
        .def("is_fitted", &auroraml::ensemble::RandomForestRegressor::is_fitted)
        .def("save", &auroraml::ensemble::RandomForestRegressor::save)
        .def("load", &auroraml::ensemble::RandomForestRegressor::load);

    // Gradient Boosting
    py::module_ gb_module = m.def_submodule("gradient_boosting", "Gradient Boosting algorithms");
    py::class_<auroraml::ensemble::GradientBoostingClassifier, auroraml::Estimator, auroraml::Classifier>(gb_module, "GradientBoostingClassifier")
        .def(py::init<int,double,int,int,int,double,int>(),
             py::arg("n_estimators")=100, py::arg("learning_rate")=0.1, py::arg("max_depth")=3,
             py::arg("min_samples_split")=2, py::arg("min_samples_leaf")=1, 
             py::arg("min_impurity_decrease")=0.0, py::arg("random_state")=-1)
        .def("fit", &auroraml::ensemble::GradientBoostingClassifier::fit)
        .def("predict_classes", &auroraml::ensemble::GradientBoostingClassifier::predict_classes)
        .def("predict_proba", &auroraml::ensemble::GradientBoostingClassifier::predict_proba)
        .def("predict", &auroraml::ensemble::GradientBoostingClassifier::predict_classes)
        .def("get_params", &auroraml::ensemble::GradientBoostingClassifier::get_params)
        .def("set_params", &auroraml::ensemble::GradientBoostingClassifier::set_params)
        .def("is_fitted", &auroraml::ensemble::GradientBoostingClassifier::is_fitted)
        .def("n_estimators", &auroraml::ensemble::GradientBoostingClassifier::n_estimators)
        .def("learning_rate", &auroraml::ensemble::GradientBoostingClassifier::learning_rate)
        .def("classes", &auroraml::ensemble::GradientBoostingClassifier::classes)
        .def("save", &auroraml::ensemble::GradientBoostingClassifier::save)
        .def("load", &auroraml::ensemble::GradientBoostingClassifier::load);

    py::class_<auroraml::ensemble::GradientBoostingRegressor, auroraml::Estimator, auroraml::Regressor>(gb_module, "GradientBoostingRegressor")
        .def(py::init<int,double,int,int,int,double,int>(),
             py::arg("n_estimators")=100, py::arg("learning_rate")=0.1, py::arg("max_depth")=3,
             py::arg("min_samples_split")=2, py::arg("min_samples_leaf")=1, 
             py::arg("min_impurity_decrease")=0.0, py::arg("random_state")=-1)
        .def("fit", &auroraml::ensemble::GradientBoostingRegressor::fit)
        .def("predict", &auroraml::ensemble::GradientBoostingRegressor::predict)
        .def("get_params", &auroraml::ensemble::GradientBoostingRegressor::get_params)
        .def("set_params", &auroraml::ensemble::GradientBoostingRegressor::set_params)
        .def("is_fitted", &auroraml::ensemble::GradientBoostingRegressor::is_fitted)
        .def("n_estimators", &auroraml::ensemble::GradientBoostingRegressor::n_estimators)
        .def("learning_rate", &auroraml::ensemble::GradientBoostingRegressor::learning_rate)
        .def("save", &auroraml::ensemble::GradientBoostingRegressor::save)
        .def("load", &auroraml::ensemble::GradientBoostingRegressor::load);

    // AdaBoost
    py::module_ adaboost_module = m.def_submodule("adaboost", "AdaBoost algorithms");
    py::class_<auroraml::ensemble::AdaBoostClassifier, auroraml::Estimator, auroraml::Classifier>(adaboost_module, "AdaBoostClassifier")
        .def(py::init<int, double, int>(),
             py::arg("n_estimators") = 50, py::arg("learning_rate") = 1.0, py::arg("random_state") = -1)
        .def("fit", &auroraml::ensemble::AdaBoostClassifier::fit)
        .def("predict", &auroraml::ensemble::AdaBoostClassifier::predict_classes)
        .def("predict_proba", &auroraml::ensemble::AdaBoostClassifier::predict_proba)
        .def("decision_function", &auroraml::ensemble::AdaBoostClassifier::decision_function)
        .def("get_params", &auroraml::ensemble::AdaBoostClassifier::get_params)
        .def("set_params", &auroraml::ensemble::AdaBoostClassifier::set_params)
        .def("is_fitted", &auroraml::ensemble::AdaBoostClassifier::is_fitted)
        .def("classes", &auroraml::ensemble::AdaBoostClassifier::classes);

    py::class_<auroraml::ensemble::AdaBoostRegressor, auroraml::Estimator, auroraml::Regressor>(adaboost_module, "AdaBoostRegressor")
        .def(py::init<int, double, std::string, int>(),
             py::arg("n_estimators") = 50, py::arg("learning_rate") = 1.0,
             py::arg("loss") = "linear", py::arg("random_state") = -1)
        .def("fit", &auroraml::ensemble::AdaBoostRegressor::fit)
        .def("predict", &auroraml::ensemble::AdaBoostRegressor::predict)
        .def("get_params", &auroraml::ensemble::AdaBoostRegressor::get_params)
        .def("set_params", &auroraml::ensemble::AdaBoostRegressor::set_params)
        .def("is_fitted", &auroraml::ensemble::AdaBoostRegressor::is_fitted);

    // XGBoost
    py::module_ xgb_module = m.def_submodule("xgboost", "XGBoost algorithms");
    py::class_<auroraml::ensemble::XGBClassifier, auroraml::Estimator, auroraml::Classifier>(xgb_module, "XGBClassifier")
        .def(py::init<int, double, int, double, double, double, int, double, double, int>(),
             py::arg("n_estimators") = 100, py::arg("learning_rate") = 0.1,
             py::arg("max_depth") = 6, py::arg("gamma") = 0.0, py::arg("reg_alpha") = 0.0,
             py::arg("reg_lambda") = 1.0, py::arg("min_child_weight") = 1,
             py::arg("subsample") = 1.0, py::arg("colsample_bytree") = 1.0,
             py::arg("random_state") = -1)
        .def("fit", &auroraml::ensemble::XGBClassifier::fit)
        .def("predict", &auroraml::ensemble::XGBClassifier::predict_classes)
        .def("predict_proba", &auroraml::ensemble::XGBClassifier::predict_proba)
        .def("decision_function", &auroraml::ensemble::XGBClassifier::decision_function)
        .def("get_params", &auroraml::ensemble::XGBClassifier::get_params)
        .def("set_params", &auroraml::ensemble::XGBClassifier::set_params)
        .def("is_fitted", &auroraml::ensemble::XGBClassifier::is_fitted)
        .def("classes", &auroraml::ensemble::XGBClassifier::classes);

    py::class_<auroraml::ensemble::XGBRegressor, auroraml::Estimator, auroraml::Regressor>(xgb_module, "XGBRegressor")
        .def(py::init<int, double, int, double, double, double, int, double, double, int>(),
             py::arg("n_estimators") = 100, py::arg("learning_rate") = 0.1,
             py::arg("max_depth") = 6, py::arg("gamma") = 0.0, py::arg("reg_alpha") = 0.0,
             py::arg("reg_lambda") = 1.0, py::arg("min_child_weight") = 1,
             py::arg("subsample") = 1.0, py::arg("colsample_bytree") = 1.0,
             py::arg("random_state") = -1)
        .def("fit", &auroraml::ensemble::XGBRegressor::fit)
        .def("predict", &auroraml::ensemble::XGBRegressor::predict)
        .def("get_params", &auroraml::ensemble::XGBRegressor::get_params)
        .def("set_params", &auroraml::ensemble::XGBRegressor::set_params)
        .def("is_fitted", &auroraml::ensemble::XGBRegressor::is_fitted);

    // CatBoost
    py::module_ catboost_module = m.def_submodule("catboost", "CatBoost algorithms");
    py::class_<auroraml::ensemble::CatBoostClassifier, auroraml::Estimator, auroraml::Classifier>(catboost_module, "CatBoostClassifier")
        .def(py::init<int, double, int, double, double, double, int>(),
             py::arg("n_estimators") = 100, py::arg("learning_rate") = 0.03,
             py::arg("max_depth") = 6, py::arg("l2_leaf_reg") = 3.0,
             py::arg("border_count") = 32.0, py::arg("bagging_temperature") = 1.0,
             py::arg("random_state") = -1)
        .def("fit", &auroraml::ensemble::CatBoostClassifier::fit)
        .def("predict", &auroraml::ensemble::CatBoostClassifier::predict_classes)
        .def("predict_proba", &auroraml::ensemble::CatBoostClassifier::predict_proba)
        .def("decision_function", &auroraml::ensemble::CatBoostClassifier::decision_function)
        .def("get_params", &auroraml::ensemble::CatBoostClassifier::get_params)
        .def("set_params", &auroraml::ensemble::CatBoostClassifier::set_params)
        .def("is_fitted", &auroraml::ensemble::CatBoostClassifier::is_fitted)
        .def("classes", &auroraml::ensemble::CatBoostClassifier::classes);

    py::class_<auroraml::ensemble::CatBoostRegressor, auroraml::Estimator, auroraml::Regressor>(catboost_module, "CatBoostRegressor")
        .def(py::init<int, double, int, double, double, double, int>(),
             py::arg("n_estimators") = 100, py::arg("learning_rate") = 0.03,
             py::arg("max_depth") = 6, py::arg("l2_leaf_reg") = 3.0,
             py::arg("border_count") = 32.0, py::arg("bagging_temperature") = 1.0,
             py::arg("random_state") = -1)
        .def("fit", &auroraml::ensemble::CatBoostRegressor::fit)
        .def("predict", &auroraml::ensemble::CatBoostRegressor::predict)
        .def("get_params", &auroraml::ensemble::CatBoostRegressor::get_params)
        .def("set_params", &auroraml::ensemble::CatBoostRegressor::set_params)
        .def("is_fitted", &auroraml::ensemble::CatBoostRegressor::is_fitted);

    py::class_<auroraml::model_selection::GridSearchCV>(model_selection_module, "GridSearchCV")
        .def(py::init<auroraml::Estimator&, const std::vector<auroraml::Params>&, const auroraml::model_selection::BaseCrossValidator&, std::string, int, bool>(),
             py::arg("estimator"), py::arg("param_grid"), py::arg("cv"), py::arg("scoring") = "accuracy",
             py::arg("n_jobs") = 1, py::arg("verbose") = false)
        .def("fit", &auroraml::model_selection::GridSearchCV::fit)
        .def("predict", &auroraml::model_selection::GridSearchCV::predict)
        .def("best_params", &auroraml::model_selection::GridSearchCV::best_params)
        .def("best_score", &auroraml::model_selection::GridSearchCV::best_score);
    
    py::class_<auroraml::model_selection::RandomizedSearchCV>(model_selection_module, "RandomizedSearchCV")
        .def(py::init<auroraml::Estimator&, const std::vector<auroraml::Params>&, const auroraml::model_selection::BaseCrossValidator&, std::string, int, int, bool>(),
             py::arg("estimator"), py::arg("param_distributions"), py::arg("cv"), py::arg("scoring") = "accuracy",
             py::arg("n_iter") = 10, py::arg("n_jobs") = 1, py::arg("verbose") = false)
        .def("fit", &auroraml::model_selection::RandomizedSearchCV::fit)
        .def("predict", &auroraml::model_selection::RandomizedSearchCV::predict)
        .def("best_params", &auroraml::model_selection::RandomizedSearchCV::best_params)
        .def("best_score", &auroraml::model_selection::RandomizedSearchCV::best_score);
}
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
#include "auroraml/pipeline.hpp"
#include "auroraml/compose.hpp"
#include "auroraml/feature_selection.hpp"
#include "auroraml/impute.hpp"
#include "auroraml/utils.hpp"
#include "auroraml/inspection.hpp"
#include "auroraml/ensemble_wrappers.hpp"
#include "auroraml/calibration.hpp"
#include "auroraml/isotonic.hpp"
#include "auroraml/discriminant_analysis.hpp"
#include "auroraml/naive_bayes_variants.hpp"
#include "auroraml/extratree.hpp"
#include "auroraml/outlier_detection.hpp"
#include "auroraml/mixture.hpp"
#include "auroraml/semi_supervised.hpp"
#include "auroraml/preprocessing_extended.hpp"
#include "auroraml/cluster_extended.hpp"

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
    
    // Classification metrics
    metrics_module.def("accuracy_score", &auroraml::metrics::accuracy_score);
    metrics_module.def("balanced_accuracy_score", &auroraml::metrics::balanced_accuracy_score);
    metrics_module.def("top_k_accuracy_score", &auroraml::metrics::top_k_accuracy_score, py::arg("y_true"), py::arg("y_score"), py::arg("k") = 5);
    metrics_module.def("roc_auc_score", 
        [](const auroraml::VectorXi& y_true, const auroraml::VectorXd& y_score) {
            return auroraml::metrics::roc_auc_score(y_true, y_score);
        }, py::arg("y_true"), py::arg("y_score"));
    metrics_module.def("roc_auc_score_multiclass", &auroraml::metrics::roc_auc_score_multiclass, py::arg("y_true"), py::arg("y_score"), py::arg("average") = "macro");
    metrics_module.def("average_precision_score", &auroraml::metrics::average_precision_score);
    metrics_module.def("log_loss", &auroraml::metrics::log_loss);
    metrics_module.def("hinge_loss", &auroraml::metrics::hinge_loss);
    metrics_module.def("cohen_kappa_score", &auroraml::metrics::cohen_kappa_score);
    metrics_module.def("matthews_corrcoef", &auroraml::metrics::matthews_corrcoef);
    metrics_module.def("hamming_loss", &auroraml::metrics::hamming_loss);
    metrics_module.def("jaccard_score", &auroraml::metrics::jaccard_score, py::arg("y_true"), py::arg("y_pred"), py::arg("average") = "macro");
    metrics_module.def("zero_one_loss", &auroraml::metrics::zero_one_loss);
    metrics_module.def("brier_score_loss", &auroraml::metrics::brier_score_loss);
    metrics_module.def("precision_score", &auroraml::metrics::precision_score, py::arg("y_true"), py::arg("y_pred"), py::arg("average") = "macro");
    metrics_module.def("recall_score", &auroraml::metrics::recall_score, py::arg("y_true"), py::arg("y_pred"), py::arg("average") = "macro");
    metrics_module.def("f1_score", &auroraml::metrics::f1_score, py::arg("y_true"), py::arg("y_pred"), py::arg("average") = "macro");
    metrics_module.def("confusion_matrix", &auroraml::metrics::confusion_matrix);
    metrics_module.def("classification_report", &auroraml::metrics::classification_report);
    
    // Regression metrics
    metrics_module.def("mean_squared_error", &auroraml::metrics::mean_squared_error);
    metrics_module.def("root_mean_squared_error", &auroraml::metrics::root_mean_squared_error);
    metrics_module.def("mean_absolute_error", &auroraml::metrics::mean_absolute_error);
    metrics_module.def("median_absolute_error", &auroraml::metrics::median_absolute_error);
    metrics_module.def("max_error", &auroraml::metrics::max_error);
    metrics_module.def("mean_poisson_deviance", &auroraml::metrics::mean_poisson_deviance);
    metrics_module.def("mean_gamma_deviance", &auroraml::metrics::mean_gamma_deviance);
    metrics_module.def("mean_tweedie_deviance", &auroraml::metrics::mean_tweedie_deviance, py::arg("y_true"), py::arg("y_pred"), py::arg("power") = 0.0);
    metrics_module.def("d2_tweedie_score", &auroraml::metrics::d2_tweedie_score, py::arg("y_true"), py::arg("y_pred"), py::arg("power") = 0.0);
    metrics_module.def("d2_pinball_score", &auroraml::metrics::d2_pinball_score, py::arg("y_true"), py::arg("y_pred"), py::arg("alpha") = 0.5);
    metrics_module.def("d2_absolute_error_score", &auroraml::metrics::d2_absolute_error_score);
    metrics_module.def("r2_score", &auroraml::metrics::r2_score);
    metrics_module.def("explained_variance_score", &auroraml::metrics::explained_variance_score);
    metrics_module.def("mean_absolute_percentage_error", &auroraml::metrics::mean_absolute_percentage_error);
    
    // Clustering metrics
    metrics_module.def("silhouette_score", &auroraml::metrics::silhouette_score);
    metrics_module.def("silhouette_samples", &auroraml::metrics::silhouette_samples);
    metrics_module.def("calinski_harabasz_score", &auroraml::metrics::calinski_harabasz_score);
    metrics_module.def("davies_bouldin_score", &auroraml::metrics::davies_bouldin_score);
    
    // Clustering comparison metrics
    metrics_module.def("adjusted_rand_score", &auroraml::metrics::adjusted_rand_score);
    metrics_module.def("adjusted_mutual_info_score", &auroraml::metrics::adjusted_mutual_info_score);
    metrics_module.def("normalized_mutual_info_score", &auroraml::metrics::normalized_mutual_info_score);
    metrics_module.def("homogeneity_score", &auroraml::metrics::homogeneity_score);
    metrics_module.def("completeness_score", &auroraml::metrics::completeness_score);
    metrics_module.def("v_measure_score", &auroraml::metrics::v_measure_score);
    metrics_module.def("fowlkes_mallows_score", &auroraml::metrics::fowlkes_mallows_score);
    
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
    
    // Pipeline module
    py::module_ pipeline_module = m.def_submodule("pipeline", "Pipeline and FeatureUnion utilities");
    
    // Helper lambda to create Pipeline from Python list of (name, estimator) tuples
    auto pipeline_init = [](py::list steps_py) {
        std::vector<std::pair<std::string, std::shared_ptr<auroraml::Estimator>>> steps;
        for (auto item : steps_py) {
            py::tuple step_tuple = py::cast<py::tuple>(item);
            if (step_tuple.size() != 2) {
                throw std::runtime_error("Each step must be a (name, estimator) tuple");
            }
            std::string name = py::cast<std::string>(step_tuple[0]);
            py::object estimator_obj = step_tuple[1];
            // Extract raw pointer and wrap in shared_ptr with no-op deleter (Python manages lifetime)
            auroraml::Estimator* estimator_ptr = estimator_obj.cast<auroraml::Estimator*>();
            std::shared_ptr<auroraml::Estimator> estimator(estimator_ptr, [](auroraml::Estimator*) {});
            steps.push_back({name, estimator});
        }
        return new auroraml::pipeline::Pipeline(steps);
    };
    
    py::class_<auroraml::pipeline::Pipeline>(pipeline_module, "Pipeline")
        .def(py::init(pipeline_init), py::arg("steps"))
        .def("fit", &auroraml::pipeline::Pipeline::fit, py::arg("X"), py::arg("y"))
        .def("transform", &auroraml::pipeline::Pipeline::transform, py::arg("X"))
        .def("predict", &auroraml::pipeline::Pipeline::predict, py::arg("X"))
        .def("predict_classes", &auroraml::pipeline::Pipeline::predict_classes, py::arg("X"))
        .def("predict_proba", &auroraml::pipeline::Pipeline::predict_proba, py::arg("X"))
        .def("fit_transform", &auroraml::pipeline::Pipeline::fit_transform, py::arg("X"), py::arg("y"))
        .def("get_params", &auroraml::pipeline::Pipeline::get_params)
        .def("set_params", &auroraml::pipeline::Pipeline::set_params)
        .def("is_fitted", &auroraml::pipeline::Pipeline::is_fitted)
        .def("get_step", &auroraml::pipeline::Pipeline::get_step, py::arg("name"))
        .def("get_step_names", &auroraml::pipeline::Pipeline::get_step_names);
    
    // Helper lambda to create FeatureUnion from Python list of (name, transformer) tuples
    auto featureunion_init = [](py::list transformers_py) {
        std::vector<std::pair<std::string, std::shared_ptr<auroraml::Transformer>>> transformers;
        for (auto item : transformers_py) {
            py::tuple transformer_tuple = py::cast<py::tuple>(item);
            if (transformer_tuple.size() != 2) {
                throw std::runtime_error("Each transformer must be a (name, transformer) tuple");
            }
            std::string name = py::cast<std::string>(transformer_tuple[0]);
            py::object transformer_obj = transformer_tuple[1];
            // Extract raw pointer and wrap in shared_ptr with no-op deleter (Python manages lifetime)
            auroraml::Transformer* transformer_ptr = transformer_obj.cast<auroraml::Transformer*>();
            std::shared_ptr<auroraml::Transformer> transformer(transformer_ptr, [](auroraml::Transformer*) {});
            transformers.push_back({name, transformer});
        }
        return new auroraml::pipeline::FeatureUnion(transformers);
    };
    
    py::class_<auroraml::pipeline::FeatureUnion>(pipeline_module, "FeatureUnion")
        .def(py::init(featureunion_init), py::arg("transformers"))
        .def("fit", &auroraml::pipeline::FeatureUnion::fit, py::arg("X"), py::arg("y"))
        .def("transform", &auroraml::pipeline::FeatureUnion::transform, py::arg("X"))
        .def("fit_transform", &auroraml::pipeline::FeatureUnion::fit_transform, py::arg("X"), py::arg("y"))
        .def("get_params", &auroraml::pipeline::FeatureUnion::get_params)
        .def("set_params", &auroraml::pipeline::FeatureUnion::set_params)
        .def("is_fitted", &auroraml::pipeline::FeatureUnion::is_fitted)
        .def("get_transformer", &auroraml::pipeline::FeatureUnion::get_transformer, py::arg("name"))
        .def("get_transformer_names", &auroraml::pipeline::FeatureUnion::get_transformer_names);
    
    // Compose module
    py::module_ compose_module = m.def_submodule("compose", "Composition utilities");
    
    // Helper lambda to create ColumnTransformer from Python list
    auto columntransformer_init = [](py::list transformers_py, const std::string& remainder = "drop", double sparse_threshold = 0.3) {
        std::vector<std::tuple<std::string, std::shared_ptr<auroraml::Transformer>, std::vector<int>>> transformers;
        for (auto item : transformers_py) {
            py::tuple transformer_tuple = py::cast<py::tuple>(item);
            if (transformer_tuple.size() != 3) {
                throw std::runtime_error("Each transformer must be a (name, transformer, column_indices) tuple");
            }
            std::string name = py::cast<std::string>(transformer_tuple[0]);
            py::object transformer_obj = transformer_tuple[1];
            auroraml::Transformer* transformer_ptr = transformer_obj.cast<auroraml::Transformer*>();
            std::shared_ptr<auroraml::Transformer> transformer(transformer_ptr, [](auroraml::Transformer*) {});
            py::list columns_py = py::cast<py::list>(transformer_tuple[2]);
            std::vector<int> column_indices;
            for (auto col : columns_py) {
                column_indices.push_back(py::cast<int>(col));
            }
            transformers.push_back(std::make_tuple(name, transformer, column_indices));
        }
        return new auroraml::compose::ColumnTransformer(transformers, remainder, sparse_threshold);
    };
    
    py::class_<auroraml::compose::ColumnTransformer>(compose_module, "ColumnTransformer")
        .def(py::init(columntransformer_init), 
             py::arg("transformers"), py::arg("remainder") = "drop", py::arg("sparse_threshold") = 0.3)
        .def("fit", &auroraml::compose::ColumnTransformer::fit, py::arg("X"), py::arg("y"))
        .def("transform", &auroraml::compose::ColumnTransformer::transform, py::arg("X"))
        .def("fit_transform", &auroraml::compose::ColumnTransformer::fit_transform, py::arg("X"), py::arg("y"))
        .def("get_params", &auroraml::compose::ColumnTransformer::get_params)
        .def("set_params", &auroraml::compose::ColumnTransformer::set_params)
        .def("is_fitted", &auroraml::compose::ColumnTransformer::is_fitted)
        .def("get_transformer", &auroraml::compose::ColumnTransformer::get_transformer, py::arg("name"))
        .def("get_transformer_names", &auroraml::compose::ColumnTransformer::get_transformer_names);
    
    // Helper lambda to create TransformedTargetRegressor
    auto transformedtargetregressor_init = [](py::object regressor_py, py::object transformer_py = py::none()) {
        auroraml::Regressor* regressor_ptr = regressor_py.cast<auroraml::Regressor*>();
        std::shared_ptr<auroraml::Regressor> regressor(regressor_ptr, [](auroraml::Regressor*) {});
        std::shared_ptr<auroraml::Transformer> transformer = nullptr;
        if (!transformer_py.is_none()) {
            auroraml::Transformer* transformer_ptr = transformer_py.cast<auroraml::Transformer*>();
            transformer = std::shared_ptr<auroraml::Transformer>(transformer_ptr, [](auroraml::Transformer*) {});
        }
        return new auroraml::compose::TransformedTargetRegressor(regressor, transformer);
    };
    
    py::class_<auroraml::compose::TransformedTargetRegressor>(compose_module, "TransformedTargetRegressor")
        .def(py::init(transformedtargetregressor_init), 
             py::arg("regressor"), py::arg("transformer") = py::none())
        .def("fit", &auroraml::compose::TransformedTargetRegressor::fit, py::arg("X"), py::arg("y"))
        .def("predict", &auroraml::compose::TransformedTargetRegressor::predict, py::arg("X"))
        .def("get_params", &auroraml::compose::TransformedTargetRegressor::get_params)
        .def("set_params", &auroraml::compose::TransformedTargetRegressor::set_params)
        .def("is_fitted", &auroraml::compose::TransformedTargetRegressor::is_fitted)
        .def("regressor", &auroraml::compose::TransformedTargetRegressor::regressor)
        .def("transformer", &auroraml::compose::TransformedTargetRegressor::transformer);
    
    // Feature selection module
    py::module_ feature_selection_module = m.def_submodule("feature_selection", "Feature selection utilities");
    
    py::class_<auroraml::feature_selection::VarianceThreshold, auroraml::Estimator, auroraml::Transformer>(feature_selection_module, "VarianceThreshold")
        .def(py::init<double>(), py::arg("threshold") = 0.0)
        .def("fit", &auroraml::feature_selection::VarianceThreshold::fit)
        .def("transform", &auroraml::feature_selection::VarianceThreshold::transform)
        .def("fit_transform", &auroraml::feature_selection::VarianceThreshold::fit_transform)
        .def("get_params", &auroraml::feature_selection::VarianceThreshold::get_params)
        .def("set_params", &auroraml::feature_selection::VarianceThreshold::set_params)
        .def("is_fitted", &auroraml::feature_selection::VarianceThreshold::is_fitted)
        .def("get_support", &auroraml::feature_selection::VarianceThreshold::get_support);
    
    // Helper lambda to create SelectKBest with scoring function
    auto selectkbest_init = [](py::object score_func_py, int k = 10) {
        // Create a C++ function wrapper for the Python scoring function
        std::function<double(const auroraml::VectorXd&, const auroraml::VectorXd&)> score_func =
            [score_func_py](const auroraml::VectorXd& X_feature, const auroraml::VectorXd& y) {
                // Convert Eigen vectors to numpy arrays
                py::array_t<double> X_arr = py::cast(X_feature);
                py::array_t<double> y_arr = py::cast(y);
                
                // Call Python scoring function
                py::object result = score_func_py(X_arr, y_arr);
                return py::cast<double>(result);
            };
        
        return new auroraml::feature_selection::SelectKBest(score_func, k);
    };
    
    py::class_<auroraml::feature_selection::SelectKBest, auroraml::Estimator, auroraml::Transformer>(feature_selection_module, "SelectKBest")
        .def(py::init(selectkbest_init), py::arg("score_func"), py::arg("k") = 10)
        .def("fit", &auroraml::feature_selection::SelectKBest::fit)
        .def("transform", &auroraml::feature_selection::SelectKBest::transform)
        .def("fit_transform", &auroraml::feature_selection::SelectKBest::fit_transform)
        .def("get_params", &auroraml::feature_selection::SelectKBest::get_params)
        .def("set_params", &auroraml::feature_selection::SelectKBest::set_params)
        .def("is_fitted", &auroraml::feature_selection::SelectKBest::is_fitted)
        .def("get_support", &auroraml::feature_selection::SelectKBest::get_support)
        .def("scores", &auroraml::feature_selection::SelectKBest::scores);
    
    // Helper lambda to create SelectPercentile with scoring function
    auto selectpercentile_init = [](py::object score_func_py, int percentile = 10) {
        std::function<double(const auroraml::VectorXd&, const auroraml::VectorXd&)> score_func =
            [score_func_py](const auroraml::VectorXd& X_feature, const auroraml::VectorXd& y) {
                py::array_t<double> X_arr = py::cast(X_feature);
                py::array_t<double> y_arr = py::cast(y);
                py::object result = score_func_py(X_arr, y_arr);
                return py::cast<double>(result);
            };
        
        return new auroraml::feature_selection::SelectPercentile(score_func, percentile);
    };
    
    py::class_<auroraml::feature_selection::SelectPercentile, auroraml::Estimator, auroraml::Transformer>(feature_selection_module, "SelectPercentile")
        .def(py::init(selectpercentile_init), py::arg("score_func"), py::arg("percentile") = 10)
        .def("fit", &auroraml::feature_selection::SelectPercentile::fit)
        .def("transform", &auroraml::feature_selection::SelectPercentile::transform)
        .def("fit_transform", &auroraml::feature_selection::SelectPercentile::fit_transform)
        .def("get_params", &auroraml::feature_selection::SelectPercentile::get_params)
        .def("set_params", &auroraml::feature_selection::SelectPercentile::set_params)
        .def("is_fitted", &auroraml::feature_selection::SelectPercentile::is_fitted)
        .def("get_support", &auroraml::feature_selection::SelectPercentile::get_support)
        .def("scores", &auroraml::feature_selection::SelectPercentile::scores);
    
    // Scoring functions
    py::module_ scores_module = feature_selection_module.def_submodule("scores", "Scoring functions");
    scores_module.def("f_classif", &auroraml::feature_selection::scores::f_classif);
    scores_module.def("f_regression", &auroraml::feature_selection::scores::f_regression);
    scores_module.def("mutual_info_classif", &auroraml::feature_selection::scores::mutual_info_classif);
    scores_module.def("mutual_info_regression", &auroraml::feature_selection::scores::mutual_info_regression);
    scores_module.def("chi2", 
        [](const auroraml::VectorXd& X_feature, const auroraml::VectorXi& y) {
            return auroraml::feature_selection::scores::chi2(X_feature, y);
        });
    
    // Impute module
    py::module_ impute_module = m.def_submodule("impute", "Imputation utilities");
    
    // Helper lambdas for optional y parameter in imputers
    auto knn_imputer_fit = [](auroraml::impute::KNNImputer& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) -> auroraml::impute::KNNImputer& {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto knn_imputer_fit_transform = [](auroraml::impute::KNNImputer& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        return self.fit_transform(X, y);
    };
    auto iterative_imputer_fit = [](auroraml::impute::IterativeImputer& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) -> auroraml::impute::IterativeImputer& {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto iterative_imputer_fit_transform = [](auroraml::impute::IterativeImputer& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        return self.fit_transform(X, y);
    };
    
    py::class_<auroraml::impute::KNNImputer, auroraml::Estimator, auroraml::Transformer>(impute_module, "KNNImputer")
        .def(py::init<int, const std::string&>(), py::arg("n_neighbors") = 5, py::arg("metric") = "euclidean")
        .def("fit", knn_imputer_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &auroraml::impute::KNNImputer::transform)
        .def("fit_transform", knn_imputer_fit_transform, py::arg("X"), py::arg("y") = py::none())
        .def("get_params", &auroraml::impute::KNNImputer::get_params)
        .def("set_params", &auroraml::impute::KNNImputer::set_params)
        .def("is_fitted", &auroraml::impute::KNNImputer::is_fitted);
    
    py::class_<auroraml::impute::IterativeImputer, auroraml::Estimator, auroraml::Transformer>(impute_module, "IterativeImputer")
        .def(py::init<int, double, int>(), py::arg("max_iter") = 10, py::arg("tol") = 1e-3, py::arg("random_state") = -1)
        .def("fit", iterative_imputer_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &auroraml::impute::IterativeImputer::transform)
        .def("fit_transform", iterative_imputer_fit_transform, py::arg("X"), py::arg("y") = py::none())
        .def("get_params", &auroraml::impute::IterativeImputer::get_params)
        .def("set_params", &auroraml::impute::IterativeImputer::set_params)
        .def("is_fitted", &auroraml::impute::IterativeImputer::is_fitted);
    
    // Utils module
    py::module_ utils_module = m.def_submodule("utils", "Utility functions");
    
    // Multiclass utilities
    py::module_ multiclass_module = utils_module.def_submodule("multiclass", "Multiclass utilities");
    multiclass_module.def("is_multiclass", &auroraml::utils::multiclass::is_multiclass);
    multiclass_module.def("unique_labels", &auroraml::utils::multiclass::unique_labels);
    multiclass_module.def("type_of_target", &auroraml::utils::multiclass::type_of_target);
    
    // Resample utilities
    py::module_ resample_module = utils_module.def_submodule("resample", "Resampling utilities");
    resample_module.def("resample", 
        [](const auroraml::MatrixXd& X, const auroraml::VectorXd& y, int n_samples, int random_state) {
            auto result = auroraml::utils::resample::resample(X, y, n_samples, random_state);
            return py::make_tuple(result.first, result.second);
        }, py::arg("X"), py::arg("y"), py::arg("n_samples") = -1, py::arg("random_state") = -1);
    resample_module.def("shuffle", &auroraml::utils::resample::shuffle, 
        py::arg("X"), py::arg("y"), py::arg("random_state") = -1);
    resample_module.def("train_test_split_stratified", 
        [](const auroraml::MatrixXd& X, const auroraml::VectorXi& y, double test_size, int random_state) {
            auto result = auroraml::utils::resample::train_test_split_stratified(X, y, test_size, random_state);
            return py::make_tuple(result.first.first, result.first.second, result.second.first, result.second.second);
        }, py::arg("X"), py::arg("y"), py::arg("test_size") = 0.25, py::arg("random_state") = -1);
    
    // Validation utilities
    py::module_ validation_module = utils_module.def_submodule("validation", "Validation utilities");
    validation_module.def("check_finite", &auroraml::utils::validation::check_finite);
    validation_module.def("check_has_nan", &auroraml::utils::validation::check_has_nan);
    validation_module.def("check_has_inf", &auroraml::utils::validation::check_has_inf);
    
    // Class weight utilities
    py::module_ class_weight_module = utils_module.def_submodule("class_weight", "Class weight utilities");
    class_weight_module.def("compute_class_weight", &auroraml::utils::class_weight::compute_class_weight);
    class_weight_module.def("compute_sample_weight", &auroraml::utils::class_weight::compute_sample_weight);
    
    // Array utilities
    py::module_ array_module = utils_module.def_submodule("array", "Array utilities");
    array_module.def("issparse", &auroraml::utils::array::issparse);
    array_module.def("shape", 
        [](const auroraml::MatrixXd& X) {
            auto shape = auroraml::utils::array::shape(X);
            return py::make_tuple(shape.first, shape.second);
        });
    
    // Inspection module
    py::module_ inspection_module = m.def_submodule("inspection", "Model inspection utilities");
    
    auto permutationimportance_init = [](py::object estimator_py, const std::string& scoring = "accuracy", int n_repeats = 5, int random_state = -1) {
        auroraml::Estimator* estimator_ptr = estimator_py.cast<auroraml::Estimator*>();
        std::shared_ptr<auroraml::Estimator> estimator(estimator_ptr, [](auroraml::Estimator*) {});
        return new auroraml::inspection::PermutationImportance(estimator, scoring, n_repeats, random_state);
    };
    
    py::class_<auroraml::inspection::PermutationImportance>(inspection_module, "PermutationImportance")
        .def(py::init(permutationimportance_init),
             py::arg("estimator"), py::arg("scoring") = "accuracy", py::arg("n_repeats") = 5, py::arg("random_state") = -1)
        .def("fit", &auroraml::inspection::PermutationImportance::fit)
        .def("feature_importances", &auroraml::inspection::PermutationImportance::feature_importances);
    
    auto partialdependence_init = [](py::object estimator_py, const std::vector<int>& features) {
        auroraml::Predictor* estimator_ptr = estimator_py.cast<auroraml::Predictor*>();
        std::shared_ptr<auroraml::Predictor> estimator(estimator_ptr, [](auroraml::Predictor*) {});
        return new auroraml::inspection::PartialDependence(estimator, features);
    };
    
    py::class_<auroraml::inspection::PartialDependence>(inspection_module, "PartialDependence")
        .def(py::init(partialdependence_init),
             py::arg("estimator"), py::arg("features"))
        .def("compute", &auroraml::inspection::PartialDependence::compute)
        .def("grid", &auroraml::inspection::PartialDependence::grid)
        .def("partial_dependence", &auroraml::inspection::PartialDependence::partial_dependence);
    
    // Additional ensemble methods
    auto baggingclassifier_init = [](py::object base_estimator_py, int n_estimators = 10, int max_samples = -1, int max_features = -1, int random_state = -1) {
        auroraml::Classifier* base_estimator_ptr = base_estimator_py.cast<auroraml::Classifier*>();
        std::shared_ptr<auroraml::Classifier> base_estimator(base_estimator_ptr, [](auroraml::Classifier*) {});
        return new auroraml::ensemble::BaggingClassifier(base_estimator, n_estimators, max_samples, max_features, random_state);
    };
    
    py::class_<auroraml::ensemble::BaggingClassifier, auroraml::Estimator, auroraml::Classifier>(rf_module, "BaggingClassifier")
        .def(py::init(baggingclassifier_init), py::arg("base_estimator"), py::arg("n_estimators") = 10, 
             py::arg("max_samples") = -1, py::arg("max_features") = -1, py::arg("random_state") = -1)
        .def("fit", &auroraml::ensemble::BaggingClassifier::fit)
        .def("predict", &auroraml::ensemble::BaggingClassifier::predict_classes)
        .def("predict_proba", &auroraml::ensemble::BaggingClassifier::predict_proba)
        .def("classes", &auroraml::ensemble::BaggingClassifier::classes);
    
    auto baggingregressor_init = [](py::object base_estimator_py, int n_estimators = 10, int max_samples = -1, int max_features = -1, int random_state = -1) {
        auroraml::Regressor* base_estimator_ptr = base_estimator_py.cast<auroraml::Regressor*>();
        std::shared_ptr<auroraml::Regressor> base_estimator(base_estimator_ptr, [](auroraml::Regressor*) {});
        return new auroraml::ensemble::BaggingRegressor(base_estimator, n_estimators, max_samples, max_features, random_state);
    };
    
    py::class_<auroraml::ensemble::BaggingRegressor, auroraml::Estimator, auroraml::Regressor>(rf_module, "BaggingRegressor")
        .def(py::init(baggingregressor_init), py::arg("base_estimator"), py::arg("n_estimators") = 10, 
             py::arg("max_samples") = -1, py::arg("max_features") = -1, py::arg("random_state") = -1)
        .def("fit", &auroraml::ensemble::BaggingRegressor::fit)
        .def("predict", &auroraml::ensemble::BaggingRegressor::predict);
    
    auto votingclassifier_init = [](py::list estimators_py, const std::string& voting = "hard") {
        std::vector<std::pair<std::string, std::shared_ptr<auroraml::Classifier>>> estimators;
        for (auto item : estimators_py) {
            py::tuple est_tuple = py::cast<py::tuple>(item);
            std::string name = py::cast<std::string>(est_tuple[0]);
            py::object estimator_obj = est_tuple[1];
            auroraml::Classifier* estimator_ptr = estimator_obj.cast<auroraml::Classifier*>();
            std::shared_ptr<auroraml::Classifier> estimator(estimator_ptr, [](auroraml::Classifier*) {});
            estimators.push_back({name, estimator});
        }
        return new auroraml::ensemble::VotingClassifier(estimators, voting);
    };
    
    py::class_<auroraml::ensemble::VotingClassifier, auroraml::Estimator, auroraml::Classifier>(rf_module, "VotingClassifier")
        .def(py::init(votingclassifier_init), py::arg("estimators"), py::arg("voting") = "hard")
        .def("fit", &auroraml::ensemble::VotingClassifier::fit)
        .def("predict", &auroraml::ensemble::VotingClassifier::predict_classes)
        .def("predict_proba", &auroraml::ensemble::VotingClassifier::predict_proba)
        .def("classes", &auroraml::ensemble::VotingClassifier::classes);
    
    auto votingregressor_init = [](py::list estimators_py) {
        std::vector<std::pair<std::string, std::shared_ptr<auroraml::Regressor>>> estimators;
        for (auto item : estimators_py) {
            py::tuple est_tuple = py::cast<py::tuple>(item);
            std::string name = py::cast<std::string>(est_tuple[0]);
            py::object estimator_obj = est_tuple[1];
            auroraml::Regressor* estimator_ptr = estimator_obj.cast<auroraml::Regressor*>();
            std::shared_ptr<auroraml::Regressor> estimator(estimator_ptr, [](auroraml::Regressor*) {});
            estimators.push_back({name, estimator});
        }
        return new auroraml::ensemble::VotingRegressor(estimators);
    };
    
    py::class_<auroraml::ensemble::VotingRegressor, auroraml::Estimator, auroraml::Regressor>(rf_module, "VotingRegressor")
        .def(py::init(votingregressor_init), py::arg("estimators"))
        .def("fit", &auroraml::ensemble::VotingRegressor::fit)
        .def("predict", &auroraml::ensemble::VotingRegressor::predict);
    
    auto stackingclassifier_init = [](py::list base_estimators_py, py::object meta_classifier_py) {
        std::vector<std::pair<std::string, std::shared_ptr<auroraml::Classifier>>> base_estimators;
        for (auto item : base_estimators_py) {
            py::tuple est_tuple = py::cast<py::tuple>(item);
            std::string name = py::cast<std::string>(est_tuple[0]);
            py::object estimator_obj = est_tuple[1];
            auroraml::Classifier* estimator_ptr = estimator_obj.cast<auroraml::Classifier*>();
            std::shared_ptr<auroraml::Classifier> estimator(estimator_ptr, [](auroraml::Classifier*) {});
            base_estimators.push_back({name, estimator});
        }
        auroraml::Classifier* meta_classifier_ptr = meta_classifier_py.cast<auroraml::Classifier*>();
        std::shared_ptr<auroraml::Classifier> meta_classifier(meta_classifier_ptr, [](auroraml::Classifier*) {});
        return new auroraml::ensemble::StackingClassifier(base_estimators, meta_classifier);
    };
    
    py::class_<auroraml::ensemble::StackingClassifier, auroraml::Estimator, auroraml::Classifier>(rf_module, "StackingClassifier")
        .def(py::init(stackingclassifier_init), py::arg("base_estimators"), py::arg("meta_classifier"))
        .def("fit", &auroraml::ensemble::StackingClassifier::fit)
        .def("predict", &auroraml::ensemble::StackingClassifier::predict_classes)
        .def("predict_proba", &auroraml::ensemble::StackingClassifier::predict_proba)
        .def("classes", &auroraml::ensemble::StackingClassifier::classes);
    
    auto stackingregressor_init = [](py::list base_estimators_py, py::object meta_regressor_py) {
        std::vector<std::pair<std::string, std::shared_ptr<auroraml::Regressor>>> base_estimators;
        for (auto item : base_estimators_py) {
            py::tuple est_tuple = py::cast<py::tuple>(item);
            std::string name = py::cast<std::string>(est_tuple[0]);
            py::object estimator_obj = est_tuple[1];
            auroraml::Regressor* estimator_ptr = estimator_obj.cast<auroraml::Regressor*>();
            std::shared_ptr<auroraml::Regressor> estimator(estimator_ptr, [](auroraml::Regressor*) {});
            base_estimators.push_back({name, estimator});
        }
        auroraml::Regressor* meta_regressor_ptr = meta_regressor_py.cast<auroraml::Regressor*>();
        std::shared_ptr<auroraml::Regressor> meta_regressor(meta_regressor_ptr, [](auroraml::Regressor*) {});
        return new auroraml::ensemble::StackingRegressor(base_estimators, meta_regressor);
    };
    
    py::class_<auroraml::ensemble::StackingRegressor, auroraml::Estimator, auroraml::Regressor>(rf_module, "StackingRegressor")
        .def(py::init(stackingregressor_init), py::arg("base_estimators"), py::arg("meta_regressor"))
        .def("fit", &auroraml::ensemble::StackingRegressor::fit)
        .def("predict", &auroraml::ensemble::StackingRegressor::predict);
    
    // Calibration module
    py::module_ calibration_module = m.def_submodule("calibration", "Probability calibration");
    
    auto calibratedclassifiercv_init = [](py::object base_estimator_py, const std::string& method = "sigmoid", int cv = 3) {
        auroraml::Classifier* base_estimator_ptr = base_estimator_py.cast<auroraml::Classifier*>();
        std::shared_ptr<auroraml::Classifier> base_estimator(base_estimator_ptr, [](auroraml::Classifier*) {});
        return new auroraml::calibration::CalibratedClassifierCV(base_estimator, method, cv);
    };
    
    py::class_<auroraml::calibration::CalibratedClassifierCV, auroraml::Estimator, auroraml::Classifier>(calibration_module, "CalibratedClassifierCV")
        .def(py::init(calibratedclassifiercv_init), py::arg("base_estimator"), py::arg("method") = "sigmoid", py::arg("cv") = 3)
        .def("fit", &auroraml::calibration::CalibratedClassifierCV::fit)
        .def("predict", &auroraml::calibration::CalibratedClassifierCV::predict_classes)
        .def("predict_proba", &auroraml::calibration::CalibratedClassifierCV::predict_proba)
        .def("classes", &auroraml::calibration::CalibratedClassifierCV::classes);
    
    // Isotonic module
    py::module_ isotonic_module = m.def_submodule("isotonic", "Isotonic regression");
    
    py::class_<auroraml::isotonic::IsotonicRegression, auroraml::Estimator, auroraml::Regressor>(isotonic_module, "IsotonicRegression")
        .def(py::init<bool>(), py::arg("increasing") = true)
        .def("fit", &auroraml::isotonic::IsotonicRegression::fit)
        .def("predict", &auroraml::isotonic::IsotonicRegression::predict)
        .def("transform", &auroraml::isotonic::IsotonicRegression::transform);
    
    // Discriminant analysis module
    py::module_ discriminant_module = m.def_submodule("discriminant_analysis", "Discriminant analysis");
    
    py::class_<auroraml::discriminant_analysis::QuadraticDiscriminantAnalysis, auroraml::Estimator, auroraml::Classifier>(discriminant_module, "QuadraticDiscriminantAnalysis")
        .def(py::init<double>(), py::arg("regularization") = 0.0)
        .def("fit", &auroraml::discriminant_analysis::QuadraticDiscriminantAnalysis::fit)
        .def("predict", &auroraml::discriminant_analysis::QuadraticDiscriminantAnalysis::predict_classes)
        .def("predict_proba", &auroraml::discriminant_analysis::QuadraticDiscriminantAnalysis::predict_proba)
        .def("classes", &auroraml::discriminant_analysis::QuadraticDiscriminantAnalysis::classes);
    
    // Additional Naive Bayes variants
    py::class_<auroraml::naive_bayes::MultinomialNB, auroraml::Estimator, auroraml::Classifier>(nb_module, "MultinomialNB")
        .def(py::init<double, bool>(), py::arg("alpha") = 1.0, py::arg("fit_prior") = true)
        .def("fit", &auroraml::naive_bayes::MultinomialNB::fit)
        .def("predict", &auroraml::naive_bayes::MultinomialNB::predict_classes)
        .def("predict_proba", &auroraml::naive_bayes::MultinomialNB::predict_proba)
        .def("classes", &auroraml::naive_bayes::MultinomialNB::classes);
    
    py::class_<auroraml::naive_bayes::BernoulliNB, auroraml::Estimator, auroraml::Classifier>(nb_module, "BernoulliNB")
        .def(py::init<double, double, bool>(), py::arg("alpha") = 1.0, py::arg("binarize") = 0.0, py::arg("fit_prior") = true)
        .def("fit", &auroraml::naive_bayes::BernoulliNB::fit)
        .def("predict", &auroraml::naive_bayes::BernoulliNB::predict_classes)
        .def("predict_proba", &auroraml::naive_bayes::BernoulliNB::predict_proba)
        .def("classes", &auroraml::naive_bayes::BernoulliNB::classes);
    
    py::class_<auroraml::naive_bayes::ComplementNB, auroraml::Estimator, auroraml::Classifier>(nb_module, "ComplementNB")
        .def(py::init<double, bool>(), py::arg("alpha") = 1.0, py::arg("fit_prior") = true)
        .def("fit", &auroraml::naive_bayes::ComplementNB::fit)
        .def("predict", &auroraml::naive_bayes::ComplementNB::predict_classes)
        .def("predict_proba", &auroraml::naive_bayes::ComplementNB::predict_proba)
        .def("classes", &auroraml::naive_bayes::ComplementNB::classes);
    
    // ExtraTree variants
    py::class_<auroraml::tree::ExtraTreeClassifier, auroraml::Estimator, auroraml::Classifier>(tree_module, "ExtraTreeClassifier")
        .def(py::init<int, int, int, int, int>(), 
             py::arg("max_depth") = -1, py::arg("min_samples_split") = 2, 
             py::arg("min_samples_leaf") = 1, py::arg("max_features") = -1, py::arg("random_state") = -1)
        .def("fit", &auroraml::tree::ExtraTreeClassifier::fit)
        .def("predict", &auroraml::tree::ExtraTreeClassifier::predict_classes)
        .def("predict_proba", &auroraml::tree::ExtraTreeClassifier::predict_proba)
        .def("classes", &auroraml::tree::ExtraTreeClassifier::classes);
    
    py::class_<auroraml::tree::ExtraTreeRegressor, auroraml::Estimator, auroraml::Regressor>(tree_module, "ExtraTreeRegressor")
        .def(py::init<int, int, int, int, int>(), 
             py::arg("max_depth") = -1, py::arg("min_samples_split") = 2, 
             py::arg("min_samples_leaf") = 1, py::arg("max_features") = -1, py::arg("random_state") = -1)
        .def("fit", &auroraml::tree::ExtraTreeRegressor::fit)
        .def("predict", &auroraml::tree::ExtraTreeRegressor::predict);
    
    // Outlier detection module
    py::module_ outlier_module = m.def_submodule("outlier_detection", "Outlier detection");
    
    // Helper lambda for optional y parameter in IsolationForest
    auto isolation_forest_fit = [](auroraml::outlier_detection::IsolationForest& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) -> auroraml::Estimator& {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        return self.fit(X, y);
    };
    
    py::class_<auroraml::outlier_detection::IsolationForest, auroraml::Estimator>(outlier_module, "IsolationForest")
        .def(py::init<int, int, double, int>(), 
             py::arg("n_estimators") = 100, py::arg("max_samples") = -1, 
             py::arg("contamination") = 0.1, py::arg("random_state") = -1)
        .def("fit", isolation_forest_fit, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("predict", &auroraml::outlier_detection::IsolationForest::predict)
        .def("decision_function", &auroraml::outlier_detection::IsolationForest::decision_function)
        .def("fit_predict", &auroraml::outlier_detection::IsolationForest::fit_predict);
    
    // Helper lambda for optional y parameter in LocalOutlierFactor
    auto lof_fit = [](auroraml::outlier_detection::LocalOutlierFactor& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) -> auroraml::Estimator& {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        return self.fit(X, y);
    };
    
    py::class_<auroraml::outlier_detection::LocalOutlierFactor, auroraml::Estimator>(outlier_module, "LocalOutlierFactor")
        .def(py::init<int, const std::string&, double>(), 
             py::arg("n_neighbors") = 20, py::arg("metric") = "euclidean", py::arg("contamination") = 0.1)
        .def("fit", lof_fit, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("predict", &auroraml::outlier_detection::LocalOutlierFactor::predict)
        .def("decision_function", &auroraml::outlier_detection::LocalOutlierFactor::decision_function)
        .def("fit_predict", &auroraml::outlier_detection::LocalOutlierFactor::fit_predict);
    
    // Mixture module
    py::module_ mixture_module = m.def_submodule("mixture", "Mixture models");
    
    // Helper lambda for optional y parameter in GaussianMixture
    auto gaussian_mixture_fit = [](auroraml::mixture::GaussianMixture& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) -> auroraml::mixture::GaussianMixture& {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    
    py::class_<auroraml::mixture::GaussianMixture, auroraml::Estimator>(mixture_module, "GaussianMixture")
        .def(py::init<int, int, double, int>(), 
             py::arg("n_components") = 1, py::arg("max_iter") = 100, 
             py::arg("tol") = 1e-3, py::arg("random_state") = -1)
        .def("fit", gaussian_mixture_fit, py::arg("X"), py::arg("y") = py::none())
        .def("predict", &auroraml::mixture::GaussianMixture::predict)
        .def("predict_proba", &auroraml::mixture::GaussianMixture::predict_proba)
        .def("score_samples", &auroraml::mixture::GaussianMixture::score_samples)
        .def("means", &auroraml::mixture::GaussianMixture::means)
        .def("covariances", &auroraml::mixture::GaussianMixture::covariances)
        .def("weights", &auroraml::mixture::GaussianMixture::weights);
    
    // Semi-supervised module
    py::module_ semi_supervised_module = m.def_submodule("semi_supervised", "Semi-supervised learning");
    
    py::class_<auroraml::semi_supervised::LabelPropagation, auroraml::Estimator, auroraml::Classifier>(semi_supervised_module, "LabelPropagation")
        .def(py::init<double, int, double, const std::string&>(), 
             py::arg("gamma") = 20.0, py::arg("max_iter") = 30, 
             py::arg("tol") = 1e-3, py::arg("kernel") = "rbf")
        .def("fit", &auroraml::semi_supervised::LabelPropagation::fit)
        .def("predict", &auroraml::semi_supervised::LabelPropagation::predict_classes)
        .def("predict_proba", &auroraml::semi_supervised::LabelPropagation::predict_proba)
        .def("classes", &auroraml::semi_supervised::LabelPropagation::classes);
    
    py::class_<auroraml::semi_supervised::LabelSpreading, auroraml::Estimator, auroraml::Classifier>(semi_supervised_module, "LabelSpreading")
        .def(py::init<double, double, int, double, const std::string&>(), 
             py::arg("alpha") = 0.2, py::arg("gamma") = 20.0, py::arg("max_iter") = 30, 
             py::arg("tol") = 1e-3, py::arg("kernel") = "rbf")
        .def("fit", &auroraml::semi_supervised::LabelSpreading::fit)
        .def("predict", &auroraml::semi_supervised::LabelSpreading::predict_classes)
        .def("predict_proba", &auroraml::semi_supervised::LabelSpreading::predict_proba)
        .def("classes", &auroraml::semi_supervised::LabelSpreading::classes);
    
    // Additional preprocessing utilities
    // Helper lambda for optional y parameter in MaxAbsScaler
    auto maxabs_scaler_fit = [](auroraml::preprocessing::MaxAbsScaler& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) -> auroraml::preprocessing::MaxAbsScaler& {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto maxabs_scaler_fit_transform = [](auroraml::preprocessing::MaxAbsScaler& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        return self.fit_transform(X, y);
    };
    
    py::class_<auroraml::preprocessing::MaxAbsScaler, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "MaxAbsScaler")
        .def(py::init<>())
        .def("fit", maxabs_scaler_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &auroraml::preprocessing::MaxAbsScaler::transform)
        .def("inverse_transform", &auroraml::preprocessing::MaxAbsScaler::inverse_transform)
        .def("fit_transform", maxabs_scaler_fit_transform, py::arg("X"), py::arg("y") = py::none())
        .def("max_abs", &auroraml::preprocessing::MaxAbsScaler::max_abs);
    
    // Helper lambda for optional y parameter in Binarizer
    auto binarizer_fit = [](auroraml::preprocessing::Binarizer& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) -> auroraml::preprocessing::Binarizer& {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        self.fit(X, y);
        return self;
    };
    auto binarizer_fit_transform = [](auroraml::preprocessing::Binarizer& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        return self.fit_transform(X, y);
    };
    
    py::class_<auroraml::preprocessing::Binarizer, auroraml::Estimator, auroraml::Transformer>(preprocessing_module, "Binarizer")
        .def(py::init<double>(), py::arg("threshold") = 0.0)
        .def("fit", binarizer_fit, py::arg("X"), py::arg("y") = py::none())
        .def("transform", &auroraml::preprocessing::Binarizer::transform)
        .def("fit_transform", binarizer_fit_transform, py::arg("X"), py::arg("y") = py::none());
    
    // Additional clustering methods
    py::class_<auroraml::cluster::SpectralClustering, auroraml::Estimator>(cluster_module, "SpectralClustering")
        .def(py::init<int, const std::string&, double, int, int>(), 
             py::arg("n_clusters") = 8, py::arg("affinity") = "rbf", 
             py::arg("gamma") = 1.0, py::arg("n_neighbors") = 10, py::arg("random_state") = -1)
        .def("fit", &auroraml::cluster::SpectralClustering::fit)
        .def("fit_predict", &auroraml::cluster::SpectralClustering::fit_predict)
        .def("labels", &auroraml::cluster::SpectralClustering::labels);
    
    // Helper lambda for optional y parameter in MiniBatchKMeans
    auto minibatch_kmeans_fit = [](auroraml::cluster::MiniBatchKMeans& self, const auroraml::MatrixXd& X, py::object y_py = py::none()) -> auroraml::Estimator& {
        auroraml::VectorXd y = y_py.is_none() ? auroraml::VectorXd::Zero(X.rows()) : py::cast<auroraml::VectorXd>(y_py);
        return self.fit(X, y);
    };
    
    py::class_<auroraml::cluster::MiniBatchKMeans, auroraml::Estimator>(cluster_module, "MiniBatchKMeans")
        .def(py::init<int, int, double, int, int>(), 
             py::arg("n_clusters") = 8, py::arg("max_iter") = 100, 
             py::arg("tol") = 1e-4, py::arg("batch_size") = 100, py::arg("random_state") = -1)
        .def("fit", minibatch_kmeans_fit, py::arg("X"), py::arg("y") = py::none(), py::return_value_policy::reference)
        .def("fit_predict", &auroraml::cluster::MiniBatchKMeans::fit_predict)
        .def("predict", &auroraml::cluster::MiniBatchKMeans::predict)
        .def("cluster_centers", &auroraml::cluster::MiniBatchKMeans::cluster_centers)
        .def("labels", &auroraml::cluster::MiniBatchKMeans::labels);
}
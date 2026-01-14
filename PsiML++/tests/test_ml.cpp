#include "../include/ml/model.h"
#include "../include/ml/dataset.h"
#include "../include/ml/metrics.h"
#include "../include/ml/preprocessing/scalar.h"
#include "../include/ml/preprocessing/normalizer.h"
#include "../include/ml/preprocessing/encoder.h"
#include "../include/ml/optimizers/gradient_descent.h"
#include "../include/ml/optimizers/momentum.h"
#include "../include/ml/optimizers/sgd.h"
#include "../include/ml/algorithms/linear_regression.h"
#include "../include/ml/algorithms/logistic_regression.h"
#include "../include/ml/algorithms/kmeans.h"
#include "../include/ml/algorithms/pca.h"
#include "../include/ml/algorithms/svm.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::math;
using namespace psi::ml;
using namespace psi::core;

// Helper function to check floating point equality
template<typename T>
bool approx_equal(T a, T b, T epsilon = T{1e-5}) {
    return std::abs(a - b) < epsilon;
}

// =============================================================================
// Dataset Tests
// =============================================================================

void test_train_test_split() {
    std::cout << "Testing train_test_split..." << std::endl;

    Matrix<float> X = {
        {1.0f, 2.0f},
        {3.0f, 4.0f},
        {5.0f, 6.0f},
        {7.0f, 8.0f},
        {9.0f, 10.0f}
    };
    Vector<float> y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto split = Dataset<float>::train_test_split(X, y, 0.4f, false, 42);

    assert(split.X_train.rows() == 3);
    assert(split.X_test.rows() == 2);
    assert(split.y_train.size() == 3);
    assert(split.y_test.size() == 2);

    std::cout << "  train_test_split: PASSED" << std::endl;
}

void test_add_bias() {
    std::cout << "Testing add_bias..." << std::endl;

    Matrix<float> X = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    Matrix<float> X_bias = Dataset<float>::add_bias(X);

    assert(X_bias.rows() == 2);
    assert(X_bias.cols() == 3);
    assert(X_bias(0, 0) == 1.0f);
    assert(X_bias(1, 0) == 1.0f);
    assert(X_bias(0, 1) == 1.0f);
    assert(X_bias(0, 2) == 2.0f);

    std::cout << "  add_bias: PASSED" << std::endl;
}

// =============================================================================
// Metrics Tests
// =============================================================================

void test_mse() {
    std::cout << "Testing mean_squared_error..." << std::endl;

    Vector<float> y_true = {1.0f, 2.0f, 3.0f, 4.0f};
    Vector<float> y_pred = {1.0f, 2.0f, 3.0f, 4.0f};

    float mse = metrics::mean_squared_error(y_true, y_pred);
    assert(approx_equal(mse, 0.0f));

    Vector<float> y_pred2 = {2.0f, 3.0f, 4.0f, 5.0f};
    float mse2 = metrics::mean_squared_error(y_true, y_pred2);
    assert(approx_equal(mse2, 1.0f));

    std::cout << "  mean_squared_error: PASSED" << std::endl;
}

void test_r2_score() {
    std::cout << "Testing r2_score..." << std::endl;

    Vector<float> y_true = {1.0f, 2.0f, 3.0f, 4.0f};
    Vector<float> y_pred = {1.0f, 2.0f, 3.0f, 4.0f};

    float r2 = metrics::r2_score(y_true, y_pred);
    assert(approx_equal(r2, 1.0f));

    std::cout << "  r2_score: PASSED" << std::endl;
}

void test_accuracy() {
    std::cout << "Testing accuracy..." << std::endl;

    Vector<float> y_true = {0.0f, 1.0f, 1.0f, 0.0f};
    Vector<float> y_pred = {0.0f, 1.0f, 1.0f, 0.0f};

    float acc = metrics::accuracy(y_true, y_pred);
    assert(approx_equal(acc, 1.0f));

    Vector<float> y_pred2 = {0.0f, 0.0f, 1.0f, 0.0f};
    float acc2 = metrics::accuracy(y_true, y_pred2);
    assert(approx_equal(acc2, 0.75f));

    std::cout << "  accuracy: PASSED" << std::endl;
}

void test_precision_recall() {
    std::cout << "Testing precision and recall..." << std::endl;

    Vector<float> y_true = {1.0f, 1.0f, 0.0f, 0.0f};
    Vector<float> y_pred = {1.0f, 0.0f, 0.0f, 0.0f};

    float prec = metrics::precision(y_true, y_pred);
    float rec = metrics::recall(y_true, y_pred);

    assert(approx_equal(prec, 1.0f));
    assert(approx_equal(rec, 0.5f));

    std::cout << "  precision and recall: PASSED" << std::endl;
}

// =============================================================================
// Preprocessing Tests
// =============================================================================

void test_standard_scaler() {
    std::cout << "Testing StandardScaler..." << std::endl;

    Matrix<float> X = {
        {1.0f, 2.0f},
        {3.0f, 4.0f},
        {5.0f, 6.0f}
    };

    preprocessing::StandardScaler<float> scaler;
    scaler.fit(X);

    assert(scaler.is_fitted());
    assert(approx_equal(scaler.mean()[0], 3.0f));
    assert(approx_equal(scaler.mean()[1], 4.0f));

    Matrix<float> X_scaled = scaler.transform(X);
    assert(approx_equal(X_scaled(1, 0), 0.0f, 1e-4f));
    assert(approx_equal(X_scaled(1, 1), 0.0f, 1e-4f));

    Matrix<float> X_inv = scaler.inverse_transform(X_scaled);
    assert(approx_equal(X_inv(0, 0), X(0, 0), 1e-4f));

    std::cout << "  StandardScaler: PASSED" << std::endl;
}

void test_minmax_scaler() {
    std::cout << "Testing MinMaxScaler..." << std::endl;

    Matrix<float> X = {
        {1.0f, 10.0f},
        {2.0f, 20.0f},
        {3.0f, 30.0f}
    };

    preprocessing::MinMaxScaler<float> scaler;
    Matrix<float> X_scaled = scaler.fit_transform(X);

    assert(scaler.is_fitted());
    assert(approx_equal(X_scaled(0, 0), 0.0f));
    assert(approx_equal(X_scaled(2, 0), 1.0f));
    assert(approx_equal(X_scaled(0, 1), 0.0f));
    assert(approx_equal(X_scaled(2, 1), 1.0f));

    std::cout << "  MinMaxScaler: PASSED" << std::endl;
}

void test_normalizer() {
    std::cout << "Testing Normalizer..." << std::endl;

    Matrix<float> X = {
        {3.0f, 4.0f},
        {1.0f, 0.0f}
    };

    preprocessing::Normalizer<float> normalizer(preprocessing::NormType::L2);
    Matrix<float> X_norm = normalizer.transform(X);

    assert(approx_equal(X_norm(0, 0), 0.6f, 1e-4f));
    assert(approx_equal(X_norm(0, 1), 0.8f, 1e-4f));
    assert(approx_equal(X_norm(1, 0), 1.0f));

    std::cout << "  Normalizer: PASSED" << std::endl;
}

void test_label_encoder() {
    std::cout << "Testing LabelEncoder..." << std::endl;

    Vector<float> y = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f};

    preprocessing::LabelEncoder<float> encoder;
    auto y_encoded = encoder.fit_transform(y);

    assert(encoder.is_fitted());
    assert(encoder.n_classes() == 3);
    assert(y_encoded[0] == 0);
    assert(y_encoded[1] == 1);
    assert(y_encoded[2] == 2);

    std::cout << "  LabelEncoder: PASSED" << std::endl;
}

// =============================================================================
// Optimizer Tests
// =============================================================================

void test_gradient_descent() {
    std::cout << "Testing GradientDescent optimizer..." << std::endl;

    optimizers::GradientDescent<float> optimizer(0.1f);

    Vector<float> weights = {1.0f, 2.0f};
    Vector<float> gradients = {0.5f, 0.5f};

    optimizer.update(weights, gradients);

    assert(approx_equal(weights[0], 0.95f));
    assert(approx_equal(weights[1], 1.95f));

    std::cout << "  GradientDescent optimizer: PASSED" << std::endl;
}

void test_momentum_optimizer() {
    std::cout << "Testing Momentum optimizer..." << std::endl;

    optimizers::Momentum<float> optimizer(0.1f, 0.9f);

    Vector<float> weights = {1.0f, 2.0f};
    Vector<float> gradients = {0.5f, 0.5f};

    optimizer.update(weights, gradients);

    assert(weights[0] < 1.0f);
    assert(weights[1] < 2.0f);

    std::cout << "  Momentum optimizer: PASSED" << std::endl;
}

void test_adam_optimizer() {
    std::cout << "Testing Adam optimizer..." << std::endl;

    optimizers::Adam<float> optimizer(0.001f);

    Vector<float> weights = {1.0f, 2.0f};
    Vector<float> gradients = {0.5f, 0.5f};

    optimizer.update(weights, gradients);

    assert(weights[0] < 1.0f);
    assert(weights[1] < 2.0f);

    std::cout << "  Adam optimizer: PASSED" << std::endl;
}

// =============================================================================
// Algorithm Tests
// =============================================================================

void test_linear_regression() {
    std::cout << "Testing LinearRegression..." << std::endl;

    // Simple linear data: y = 2*x + 1
    Matrix<float> X = {
        {1.0f}, {2.0f}, {3.0f}, {4.0f}, {5.0f}
    };
    Vector<float> y = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};

    algorithms::LinearRegression<float> model;
    model.fit(X, y);

    assert(model.is_trained());

    Vector<float> predictions = model.predict(X);
    float r2 = model.score(X, y);

    assert(r2 > 0.99f);

    std::cout << "  LinearRegression: PASSED" << std::endl;
}

void test_ridge_regression() {
    std::cout << "Testing RidgeRegression..." << std::endl;

    Matrix<float> X = {
        {1.0f}, {2.0f}, {3.0f}, {4.0f}, {5.0f}
    };
    Vector<float> y = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};

    algorithms::RidgeRegression<float> model(0.1f);
    model.fit(X, y);

    assert(model.is_trained());

    float r2 = model.score(X, y);
    assert(r2 > 0.95f);

    std::cout << "  RidgeRegression: PASSED" << std::endl;
}

void test_logistic_regression() {
    std::cout << "Testing LogisticRegression..." << std::endl;

    // Simple binary classification
    Matrix<float> X = {
        {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f},
        {6.0f, 6.0f}, {7.0f, 7.0f}, {8.0f, 8.0f}
    };
    Vector<float> y = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

    algorithms::LogisticRegression<float> model(0.5f, 1000, 1e-6f, 0.0f, true);
    model.fit(X, y);

    assert(model.is_trained());

    Vector<float> predictions = model.predict(X);
    float acc = model.score(X, y);

    assert(acc >= 0.5f);

    std::cout << "  LogisticRegression: PASSED" << std::endl;
}

void test_kmeans() {
    std::cout << "Testing KMeans..." << std::endl;

    // Two clear clusters
    Matrix<float> X = {
        {1.0f, 1.0f}, {1.5f, 1.5f}, {2.0f, 2.0f},
        {8.0f, 8.0f}, {8.5f, 8.5f}, {9.0f, 9.0f}
    };

    algorithms::KMeans<float> model(2, 100, 1e-4f, 3, 42);
    model.fit(X);

    assert(model.is_fitted());
    assert(model.n_clusters() == 2);

    auto labels = model.labels();
    assert(labels.size() == 6);

    // First three should be in one cluster, last three in another
    assert(labels[0] == labels[1] && labels[1] == labels[2]);
    assert(labels[3] == labels[4] && labels[4] == labels[5]);
    assert(labels[0] != labels[3]);

    std::cout << "  KMeans: PASSED" << std::endl;
}

void test_pca() {
    std::cout << "Testing PCA..." << std::endl;

    Matrix<float> X = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f},
        {10.0f, 11.0f, 12.0f}
    };

    algorithms::PCA<float> pca(2);
    pca.fit(X);

    assert(pca.is_fitted());

    Matrix<float> X_transformed = pca.transform(X);
    assert(X_transformed.cols() == 2);
    assert(X_transformed.rows() == 4);

    Matrix<float> X_reconstructed = pca.inverse_transform(X_transformed);
    assert(X_reconstructed.cols() == 3);
    assert(X_reconstructed.rows() == 4);

    std::cout << "  PCA: PASSED" << std::endl;
}

void test_linear_svm() {
    std::cout << "Testing LinearSVM..." << std::endl;

    // Simple linearly separable data
    Matrix<float> X = {
        {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 1.0f},
        {6.0f, 6.0f}, {7.0f, 7.0f}, {6.0f, 8.0f}
    };
    Vector<float> y = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

    algorithms::LinearSVM<float> model(1.0f, 0.01f, 1000);
    model.fit(X, y);

    assert(model.is_trained());

    float acc = model.score(X, y);
    assert(acc >= 0.5f);

    std::cout << "  LinearSVM: PASSED" << std::endl;
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::cout << "\n=== ML Module Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        // Dataset tests
        std::cout << "--- Dataset Tests ---" << std::endl;
        test_train_test_split();
        test_add_bias();
        std::cout << std::endl;

        // Metrics tests
        std::cout << "--- Metrics Tests ---" << std::endl;
        test_mse();
        test_r2_score();
        test_accuracy();
        test_precision_recall();
        std::cout << std::endl;

        // Preprocessing tests
        std::cout << "--- Preprocessing Tests ---" << std::endl;
        test_standard_scaler();
        test_minmax_scaler();
        test_normalizer();
        test_label_encoder();
        std::cout << std::endl;

        // Optimizer tests
        std::cout << "--- Optimizer Tests ---" << std::endl;
        test_gradient_descent();
        test_momentum_optimizer();
        test_adam_optimizer();
        std::cout << std::endl;

        // Algorithm tests
        std::cout << "--- Algorithm Tests ---" << std::endl;
        test_linear_regression();
        test_ridge_regression();
        test_logistic_regression();
        test_kmeans();
        test_pca();
        test_linear_svm();
        std::cout << std::endl;

        std::cout << "=== All ML Module Tests PASSED ===" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

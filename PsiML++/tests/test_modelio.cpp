#include "../include/utils/model_io.h"
#include "../include/utils/data_loader.h"
#include "../include/ml/algorithms/linear_regression.h"
#include "../include/ml/preprocessing/scalar.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::utils;
using namespace psi::math;
using namespace psi::ml;
using namespace psi::core;

template<typename T>
bool approx_equal(T a, T b, T epsilon = T{1e-5}) {
    return std::abs(a - b) < epsilon;
}

// =============================================================================
// Test: Save and Load Weights (Text Format)
// =============================================================================

void test_weights_text_io() {
    std::cout << "Testing weights text I/O..." << std::endl;

    // Create test weights
    Vector<float> weights(5);
    weights[0] = 1.5f;
    weights[1] = -2.3f;
    weights[2] = 0.0f;
    weights[3] = 4.567f;
    weights[4] = -0.001f;

    // Create metadata
    ModelMetadata meta;
    meta.model_type = "LinearRegression";
    meta.version = "1.0";
    meta.params["learning_rate"] = "0.01";
    meta.params["iterations"] = "1000";

    // Save
    ModelIO<float>::save_weights_text("temp_weights.txt", weights, meta);

    // Load
    ModelMetadata loaded_meta;
    Vector<float> loaded = ModelIO<float>::load_weights_text("temp_weights.txt", &loaded_meta);

    // Verify
    assert(loaded.size() == 5);
    assert(approx_equal(loaded[0], 1.5f));
    assert(approx_equal(loaded[1], -2.3f));
    assert(approx_equal(loaded[3], 4.567f));
    assert(loaded_meta.model_type == "LinearRegression");

    // Clean up
    std::remove("temp_weights.txt");

    std::cout << "  weights text I/O: PASSED" << std::endl;
}

// =============================================================================
// Test: Save and Load Weights (Binary Format)
// =============================================================================

void test_weights_binary_io() {
    std::cout << "Testing weights binary I/O..." << std::endl;

    // Create test weights
    Vector<float> weights(1000);
    for (usize i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>(i) * 0.01f - 5.0f;
    }

    ModelMetadata meta;
    meta.model_type = "NeuralNetwork";
    meta.version = "2.0";
    meta.params["layers"] = "3";
    meta.params["activation"] = "relu";

    // Save
    ModelIO<float>::save_weights_binary("temp_weights.bin", weights, meta);

    // Load
    ModelMetadata loaded_meta;
    Vector<float> loaded = ModelIO<float>::load_weights_binary("temp_weights.bin", &loaded_meta);

    // Verify
    assert(loaded.size() == 1000);
    for (usize i = 0; i < 10; ++i) {
        assert(approx_equal(loaded[i], weights[i]));
    }
    assert(loaded_meta.model_type == "NeuralNetwork");
    assert(loaded_meta.params["layers"] == "3");

    // Clean up
    std::remove("temp_weights.bin");

    std::cout << "  weights binary I/O: PASSED" << std::endl;
}

// =============================================================================
// Test: Save and Load Matrix (Text Format)
// =============================================================================

void test_matrix_text_io() {
    std::cout << "Testing matrix text I/O..." << std::endl;

    Matrix<float> original(3, 4);
    for (usize i = 0; i < original.rows(); ++i) {
        for (usize j = 0; j < original.cols(); ++j) {
            original(i, j) = static_cast<float>(i * 10 + j) + 0.5f;
        }
    }

    // Save
    ModelIO<float>::save_matrix_text("temp_matrix.txt", original, "coefficients");

    // Load
    Matrix<float> loaded = ModelIO<float>::load_matrix_text("temp_matrix.txt");

    // Verify
    assert(loaded.rows() == 3);
    assert(loaded.cols() == 4);
    for (usize i = 0; i < loaded.rows(); ++i) {
        for (usize j = 0; j < loaded.cols(); ++j) {
            assert(approx_equal(loaded(i, j), original(i, j)));
        }
    }

    // Clean up
    std::remove("temp_matrix.txt");

    std::cout << "  matrix text I/O: PASSED" << std::endl;
}

// =============================================================================
// Test: Save and Load Matrix (Binary Format)
// =============================================================================

void test_matrix_binary_io() {
    std::cout << "Testing matrix binary I/O..." << std::endl;

    Matrix<double> original(100, 50);
    for (usize i = 0; i < original.rows(); ++i) {
        for (usize j = 0; j < original.cols(); ++j) {
            original(i, j) = static_cast<double>(i * 1000 + j) * 0.001;
        }
    }

    // Save
    ModelIO<double>::save_matrix_binary("temp_matrix.bin", original);

    // Load
    Matrix<double> loaded = ModelIO<double>::load_matrix_binary("temp_matrix.bin");

    // Verify
    assert(loaded.rows() == 100);
    assert(loaded.cols() == 50);
    for (usize i = 0; i < 10; ++i) {
        for (usize j = 0; j < 10; ++j) {
            assert(approx_equal(loaded(i, j), original(i, j), 1e-10));
        }
    }

    // Clean up
    std::remove("temp_matrix.bin");

    std::cout << "  matrix binary I/O: PASSED" << std::endl;
}

// =============================================================================
// Test: Complete Model Save/Load
// =============================================================================

void test_complete_model_io() {
    std::cout << "Testing complete model I/O..." << std::endl;

    // Create model components
    Vector<float> weights(10);
    Vector<float> scaler_mean(3);
    Vector<float> scaler_std(3);

    for (usize i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>(i) * 0.1f;
    }
    scaler_mean[0] = 100.0f; scaler_mean[1] = 50.0f; scaler_mean[2] = 25.0f;
    scaler_std[0] = 10.0f; scaler_std[1] = 5.0f; scaler_std[2] = 2.5f;

    ModelMetadata meta;
    meta.model_type = "LinearRegression";
    meta.version = "1.0";
    meta.params["fit_intercept"] = "true";
    meta.params["n_features"] = "3";

    // Save
    ModelIO<float>::save_model("temp_model.bin", weights, scaler_mean, scaler_std, meta);

    // Load
    Vector<float> loaded_weights;
    Vector<float> loaded_mean;
    Vector<float> loaded_std;
    ModelMetadata loaded_meta;

    ModelIO<float>::load_model("temp_model.bin", loaded_weights, loaded_mean, loaded_std, loaded_meta);

    // Verify weights
    assert(loaded_weights.size() == 10);
    for (usize i = 0; i < loaded_weights.size(); ++i) {
        assert(approx_equal(loaded_weights[i], weights[i]));
    }

    // Verify scaler parameters
    assert(loaded_mean.size() == 3);
    assert(loaded_std.size() == 3);
    assert(approx_equal(loaded_mean[0], 100.0f));
    assert(approx_equal(loaded_std[2], 2.5f));

    // Verify metadata
    assert(loaded_meta.model_type == "LinearRegression");
    assert(loaded_meta.params["fit_intercept"] == "true");
    assert(loaded_meta.params["n_features"] == "3");

    // Clean up
    std::remove("temp_model.bin");

    std::cout << "  complete model I/O: PASSED" << std::endl;
}

// =============================================================================
// Test: Full Pipeline - Train, Save, Load, Predict
// =============================================================================

void test_train_save_load_predict() {
    std::cout << "Testing train-save-load-predict pipeline..." << std::endl;

    // Generate synthetic data
    auto data = DataLoader<float>::make_regression(100, 3, 0.05f, 42);

    // Split data
    auto split = Dataset<float>::train_test_split(data.X, data.y, 0.2f, true, 42);

    // Normalize
    preprocessing::StandardScaler<float> scaler;
    Matrix<float> X_train_scaled = scaler.fit_transform(split.X_train);
    Matrix<float> X_test_scaled = scaler.transform(split.X_test);

    // Train model
    algorithms::LinearRegression<float> model;
    model.fit(X_train_scaled, split.y_train);

    // Get predictions before saving
    Vector<float> predictions_before = model.predict(X_test_scaled);
    float score_before = model.score(X_test_scaled, split.y_test);

    // Save model state
    ModelMetadata meta;
    meta.model_type = "LinearRegression";
    meta.version = "1.0";
    meta.params["n_features"] = std::to_string(data.n_features);

    ModelIO<float>::save_model(
        "temp_pipeline_model.bin",
        model.weights(),
        scaler.mean(),
        scaler.std(),
        meta);

    std::cout << "  Model saved successfully" << std::endl;

    // Load model state
    Vector<float> loaded_weights;
    Vector<float> loaded_mean;
    Vector<float> loaded_std;
    ModelMetadata loaded_meta;

    ModelIO<float>::load_model(
        "temp_pipeline_model.bin",
        loaded_weights,
        loaded_mean,
        loaded_std,
        loaded_meta);

    std::cout << "  Model loaded successfully" << std::endl;
    std::cout << "  Model type: " << loaded_meta.model_type << std::endl;
    std::cout << "  Features: " << loaded_meta.params["n_features"] << std::endl;

    // Create new scaler with loaded parameters
    preprocessing::StandardScaler<float> loaded_scaler;
    // We need to verify the loaded parameters match
    assert(loaded_mean.size() == scaler.mean().size());
    assert(loaded_std.size() == scaler.std().size());

    for (usize i = 0; i < loaded_mean.size(); ++i) {
        assert(approx_equal(loaded_mean[i], scaler.mean()[i]));
        assert(approx_equal(loaded_std[i], scaler.std()[i]));
    }

    // Verify weights match
    assert(loaded_weights.size() == model.weights().size());
    for (usize i = 0; i < loaded_weights.size(); ++i) {
        assert(approx_equal(loaded_weights[i], model.weights()[i]));
    }

    std::cout << "  Original score: " << score_before << std::endl;
    std::cout << "  Weights verified to match" << std::endl;

    // Clean up
    std::remove("temp_pipeline_model.bin");

    std::cout << "  train-save-load-predict pipeline: PASSED" << std::endl;
}

// =============================================================================
// Test: Double precision
// =============================================================================

void test_double_precision() {
    std::cout << "Testing double precision I/O..." << std::endl;

    Vector<double> weights(5);
    weights[0] = 1.23456789012345;
    weights[1] = -9.87654321098765;
    weights[2] = 0.000000001;
    weights[3] = 1234567890.123456;
    weights[4] = -0.000000000001;

    ModelMetadata meta;
    meta.model_type = "HighPrecisionModel";

    // Save and load binary (should preserve precision)
    ModelIO<double>::save_weights_binary("temp_double.bin", weights, meta);
    Vector<double> loaded = ModelIO<double>::load_weights_binary("temp_double.bin");

    // Verify high precision
    for (usize i = 0; i < weights.size(); ++i) {
        assert(loaded[i] == weights[i]);  // Should be exact for binary
    }

    std::remove("temp_double.bin");

    std::cout << "  double precision I/O: PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== ModelIO Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_weights_text_io();
        test_weights_binary_io();
        test_matrix_text_io();
        test_matrix_binary_io();
        test_complete_model_io();
        test_double_precision();
        test_train_save_load_predict();

        std::cout << std::endl;
        std::cout << "=== All ModelIO Tests PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

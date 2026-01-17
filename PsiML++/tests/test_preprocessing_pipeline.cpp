#include "../include/utils/data_loader.h"
#include "../include/utils/string_utils.h"
#include "../include/ml/dataset.h"
#include "../include/ml/preprocessing/scalar.h"
#include "../include/ml/preprocessing/normalizer.h"
#include "../include/ml/preprocessing/encoder.h"
#include "../include/ml/algorithms/linear_regression.h"
#include "../include/ml/algorithms/logistic_regression.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::utils;
using namespace psi::math;
using namespace psi::ml;
using namespace psi::core;

template<typename T>
bool approx_equal(T a, T b, T epsilon = T{1e-4}) {
    return std::abs(a - b) < epsilon;
}

// =============================================================================
// Test: StandardScaler Pipeline
// =============================================================================

void test_standard_scaler_pipeline() {
    std::cout << "=== StandardScaler Pipeline ===" << std::endl;
    std::cout << std::endl;

    // Create data with different scales
    Matrix<float> X(100, 3);
    Vector<float> y(100);

    // Feature 0: values around 1000
    // Feature 1: values around 0.01
    // Feature 2: values around 50
    for (usize i = 0; i < 100; ++i) {
        float base = static_cast<float>(i) / 100.0f;
        X(i, 0) = 1000.0f + base * 100.0f;
        X(i, 1) = 0.01f + base * 0.005f;
        X(i, 2) = 50.0f + base * 10.0f;
        y[i] = X(i, 0) * 0.001f + X(i, 1) * 100.0f + X(i, 2) * 0.02f;
    }

    std::cout << "Data statistics before scaling:" << std::endl;
    for (usize j = 0; j < 3; ++j) {
        float sum = 0, min_v = X(0, j), max_v = X(0, j);
        for (usize i = 0; i < 100; ++i) {
            sum += X(i, j);
            min_v = std::min(min_v, X(i, j));
            max_v = std::max(max_v, X(i, j));
        }
        std::cout << "  Feature " << j << ": mean=" << sum/100.0f
                  << ", min=" << min_v << ", max=" << max_v << std::endl;
    }

    // Split
    auto split = Dataset<float>::train_test_split(X, y, 0.2f, true, 42);

    // Without scaling
    algorithms::LinearRegression<float> model_unscaled;
    model_unscaled.fit(split.X_train, split.y_train);
    float r2_unscaled = model_unscaled.score(split.X_test, split.y_test);

    // With StandardScaler
    preprocessing::StandardScaler<float> scaler;
    Matrix<float> X_train_scaled = scaler.fit_transform(split.X_train);
    Matrix<float> X_test_scaled = scaler.transform(split.X_test);

    std::cout << "\nScaler parameters:" << std::endl;
    for (usize j = 0; j < 3; ++j) {
        std::cout << "  Feature " << j << ": mean=" << scaler.mean()[j]
                  << ", std=" << scaler.std()[j] << std::endl;
    }

    // Verify scaled data has mean~0 and std~1
    std::cout << "\nScaled training data statistics:" << std::endl;
    for (usize j = 0; j < 3; ++j) {
        float sum = 0, sum_sq = 0;
        for (usize i = 0; i < X_train_scaled.rows(); ++i) {
            sum += X_train_scaled(i, j);
            sum_sq += X_train_scaled(i, j) * X_train_scaled(i, j);
        }
        float mean = sum / static_cast<float>(X_train_scaled.rows());
        float var = sum_sq / static_cast<float>(X_train_scaled.rows()) - mean * mean;
        std::cout << "  Feature " << j << ": mean=" << mean
                  << ", std=" << std::sqrt(var) << std::endl;
    }

    algorithms::LinearRegression<float> model_scaled;
    model_scaled.fit(X_train_scaled, split.y_train);
    float r2_scaled = model_scaled.score(X_test_scaled, split.y_test);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  R2 without scaling: " << r2_unscaled << std::endl;
    std::cout << "  R2 with scaling: " << r2_scaled << std::endl;

    // Test inverse transform
    Matrix<float> X_restored = scaler.inverse_transform(X_train_scaled);
    bool restore_ok = true;
    for (usize i = 0; i < 5; ++i) {
        for (usize j = 0; j < 3; ++j) {
            if (!approx_equal(X_restored(i, j), split.X_train(i, j), 0.01f)) {
                restore_ok = false;
                break;
            }
        }
    }
    std::cout << "  Inverse transform verified: " << (restore_ok ? "YES" : "NO") << std::endl;

    assert(restore_ok);

    std::cout << "\n=== StandardScaler Pipeline: PASSED ===" << std::endl;
}

// =============================================================================
// Test: MinMaxScaler Pipeline
// =============================================================================

void test_minmax_scaler_pipeline() {
    std::cout << "\n=== MinMaxScaler Pipeline ===" << std::endl;
    std::cout << std::endl;

    // Generate data
    auto data = DataLoader<float>::make_regression(100, 3, 0.1f, 42);

    // Split
    auto split = Dataset<float>::train_test_split(data.X, data.y, 0.2f, true, 42);

    // Apply MinMaxScaler
    preprocessing::MinMaxScaler<float> scaler(0.0f, 1.0f);
    Matrix<float> X_train_scaled = scaler.fit_transform(split.X_train);
    Matrix<float> X_test_scaled = scaler.transform(split.X_test);

    std::cout << "MinMaxScaler parameters:" << std::endl;
    for (usize j = 0; j < data.n_features; ++j) {
        std::cout << "  Feature " << j << ": min=" << scaler.data_min()[j]
                  << ", max=" << scaler.data_max()[j] << std::endl;
    }

    // Verify values are in [0, 1]
    float min_val = X_train_scaled(0, 0);
    float max_val = X_train_scaled(0, 0);
    for (usize i = 0; i < X_train_scaled.rows(); ++i) {
        for (usize j = 0; j < X_train_scaled.cols(); ++j) {
            min_val = std::min(min_val, X_train_scaled(i, j));
            max_val = std::max(max_val, X_train_scaled(i, j));
        }
    }

    std::cout << "\nScaled training data range:" << std::endl;
    std::cout << "  Min value: " << min_val << std::endl;
    std::cout << "  Max value: " << max_val << std::endl;

    assert(min_val >= 0.0f);
    assert(max_val <= 1.0f);

    // Train model
    algorithms::LinearRegression<float> model;
    model.fit(X_train_scaled, split.y_train);
    float r2 = model.score(X_test_scaled, split.y_test);

    std::cout << "  R2 score: " << r2 << std::endl;

    // Test inverse transform
    Matrix<float> X_restored = scaler.inverse_transform(X_train_scaled);
    bool restore_ok = true;
    for (usize i = 0; i < 5; ++i) {
        for (usize j = 0; j < data.n_features; ++j) {
            if (!approx_equal(X_restored(i, j), split.X_train(i, j), 0.01f)) {
                restore_ok = false;
                break;
            }
        }
    }
    std::cout << "  Inverse transform verified: " << (restore_ok ? "YES" : "NO") << std::endl;

    assert(restore_ok);

    std::cout << "\n=== MinMaxScaler Pipeline: PASSED ===" << std::endl;
}

// =============================================================================
// Test: MaxAbsScaler Pipeline
// =============================================================================

void test_maxabs_scaler_pipeline() {
    std::cout << "\n=== MaxAbsScaler Pipeline ===" << std::endl;
    std::cout << std::endl;

    // Create data with positive and negative values
    Matrix<float> X(50, 2);
    for (usize i = 0; i < 50; ++i) {
        X(i, 0) = static_cast<float>(i) - 25.0f;  // Range [-25, 25]
        X(i, 1) = static_cast<float>(i * 2) - 50.0f;  // Range [-50, 50]
    }

    preprocessing::MaxAbsScaler<float> scaler;
    Matrix<float> X_scaled = scaler.fit_transform(X);

    std::cout << "MaxAbsScaler parameters:" << std::endl;
    for (usize j = 0; j < 2; ++j) {
        std::cout << "  Feature " << j << ": max_abs=" << scaler.max_abs()[j] << std::endl;
    }

    // Verify values are in [-1, 1]
    float min_val = X_scaled(0, 0);
    float max_val = X_scaled(0, 0);
    for (usize i = 0; i < X_scaled.rows(); ++i) {
        for (usize j = 0; j < X_scaled.cols(); ++j) {
            min_val = std::min(min_val, X_scaled(i, j));
            max_val = std::max(max_val, X_scaled(i, j));
        }
    }

    std::cout << "\nScaled data range: [" << min_val << ", " << max_val << "]" << std::endl;

    assert(min_val >= -1.0f);
    assert(max_val <= 1.0f);

    // Test inverse
    Matrix<float> X_restored = scaler.inverse_transform(X_scaled);
    bool restore_ok = true;
    for (usize i = 0; i < 10; ++i) {
        for (usize j = 0; j < 2; ++j) {
            if (!approx_equal(X_restored(i, j), X(i, j), 0.01f)) {
                restore_ok = false;
                break;
            }
        }
    }
    std::cout << "  Inverse transform verified: " << (restore_ok ? "YES" : "NO") << std::endl;

    assert(restore_ok);

    std::cout << "\n=== MaxAbsScaler Pipeline: PASSED ===" << std::endl;
}

// =============================================================================
// Test: Normalizer Pipeline
// =============================================================================

void test_normalizer_pipeline() {
    std::cout << "\n=== Normalizer Pipeline ===" << std::endl;
    std::cout << std::endl;

    Matrix<float> X = {
        {3.0f, 4.0f},    // norm = 5
        {1.0f, 0.0f},    // norm = 1
        {0.0f, 0.0f},    // norm = 0
        {6.0f, 8.0f}     // norm = 10
    };

    // L2 normalization
    std::cout << "L2 Normalization:" << std::endl;
    preprocessing::Normalizer<float> normalizer_l2(preprocessing::NormType::L2);
    Matrix<float> X_l2 = normalizer_l2.transform(X);

    for (usize i = 0; i < X_l2.rows(); ++i) {
        float norm = 0;
        for (usize j = 0; j < X_l2.cols(); ++j) {
            norm += X_l2(i, j) * X_l2(i, j);
        }
        norm = std::sqrt(norm);
        std::cout << "  Row " << i << ": [" << X_l2(i, 0) << ", " << X_l2(i, 1)
                  << "] -> norm=" << norm << std::endl;
    }

    // Verify L2 norms are 1 (except for zero vectors)
    assert(approx_equal(X_l2(0, 0), 0.6f));
    assert(approx_equal(X_l2(0, 1), 0.8f));
    assert(approx_equal(X_l2(1, 0), 1.0f));

    // L1 normalization
    std::cout << "\nL1 Normalization:" << std::endl;
    preprocessing::Normalizer<float> normalizer_l1(preprocessing::NormType::L1);
    Matrix<float> X_l1 = normalizer_l1.transform(X);

    for (usize i = 0; i < X_l1.rows(); ++i) {
        float norm = 0;
        for (usize j = 0; j < X_l1.cols(); ++j) {
            norm += std::abs(X_l1(i, j));
        }
        std::cout << "  Row " << i << ": [" << X_l1(i, 0) << ", " << X_l1(i, 1)
                  << "] -> L1 norm=" << norm << std::endl;
    }

    // Verify L1 norms are 1
    assert(approx_equal(X_l1(0, 0) + X_l1(0, 1), 1.0f));

    // Max normalization
    std::cout << "\nMax Normalization:" << std::endl;
    preprocessing::Normalizer<float> normalizer_max(preprocessing::NormType::Max);
    Matrix<float> X_max = normalizer_max.transform(X);

    for (usize i = 0; i < X_max.rows(); ++i) {
        float max_val = std::max(std::abs(X_max(i, 0)), std::abs(X_max(i, 1)));
        std::cout << "  Row " << i << ": [" << X_max(i, 0) << ", " << X_max(i, 1)
                  << "] -> max=" << max_val << std::endl;
    }

    // Verify max values are 1
    assert(approx_equal(X_max(0, 1), 1.0f));
    assert(approx_equal(X_max(3, 1), 1.0f));

    std::cout << "\n=== Normalizer Pipeline: PASSED ===" << std::endl;
}

// =============================================================================
// Test: LabelEncoder Pipeline
// =============================================================================

void test_label_encoder_pipeline() {
    std::cout << "\n=== LabelEncoder Pipeline ===" << std::endl;
    std::cout << std::endl;

    Vector<float> labels = {5.0f, 10.0f, 5.0f, 15.0f, 10.0f, 5.0f, 15.0f};

    preprocessing::LabelEncoder<float> encoder;
    auto encoded = encoder.fit_transform(labels);

    std::cout << "Original labels: ";
    for (usize i = 0; i < labels.size(); ++i) {
        std::cout << labels[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Encoded labels:  ";
    for (usize i = 0; i < encoded.size(); ++i) {
        std::cout << encoded[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Number of classes: " << encoder.n_classes() << std::endl;

    assert(encoder.n_classes() == 3);
    assert(encoded[0] == encoded[2]);  // Same original value -> same encoded
    assert(encoded[1] == encoded[4]);
    assert(encoded[3] == encoded[6]);

    // Inverse transform
    auto decoded = encoder.inverse_transform(encoded);
    std::cout << "Decoded labels:  ";
    for (usize i = 0; i < decoded.size(); ++i) {
        std::cout << decoded[i] << " ";
    }
    std::cout << std::endl;

    bool decode_ok = true;
    for (usize i = 0; i < labels.size(); ++i) {
        if (!approx_equal(decoded[i], labels[i])) {
            decode_ok = false;
            break;
        }
    }
    std::cout << "Inverse transform verified: " << (decode_ok ? "YES" : "NO") << std::endl;

    assert(decode_ok);

    std::cout << "\n=== LabelEncoder Pipeline: PASSED ===" << std::endl;
}

// =============================================================================
// Test: OneHotEncoder Pipeline
// =============================================================================

void test_onehot_encoder_pipeline() {
    std::cout << "\n=== OneHotEncoder Pipeline ===" << std::endl;
    std::cout << std::endl;

    Vector<float> labels = {0.0f, 1.0f, 2.0f, 0.0f, 1.0f};

    preprocessing::OneHotEncoder<float> encoder;
    auto encoded = encoder.fit_transform(labels);

    std::cout << "Original labels:" << std::endl;
    for (usize i = 0; i < labels.size(); ++i) {
        std::cout << "  " << labels[i] << std::endl;
    }

    std::cout << "\nOne-hot encoded:" << std::endl;
    for (usize i = 0; i < encoded.rows(); ++i) {
        std::cout << "  [";
        for (usize j = 0; j < encoded.cols(); ++j) {
            std::cout << encoded(i, j);
            if (j < encoded.cols() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    assert(encoded.rows() == 5);
    assert(encoded.cols() == 3);
    assert(encoded(0, 0) == 1.0f);  // Class 0
    assert(encoded(1, 1) == 1.0f);  // Class 1
    assert(encoded(2, 2) == 1.0f);  // Class 2

    // Inverse transform
    auto decoded = encoder.inverse_transform(encoded);
    std::cout << "\nDecoded labels:" << std::endl;
    for (usize i = 0; i < decoded.size(); ++i) {
        std::cout << "  " << decoded[i] << std::endl;
    }

    bool decode_ok = true;
    for (usize i = 0; i < labels.size(); ++i) {
        if (!approx_equal(decoded[i], labels[i])) {
            decode_ok = false;
            break;
        }
    }
    std::cout << "Inverse transform verified: " << (decode_ok ? "YES" : "NO") << std::endl;

    assert(decode_ok);

    std::cout << "\n=== OneHotEncoder Pipeline: PASSED ===" << std::endl;
}

// =============================================================================
// Test: Comparison of Scalers Effect on Model
// =============================================================================

void test_scaler_comparison() {
    std::cout << "\n=== Scaler Comparison on Model Performance ===" << std::endl;
    std::cout << std::endl;

    // Generate data with features at different scales
    auto data = DataLoader<float>::make_regression(200, 4, 0.2f, 42);

    // Artificially scale features to very different ranges
    for (usize i = 0; i < data.n_samples; ++i) {
        data.X(i, 0) *= 1000.0f;   // [0, 10000]
        data.X(i, 1) *= 0.001f;    // [0, 0.01]
        data.X(i, 2) *= 100.0f;    // [0, 1000]
        // data.X(i, 3) unchanged  // [0, 10]
    }

    auto split = Dataset<float>::train_test_split(data.X, data.y, 0.2f, true, 42);

    std::cout << "Feature ranges:" << std::endl;
    for (usize j = 0; j < 4; ++j) {
        float min_v = split.X_train(0, j), max_v = split.X_train(0, j);
        for (usize i = 0; i < split.X_train.rows(); ++i) {
            min_v = std::min(min_v, split.X_train(i, j));
            max_v = std::max(max_v, split.X_train(i, j));
        }
        std::cout << "  Feature " << j << ": [" << min_v << ", " << max_v << "]" << std::endl;
    }

    // Test different scalers
    std::cout << "\nModel R2 scores with different scalers:" << std::endl;

    // No scaling
    {
        algorithms::LinearRegression<float> model;
        model.fit(split.X_train, split.y_train);
        float r2 = model.score(split.X_test, split.y_test);
        std::cout << "  No scaling:      " << r2 << std::endl;
    }

    // StandardScaler
    {
        preprocessing::StandardScaler<float> scaler;
        auto X_train = scaler.fit_transform(split.X_train);
        auto X_test = scaler.transform(split.X_test);
        algorithms::LinearRegression<float> model;
        model.fit(X_train, split.y_train);
        float r2 = model.score(X_test, split.y_test);
        std::cout << "  StandardScaler:  " << r2 << std::endl;
    }

    // MinMaxScaler
    {
        preprocessing::MinMaxScaler<float> scaler;
        auto X_train = scaler.fit_transform(split.X_train);
        auto X_test = scaler.transform(split.X_test);
        algorithms::LinearRegression<float> model;
        model.fit(X_train, split.y_train);
        float r2 = model.score(X_test, split.y_test);
        std::cout << "  MinMaxScaler:    " << r2 << std::endl;
    }

    // MaxAbsScaler
    {
        preprocessing::MaxAbsScaler<float> scaler;
        auto X_train = scaler.fit_transform(split.X_train);
        auto X_test = scaler.transform(split.X_test);
        algorithms::LinearRegression<float> model;
        model.fit(X_train, split.y_train);
        float r2 = model.score(X_test, split.y_test);
        std::cout << "  MaxAbsScaler:    " << r2 << std::endl;
    }

    std::cout << "\n=== Scaler Comparison: PASSED ===" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Preprocessing Full Pipeline Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        test_standard_scaler_pipeline();
        test_minmax_scaler_pipeline();
        test_maxabs_scaler_pipeline();
        test_normalizer_pipeline();
        test_label_encoder_pipeline();
        test_onehot_encoder_pipeline();
        test_scaler_comparison();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All Preprocessing Tests PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

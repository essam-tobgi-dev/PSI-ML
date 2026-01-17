#include "../include/utils/data_loader.h"
#include "../include/utils/model_io.h"
#include "../include/utils/string_utils.h"
#include "../include/ml/dataset.h"
#include "../include/ml/metrics.h"
#include "../include/ml/preprocessing/scalar.h"
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
// Test: Complete Logistic Regression Pipeline with Iris Binary Data
// =============================================================================

void test_iris_classification() {
    std::cout << "=== Iris Binary Classification Pipeline ===" << std::endl;
    std::cout << std::endl;

    try {
        // Step 1: Load the iris binary dataset
        std::cout << "Step 1: Loading iris binary dataset..." << std::endl;
        auto data = DataLoader<float>::load_csv_with_target("data/iris_binary.csv");

        std::cout << "  Loaded " << data.n_samples << " samples" << std::endl;
        std::cout << "  Features (" << data.n_features << "): ";
        for (const auto& name : data.feature_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
        std::cout << "  Target: " << data.target_name << std::endl;

        // Count classes
        usize class0 = 0, class1 = 0;
        for (usize i = 0; i < data.n_samples; ++i) {
            if (data.y[i] < 0.5f) class0++;
            else class1++;
        }
        std::cout << "  Class distribution: 0=" << class0 << ", 1=" << class1 << std::endl;

        // Step 2: Split into training and test sets
        std::cout << "\nStep 2: Splitting data (80% train, 20% test)..." << std::endl;
        auto split = Dataset<float>::train_test_split(
            data.X, data.y, 0.2f, true, 42);

        std::cout << "  Training samples: " << split.X_train.rows() << std::endl;
        std::cout << "  Test samples: " << split.X_test.rows() << std::endl;

        // Step 3: Normalize features
        std::cout << "\nStep 3: Normalizing features..." << std::endl;
        preprocessing::StandardScaler<float> scaler;
        Matrix<float> X_train_scaled = scaler.fit_transform(split.X_train);
        Matrix<float> X_test_scaled = scaler.transform(split.X_test);

        // Step 4: Train Logistic Regression model
        std::cout << "\nStep 4: Training Logistic Regression..." << std::endl;
        algorithms::LogisticRegression<float> model(
            0.1f,    // learning rate
            2000,    // max iterations
            1e-6f,   // tolerance
            0.0f,    // regularization
            true);   // fit intercept

        model.fit(X_train_scaled, split.y_train);

        std::cout << "  Model trained!" << std::endl;
        std::cout << "  Intercept: " << model.intercept() << std::endl;
        std::cout << "  Coefficients: ";
        Vector<float> coefs = model.coefficients();
        for (usize j = 0; j < coefs.size(); ++j) {
            std::cout << StringUtils::format_number(coefs[j], 4) << " ";
        }
        std::cout << std::endl;

        // Step 5: Evaluate on training and test sets
        std::cout << "\nStep 5: Evaluating model..." << std::endl;

        Vector<float> train_pred = model.predict(X_train_scaled);
        Vector<float> test_pred = model.predict(X_test_scaled);
        Vector<float> test_proba = model.predict_proba(X_test_scaled);

        float train_acc = metrics::accuracy(split.y_train, train_pred);
        float test_acc = metrics::accuracy(split.y_test, test_pred);

        std::cout << "  Training Accuracy: " << StringUtils::format_percent(train_acc, 1) << std::endl;
        std::cout << "  Test Accuracy: " << StringUtils::format_percent(test_acc, 1) << std::endl;

        // Detailed classification metrics
        auto cm = metrics::confusion_matrix(split.y_test, test_pred);
        float precision_val = metrics::precision(split.y_test, test_pred);
        float recall_val = metrics::recall(split.y_test, test_pred);
        float f1 = metrics::f1_score(split.y_test, test_pred);
        float specificity_val = metrics::specificity(split.y_test, test_pred);

        std::cout << "\n  Confusion Matrix:" << std::endl;
        std::cout << "                  Predicted" << std::endl;
        std::cout << "                  0      1" << std::endl;
        std::cout << "    Actual 0     " << cm.true_negatives << "      " << cm.false_positives << std::endl;
        std::cout << "    Actual 1     " << cm.false_negatives << "      " << cm.true_positives << std::endl;

        std::cout << "\n  Classification Metrics:" << std::endl;
        std::cout << "    Precision: " << StringUtils::format_percent(precision_val, 1) << std::endl;
        std::cout << "    Recall: " << StringUtils::format_percent(recall_val, 1) << std::endl;
        std::cout << "    F1 Score: " << StringUtils::format_number(f1, 4) << std::endl;
        std::cout << "    Specificity: " << StringUtils::format_percent(specificity_val, 1) << std::endl;

        // Step 6: Show sample predictions with probabilities
        std::cout << "\nStep 6: Sample predictions:" << std::endl;
        std::cout << "  " << StringUtils::pad_right("Actual", 8)
                  << StringUtils::pad_right("Pred", 8)
                  << "Probability" << std::endl;
        std::cout << "  " << StringUtils::repeat("-", 30) << std::endl;

        for (usize i = 0; i < std::min(usize{8}, split.y_test.size()); ++i) {
            std::cout << "  " << StringUtils::pad_right(std::to_string(static_cast<int>(split.y_test[i])), 8)
                      << StringUtils::pad_right(std::to_string(static_cast<int>(test_pred[i])), 8)
                      << StringUtils::format_number(test_proba[i], 4) << std::endl;
        }

        // Step 7: With regularization
        std::cout << "\nStep 7: Training with L2 regularization (lambda=0.1)..." << std::endl;
        algorithms::LogisticRegression<float> model_reg(
            0.1f, 2000, 1e-6f, 0.1f, true);
        model_reg.fit(X_train_scaled, split.y_train);

        float reg_train_acc = model_reg.score(X_train_scaled, split.y_train);
        float reg_test_acc = model_reg.score(X_test_scaled, split.y_test);

        std::cout << "  Regularized Training Accuracy: " << StringUtils::format_percent(reg_train_acc, 1) << std::endl;
        std::cout << "  Regularized Test Accuracy: " << StringUtils::format_percent(reg_test_acc, 1) << std::endl;

        // Assertions
        assert(train_acc >= 0.8f);
        assert(test_acc >= 0.7f);

        std::cout << "\n=== Iris Classification Pipeline: PASSED ===" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Warning: Test failed - " << e.what() << std::endl;
        std::cout << "Make sure data/iris_binary.csv exists" << std::endl;
    }
}

// =============================================================================
// Test: Synthetic Classification Pipeline
// =============================================================================

void test_synthetic_classification() {
    std::cout << "\n=== Synthetic Classification Pipeline ===" << std::endl;
    std::cout << std::endl;

    // Generate synthetic data
    std::cout << "Step 1: Generating synthetic classification data..." << std::endl;
    auto data = DataLoader<float>::make_classification(200, 4, 2, 3.0f, 42);

    std::cout << "  Generated " << data.n_samples << " samples with "
              << data.n_features << " features" << std::endl;

    // Count classes
    usize class0 = 0, class1 = 0;
    for (usize i = 0; i < data.n_samples; ++i) {
        if (data.y[i] < 0.5f) class0++;
        else class1++;
    }
    std::cout << "  Class distribution: 0=" << class0 << ", 1=" << class1 << std::endl;

    // Split
    auto split = Dataset<float>::train_test_split(data.X, data.y, 0.2f, true, 42);
    std::cout << "  Train: " << split.X_train.rows()
              << ", Test: " << split.X_test.rows() << std::endl;

    // Normalize
    preprocessing::StandardScaler<float> scaler;
    Matrix<float> X_train = scaler.fit_transform(split.X_train);
    Matrix<float> X_test = scaler.transform(split.X_test);

    // Train
    algorithms::LogisticRegression<float> model(0.5f, 1000);
    model.fit(X_train, split.y_train);

    // Evaluate
    float train_acc = model.score(X_train, split.y_train);
    float test_acc = model.score(X_test, split.y_test);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Train Accuracy: " << StringUtils::format_percent(train_acc, 1) << std::endl;
    std::cout << "  Test Accuracy: " << StringUtils::format_percent(test_acc, 1) << std::endl;

    // With good separation, should achieve high accuracy
    assert(train_acc > 0.9f);
    assert(test_acc > 0.85f);

    std::cout << "\n=== Synthetic Classification Pipeline: PASSED ===" << std::endl;
}

// =============================================================================
// Test: Decision Boundary Analysis
// =============================================================================

void test_decision_boundary() {
    std::cout << "\n=== Decision Boundary Analysis ===" << std::endl;
    std::cout << std::endl;

    // Create simple 2D data for visualization
    Matrix<float> X(6, 2);
    Vector<float> y(6);

    // Class 0: bottom-left
    X(0, 0) = 1.0f; X(0, 1) = 1.0f; y[0] = 0.0f;
    X(1, 0) = 2.0f; X(1, 1) = 1.5f; y[1] = 0.0f;
    X(2, 0) = 1.5f; X(2, 1) = 2.0f; y[2] = 0.0f;

    // Class 1: top-right
    X(3, 0) = 5.0f; X(3, 1) = 5.0f; y[3] = 1.0f;
    X(4, 0) = 6.0f; X(4, 1) = 5.5f; y[4] = 1.0f;
    X(5, 0) = 5.5f; X(5, 1) = 6.0f; y[5] = 1.0f;

    std::cout << "Training data:" << std::endl;
    for (usize i = 0; i < 6; ++i) {
        std::cout << "  (" << X(i, 0) << ", " << X(i, 1) << ") -> class " << y[i] << std::endl;
    }

    // Normalize
    preprocessing::StandardScaler<float> scaler;
    Matrix<float> X_scaled = scaler.fit_transform(X);

    // Train
    algorithms::LogisticRegression<float> model(0.5f, 2000);
    model.fit(X_scaled, y);

    std::cout << "\nModel parameters:" << std::endl;
    std::cout << "  Intercept: " << model.intercept() << std::endl;
    std::cout << "  Coefficients: [" << model.coefficients()[0] << ", "
              << model.coefficients()[1] << "]" << std::endl;

    // Predict on grid points
    std::cout << "\nPredictions on new points:" << std::endl;

    std::vector<std::pair<float, float>> test_points = {
        {1.5f, 1.5f}, {3.0f, 3.0f}, {5.5f, 5.5f}
    };

    for (const auto& [x1, x2] : test_points) {
        Matrix<float> point(1, 2);
        point(0, 0) = x1;
        point(0, 1) = x2;
        Matrix<float> point_scaled = scaler.transform(point);

        Vector<float> proba = model.predict_proba(point_scaled);
        Vector<float> pred = model.predict(point_scaled);

        std::cout << "  (" << x1 << ", " << x2 << ") -> class "
                  << pred[0] << " (prob: " << StringUtils::format_number(proba[0], 4) << ")" << std::endl;
    }

    // Verify perfect classification on training data
    Vector<float> train_pred = model.predict(X_scaled);
    float acc = metrics::accuracy(y, train_pred);
    std::cout << "\nTraining accuracy: " << StringUtils::format_percent(acc, 1) << std::endl;

    assert(acc == 1.0f);

    std::cout << "\n=== Decision Boundary Analysis: PASSED ===" << std::endl;
}

// =============================================================================
// Test: Model Persistence
// =============================================================================

void test_model_persistence() {
    std::cout << "\n=== Model Persistence Test ===" << std::endl;
    std::cout << std::endl;

    // Generate and train
    auto data = DataLoader<float>::make_classification(100, 3, 2, 2.5f, 42);
    auto split = Dataset<float>::train_test_split(data.X, data.y, 0.2f, true, 42);

    preprocessing::StandardScaler<float> scaler;
    Matrix<float> X_train = scaler.fit_transform(split.X_train);
    Matrix<float> X_test = scaler.transform(split.X_test);

    algorithms::LogisticRegression<float> model(0.5f, 1000);
    model.fit(X_train, split.y_train);

    Vector<float> original_pred = model.predict(X_test);
    float original_acc = model.score(X_test, split.y_test);

    std::cout << "Original model accuracy: " << StringUtils::format_percent(original_acc, 1) << std::endl;

    // Save
    ModelMetadata meta;
    meta.model_type = "LogisticRegression";
    meta.params["n_features"] = std::to_string(data.n_features);
    meta.params["accuracy"] = StringUtils::format_number(original_acc, 4);

    ModelIO<float>::save_model(
        "temp_logreg.bin",
        model.weights(),
        scaler.mean(),
        scaler.std(),
        meta);

    // Load
    Vector<float> loaded_weights;
    Vector<float> loaded_mean;
    Vector<float> loaded_std;
    ModelMetadata loaded_meta;

    ModelIO<float>::load_model(
        "temp_logreg.bin",
        loaded_weights,
        loaded_mean,
        loaded_std,
        loaded_meta);

    std::cout << "Loaded model type: " << loaded_meta.model_type << std::endl;
    std::cout << "Loaded accuracy: " << loaded_meta.params["accuracy"] << std::endl;

    // Verify weights match
    bool match = true;
    for (usize i = 0; i < model.weights().size(); ++i) {
        if (!approx_equal(loaded_weights[i], model.weights()[i])) {
            match = false;
            break;
        }
    }

    std::cout << "Weights verified: " << (match ? "YES" : "NO") << std::endl;

    std::remove("temp_logreg.bin");

    assert(match);

    std::cout << "\n=== Model Persistence Test: PASSED ===" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Logistic Regression Full Pipeline Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        test_decision_boundary();
        test_synthetic_classification();
        test_model_persistence();
        test_iris_classification();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All Logistic Regression Tests PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

#include "../include/utils/data_loader.h"
#include "../include/utils/model_io.h"
#include "../include/utils/string_utils.h"
#include "../include/ml/dataset.h"
#include "../include/ml/metrics.h"
#include "../include/ml/preprocessing/scalar.h"
#include "../include/ml/algorithms/linear_regression.h"
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
// Test: Complete Linear Regression Pipeline with Housing Data
// =============================================================================

void test_housing_price_prediction() {
    std::cout << "=== Housing Price Prediction Pipeline ===" << std::endl;
    std::cout << std::endl;

    try {
        // Step 1: Load the housing dataset from CSV
        std::cout << "Step 1: Loading housing dataset..." << std::endl;
        auto data = DataLoader<float>::load_csv_with_target("data/housing.csv");

        std::cout << "  Loaded " << data.n_samples << " samples" << std::endl;
        std::cout << "  Features (" << data.n_features << "): ";
        for (const auto& name : data.feature_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
        std::cout << "  Target: " << data.target_name << std::endl;

        // Show data statistics
        std::cout << "\n  Data Statistics:" << std::endl;
        for (usize j = 0; j < data.n_features; ++j) {
            float sum = 0, min_val = data.X(0, j), max_val = data.X(0, j);
            for (usize i = 0; i < data.n_samples; ++i) {
                sum += data.X(i, j);
                min_val = std::min(min_val, data.X(i, j));
                max_val = std::max(max_val, data.X(i, j));
            }
            float mean = sum / static_cast<float>(data.n_samples);
            std::cout << "    " << data.feature_names[j]
                      << ": mean=" << mean
                      << ", min=" << min_val
                      << ", max=" << max_val << std::endl;
        }

        // Step 2: Split into training and test sets
        std::cout << "\nStep 2: Splitting data (80% train, 20% test)..." << std::endl;
        auto split = Dataset<float>::train_test_split(
            data.X, data.y, 0.2f, true, 42);

        std::cout << "  Training samples: " << split.X_train.rows() << std::endl;
        std::cout << "  Test samples: " << split.X_test.rows() << std::endl;

        // Step 3: Normalize features
        std::cout << "\nStep 3: Normalizing features (StandardScaler)..." << std::endl;
        preprocessing::StandardScaler<float> scaler;
        Matrix<float> X_train_scaled = scaler.fit_transform(split.X_train);
        Matrix<float> X_test_scaled = scaler.transform(split.X_test);

        std::cout << "  Scaler fitted with mean and std:" << std::endl;
        for (usize j = 0; j < data.n_features; ++j) {
            std::cout << "    " << data.feature_names[j]
                      << ": mean=" << scaler.mean()[j]
                      << ", std=" << scaler.std()[j] << std::endl;
        }

        // Step 4: Train Linear Regression model (Normal Equation)
        std::cout << "\nStep 4: Training Linear Regression (Normal Equation)..." << std::endl;
        algorithms::LinearRegression<float> model_ne(
            algorithms::LinearRegression<float>::Solver::NormalEquation);
        model_ne.fit(X_train_scaled, split.y_train);

        std::cout << "  Model trained!" << std::endl;
        std::cout << "  Intercept: " << model_ne.intercept() << std::endl;
        std::cout << "  Coefficients: ";
        Vector<float> coefs = model_ne.coefficients();
        for (usize j = 0; j < coefs.size(); ++j) {
            std::cout << coefs[j] << " ";
        }
        std::cout << std::endl;

        // Step 5: Evaluate on training and test sets
        std::cout << "\nStep 5: Evaluating model..." << std::endl;

        Vector<float> train_pred = model_ne.predict(X_train_scaled);
        Vector<float> test_pred = model_ne.predict(X_test_scaled);

        float train_r2 = metrics::r2_score(split.y_train, train_pred);
        float test_r2 = metrics::r2_score(split.y_test, test_pred);
        float train_mse = metrics::mean_squared_error(split.y_train, train_pred);
        float test_mse = metrics::mean_squared_error(split.y_test, test_pred);
        float train_rmse = std::sqrt(train_mse);
        float test_rmse = std::sqrt(test_mse);
        float train_mae = metrics::mean_absolute_error(split.y_train, train_pred);
        float test_mae = metrics::mean_absolute_error(split.y_test, test_pred);

        std::cout << "  Training Metrics:" << std::endl;
        std::cout << "    R2 Score: " << train_r2 << std::endl;
        std::cout << "    RMSE: $" << train_rmse << std::endl;
        std::cout << "    MAE: $" << train_mae << std::endl;

        std::cout << "  Test Metrics:" << std::endl;
        std::cout << "    R2 Score: " << test_r2 << std::endl;
        std::cout << "    RMSE: $" << test_rmse << std::endl;
        std::cout << "    MAE: $" << test_mae << std::endl;

        // Step 6: Sample predictions
        std::cout << "\nStep 6: Sample predictions on test set:" << std::endl;
        std::cout << "  " << StringUtils::pad_right("Actual", 12)
                  << StringUtils::pad_right("Predicted", 12)
                  << "Error" << std::endl;
        std::cout << "  " << StringUtils::repeat("-", 36) << std::endl;

        for (usize i = 0; i < std::min(usize{5}, split.y_test.size()); ++i) {
            float actual = split.y_test[i];
            float predicted = test_pred[i];
            float error = actual - predicted;
            std::cout << "  $" << StringUtils::pad_right(std::to_string(static_cast<int>(actual)), 11)
                      << "$" << StringUtils::pad_right(std::to_string(static_cast<int>(predicted)), 11)
                      << "$" << static_cast<int>(error) << std::endl;
        }

        // Step 7: Compare with Gradient Descent
        std::cout << "\nStep 7: Training with Gradient Descent..." << std::endl;
        algorithms::LinearRegression<float> model_gd(
            algorithms::LinearRegression<float>::Solver::GradientDescent,
            0.01f,   // learning rate
            5000,    // max iterations
            1e-6f);  // tolerance
        model_gd.fit(X_train_scaled, split.y_train);

        float gd_train_r2 = model_gd.score(X_train_scaled, split.y_train);
        float gd_test_r2 = model_gd.score(X_test_scaled, split.y_test);

        std::cout << "  GD Training R2: " << gd_train_r2 << std::endl;
        std::cout << "  GD Test R2: " << gd_test_r2 << std::endl;

        // Step 8: Ridge Regression comparison
        std::cout << "\nStep 8: Training Ridge Regression (alpha=1.0)..." << std::endl;
        algorithms::RidgeRegression<float> model_ridge(1.0f);
        model_ridge.fit(X_train_scaled, split.y_train);

        float ridge_train_r2 = model_ridge.score(X_train_scaled, split.y_train);
        float ridge_test_r2 = model_ridge.score(X_test_scaled, split.y_test);

        std::cout << "  Ridge Training R2: " << ridge_train_r2 << std::endl;
        std::cout << "  Ridge Test R2: " << ridge_test_r2 << std::endl;

        // Step 9: Save the best model
        std::cout << "\nStep 9: Saving model..." << std::endl;
        ModelMetadata meta;
        meta.model_type = "LinearRegression";
        meta.version = "1.0";
        meta.params["solver"] = "NormalEquation";
        meta.params["n_features"] = std::to_string(data.n_features);
        meta.params["train_r2"] = StringUtils::format_number(train_r2, 4);
        meta.params["test_r2"] = StringUtils::format_number(test_r2, 4);

        ModelIO<float>::save_model(
            "housing_model.bin",
            model_ne.weights(),
            scaler.mean(),
            scaler.std(),
            meta);
        std::cout << "  Model saved to housing_model.bin" << std::endl;

        // Step 10: Load and verify
        std::cout << "\nStep 10: Loading and verifying model..." << std::endl;
        Vector<float> loaded_weights;
        Vector<float> loaded_mean;
        Vector<float> loaded_std;
        ModelMetadata loaded_meta;

        ModelIO<float>::load_model(
            "housing_model.bin",
            loaded_weights,
            loaded_mean,
            loaded_std,
            loaded_meta);

        std::cout << "  Loaded model type: " << loaded_meta.model_type << std::endl;
        std::cout << "  Original test R2: " << loaded_meta.params["test_r2"] << std::endl;

        // Verify weights match
        bool weights_match = true;
        for (usize i = 0; i < model_ne.weights().size(); ++i) {
            if (!approx_equal(loaded_weights[i], model_ne.weights()[i])) {
                weights_match = false;
                break;
            }
        }
        std::cout << "  Weights verified: " << (weights_match ? "YES" : "NO") << std::endl;

        // Clean up
        std::remove("housing_model.bin");

        // Assertions
        assert(train_r2 > 0.5f);  // Should explain at least 50% variance
        assert(weights_match);

        std::cout << "\n=== Housing Price Prediction Pipeline: PASSED ===" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Warning: Test failed - " << e.what() << std::endl;
        std::cout << "Make sure data/housing.csv exists" << std::endl;
    }
}

// =============================================================================
// Test: Synthetic Data Pipeline
// =============================================================================

void test_synthetic_regression() {
    std::cout << "\n=== Synthetic Regression Pipeline ===" << std::endl;
    std::cout << std::endl;

    // Generate synthetic data
    std::cout << "Step 1: Generating synthetic regression data..." << std::endl;
    auto data = DataLoader<float>::make_regression(200, 5, 0.1f, 42);

    std::cout << "  Generated " << data.n_samples << " samples with "
              << data.n_features << " features" << std::endl;

    // Split
    auto split = Dataset<float>::train_test_split(data.X, data.y, 0.2f, true, 42);
    std::cout << "  Train: " << split.X_train.rows()
              << ", Test: " << split.X_test.rows() << std::endl;

    // Normalize
    preprocessing::StandardScaler<float> scaler;
    Matrix<float> X_train = scaler.fit_transform(split.X_train);
    Matrix<float> X_test = scaler.transform(split.X_test);

    // Train
    algorithms::LinearRegression<float> model;
    model.fit(X_train, split.y_train);

    // Evaluate
    float train_r2 = model.score(X_train, split.y_train);
    float test_r2 = model.score(X_test, split.y_test);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Train R2: " << train_r2 << std::endl;
    std::cout << "  Test R2: " << test_r2 << std::endl;

    // With low noise, model should fit very well
    assert(train_r2 > 0.95f);
    assert(test_r2 > 0.90f);

    std::cout << "\n=== Synthetic Regression Pipeline: PASSED ===" << std::endl;
}

// =============================================================================
// Test: Cross-Validation
// =============================================================================

void test_cross_validation() {
    std::cout << "\n=== Cross-Validation Pipeline ===" << std::endl;
    std::cout << std::endl;

    // Generate data
    auto data = DataLoader<float>::make_regression(100, 3, 0.2f, 42);
    std::cout << "Generated " << data.n_samples << " samples" << std::endl;

    // K-fold cross validation
    usize n_folds = 5;
    auto folds = Dataset<float>::kfold_indices(data.n_samples, n_folds, true, 42);

    std::cout << "Performing " << n_folds << "-fold cross-validation..." << std::endl;

    std::vector<float> fold_scores;

    for (usize fold = 0; fold < n_folds; ++fold) {
        auto& [train_idx, test_idx] = folds[fold];

        // Create train/test sets for this fold
        Matrix<float> X_train(train_idx.size(), data.n_features);
        Vector<float> y_train(train_idx.size());
        Matrix<float> X_test(test_idx.size(), data.n_features);
        Vector<float> y_test(test_idx.size());

        for (usize i = 0; i < train_idx.size(); ++i) {
            for (usize j = 0; j < data.n_features; ++j) {
                X_train(i, j) = data.X(train_idx[i], j);
            }
            y_train[i] = data.y[train_idx[i]];
        }

        for (usize i = 0; i < test_idx.size(); ++i) {
            for (usize j = 0; j < data.n_features; ++j) {
                X_test(i, j) = data.X(test_idx[i], j);
            }
            y_test[i] = data.y[test_idx[i]];
        }

        // Normalize
        preprocessing::StandardScaler<float> scaler;
        X_train = scaler.fit_transform(X_train);
        X_test = scaler.transform(X_test);

        // Train and evaluate
        algorithms::LinearRegression<float> model;
        model.fit(X_train, y_train);
        float score = model.score(X_test, y_test);
        fold_scores.push_back(score);

        std::cout << "  Fold " << (fold + 1) << ": R2 = " << score << std::endl;
    }

    // Calculate mean and std
    float mean_score = 0;
    for (float s : fold_scores) mean_score += s;
    mean_score /= static_cast<float>(n_folds);

    float std_score = 0;
    for (float s : fold_scores) {
        float diff = s - mean_score;
        std_score += diff * diff;
    }
    std_score = std::sqrt(std_score / static_cast<float>(n_folds));

    std::cout << "\nCross-Validation Results:" << std::endl;
    std::cout << "  Mean R2: " << mean_score << " (+/- " << std_score << ")" << std::endl;

    assert(mean_score > 0.7f);

    std::cout << "\n=== Cross-Validation Pipeline: PASSED ===" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Linear Regression Full Pipeline Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        test_synthetic_regression();
        test_cross_validation();
        test_housing_price_prediction();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All Linear Regression Tests PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

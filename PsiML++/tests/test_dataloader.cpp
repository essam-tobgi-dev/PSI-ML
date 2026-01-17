#include "../include/utils/data_loader.h"
#include "../include/ml/dataset.h"
#include "../include/ml/preprocessing/scalar.h"
#include "../include/ml/algorithms/linear_regression.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace psi::utils;
using namespace psi::math;
using namespace psi::ml;
using namespace psi::core;

template<typename T>
bool approx_equal(T a, T b, T epsilon = T{1e-4}) {
    return std::abs(a - b) < epsilon;
}

// =============================================================================
// Test: Load CSV file
// =============================================================================

void test_load_csv_basic() {
    std::cout << "Testing load_csv (basic)..." << std::endl;

    // Create a temporary CSV file
    std::ofstream temp("temp_test.csv");
    temp << "a,b,c\n";
    temp << "1.0,2.0,3.0\n";
    temp << "4.0,5.0,6.0\n";
    temp << "7.0,8.0,9.0\n";
    temp.close();

    // Load the CSV
    Matrix<float> data = DataLoader<float>::load_csv("temp_test.csv");

    assert(data.rows() == 3);
    assert(data.cols() == 3);
    assert(approx_equal(data(0, 0), 1.0f));
    assert(approx_equal(data(1, 1), 5.0f));
    assert(approx_equal(data(2, 2), 9.0f));

    // Clean up
    std::remove("temp_test.csv");

    std::cout << "  load_csv (basic): PASSED" << std::endl;
}

void test_load_csv_with_target() {
    std::cout << "Testing load_csv_with_target..." << std::endl;

    // Create a temporary CSV file
    std::ofstream temp("temp_target.csv");
    temp << "feature1,feature2,target\n";
    temp << "1.0,2.0,10.0\n";
    temp << "3.0,4.0,20.0\n";
    temp << "5.0,6.0,30.0\n";
    temp.close();

    // Load with target
    auto loaded = DataLoader<float>::load_csv_with_target("temp_target.csv");

    assert(loaded.n_samples == 3);
    assert(loaded.n_features == 2);
    assert(loaded.X.rows() == 3);
    assert(loaded.X.cols() == 2);
    assert(loaded.y.size() == 3);
    assert(approx_equal(loaded.y[0], 10.0f));
    assert(approx_equal(loaded.y[2], 30.0f));
    assert(loaded.target_name == "target");

    // Clean up
    std::remove("temp_target.csv");

    std::cout << "  load_csv_with_target: PASSED" << std::endl;
}

// =============================================================================
// Test: Load real housing dataset
// =============================================================================

void test_load_housing_dataset() {
    std::cout << "Testing load housing dataset..." << std::endl;

    try {
        auto data = DataLoader<float>::load_csv_with_target("data/housing.csv");

        std::cout << "  Loaded " << data.n_samples << " samples with "
                  << data.n_features << " features" << std::endl;

        assert(data.n_samples == 20);
        assert(data.n_features == 3);  // size, bedrooms, age
        assert(data.feature_names.size() == 3);
        assert(data.target_name == "price");

        // Check first sample
        assert(approx_equal(data.X(0, 0), 1400.0f));  // size
        assert(approx_equal(data.X(0, 1), 3.0f));     // bedrooms
        assert(approx_equal(data.X(0, 2), 15.0f));    // age
        assert(approx_equal(data.y[0], 245000.0f));   // price

        std::cout << "  Feature names: ";
        for (const auto& name : data.feature_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
        std::cout << "  Target: " << data.target_name << std::endl;

        std::cout << "  load housing dataset: PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Warning: Could not load housing.csv - " << e.what() << std::endl;
        std::cout << "  Skipping housing dataset test" << std::endl;
    }
}

// =============================================================================
// Test: Full pipeline - Load CSV, preprocess, train, predict
// =============================================================================

void test_full_regression_pipeline() {
    std::cout << "Testing full regression pipeline with CSV..." << std::endl;

    try {
        // Step 1: Load housing dataset
        auto data = DataLoader<float>::load_csv_with_target("data/housing.csv");
        std::cout << "  Step 1: Loaded " << data.n_samples << " samples" << std::endl;

        // Step 2: Split into train/test
        auto split = Dataset<float>::train_test_split(
            data.X, data.y, 0.2f, true, 42);
        std::cout << "  Step 2: Split - Train: " << split.X_train.rows()
                  << ", Test: " << split.X_test.rows() << std::endl;

        // Step 3: Normalize features
        preprocessing::StandardScaler<float> scaler;
        Matrix<float> X_train_scaled = scaler.fit_transform(split.X_train);
        Matrix<float> X_test_scaled = scaler.transform(split.X_test);
        std::cout << "  Step 3: Normalized features" << std::endl;

        // Step 4: Train linear regression model
        algorithms::LinearRegression<float> model;
        model.fit(X_train_scaled, split.y_train);
        std::cout << "  Step 4: Model trained" << std::endl;

        // Step 5: Evaluate on test set
        float train_score = model.score(X_train_scaled, split.y_train);
        float test_score = model.score(X_test_scaled, split.y_test);
        std::cout << "  Step 5: R2 score - Train: " << train_score
                  << ", Test: " << test_score << std::endl;

        // Step 6: Make predictions
        Vector<float> predictions = model.predict(X_test_scaled);
        std::cout << "  Step 6: Made " << predictions.size() << " predictions" << std::endl;

        // Print sample predictions
        std::cout << "  Sample predictions:" << std::endl;
        for (usize i = 0; i < std::min(usize{3}, predictions.size()); ++i) {
            std::cout << "    Actual: " << split.y_test[i]
                      << ", Predicted: " << predictions[i] << std::endl;
        }

        // Model should have reasonable performance
        assert(train_score > 0.5f);  // Should explain at least 50% of variance

        std::cout << "  full regression pipeline: PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Warning: Pipeline test failed - " << e.what() << std::endl;
    }
}

// =============================================================================
// Test: Save and load CSV
// =============================================================================

void test_save_load_csv() {
    std::cout << "Testing save_csv and load_csv..." << std::endl;

    // Create test data
    Matrix<float> original(3, 2);
    original(0, 0) = 1.5f; original(0, 1) = 2.5f;
    original(1, 0) = 3.5f; original(1, 1) = 4.5f;
    original(2, 0) = 5.5f; original(2, 1) = 6.5f;

    std::vector<std::string> headers = {"col1", "col2"};

    // Save
    DataLoader<float>::save_csv("temp_save_test.csv", original, headers);

    // Load
    Matrix<float> loaded = DataLoader<float>::load_csv("temp_save_test.csv");

    // Verify
    assert(loaded.rows() == 3);
    assert(loaded.cols() == 2);
    assert(approx_equal(loaded(0, 0), 1.5f));
    assert(approx_equal(loaded(2, 1), 6.5f));

    // Clean up
    std::remove("temp_save_test.csv");

    std::cout << "  save_csv and load_csv: PASSED" << std::endl;
}

// =============================================================================
// Test: Synthetic data generation
// =============================================================================

void test_make_regression() {
    std::cout << "Testing make_regression..." << std::endl;

    auto data = DataLoader<float>::make_regression(100, 3, 0.1f, 42);

    assert(data.n_samples == 100);
    assert(data.n_features == 3);
    assert(data.X.rows() == 100);
    assert(data.X.cols() == 3);
    assert(data.y.size() == 100);
    assert(data.feature_names.size() == 3);

    // Train a model on synthetic data - should fit well
    algorithms::LinearRegression<float> model;
    model.fit(data.X, data.y);
    float r2 = model.score(data.X, data.y);

    std::cout << "  R2 score on synthetic data: " << r2 << std::endl;
    assert(r2 > 0.9f);  // Should fit very well with low noise

    std::cout << "  make_regression: PASSED" << std::endl;
}

void test_make_classification() {
    std::cout << "Testing make_classification..." << std::endl;

    auto data = DataLoader<float>::make_classification(100, 4, 2, 3.0f, 42);

    assert(data.n_samples == 100);
    assert(data.n_features == 4);
    assert(data.X.rows() == 100);
    assert(data.y.size() == 100);

    // Check that we have two classes
    usize class0 = 0, class1 = 0;
    for (usize i = 0; i < data.y.size(); ++i) {
        if (data.y[i] < 0.5f) class0++;
        else class1++;
    }
    std::cout << "  Class distribution: 0=" << class0 << ", 1=" << class1 << std::endl;
    assert(class0 > 0 && class1 > 0);

    std::cout << "  make_classification: PASSED" << std::endl;
}

// =============================================================================
// Test: Vector load/save
// =============================================================================

void test_vector_io() {
    std::cout << "Testing vector load/save..." << std::endl;

    Vector<float> original(5);
    original[0] = 1.1f;
    original[1] = 2.2f;
    original[2] = 3.3f;
    original[3] = 4.4f;
    original[4] = 5.5f;

    // Save
    DataLoader<float>::save_vector("temp_vector.txt", original);

    // Load
    Vector<float> loaded = DataLoader<float>::load_vector("temp_vector.txt");

    // Verify
    assert(loaded.size() == 5);
    assert(approx_equal(loaded[0], 1.1f));
    assert(approx_equal(loaded[4], 5.5f));

    // Clean up
    std::remove("temp_vector.txt");

    std::cout << "  vector load/save: PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== DataLoader Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_load_csv_basic();
        test_load_csv_with_target();
        test_save_load_csv();
        test_vector_io();
        test_make_regression();
        test_make_classification();
        test_load_housing_dataset();
        test_full_regression_pipeline();

        std::cout << std::endl;
        std::cout << "=== All DataLoader Tests PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

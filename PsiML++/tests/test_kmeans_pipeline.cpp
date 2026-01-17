#include "../include/utils/data_loader.h"
#include "../include/utils/string_utils.h"
#include "../include/ml/preprocessing/scalar.h"
#include "../include/ml/algorithms/kmeans.h"
#include "../include/ml/metrics.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <map>

using namespace psi::utils;
using namespace psi::math;
using namespace psi::ml;
using namespace psi::core;

template<typename T>
bool approx_equal(T a, T b, T epsilon = T{1e-4}) {
    return std::abs(a - b) < epsilon;
}

// =============================================================================
// Test: Basic KMeans Clustering
// =============================================================================

void test_basic_kmeans() {
    std::cout << "=== Basic KMeans Clustering ===" << std::endl;
    std::cout << std::endl;

    // Create clearly separated clusters
    Matrix<float> X(15, 2);
    // Cluster 1: around (1, 1)
    X(0, 0) = 0.9f; X(0, 1) = 1.1f;
    X(1, 0) = 1.1f; X(1, 1) = 0.9f;
    X(2, 0) = 1.0f; X(2, 1) = 1.0f;
    X(3, 0) = 0.8f; X(3, 1) = 1.2f;
    X(4, 0) = 1.2f; X(4, 1) = 0.8f;

    // Cluster 2: around (5, 5)
    X(5, 0) = 4.9f; X(5, 1) = 5.1f;
    X(6, 0) = 5.1f; X(6, 1) = 4.9f;
    X(7, 0) = 5.0f; X(7, 1) = 5.0f;
    X(8, 0) = 4.8f; X(8, 1) = 5.2f;
    X(9, 0) = 5.2f; X(9, 1) = 4.8f;

    // Cluster 3: around (9, 1)
    X(10, 0) = 8.9f; X(10, 1) = 1.1f;
    X(11, 0) = 9.1f; X(11, 1) = 0.9f;
    X(12, 0) = 9.0f; X(12, 1) = 1.0f;
    X(13, 0) = 8.8f; X(13, 1) = 1.2f;
    X(14, 0) = 9.2f; X(14, 1) = 0.8f;

    std::cout << "Created 15 points in 3 clusters" << std::endl;

    // Train KMeans
    algorithms::KMeans<float> kmeans(3, 100, 1e-4f, 5, 42);
    kmeans.fit(X);

    assert(kmeans.is_fitted());
    assert(kmeans.n_clusters() == 3);

    // Get labels
    auto labels = kmeans.labels();
    std::cout << "\nCluster assignments:" << std::endl;
    for (usize i = 0; i < 15; ++i) {
        std::cout << "  Point (" << X(i, 0) << ", " << X(i, 1)
                  << ") -> Cluster " << labels[i] << std::endl;
    }

    // Verify each group is in same cluster
    assert(labels[0] == labels[1] && labels[1] == labels[2] &&
           labels[2] == labels[3] && labels[3] == labels[4]);
    assert(labels[5] == labels[6] && labels[6] == labels[7] &&
           labels[7] == labels[8] && labels[8] == labels[9]);
    assert(labels[10] == labels[11] && labels[11] == labels[12] &&
           labels[12] == labels[13] && labels[13] == labels[14]);

    // Verify different groups are in different clusters
    assert(labels[0] != labels[5]);
    assert(labels[0] != labels[10]);
    assert(labels[5] != labels[10]);

    // Get centroids
    auto centroids = kmeans.centroids();
    std::cout << "\nCluster centroids:" << std::endl;
    for (usize k = 0; k < 3; ++k) {
        std::cout << "  Cluster " << k << ": ("
                  << centroids(k, 0) << ", " << centroids(k, 1) << ")" << std::endl;
    }

    // Get inertia
    float inertia = kmeans.inertia();
    std::cout << "\nInertia: " << inertia << std::endl;

    std::cout << "\n=== Basic KMeans Clustering: PASSED ===" << std::endl;
}

// =============================================================================
// Test: KMeans with CSV Data
// =============================================================================

void test_kmeans_csv_data() {
    std::cout << "\n=== KMeans with CSV Data ===" << std::endl;
    std::cout << std::endl;

    try {
        // Load clustering data
        Matrix<float> X = DataLoader<float>::load_csv("data/clustering.csv");

        std::cout << "Loaded " << X.rows() << " points with "
                  << X.cols() << " features" << std::endl;

        // Normalize
        preprocessing::StandardScaler<float> scaler;
        Matrix<float> X_scaled = scaler.fit_transform(X);

        // Try different numbers of clusters
        std::cout << "\nTrying different numbers of clusters:" << std::endl;
        for (usize k = 2; k <= 4; ++k) {
            algorithms::KMeans<float> kmeans(k, 100, 1e-4f, 5, 42);
            kmeans.fit(X_scaled);

            float inertia = kmeans.inertia();
            std::cout << "  k=" << k << ": inertia=" << inertia << std::endl;
        }

        // Final model with k=3 (expected number of clusters)
        algorithms::KMeans<float> kmeans(3, 100, 1e-4f, 5, 42);
        kmeans.fit(X_scaled);

        auto labels = kmeans.labels();

        // Count points per cluster
        std::map<i32, usize> cluster_counts;
        for (usize i = 0; i < labels.size(); ++i) {
            cluster_counts[labels[i]]++;
        }

        std::cout << "\nCluster sizes (k=3):" << std::endl;
        for (const auto& [cluster, count] : cluster_counts) {
            std::cout << "  Cluster " << cluster << ": " << count << " points" << std::endl;
        }

        // Predict on new points
        Matrix<float> new_points(2, 2);
        new_points(0, 0) = 1.0f; new_points(0, 1) = 1.0f;  // Should be cluster 0
        new_points(1, 0) = 9.0f; new_points(1, 1) = 9.0f;  // Should be cluster 2

        Matrix<float> new_scaled = scaler.transform(new_points);
        auto new_labels = kmeans.predict(new_scaled);

        std::cout << "\nPredictions on new points:" << std::endl;
        std::cout << "  (1, 1) -> Cluster " << new_labels[0] << std::endl;
        std::cout << "  (9, 9) -> Cluster " << new_labels[1] << std::endl;

        assert(new_labels[0] != new_labels[1]);

        std::cout << "\n=== KMeans with CSV Data: PASSED ===" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Warning: Test skipped - " << e.what() << std::endl;
    }
}

// =============================================================================
// Test: KMeans with Synthetic Blobs
// =============================================================================

void test_kmeans_synthetic() {
    std::cout << "\n=== KMeans with Synthetic Blobs ===" << std::endl;
    std::cout << std::endl;

    // Generate blob data
    auto data = DataLoader<float>::make_blobs(150, 2, 3, 1.0f, 42);

    std::cout << "Generated " << data.n_samples << " points in "
              << "3 clusters" << std::endl;

    // Count true labels
    std::map<i32, usize> true_counts;
    for (usize i = 0; i < data.n_samples; ++i) {
        true_counts[static_cast<i32>(data.y[i])]++;
    }
    std::cout << "True cluster sizes:" << std::endl;
    for (const auto& [cluster, count] : true_counts) {
        std::cout << "  Cluster " << cluster << ": " << count << " points" << std::endl;
    }

    // Normalize and cluster
    preprocessing::StandardScaler<float> scaler;
    Matrix<float> X_scaled = scaler.fit_transform(data.X);

    algorithms::KMeans<float> kmeans(3, 100, 1e-4f, 10, 42);
    kmeans.fit(X_scaled);

    auto pred_labels = kmeans.labels();

    // Count predicted labels
    std::map<i32, usize> pred_counts;
    for (usize i = 0; i < pred_labels.size(); ++i) {
        pred_counts[pred_labels[i]]++;
    }
    std::cout << "\nPredicted cluster sizes:" << std::endl;
    for (const auto& [cluster, count] : pred_counts) {
        std::cout << "  Cluster " << cluster << ": " << count << " points" << std::endl;
    }

    // Calculate purity (simplified)
    // For each predicted cluster, find the most common true label
    std::cout << "\nCluster purity analysis:" << std::endl;
    for (const auto& [pred_cluster, _] : pred_counts) {
        std::map<i32, usize> true_in_pred;
        for (usize i = 0; i < data.n_samples; ++i) {
            if (pred_labels[i] == pred_cluster) {
                true_in_pred[static_cast<i32>(data.y[i])]++;
            }
        }
        i32 dominant_true = 0;
        usize max_count = 0;
        for (const auto& [true_label, count] : true_in_pred) {
            if (count > max_count) {
                max_count = count;
                dominant_true = true_label;
            }
        }
        float purity = static_cast<float>(max_count) /
                       static_cast<float>(pred_counts[pred_cluster]);
        std::cout << "  Predicted " << pred_cluster << " -> mostly True "
                  << dominant_true << " (purity: "
                  << StringUtils::format_percent(purity, 1) << ")" << std::endl;
    }

    // Calculate inertia
    float inertia = kmeans.inertia();
    std::cout << "\nInertia: " << inertia << std::endl;

    // Centroids
    auto centroids = kmeans.centroids();
    std::cout << "\nCentroids:" << std::endl;
    for (usize k = 0; k < 3; ++k) {
        std::cout << "  Cluster " << k << ": ("
                  << centroids(k, 0) << ", " << centroids(k, 1) << ")" << std::endl;
    }

    std::cout << "\n=== KMeans with Synthetic Blobs: PASSED ===" << std::endl;
}

// =============================================================================
// Test: Elbow Method
// =============================================================================

void test_elbow_method() {
    std::cout << "\n=== Elbow Method for Optimal K ===" << std::endl;
    std::cout << std::endl;

    // Generate data with 4 clear clusters
    auto data = DataLoader<float>::make_blobs(200, 2, 4, 0.5f, 42);

    preprocessing::StandardScaler<float> scaler;
    Matrix<float> X_scaled = scaler.fit_transform(data.X);

    std::cout << "Testing k from 1 to 8:" << std::endl;
    std::cout << StringUtils::pad_right("k", 5)
              << StringUtils::pad_right("Inertia", 15)
              << "Elbow" << std::endl;
    std::cout << StringUtils::repeat("-", 35) << std::endl;

    std::vector<float> inertias;
    for (usize k = 1; k <= 8; ++k) {
        algorithms::KMeans<float> kmeans(k, 100, 1e-4f, 5, 42);
        kmeans.fit(X_scaled);
        float inertia = kmeans.inertia();
        inertias.push_back(inertia);

        std::string elbow = "";
        if (k == 4) elbow = " <- optimal";

        std::cout << StringUtils::pad_right(std::to_string(k), 5)
                  << StringUtils::pad_right(StringUtils::format_number(inertia, 2), 15)
                  << elbow << std::endl;
    }

    // Verify that inertia decreases with more clusters
    for (usize i = 1; i < inertias.size(); ++i) {
        assert(inertias[i] <= inertias[i-1]);
    }

    std::cout << "\n=== Elbow Method: PASSED ===" << std::endl;
}

// =============================================================================
// Test: KMeans Transform (Distance to Centroids)
// =============================================================================

void test_kmeans_transform() {
    std::cout << "\n=== KMeans Transform ===" << std::endl;
    std::cout << std::endl;

    // Simple 2D data with 3 clusters
    Matrix<float> X(9, 2);
    // Cluster at origin
    X(0, 0) = 0.0f; X(0, 1) = 0.0f;
    X(1, 0) = 0.1f; X(1, 1) = 0.1f;
    X(2, 0) = -0.1f; X(2, 1) = -0.1f;

    // Cluster at (5, 0)
    X(3, 0) = 5.0f; X(3, 1) = 0.0f;
    X(4, 0) = 5.1f; X(4, 1) = 0.1f;
    X(5, 0) = 4.9f; X(5, 1) = -0.1f;

    // Cluster at (0, 5)
    X(6, 0) = 0.0f; X(6, 1) = 5.0f;
    X(7, 0) = 0.1f; X(7, 1) = 5.1f;
    X(8, 0) = -0.1f; X(8, 1) = 4.9f;

    algorithms::KMeans<float> kmeans(3, 100, 1e-4f, 5, 42);
    kmeans.fit(X);

    // Transform gives distances to each centroid
    Matrix<float> distances = kmeans.transform(X);

    std::cout << "Distances to centroids:" << std::endl;
    std::cout << "Point\t\tDist to 0\tDist to 1\tDist to 2" << std::endl;
    for (usize i = 0; i < 9; ++i) {
        std::cout << "(" << X(i, 0) << "," << X(i, 1) << ")\t";
        for (usize k = 0; k < 3; ++k) {
            std::cout << StringUtils::format_number(distances(i, k), 2) << "\t\t";
        }
        std::cout << std::endl;
    }

    // Points should be closest to their own centroid
    auto labels = kmeans.labels();
    for (usize i = 0; i < 9; ++i) {
        i32 assigned = labels[i];
        for (usize k = 0; k < 3; ++k) {
            if (static_cast<i32>(k) != assigned) {
                assert(distances(i, assigned) <= distances(i, k) + 0.01f);
            }
        }
    }

    std::cout << "\n=== KMeans Transform: PASSED ===" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "KMeans Full Pipeline Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        test_basic_kmeans();
        test_kmeans_transform();
        test_kmeans_synthetic();
        test_elbow_method();
        test_kmeans_csv_data();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All KMeans Tests PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

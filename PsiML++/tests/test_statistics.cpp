#include "../include/math/linalg/statistics.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::math;
using namespace psi::math::ops;

constexpr float TOL = 1e-5f;

template<typename T>
bool approx_equal(T a, T b, T tol = TOL) {
    return std::abs(a - b) < tol;
}

void test_vector_mean() {
    std::cout << "Testing vector mean..." << std::endl;

    Vector<float> v({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    float m = mean(v);
    assert(approx_equal(m, 3.0f));

    std::cout << "  Vector mean: PASSED" << std::endl;
}

void test_vector_variance() {
    std::cout << "Testing vector variance..." << std::endl;

    Vector<float> v({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    float var = variance(v);  // Sample variance
    // Var = sum((x_i - mean)^2) / (n-1) = ((4+1+0+1+4)/4) = 2.5
    assert(approx_equal(var, 2.5f));

    std::cout << "  Vector variance: PASSED" << std::endl;
}

void test_vector_stddev() {
    std::cout << "Testing vector standard deviation..." << std::endl;

    Vector<float> v({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    float std = stddev(v);
    assert(approx_equal(std, std::sqrt(2.5f)));

    std::cout << "  Vector stddev: PASSED" << std::endl;
}

void test_vector_median() {
    std::cout << "Testing vector median..." << std::endl;

    Vector<float> v1({1.0f, 3.0f, 2.0f, 5.0f, 4.0f});
    float median1 = median(v1);
    assert(approx_equal(median1, 3.0f));

    Vector<float> v2({1.0f, 2.0f, 3.0f, 4.0f});
    float median2 = median(v2);
    assert(approx_equal(median2, 2.5f));

    std::cout << "  Vector median: PASSED" << std::endl;
}

void test_vector_percentile() {
    std::cout << "Testing vector percentile..." << std::endl;

    Vector<float> v({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    double p25 = percentile(v, 25.0);
    double p50 = percentile(v, 50.0);
    double p75 = percentile(v, 75.0);

    assert(approx_equal(static_cast<float>(p25), 2.0f));
    assert(approx_equal(static_cast<float>(p50), 3.0f));
    assert(approx_equal(static_cast<float>(p75), 4.0f));

    std::cout << "  Vector percentile: PASSED" << std::endl;
}

void test_matrix_mean() {
    std::cout << "Testing matrix mean..." << std::endl;

    Matrix<float> m(2, 3);
    m(0, 0) = 1.0f; m(0, 1) = 2.0f; m(0, 2) = 3.0f;
    m(1, 0) = 4.0f; m(1, 1) = 5.0f; m(1, 2) = 6.0f;

    float mean_val = mean(m);
    assert(approx_equal(mean_val, 3.5f));

    std::cout << "  Matrix mean: PASSED" << std::endl;
}

void test_mean_axis() {
    std::cout << "Testing mean along axis..." << std::endl;

    Matrix<float> m(2, 3);
    m(0, 0) = 1.0f; m(0, 1) = 2.0f; m(0, 2) = 3.0f;
    m(1, 0) = 4.0f; m(1, 1) = 5.0f; m(1, 2) = 6.0f;

    // Mean along rows (axis=0): [2.5, 3.5, 4.5]
    Vector<float> mean_rows = mean_axis(m, 0);
    assert(mean_rows.size() == 3);
    assert(approx_equal(mean_rows[0], 2.5f));
    assert(approx_equal(mean_rows[1], 3.5f));
    assert(approx_equal(mean_rows[2], 4.5f));

    // Mean along columns (axis=1): [2.0, 5.0]
    Vector<float> mean_cols = mean_axis(m, 1);
    assert(mean_cols.size() == 2);
    assert(approx_equal(mean_cols[0], 2.0f));
    assert(approx_equal(mean_cols[1], 5.0f));

    std::cout << "  Mean along axis: PASSED" << std::endl;
}

void test_covariance() {
    std::cout << "Testing covariance..." << std::endl;

    Vector<float> x({1.0f, 2.0f, 3.0f});
    Vector<float> y({2.0f, 4.0f, 6.0f});

    float cov = covariance(x, y);
    assert(cov > 0.0f);  // Positive covariance

    std::cout << "  Covariance: PASSED" << std::endl;
}

void test_correlation() {
    std::cout << "Testing correlation..." << std::endl;

    Vector<double> x({1.0, 2.0, 3.0, 4.0, 5.0});
    Vector<double> y({2.0, 4.0, 6.0, 8.0, 10.0});

    double corr = correlation(x, y);
    assert(approx_equal(static_cast<float>(corr), 1.0f, 1e-4f));  // Perfect correlation

    std::cout << "  Correlation: PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PsiML++ Statistics Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        test_vector_mean();
        test_vector_variance();
        test_vector_stddev();
        test_vector_median();
        test_vector_percentile();
        test_matrix_mean();
        test_mean_axis();
        test_covariance();
        test_correlation();

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "All statistics tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

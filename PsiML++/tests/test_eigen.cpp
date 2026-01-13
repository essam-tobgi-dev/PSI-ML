#include "../include/math/linalg/eigen.h"
#include <iostream>
#include <cassert>

using namespace psi::math;
using namespace psi::math::linalg;

constexpr float TOL = 1e-3f;

bool approx_equal(float a, float b) {
    return std::abs(a - b) < TOL;
}

void test_power_iteration() {
    std::cout << "Testing power iteration..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=2; A(0,1)=1;
    A(1,0)=1; A(1,1)=2;

    auto [eigenvalue, eigenvector] = power_iteration(A, 100, 1e-6f);
    assert(eigenvalue > 2.5f && eigenvalue < 3.5f);  // Dominant eigenvalue ~3
    std::cout << "  Power iteration: PASSED" << std::endl;
}

void test_inverse_power_iteration() {
    std::cout << "Testing inverse power iteration..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=4; A(0,1)=1;
    A(1,0)=1; A(1,1)=3;

    auto [eigenvalue, eigenvector] = inverse_power_iteration(A, 100, 1e-6f);
    // Smallest eigenvalue should be around 2.38
    assert(eigenvalue > 2.0f && eigenvalue < 3.0f);
    std::cout << "  Inverse power iteration: PASSED" << std::endl;
}

void test_jacobi_eigenvalue() {
    std::cout << "Testing Jacobi eigenvalue..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=4; A(0,1)=1;
    A(1,0)=1; A(1,1)=3;

    auto result = jacobi_eigenvalue(A, 100, 1e-6f);
    assert(result.converged);
    assert(result.eigenvalues.size() == 2);

    // Eigenvalues should be approximately 4.618 and 2.382
    assert(result.eigenvalues[0] > 4.0f && result.eigenvalues[0] < 5.0f);
    assert(result.eigenvalues[1] > 2.0f && result.eigenvalues[1] < 3.0f);

    std::cout << "  Jacobi eigenvalue: PASSED" << std::endl;
}

void test_qr_algorithm() {
    std::cout << "Testing QR algorithm..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=2; A(0,1)=1;
    A(1,0)=1; A(1,1)=2;

    auto result = qr_algorithm(A, 100, 1e-6f);
    assert(result.converged);
    assert(result.eigenvalues.size() == 2);

    std::cout << "  QR algorithm: PASSED" << std::endl;
}

void test_rayleigh_quotient() {
    std::cout << "Testing Rayleigh quotient..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=2; A(0,1)=0;
    A(1,0)=0; A(1,1)=3;

    Vector<float> x({1, 0});
    float rq = rayleigh_quotient(A, x);
    assert(approx_equal(rq, 2.0f));
    std::cout << "  Rayleigh quotient: PASSED" << std::endl;
}

void test_is_positive_definite() {
    std::cout << "Testing is_positive_definite..." << std::endl;

    // Positive definite matrix
    Matrix<float> A(2, 2);
    A(0,0)=4; A(0,1)=1;
    A(1,0)=1; A(1,1)=3;

    bool pd = is_positive_definite(A, 1e-6f);
    assert(pd);

    // Non-positive definite matrix
    Matrix<float> B(2, 2);
    B(0,0)=1; B(0,1)=2;
    B(1,0)=2; B(1,1)=1;

    bool not_pd = is_positive_definite(B, 1e-6f);
    assert(!not_pd);

    std::cout << "  Is positive definite: PASSED" << std::endl;
}

void test_hessenberg_reduction() {
    std::cout << "Testing Hessenberg reduction..." << std::endl;

    Matrix<float> A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=2; A(1,1)=3; A(1,2)=4;
    A(2,0)=3; A(2,1)=4; A(2,2)=5;

    Matrix<float> Q = Matrix<float>::identity(3, 0);
    hessenberg_reduction(A, Q);

    // Check that A is in Hessenberg form (elements below first subdiagonal are zero)
    assert(std::abs(A(2,0)) < TOL);

    std::cout << "  Hessenberg reduction: PASSED" << std::endl;
}

void test_pca() {
    std::cout << "Testing PCA..." << std::endl;

    // Create simple 2D data
    Matrix<float> data(3, 2);
    data(0,0)=1; data(0,1)=2;
    data(1,0)=2; data(1,1)=3;
    data(2,0)=3; data(2,1)=4;

    auto result = pca(data, true);

    // Check that we have 2 principal components
    assert(result.principal_components.rows() == 2);
    assert(result.principal_components.cols() == 2);
    assert(result.explained_variance.size() == 2);
    assert(result.explained_variance_ratio.size() == 2);

    // Variance ratios should sum to 1
    float sum_ratio = result.explained_variance_ratio.sum();
    assert(approx_equal(sum_ratio, 1.0f));

    std::cout << "  PCA: PASSED" << std::endl;
}

void test_generalized_eigenvalue() {
    std::cout << "Testing generalized eigenvalue..." << std::endl;

    // A*x = lambda*B*x
    Matrix<float> A(2, 2);
    A(0,0)=6; A(0,1)=2;
    A(1,0)=2; A(1,1)=3;

    Matrix<float> B(2, 2);
    B(0,0)=4; B(0,1)=1;
    B(1,0)=1; B(1,1)=3;

    auto result = generalized_eigenvalue(A, B, 100, 1e-6f);
    assert(result.converged);
    assert(result.eigenvalues.size() == 2);

    std::cout << "  Generalized eigenvalue: PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PsiML++ Eigenvalue Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        test_power_iteration();
        test_inverse_power_iteration();
        test_jacobi_eigenvalue();
        test_qr_algorithm();
        test_rayleigh_quotient();
        test_is_positive_definite();
        test_hessenberg_reduction();
        test_pca();
        test_generalized_eigenvalue();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All eigenvalue tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

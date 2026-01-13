#include "../include/math/linalg/solvers.h"
#include <iostream>
#include <cassert>

using namespace psi::math;
using namespace psi::math::linalg;

constexpr float TOL = 1e-4f;

bool approx_equal(float a, float b) {
    return std::abs(a - b) < TOL;
}

void test_lu_solve() {
    std::cout << "Testing LU solve..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=3; A(0,1)=1;
    A(1,0)=1; A(1,1)=2;

    Vector<float> b({9, 8});

    auto x = solve(A, b);
    assert(approx_equal(x[0], 2.0f));
    assert(approx_equal(x[1], 3.0f));
    std::cout << "  LU solve: PASSED" << std::endl;
}

void test_cholesky_solve() {
    std::cout << "Testing Cholesky solve..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=4; A(0,1)=2;
    A(1,0)=2; A(1,1)=3;

    Vector<float> b({14, 11});

    auto chol = cholesky_decomposition(A);
    auto x = cholesky_solve(chol, b);
    assert(approx_equal(x[0], 2.0f));
    assert(approx_equal(x[1], 3.0f));
    std::cout << "  Cholesky solve: PASSED" << std::endl;
}

void test_conjugate_gradient() {
    std::cout << "Testing conjugate gradient..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=4; A(0,1)=1;
    A(1,0)=1; A(1,1)=3;

    Vector<float> b({1, 2});
    Vector<float> x0({0, 0});

    auto result = conjugate_gradient(A, b, x0, 100, 1e-6f);
    assert(result.converged);
    std::cout << "  Conjugate gradient: PASSED" << std::endl;
}

void test_determinant() {
    std::cout << "Testing determinant..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=3; A(0,1)=7;
    A(1,0)=1; A(1,1)=5;

    float det = determinant(A);
    assert(approx_equal(det, 8.0f));  // 3*5 - 7*1 = 8
    std::cout << "  Determinant: PASSED" << std::endl;
}

void test_invert() {
    std::cout << "Testing matrix inversion..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=4; A(0,1)=7;
    A(1,0)=2; A(1,1)=6;

    auto A_inv = invert(A);
    auto I = linalg::matmul(A, A_inv);

    assert(approx_equal(I(0,0), 1.0f));
    assert(approx_equal(I(1,1), 1.0f));
    assert(approx_equal(I(0,1), 0.0f));
    std::cout << "  Matrix inversion: PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PsiML++ Solvers Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        test_lu_solve();
        test_cholesky_solve();
        test_conjugate_gradient();
        test_determinant();
        test_invert();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All solvers tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

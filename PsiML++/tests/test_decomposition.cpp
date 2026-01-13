#include "../include/math/linalg/decomposition.h"
#include <iostream>
#include <cassert>

using namespace psi::math;
using namespace psi::math::linalg;

constexpr float TOL = 1e-4f;

bool approx_equal(float a, float b) {
    return std::abs(a - b) < TOL;
}

void test_lu_decomposition() {
    std::cout << "Testing LU decomposition..." << std::endl;

    Matrix<float> A(3, 3);
    A(0,0)=2; A(0,1)=1; A(0,2)=1;
    A(1,0)=4; A(1,1)=3; A(1,2)=3;
    A(2,0)=8; A(2,1)=7; A(2,2)=9;

    auto lu = lu_decomposition(A);
    assert(!lu.is_singular);
    std::cout << "  LU decomposition: PASSED" << std::endl;
}

void test_qr_decomposition() {
    std::cout << "Testing QR decomposition..." << std::endl;

    Matrix<float> A(3, 2);
    A(0,0)=1; A(0,1)=1;
    A(1,0)=1; A(1,1)=2;
    A(2,0)=1; A(2,1)=3;

    auto qr = qr_decomposition(A);
    assert(qr.Q.rows() == 3 && qr.Q.cols() == 3);
    assert(qr.R.rows() == 3 && qr.R.cols() == 2);
    std::cout << "  QR decomposition: PASSED" << std::endl;
}

void test_cholesky_decomposition() {
    std::cout << "Testing Cholesky decomposition..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=4; A(0,1)=2;
    A(1,0)=2; A(1,1)=3;

    auto chol = cholesky_decomposition(A);
    assert(chol.is_positive_definite);
    std::cout << "  Cholesky decomposition: PASSED" << std::endl;
}

void test_svd_decomposition() {
    std::cout << "Testing SVD decomposition..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=3; A(0,1)=2;
    A(1,0)=2; A(1,1)=3;

    auto svd = svd_decomposition(A, 100, 1e-6f);
    assert(svd.S.size() == 2);
    assert(svd.S[0] > 0.0f);
    std::cout << "  SVD decomposition: PASSED" << std::endl;
}

void test_matrix_rank() {
    std::cout << "Testing matrix rank..." << std::endl;

    Matrix<float> A(2, 2);
    A(0,0)=1; A(0,1)=2;
    A(1,0)=2; A(1,1)=4;  // Rank 1 matrix

    auto rank = matrix_rank(A, 0.1f);
    assert(rank == 1);
    std::cout << "  Matrix rank: PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PsiML++ Decomposition Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        test_lu_decomposition();
        test_qr_decomposition();
        test_cholesky_decomposition();
        test_svd_decomposition();
        test_matrix_rank();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All decomposition tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

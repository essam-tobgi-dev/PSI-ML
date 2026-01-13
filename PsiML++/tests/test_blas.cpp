#include "../include/math/linalg/blas.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::math;
using namespace psi::math::linalg;
using namespace psi::core;

// Tolerance for floating-point comparisons
constexpr float FLOAT_TOL = 1e-5f;
constexpr double DOUBLE_TOL = 1e-10;

template<typename T>
bool approx_equal(T a, T b, T tolerance) {
    return std::abs(a - b) < tolerance;
}

void test_dot_product() {
    std::cout << "Testing dot product..." << std::endl;

    Vector<float> x({1.0f, 2.0f, 3.0f});
    Vector<float> y({4.0f, 5.0f, 6.0f});

    float result = dot(x, y);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert(approx_equal(result, 32.0f, FLOAT_TOL));

    Vector<double> xd({1.0, 2.0, 3.0, 4.0});
    Vector<double> yd({2.0, 3.0, 4.0, 5.0});
    double resultd = dot(xd, yd);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    assert(approx_equal(resultd, 40.0, DOUBLE_TOL));

    std::cout << "  Dot product: PASSED" << std::endl;
}

void test_norm() {
    std::cout << "Testing vector norm..." << std::endl;

    Vector<float> x({3.0f, 4.0f});
    float result = norm(x);
    // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    assert(approx_equal(result, 5.0f, FLOAT_TOL));

    Vector<double> xd({1.0, 2.0, 2.0});
    double resultd = norm(xd);
    // sqrt(1 + 4 + 4) = sqrt(9) = 3
    assert(approx_equal(resultd, 3.0, DOUBLE_TOL));

    std::cout << "  Norm: PASSED" << std::endl;
}

void test_asum() {
    std::cout << "Testing asum (sum of absolute values)..." << std::endl;

    Vector<float> x({1.0f, -2.0f, 3.0f, -4.0f});
    float result = asum(x);
    // |1| + |-2| + |3| + |-4| = 1 + 2 + 3 + 4 = 10
    assert(approx_equal(result, 10.0f, FLOAT_TOL));

    std::cout << "  Asum: PASSED" << std::endl;
}

void test_iamax() {
    std::cout << "Testing iamax (index of max absolute value)..." << std::endl;

    Vector<float> x({1.0f, -5.0f, 3.0f, -4.0f});
    index_t idx = iamax(x);
    // Max absolute value is |-5| = 5 at index 1
    assert(idx == 1);

    Vector<double> xd({2.0, 3.0, -8.0, 1.0});
    index_t idxd = iamax(xd);
    // Max absolute value is |-8| = 8 at index 2
    assert(idxd == 2);

    std::cout << "  Iamax: PASSED" << std::endl;
}

void test_scal() {
    std::cout << "Testing scalar multiplication..." << std::endl;

    Vector<float> x({1.0f, 2.0f, 3.0f});
    Vector<float> result = scal(2.0f, x);

    assert(approx_equal(result[0], 2.0f, FLOAT_TOL));
    assert(approx_equal(result[1], 4.0f, FLOAT_TOL));
    assert(approx_equal(result[2], 6.0f, FLOAT_TOL));

    std::cout << "  Scal: PASSED" << std::endl;
}

void test_vector_add_sub() {
    std::cout << "Testing vector addition and subtraction..." << std::endl;

    Vector<float> x({1.0f, 2.0f, 3.0f});
    Vector<float> y({4.0f, 5.0f, 6.0f});

    // Addition
    Vector<float> sum = add(x, y);
    assert(approx_equal(sum[0], 5.0f, FLOAT_TOL));
    assert(approx_equal(sum[1], 7.0f, FLOAT_TOL));
    assert(approx_equal(sum[2], 9.0f, FLOAT_TOL));

    // Subtraction
    Vector<float> diff = sub(y, x);
    assert(approx_equal(diff[0], 3.0f, FLOAT_TOL));
    assert(approx_equal(diff[1], 3.0f, FLOAT_TOL));
    assert(approx_equal(diff[2], 3.0f, FLOAT_TOL));

    std::cout << "  Vector add/sub: PASSED" << std::endl;
}

void test_axpy() {
    std::cout << "Testing AXPY (y = alpha*x + y)..." << std::endl;

    Vector<float> x({1.0f, 2.0f, 3.0f});
    Vector<float> y({4.0f, 5.0f, 6.0f});

    axpy(2.0f, x, y);
    // y = 2*x + y = 2*[1,2,3] + [4,5,6] = [2,4,6] + [4,5,6] = [6,9,12]
    assert(approx_equal(y[0], 6.0f, FLOAT_TOL));
    assert(approx_equal(y[1], 9.0f, FLOAT_TOL));
    assert(approx_equal(y[2], 12.0f, FLOAT_TOL));

    std::cout << "  AXPY: PASSED" << std::endl;
}

void test_copy_swap() {
    std::cout << "Testing copy and swap..." << std::endl;

    Vector<float> x({1.0f, 2.0f, 3.0f});
    Vector<float> y(3);

    // Copy
    copy(x, y);
    assert(approx_equal(y[0], 1.0f, FLOAT_TOL));
    assert(approx_equal(y[1], 2.0f, FLOAT_TOL));
    assert(approx_equal(y[2], 3.0f, FLOAT_TOL));

    // Swap
    Vector<float> a({10.0f, 20.0f, 30.0f});
    Vector<float> b({40.0f, 50.0f, 60.0f});
    swap(a, b);
    assert(approx_equal(a[0], 40.0f, FLOAT_TOL));
    assert(approx_equal(b[0], 10.0f, FLOAT_TOL));

    std::cout << "  Copy and swap: PASSED" << std::endl;
}

void test_matvec() {
    std::cout << "Testing matrix-vector multiplication..." << std::endl;

    Matrix<float> A(2, 3);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    Vector<float> x({1.0f, 2.0f, 3.0f});

    Vector<float> y = matvec(A, x);
    // [1,2,3] * [1,2,3]^T = 1*1 + 2*2 + 3*3 = 14
    // [4,5,6] * [1,2,3]^T = 4*1 + 5*2 + 6*3 = 32
    assert(y.size() == 2);
    assert(approx_equal(y[0], 14.0f, FLOAT_TOL));
    assert(approx_equal(y[1], 32.0f, FLOAT_TOL));

    std::cout << "  Matrix-vector multiplication: PASSED" << std::endl;
}

void test_matvec_trans() {
    std::cout << "Testing matrix-vector multiplication with transpose..." << std::endl;

    Matrix<float> A(2, 3);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    Vector<float> x({1.0f, 2.0f});

    Vector<float> y = matvec_trans(A, x);
    // A^T * x:
    // [1,4] * [1,2]^T = 1*1 + 4*2 = 9
    // [2,5] * [1,2]^T = 2*1 + 5*2 = 12
    // [3,6] * [1,2]^T = 3*1 + 6*2 = 15
    assert(y.size() == 3);
    assert(approx_equal(y[0], 9.0f, FLOAT_TOL));
    assert(approx_equal(y[1], 12.0f, FLOAT_TOL));
    assert(approx_equal(y[2], 15.0f, FLOAT_TOL));

    std::cout << "  Matrix-vector transpose multiplication: PASSED" << std::endl;
}

void test_gemv() {
    std::cout << "Testing GEMV (general matrix-vector multiplication)..." << std::endl;

    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    Vector<float> x({1.0f, 1.0f});
    Vector<float> y({1.0f, 1.0f});

    // y = 2*A*x + 3*y
    Vector<float> result = gemv(false, 2.0f, A, x, 3.0f, y);
    // A*x = [1+2, 3+4] = [3, 7]
    // 2*[3,7] + 3*[1,1] = [6,14] + [3,3] = [9,17]
    assert(approx_equal(result[0], 9.0f, FLOAT_TOL));
    assert(approx_equal(result[1], 17.0f, FLOAT_TOL));

    std::cout << "  GEMV: PASSED" << std::endl;
}

void test_ger() {
    std::cout << "Testing GER (rank-1 update)..." << std::endl;

    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    Vector<float> x({1.0f, 2.0f});
    Vector<float> y({3.0f, 4.0f});

    // A = A + 1.0 * x * y^T
    ger(1.0f, x, y, A);
    // x*y^T = [1,2]^T * [3,4] = [[3,4], [6,8]]
    // A + x*y^T = [[1,2],[3,4]] + [[3,4],[6,8]] = [[4,6],[9,12]]
    assert(approx_equal(A(0, 0), 4.0f, FLOAT_TOL));
    assert(approx_equal(A(0, 1), 6.0f, FLOAT_TOL));
    assert(approx_equal(A(1, 0), 9.0f, FLOAT_TOL));
    assert(approx_equal(A(1, 1), 12.0f, FLOAT_TOL));

    std::cout << "  GER: PASSED" << std::endl;
}

void test_matmul() {
    std::cout << "Testing matrix-matrix multiplication..." << std::endl;

    Matrix<float> A(2, 3);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    Matrix<float> B(3, 2);
    B(0, 0) = 7.0f;  B(0, 1) = 8.0f;
    B(1, 0) = 9.0f;  B(1, 1) = 10.0f;
    B(2, 0) = 11.0f; B(2, 1) = 12.0f;

    Matrix<float> C = linalg::matmul(A, B);
    // C(0,0) = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C(0,1) = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C(1,0) = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C(1,1) = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

    assert(C.rows() == 2 && C.cols() == 2);
    assert(approx_equal(C(0, 0), 58.0f, FLOAT_TOL));
    assert(approx_equal(C(0, 1), 64.0f, FLOAT_TOL));
    assert(approx_equal(C(1, 0), 139.0f, FLOAT_TOL));
    assert(approx_equal(C(1, 1), 154.0f, FLOAT_TOL));

    std::cout << "  Matrix multiplication: PASSED" << std::endl;
}

void test_gemm() {
    std::cout << "Testing GEMM (general matrix-matrix multiplication)..." << std::endl;

    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    Matrix<float> B(2, 2);
    B(0, 0) = 5.0f; B(0, 1) = 6.0f;
    B(1, 0) = 7.0f; B(1, 1) = 8.0f;

    Matrix<float> C(2, 2);
    C(0, 0) = 1.0f; C(0, 1) = 1.0f;
    C(1, 0) = 1.0f; C(1, 1) = 1.0f;

    // C = 2*A*B + 3*C
    Matrix<float> result = gemm(false, false, 2.0f, A, B, 3.0f, C);
    // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //     = [[19, 22], [43, 50]]
    // 2*A*B + 3*C = 2*[[19,22],[43,50]] + 3*[[1,1],[1,1]]
    //              = [[38,44],[86,100]] + [[3,3],[3,3]]
    //              = [[41,47],[89,103]]

    assert(approx_equal(result(0, 0), 41.0f, FLOAT_TOL));
    assert(approx_equal(result(0, 1), 47.0f, FLOAT_TOL));
    assert(approx_equal(result(1, 0), 89.0f, FLOAT_TOL));
    assert(approx_equal(result(1, 1), 103.0f, FLOAT_TOL));

    std::cout << "  GEMM: PASSED" << std::endl;
}

void test_matrix_add_sub() {
    std::cout << "Testing matrix addition and subtraction..." << std::endl;

    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    Matrix<float> B(2, 2);
    B(0, 0) = 5.0f; B(0, 1) = 6.0f;
    B(1, 0) = 7.0f; B(1, 1) = 8.0f;

    // Addition
    Matrix<float> sum = matadd(A, B);
    assert(approx_equal(sum(0, 0), 6.0f, FLOAT_TOL));
    assert(approx_equal(sum(0, 1), 8.0f, FLOAT_TOL));
    assert(approx_equal(sum(1, 0), 10.0f, FLOAT_TOL));
    assert(approx_equal(sum(1, 1), 12.0f, FLOAT_TOL));

    // Subtraction
    Matrix<float> diff = matsub(B, A);
    assert(approx_equal(diff(0, 0), 4.0f, FLOAT_TOL));
    assert(approx_equal(diff(0, 1), 4.0f, FLOAT_TOL));
    assert(approx_equal(diff(1, 0), 4.0f, FLOAT_TOL));
    assert(approx_equal(diff(1, 1), 4.0f, FLOAT_TOL));

    std::cout << "  Matrix add/sub: PASSED" << std::endl;
}

void test_transpose() {
    std::cout << "Testing matrix transpose..." << std::endl;

    Matrix<float> A(2, 3);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    Matrix<float> At = transpose(A);

    assert(At.rows() == 3 && At.cols() == 2);
    assert(approx_equal(At(0, 0), 1.0f, FLOAT_TOL));
    assert(approx_equal(At(0, 1), 4.0f, FLOAT_TOL));
    assert(approx_equal(At(1, 0), 2.0f, FLOAT_TOL));
    assert(approx_equal(At(1, 1), 5.0f, FLOAT_TOL));
    assert(approx_equal(At(2, 0), 3.0f, FLOAT_TOL));
    assert(approx_equal(At(2, 1), 6.0f, FLOAT_TOL));

    std::cout << "  Matrix transpose: PASSED" << std::endl;
}

void test_matscal() {
    std::cout << "Testing matrix scalar multiplication..." << std::endl;

    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    Matrix<float> result = matscal(2.0f, A);

    assert(approx_equal(result(0, 0), 2.0f, FLOAT_TOL));
    assert(approx_equal(result(0, 1), 4.0f, FLOAT_TOL));
    assert(approx_equal(result(1, 0), 6.0f, FLOAT_TOL));
    assert(approx_equal(result(1, 1), 8.0f, FLOAT_TOL));

    std::cout << "  Matrix scalar multiplication: PASSED" << std::endl;
}

void test_outer_product() {
    std::cout << "Testing outer product..." << std::endl;

    Vector<float> x({1.0f, 2.0f, 3.0f});
    Vector<float> y({4.0f, 5.0f});

    Matrix<float> result = outer(x, y);
    // [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
    // = [[4, 5], [8, 10], [12, 15]]

    assert(result.rows() == 3 && result.cols() == 2);
    assert(approx_equal(result(0, 0), 4.0f, FLOAT_TOL));
    assert(approx_equal(result(0, 1), 5.0f, FLOAT_TOL));
    assert(approx_equal(result(1, 0), 8.0f, FLOAT_TOL));
    assert(approx_equal(result(1, 1), 10.0f, FLOAT_TOL));
    assert(approx_equal(result(2, 0), 12.0f, FLOAT_TOL));
    assert(approx_equal(result(2, 1), 15.0f, FLOAT_TOL));

    std::cout << "  Outer product: PASSED" << std::endl;
}

void test_trace() {
    std::cout << "Testing matrix trace..." << std::endl;

    Matrix<float> A(3, 3);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;
    A(2, 0) = 7.0f; A(2, 1) = 8.0f; A(2, 2) = 9.0f;

    float tr = trace(A);
    // trace = 1 + 5 + 9 = 15
    assert(approx_equal(tr, 15.0f, FLOAT_TOL));

    std::cout << "  Matrix trace: PASSED" << std::endl;
}

void test_frobenius_norm() {
    std::cout << "Testing Frobenius norm..." << std::endl;

    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 2.0f; A(1, 1) = 2.0f;

    float fn = frobenius_norm(A);
    // sqrt(1^2 + 2^2 + 2^2 + 2^2) = sqrt(1 + 4 + 4 + 4) = sqrt(13)
    assert(approx_equal(fn, std::sqrt(13.0f), FLOAT_TOL));

    std::cout << "  Frobenius norm: PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PsiML++ BLAS Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        // Level 1 tests
        std::cout << "Level 1 BLAS (Vector-Vector Operations):" << std::endl;
        test_dot_product();
        test_norm();
        test_asum();
        test_iamax();
        test_scal();
        test_vector_add_sub();
        test_axpy();
        test_copy_swap();
        std::cout << std::endl;

        // Level 2 tests
        std::cout << "Level 2 BLAS (Matrix-Vector Operations):" << std::endl;
        test_matvec();
        test_matvec_trans();
        test_gemv();
        test_ger();
        std::cout << std::endl;

        // Level 3 tests
        std::cout << "Level 3 BLAS (Matrix-Matrix Operations):" << std::endl;
        test_matmul();
        test_gemm();
        test_matrix_add_sub();
        test_transpose();
        test_matscal();
        std::cout << std::endl;

        // Utility tests
        std::cout << "Utility Functions:" << std::endl;
        test_outer_product();
        test_trace();
        test_frobenius_norm();
        std::cout << std::endl;

        std::cout << "========================================" << std::endl;
        std::cout << "All BLAS tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

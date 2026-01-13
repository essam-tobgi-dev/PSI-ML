#include "../include/math/matrix.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::math;
using namespace psi::core;

void test_matrix_construction() {
    std::cout << "Testing Matrix construction..." << std::endl;

    // Default constructor
    Matrix<float> m1;
    assert(m1.rows() == 0);
    assert(m1.cols() == 0);
    assert(m1.empty());

    // Size constructor
    Matrix<float> m2(3, 4);
    assert(m2.rows() == 3);
    assert(m2.cols() == 4);
    assert(!m2.empty());
    assert(m2.size() == 12);

    // Size with value constructor
    Matrix<float> m3(2, 3, 5.0f);
    assert(m3(0, 0) == 5.0f);
    assert(m3(1, 2) == 5.0f);

    // Initializer list constructor
    Matrix<float> m4 = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    assert(m4.rows() == 2);
    assert(m4.cols() == 3);
    assert(m4(0, 0) == 1.0f);
    assert(m4(1, 2) == 6.0f);

    // Copy constructor
    Matrix<float> m5(m4);
    assert(m5.rows() == m4.rows());
    assert(m5(0, 0) == m4(0, 0));

    std::cout << "  Matrix construction: PASSED" << std::endl;
}

void test_matrix_element_access() {
    std::cout << "Testing Matrix element access..." << std::endl;

    Matrix<int> m = {
        {1, 2, 3},
        {4, 5, 6}
    };

    // Operator()
    assert(m(0, 0) == 1);
    assert(m(1, 2) == 6);

    // at()
    assert(m.at(0, 1) == 2);

    // Flat indexing operator[]
    assert(m[0] == 1);
    assert(m[5] == 6);

    // Modify elements
    m(1, 1) = 10;
    assert(m(1, 1) == 10);

    // data()
    int* ptr = m.data();
    assert(ptr != nullptr);
    assert(ptr[0] == 1);

    std::cout << "  Matrix element access: PASSED" << std::endl;
}

void test_matrix_properties() {
    std::cout << "Testing Matrix properties..." << std::endl;

    Matrix<float> m1(3, 3);
    assert(m1.is_square());

    Matrix<float> m2(3, 4);
    assert(!m2.is_square());

    assert(m1.rows() == 3);
    assert(m1.cols() == 3);
    assert(m1.size() == 9);

    std::cout << "  Matrix properties: PASSED" << std::endl;
}

void test_matrix_row_col_operations() {
    std::cout << "Testing Matrix row/column operations..." << std::endl;

    Matrix<float> m = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    // get_row()
    Vector<float> row1 = m.get_row(1);
    assert(row1.size() == 3);
    assert(row1[0] == 4.0f);
    assert(row1[2] == 6.0f);

    // get_col()
    Vector<float> col1 = m.get_col(1);
    assert(col1.size() == 2);
    assert(col1[0] == 2.0f);
    assert(col1[1] == 5.0f);

    // set_row()
    Vector<float> new_row = {7.0f, 8.0f, 9.0f};
    m.set_row(0, new_row);
    assert(m(0, 0) == 7.0f);
    assert(m(0, 2) == 9.0f);

    // set_col()
    Vector<float> new_col = {10.0f, 11.0f};
    m.set_col(1, new_col);
    assert(m(0, 1) == 10.0f);
    assert(m(1, 1) == 11.0f);

    std::cout << "  Matrix row/column operations: PASSED" << std::endl;
}

void test_matrix_modifiers() {
    std::cout << "Testing Matrix modifiers..." << std::endl;

    Matrix<float> m(2, 2, 1.0f);

    // fill()
    m.fill(3.0f);
    assert(m(0, 0) == 3.0f);
    assert(m(1, 1) == 3.0f);

    // resize()
    m.resize(3, 4, 5.0f);
    assert(m.rows() == 3);
    assert(m.cols() == 4);
    assert(m(2, 3) == 5.0f);

    // clear()
    m.clear();
    assert(m.empty());

    std::cout << "  Matrix modifiers: PASSED" << std::endl;
}

void test_matrix_transpose() {
    std::cout << "Testing Matrix transpose..." << std::endl;

    Matrix<int> m = {
        {1, 2, 3},
        {4, 5, 6}
    };

    // transpose()
    Matrix<int> mt = m.transpose();
    assert(mt.rows() == 3);
    assert(mt.cols() == 2);
    assert(mt(0, 0) == 1);
    assert(mt(2, 1) == 6);
    assert(mt(1, 0) == 2);

    // transpose_inplace() on square matrix
    Matrix<int> sq = {
        {1, 2},
        {3, 4}
    };
    sq.transpose_inplace();
    assert(sq(0, 1) == 3);
    assert(sq(1, 0) == 2);

    std::cout << "  Matrix transpose: PASSED" << std::endl;
}

void test_matrix_math_ops() {
    std::cout << "Testing Matrix mathematical operations..." << std::endl;

    Matrix<float> m = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    // trace()
    float trace = m.trace();
    assert(trace == 5.0f);

    // sum()
    float sum = m.sum();
    assert(sum == 10.0f);

    // mean()
    float mean = m.mean();
    assert(mean == 2.5f);

    // min() and max()
    assert(m.min() == 1.0f);
    assert(m.max() == 4.0f);

    // frobenius_norm()
    float norm = m.frobenius_norm();
    float expected = std::sqrt(1.0f + 4.0f + 9.0f + 16.0f);
    assert(std::abs(norm - expected) < 1e-5f);

    std::cout << "  Matrix mathematical operations: PASSED" << std::endl;
}

void test_matrix_arithmetic() {
    std::cout << "Testing Matrix arithmetic..." << std::endl;

    Matrix<float> m1 = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    Matrix<float> m2 = {
        {5.0f, 6.0f},
        {7.0f, 8.0f}
    };

    // Addition
    Matrix<float> m3 = m1 + m2;
    assert(m3(0, 0) == 6.0f);
    assert(m3(1, 1) == 12.0f);

    // Subtraction
    Matrix<float> m4 = m2 - m1;
    assert(m4(0, 0) == 4.0f);
    assert(m4(1, 1) == 4.0f);

    // Element-wise multiplication
    Matrix<float> m5 = m1 * m2;
    assert(m5(0, 0) == 5.0f);
    assert(m5(1, 1) == 32.0f);

    // Element-wise division
    Matrix<float> m6 = m2 / m1;
    assert(m6(0, 0) == 5.0f);
    assert(m6(1, 1) == 2.0f);

    // Negation
    Matrix<float> m7 = -m1;
    assert(m7(0, 0) == -1.0f);
    assert(m7(1, 1) == -4.0f);

    std::cout << "  Matrix arithmetic: PASSED" << std::endl;
}

void test_matrix_scalar_ops() {
    std::cout << "Testing Matrix scalar operations..." << std::endl;

    Matrix<float> m = {
        {2.0f, 4.0f},
        {6.0f, 8.0f}
    };

    // Scalar addition
    Matrix<float> m1 = m + 1.0f;
    assert(m1(0, 0) == 3.0f);

    // Scalar multiplication
    Matrix<float> m2 = m * 2.0f;
    assert(m2(0, 0) == 4.0f);
    assert(m2(1, 1) == 16.0f);

    Matrix<float> m3 = 2.0f * m;
    assert(m3(0, 0) == 4.0f);

    // Scalar division
    Matrix<float> m4 = m / 2.0f;
    assert(m4(0, 0) == 1.0f);
    assert(m4(1, 1) == 4.0f);

    std::cout << "  Matrix scalar operations: PASSED" << std::endl;
}

void test_matrix_vector_mult() {
    std::cout << "Testing Matrix-Vector multiplication..." << std::endl;

    Matrix<float> m = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    Vector<float> v = {5.0f, 6.0f};

    // Matrix * Vector
    Vector<float> result = m * v;
    assert(result.size() == 2);
    assert(result[0] == 17.0f);  // 1*5 + 2*6 = 17
    assert(result[1] == 39.0f);  // 3*5 + 4*6 = 39

    std::cout << "  Matrix-Vector multiplication: PASSED" << std::endl;
}

void test_matrix_multiplication() {
    std::cout << "Testing Matrix-Matrix multiplication..." << std::endl;

    Matrix<float> m1 = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    Matrix<float> m2 = {
        {5.0f, 6.0f},
        {7.0f, 8.0f}
    };

    // Matrix multiplication using matmul()
    Matrix<float> result = matmul(m1, m2);
    assert(result.rows() == 2);
    assert(result.cols() == 2);
    assert(result(0, 0) == 19.0f);  // 1*5 + 2*7 = 19
    assert(result(0, 1) == 22.0f);  // 1*6 + 2*8 = 22
    assert(result(1, 0) == 43.0f);  // 3*5 + 4*7 = 43
    assert(result(1, 1) == 50.0f);  // 3*6 + 4*8 = 50

    std::cout << "  Matrix-Matrix multiplication: PASSED" << std::endl;
}

void test_matrix_factory_methods() {
    std::cout << "Testing Matrix factory methods..." << std::endl;

    // zeros()
    Matrix<float> m1 = Matrix<float>::zeros(2, 3);
    assert(m1.rows() == 2);
    assert(m1.cols() == 3);
    assert(m1(0, 0) == 0.0f);
    assert(m1(1, 2) == 0.0f);

    // ones()
    Matrix<float> m2 = Matrix<float>::ones(3, 2);
    assert(m2(0, 0) == 1.0f);
    assert(m2(2, 1) == 1.0f);

    // identity()
    Matrix<float> m3 = Matrix<float>::identity(3);
    assert(m3.is_square());
    assert(m3(0, 0) == 1.0f);
    assert(m3(1, 1) == 1.0f);
    assert(m3(2, 2) == 1.0f);
    assert(m3(0, 1) == 0.0f);
    assert(m3(1, 0) == 0.0f);

    // diagonal()
    Vector<float> diag_vec = {2.0f, 3.0f, 4.0f};
    Matrix<float> m4 = Matrix<float>::diagonal(diag_vec);
    assert(m4.is_square());
    assert(m4(0, 0) == 2.0f);
    assert(m4(1, 1) == 3.0f);
    assert(m4(2, 2) == 4.0f);
    assert(m4(0, 1) == 0.0f);

    std::cout << "  Matrix factory methods: PASSED" << std::endl;
}

void test_matrix_comparison() {
    std::cout << "Testing Matrix comparison..." << std::endl;

    Matrix<int> m1 = {
        {1, 2},
        {3, 4}
    };

    Matrix<int> m2 = {
        {1, 2},
        {3, 4}
    };

    Matrix<int> m3 = {
        {1, 2},
        {3, 5}
    };

    assert(m1 == m2);
    assert(m1 != m3);

    std::cout << "  Matrix comparison: PASSED" << std::endl;
}

void test_matrix_stream() {
    std::cout << "Testing Matrix stream output..." << std::endl;

    Matrix<int> m = {
        {1, 2, 3},
        {4, 5, 6}
    };

    std::cout << "  Matrix:" << std::endl;
    std::cout << "  " << m << std::endl;

    std::cout << "  Matrix stream output: PASSED" << std::endl;
}

int main() {
    std::cout << "\n=== Matrix Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_matrix_construction();
        std::cout << std::endl;

        test_matrix_element_access();
        std::cout << std::endl;

        test_matrix_properties();
        std::cout << std::endl;

        test_matrix_row_col_operations();
        std::cout << std::endl;

        test_matrix_modifiers();
        std::cout << std::endl;

        test_matrix_transpose();
        std::cout << std::endl;

        test_matrix_math_ops();
        std::cout << std::endl;

        test_matrix_arithmetic();
        std::cout << std::endl;

        test_matrix_scalar_ops();
        std::cout << std::endl;

        test_matrix_vector_mult();
        std::cout << std::endl;

        test_matrix_multiplication();
        std::cout << std::endl;

        test_matrix_factory_methods();
        std::cout << std::endl;

        test_matrix_comparison();
        std::cout << std::endl;

        test_matrix_stream();
        std::cout << std::endl;

        std::cout << "=== All Matrix Tests PASSED ===" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

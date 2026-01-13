#include "../include/math/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::math;
using namespace psi::core;

void test_tensor_construction() {
    std::cout << "Testing Tensor construction..." << std::endl;

    // Default constructor
    Tensor<float> t1;
    assert(t1.empty());
    assert(t1.ndim() == 0);

    // Shape constructor
    Tensor<float> t2({2, 3, 4});
    assert(t2.ndim() == 3);
    assert(t2.size(0) == 2);
    assert(t2.size(1) == 3);
    assert(t2.size(2) == 4);
    assert(t2.size() == 24);

    // Shape with value constructor
    Tensor<float> t3({2, 2}, 5.0f);
    assert(t3(0, 0) == 5.0f);
    assert(t3(1, 1) == 5.0f);

    // From Vector
    Vector<float> v = {1.0f, 2.0f, 3.0f};
    Tensor<float> t4(v);
    assert(t4.ndim() == 1);
    assert(t4.size() == 3);

    // From Matrix
    Matrix<float> m(2, 3, 1.0f);
    Tensor<float> t5(m);
    assert(t5.ndim() == 2);
    assert(t5.shape()[0] == 2);
    assert(t5.shape()[1] == 3);

    std::cout << "  Tensor construction: PASSED" << std::endl;
}

void test_tensor_element_access() {
    std::cout << "Testing Tensor element access..." << std::endl;

    Tensor<int> t({2, 3, 2});

    // Fill with some values using flat indexing
    for (usize i = 0; i < t.size(); ++i) {
        t[i] = static_cast<int>(i);
    }

    // Multi-dimensional indexing
    assert(t(0, 0, 0) == 0);
    assert(t(0, 0, 1) == 1);
    assert(t(1, 2, 1) == 11);

    // Flat indexing
    assert(t[0] == 0);
    assert(t[11] == 11);

    std::cout << "  Tensor element access: PASSED" << std::endl;
}

void test_tensor_shape_operations() {
    std::cout << "Testing Tensor shape operations..." << std::endl;

    Tensor<float> t({2, 3, 4});

    // Test shape and strides
    assert(t.shape().size() == 3);
    assert(t.strides().size() == 3);

    // reshape()
    Tensor<float> t2 = t.reshape({4, 6});
    assert(t2.shape()[0] == 4);
    assert(t2.shape()[1] == 6);
    assert(t2.size() == 24);

    // reshape_inplace()
    Tensor<float> t3({2, 6});
    t3.reshape_inplace({3, 4});
    assert(t3.shape()[0] == 3);
    assert(t3.shape()[1] == 4);

    // squeeze()
    Tensor<float> t4({2, 1, 3, 1});
    Tensor<float> t5 = t4.squeeze();
    assert(t5.ndim() == 2);
    assert(t5.shape()[0] == 2);
    assert(t5.shape()[1] == 3);

    // unsqueeze()
    Tensor<float> t6({2, 3});
    Tensor<float> t7 = t6.unsqueeze(1);
    assert(t7.ndim() == 3);
    assert(t7.shape()[0] == 2);
    assert(t7.shape()[1] == 1);
    assert(t7.shape()[2] == 3);

    std::cout << "  Tensor shape operations: PASSED" << std::endl;
}

void test_tensor_transpose() {
    std::cout << "Testing Tensor transpose..." << std::endl;

    Tensor<float> t({2, 3, 4});

    // Fill with sequential values
    for (usize i = 0; i < t.size(); ++i) {
        t[i] = static_cast<float>(i);
    }

    // Transpose dimensions 0 and 1
    Tensor<float> t2 = t.transpose(0, 1);
    assert(t2.shape()[0] == 3);
    assert(t2.shape()[1] == 2);
    assert(t2.shape()[2] == 4);

    // Transpose with same dimension (should return copy)
    Tensor<float> t3 = t.transpose(1, 1);
    assert(t3.shape() == t.shape());

    std::cout << "  Tensor transpose: PASSED" << std::endl;
}

void test_tensor_view_conversion() {
    std::cout << "Testing Tensor view conversions..." << std::endl;

    // as_vector()
    Tensor<float> t1({5});
    for (usize i = 0; i < 5; ++i) {
        t1[i] = static_cast<float>(i);
    }
    Vector<float> v = t1.as_vector();
    assert(v.size() == 5);
    assert(v[0] == 0.0f);
    assert(v[4] == 4.0f);

    // as_matrix()
    Tensor<float> t2({2, 3});
    for (usize i = 0; i < 6; ++i) {
        t2[i] = static_cast<float>(i);
    }
    Matrix<float> m = t2.as_matrix();
    assert(m.rows() == 2);
    assert(m.cols() == 3);
    assert(m(0, 0) == 0.0f);
    assert(m(1, 2) == 5.0f);

    std::cout << "  Tensor view conversions: PASSED" << std::endl;
}

void test_tensor_modifiers() {
    std::cout << "Testing Tensor modifiers..." << std::endl;

    Tensor<float> t({2, 2}, 1.0f);

    // fill()
    t.fill(3.0f);
    assert(t(0, 0) == 3.0f);
    assert(t(1, 1) == 3.0f);

    // resize()
    t.resize({3, 3}, 5.0f);
    assert(t.size(0) == 3);
    assert(t.size(1) == 3);
    assert(t(2, 2) == 5.0f);

    // clear()
    t.clear();
    assert(t.empty());
    assert(t.ndim() == 0);

    std::cout << "  Tensor modifiers: PASSED" << std::endl;
}

void test_tensor_math_ops() {
    std::cout << "Testing Tensor mathematical operations..." << std::endl;

    Tensor<float> t({2, 2});
    t[0] = 1.0f;
    t[1] = 2.0f;
    t[2] = 3.0f;
    t[3] = 4.0f;

    // sum()
    float sum = t.sum();
    assert(sum == 10.0f);

    // mean()
    float mean = t.mean();
    assert(mean == 2.5f);

    // min() and max()
    assert(t.min() == 1.0f);
    assert(t.max() == 4.0f);

    // norm()
    float norm = t.norm();
    float expected = std::sqrt(1.0f + 4.0f + 9.0f + 16.0f);
    assert(std::abs(norm - expected) < 1e-5f);

    std::cout << "  Tensor mathematical operations: PASSED" << std::endl;
}

void test_tensor_arithmetic() {
    std::cout << "Testing Tensor arithmetic..." << std::endl;

    Tensor<float> t1({2, 2});
    t1[0] = 1.0f; t1[1] = 2.0f;
    t1[2] = 3.0f; t1[3] = 4.0f;

    Tensor<float> t2({2, 2});
    t2[0] = 5.0f; t2[1] = 6.0f;
    t2[2] = 7.0f; t2[3] = 8.0f;

    // Addition
    Tensor<float> t3 = t1 + t2;
    assert(t3[0] == 6.0f);
    assert(t3[3] == 12.0f);

    // Subtraction
    Tensor<float> t4 = t2 - t1;
    assert(t4[0] == 4.0f);
    assert(t4[3] == 4.0f);

    // Element-wise multiplication
    Tensor<float> t5 = t1 * t2;
    assert(t5[0] == 5.0f);
    assert(t5[3] == 32.0f);

    // Element-wise division
    Tensor<float> t6 = t2 / t1;
    assert(t6[0] == 5.0f);
    assert(t6[3] == 2.0f);

    // Negation
    Tensor<float> t7 = -t1;
    assert(t7[0] == -1.0f);
    assert(t7[3] == -4.0f);

    std::cout << "  Tensor arithmetic: PASSED" << std::endl;
}

void test_tensor_scalar_ops() {
    std::cout << "Testing Tensor scalar operations..." << std::endl;

    Tensor<float> t({2, 2});
    t[0] = 2.0f; t[1] = 4.0f;
    t[2] = 6.0f; t[3] = 8.0f;

    // Scalar addition
    Tensor<float> t1 = t + 1.0f;
    assert(t1[0] == 3.0f);

    Tensor<float> t2 = 1.0f + t;
    assert(t2[0] == 3.0f);

    // Scalar multiplication
    Tensor<float> t3 = t * 2.0f;
    assert(t3[0] == 4.0f);

    Tensor<float> t4 = 2.0f * t;
    assert(t4[0] == 4.0f);

    // Scalar division
    Tensor<float> t5 = t / 2.0f;
    assert(t5[0] == 1.0f);
    assert(t5[3] == 4.0f);

    std::cout << "  Tensor scalar operations: PASSED" << std::endl;
}

void test_tensor_factory_methods() {
    std::cout << "Testing Tensor factory methods..." << std::endl;

    // zeros()
    Tensor<float> t1 = Tensor<float>::zeros({2, 3});
    assert(t1.size() == 6);
    assert(t1[0] == 0.0f);
    assert(t1[5] == 0.0f);

    // ones()
    Tensor<float> t2 = Tensor<float>::ones({2, 2});
    assert(t2[0] == 1.0f);
    assert(t2[3] == 1.0f);

    // full()
    Tensor<float> t3 = Tensor<float>::full({2, 3}, 7.0f);
    assert(t3[0] == 7.0f);
    assert(t3[5] == 7.0f);

    std::cout << "  Tensor factory methods: PASSED" << std::endl;
}

void test_tensor_comparison() {
    std::cout << "Testing Tensor comparison..." << std::endl;

    Tensor<int> t1({2, 2});
    t1[0] = 1; t1[1] = 2;
    t1[2] = 3; t1[3] = 4;

    Tensor<int> t2({2, 2});
    t2[0] = 1; t2[1] = 2;
    t2[2] = 3; t2[3] = 4;

    Tensor<int> t3({2, 2});
    t3[0] = 1; t3[1] = 2;
    t3[2] = 3; t3[3] = 5;

    assert(t1 == t2);
    assert(t1 != t3);

    std::cout << "  Tensor comparison: PASSED" << std::endl;
}

void test_tensor_stream() {
    std::cout << "Testing Tensor stream output..." << std::endl;

    Tensor<int> t({2, 3});
    for (usize i = 0; i < t.size(); ++i) {
        t[i] = static_cast<int>(i);
    }

    std::cout << "  " << t << std::endl;

    std::cout << "  Tensor stream output: PASSED" << std::endl;
}

int main() {
    std::cout << "\n=== Tensor Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_tensor_construction();
        std::cout << std::endl;

        test_tensor_element_access();
        std::cout << std::endl;

        test_tensor_shape_operations();
        std::cout << std::endl;

        test_tensor_transpose();
        std::cout << std::endl;

        test_tensor_view_conversion();
        std::cout << std::endl;

        test_tensor_modifiers();
        std::cout << std::endl;

        test_tensor_math_ops();
        std::cout << std::endl;

        test_tensor_arithmetic();
        std::cout << std::endl;

        test_tensor_scalar_ops();
        std::cout << std::endl;

        test_tensor_factory_methods();
        std::cout << std::endl;

        test_tensor_comparison();
        std::cout << std::endl;

        test_tensor_stream();
        std::cout << std::endl;

        std::cout << "=== All Tensor Tests PASSED ===" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

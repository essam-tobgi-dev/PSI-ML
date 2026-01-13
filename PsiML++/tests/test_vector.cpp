#include "../include/math/vector.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::math;
using namespace psi::core;

void test_vector_construction() {
    std::cout << "Testing Vector construction..." << std::endl;

    // Default constructor
    Vector<float> v1;
    assert(v1.size() == 0);
    assert(v1.empty());

    // Size constructor
    Vector<float> v2(5);
    assert(v2.size() == 5);
    assert(!v2.empty());

    // Size with value constructor
    Vector<float> v3(4, 2.5f);
    assert(v3.size() == 4);
    for (usize i = 0; i < 4; ++i) {
        assert(v3[i] == 2.5f);
    }

    // Initializer list constructor
    Vector<float> v4 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    assert(v4.size() == 5);
    assert(v4[0] == 1.0f);
    assert(v4[4] == 5.0f);

    // Copy constructor
    Vector<float> v5(v4);
    assert(v5.size() == v4.size());
    assert(v5[0] == v4[0]);

    // Move constructor
    Vector<float> v6(std::move(v5));
    assert(v6.size() == 5);
    assert(v5.size() == 0);

    std::cout << "  Vector construction: PASSED" << std::endl;
}

void test_vector_element_access() {
    std::cout << "Testing Vector element access..." << std::endl;

    Vector<int> v = {10, 20, 30, 40, 50};

    // Operator[]
    assert(v[0] == 10);
    assert(v[4] == 50);

    // at()
    assert(v.at(2) == 30);

    // front() and back()
    assert(v.front() == 10);
    assert(v.back() == 50);

    // Modify elements
    v[1] = 25;
    assert(v[1] == 25);

    // data()
    int* ptr = v.data();
    assert(ptr != nullptr);
    assert(ptr[0] == 10);

    std::cout << "  Vector element access: PASSED" << std::endl;
}

void test_vector_modifiers() {
    std::cout << "Testing Vector modifiers..." << std::endl;

    Vector<float> v(5, 1.0f);

    // fill()
    v.fill(3.0f);
    for (usize i = 0; i < v.size(); ++i) {
        assert(v[i] == 3.0f);
    }

    // resize()
    v.resize(10, 5.0f);
    assert(v.size() == 10);
    assert(v[9] == 5.0f);

    v.resize(3);
    assert(v.size() == 3);

    // clear()
    v.clear();
    assert(v.empty());
    assert(v.size() == 0);

    std::cout << "  Vector modifiers: PASSED" << std::endl;
}

void test_vector_arithmetic() {
    std::cout << "Testing Vector arithmetic..." << std::endl;

    Vector<float> v1 = {1.0f, 2.0f, 3.0f};
    Vector<float> v2 = {4.0f, 5.0f, 6.0f};

    // Addition
    Vector<float> v3 = v1 + v2;
    assert(v3[0] == 5.0f);
    assert(v3[1] == 7.0f);
    assert(v3[2] == 9.0f);

    // Subtraction
    Vector<float> v4 = v2 - v1;
    assert(v4[0] == 3.0f);
    assert(v4[1] == 3.0f);
    assert(v4[2] == 3.0f);

    // Element-wise multiplication
    Vector<float> v5 = v1 * v2;
    assert(v5[0] == 4.0f);
    assert(v5[1] == 10.0f);
    assert(v5[2] == 18.0f);

    // Element-wise division
    Vector<float> v6 = v2 / v1;
    assert(v6[0] == 4.0f);
    assert(v6[1] == 2.5f);
    assert(v6[2] == 2.0f);

    // Negation
    Vector<float> v7 = -v1;
    assert(v7[0] == -1.0f);
    assert(v7[2] == -3.0f);

    std::cout << "  Vector arithmetic: PASSED" << std::endl;
}

void test_vector_scalar_ops() {
    std::cout << "Testing Vector scalar operations..." << std::endl;

    Vector<float> v = {2.0f, 4.0f, 6.0f};

    // Scalar addition
    Vector<float> v1 = v + 3.0f;
    assert(v1[0] == 5.0f);
    assert(v1[1] == 7.0f);

    Vector<float> v2 = 3.0f + v;
    assert(v2[0] == 5.0f);

    // Scalar subtraction
    Vector<float> v3 = v - 1.0f;
    assert(v3[0] == 1.0f);
    assert(v3[2] == 5.0f);

    // Scalar multiplication
    Vector<float> v4 = v * 2.0f;
    assert(v4[0] == 4.0f);
    assert(v4[1] == 8.0f);

    Vector<float> v5 = 2.0f * v;
    assert(v5[0] == 4.0f);

    // Scalar division
    Vector<float> v6 = v / 2.0f;
    assert(v6[0] == 1.0f);
    assert(v6[2] == 3.0f);

    // In-place operations
    v += 1.0f;
    assert(v[0] == 3.0f);

    v *= 2.0f;
    assert(v[0] == 6.0f);

    std::cout << "  Vector scalar operations: PASSED" << std::endl;
}

void test_vector_math_ops() {
    std::cout << "Testing Vector mathematical operations..." << std::endl;

    Vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};

    // sum()
    float sum = v.sum();
    assert(sum == 10.0f);

    // mean()
    float mean = v.mean();
    assert(mean == 2.5f);

    // min() and max()
    assert(v.min() == 1.0f);
    assert(v.max() == 4.0f);

    // norm() (Euclidean norm)
    float norm = v.norm();
    float expected_norm = std::sqrt(1.0f + 4.0f + 9.0f + 16.0f);
    assert(std::abs(norm - expected_norm) < 1e-5f);

    // norm_squared()
    float norm_sq = v.norm_squared();
    assert(norm_sq == 30.0f);

    // dot product
    Vector<float> v2 = {1.0f, 0.0f, 1.0f, 0.0f};
    float dot = v.dot(v2);
    assert(dot == 4.0f);  // 1*1 + 2*0 + 3*1 + 4*0 = 4

    // normalize()
    Vector<float> v3 = {3.0f, 4.0f};
    v3.normalize();
    assert(std::abs(v3.norm() - 1.0f) < 1e-5f);

    std::cout << "  Vector mathematical operations: PASSED" << std::endl;
}

void test_vector_comparison() {
    std::cout << "Testing Vector comparison..." << std::endl;

    Vector<int> v1 = {1, 2, 3};
    Vector<int> v2 = {1, 2, 3};
    Vector<int> v3 = {1, 2, 4};

    assert(v1 == v2);
    assert(v1 != v3);

    std::cout << "  Vector comparison: PASSED" << std::endl;
}

void test_vector_iterators() {
    std::cout << "Testing Vector iterators..." << std::endl;

    Vector<int> v = {1, 2, 3, 4, 5};

    // Range-based for loop
    int sum = 0;
    for (int val : v) {
        sum += val;
    }
    assert(sum == 15);

    // Iterator modification
    for (auto it = v.begin(); it != v.end(); ++it) {
        *it *= 2;
    }
    assert(v[0] == 2);
    assert(v[4] == 10);

    std::cout << "  Vector iterators: PASSED" << std::endl;
}

void test_vector_apply_map() {
    std::cout << "Testing Vector apply and map..." << std::endl;

    Vector<float> v = {1.0f, 2.0f, 3.0f};

    // apply() - modifies in place
    v.apply([](float x) { return x * x; });
    assert(v[0] == 1.0f);
    assert(v[1] == 4.0f);
    assert(v[2] == 9.0f);

    // map() - returns new vector
    Vector<float> v2 = {1.0f, 2.0f, 3.0f};
    Vector<float> v3 = v2.map([](float x) { return x + 10.0f; });
    assert(v3[0] == 11.0f);
    assert(v2[0] == 1.0f);  // Original unchanged

    std::cout << "  Vector apply and map: PASSED" << std::endl;
}

void test_vector_stream() {
    std::cout << "Testing Vector stream output..." << std::endl;

    Vector<int> v = {1, 2, 3};
    std::cout << "  Vector: " << v << std::endl;

    std::cout << "  Vector stream output: PASSED" << std::endl;
}

int main() {
    std::cout << "\n=== Vector Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_vector_construction();
        std::cout << std::endl;

        test_vector_element_access();
        std::cout << std::endl;

        test_vector_modifiers();
        std::cout << std::endl;

        test_vector_arithmetic();
        std::cout << std::endl;

        test_vector_scalar_ops();
        std::cout << std::endl;

        test_vector_math_ops();
        std::cout << std::endl;

        test_vector_comparison();
        std::cout << std::endl;

        test_vector_iterators();
        std::cout << std::endl;

        test_vector_apply_map();
        std::cout << std::endl;

        test_vector_stream();
        std::cout << std::endl;

        std::cout << "=== All Vector Tests PASSED ===" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

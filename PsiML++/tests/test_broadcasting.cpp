#include "../include/math/ops/broadcasting.h"
#include <iostream>
#include <cassert>

using namespace psi::math;
using namespace psi::math::ops;

constexpr float TOL = 1e-5f;

bool approx_equal(float a, float b) {
    return std::abs(a - b) < TOL;
}

void test_are_broadcastable() {
    std::cout << "Testing are_broadcastable..." << std::endl;

    Shape s1 = {2, 3};
    Shape s2 = {3};
    assert(are_broadcastable(s1, s2));

    Shape s3 = {2, 1};
    Shape s4 = {1, 3};
    assert(are_broadcastable(s3, s4));

    std::cout << "  Are broadcastable: PASSED" << std::endl;
}

void test_broadcast_shape() {
    std::cout << "Testing broadcast_shape..." << std::endl;

    Shape s1 = {2, 1};
    Shape s2 = {1, 3};
    Shape result = broadcast_shape(s1, s2);

    assert(result.size() == 2);
    assert(result[0] == 2 && result[1] == 3);

    std::cout << "  Broadcast shape: PASSED" << std::endl;
}

void test_broadcast_add() {
    std::cout << "Testing broadcast_add..." << std::endl;

    Tensor<float> a({2, 1});
    a[0] = 1.0f;
    a[1] = 2.0f;

    Tensor<float> b({1, 3});
    b[0] = 10.0f;
    b[1] = 20.0f;
    b[2] = 30.0f;

    auto c = broadcast_add(a, b);
    assert(c.shape()[0] == 2 && c.shape()[1] == 3);
    assert(approx_equal(c(0, 0), 11.0f));
    assert(approx_equal(c(0, 1), 21.0f));
    assert(approx_equal(c(1, 0), 12.0f));

    std::cout << "  Broadcast add: PASSED" << std::endl;
}

void test_broadcast_multiply() {
    std::cout << "Testing broadcast_multiply..." << std::endl;

    Tensor<float> a({3});
    a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f;

    Tensor<float> b({2, 3});
    for (psi::core::usize i = 0; i < 6; ++i) b[i] = static_cast<float>(i + 1);

    auto c = broadcast_multiply(a, b);
    assert(c.shape()[0] == 2 && c.shape()[1] == 3);
    assert(approx_equal(c(0, 0), 1.0f));
    assert(approx_equal(c(0, 1), 4.0f));

    std::cout << "  Broadcast multiply: PASSED" << std::endl;
}

void test_expand_dims() {
    std::cout << "Testing expand_dims..." << std::endl;

    Tensor<float> t({3});
    t[0] = 1.0f; t[1] = 2.0f; t[2] = 3.0f;

    auto expanded = expand_dims(t, 0);
    assert(expanded.ndim() == 2);
    assert(expanded.shape()[0] == 1 && expanded.shape()[1] == 3);

    std::cout << "  Expand dims: PASSED" << std::endl;
}

void test_squeeze_dims() {
    std::cout << "Testing squeeze_dims..." << std::endl;

    Tensor<float> t({1, 3, 1});
    t[0] = 1.0f; t[1] = 2.0f; t[2] = 3.0f;

    auto squeezed = squeeze_dims(t);
    assert(squeezed.ndim() == 1 || (squeezed.ndim() == 2 && squeezed.shape().size() == 1));

    std::cout << "  Squeeze dims: PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PsiML++ Broadcasting Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        test_are_broadcastable();
        test_broadcast_shape();
        test_broadcast_add();
        test_broadcast_multiply();
        test_expand_dims();
        test_squeeze_dims();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All broadcasting tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

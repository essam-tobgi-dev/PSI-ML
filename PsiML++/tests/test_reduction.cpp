#include "../include/math/ops/reduction.h"
#include <iostream>
#include <cassert>

using namespace psi::math;
using namespace psi::math::ops;

constexpr float TOL = 1e-5f;

bool approx_equal(float a, float b) {
    return std::abs(a - b) < TOL;
}

void test_reduce_sum() {
    std::cout << "Testing reduce_sum..." << std::endl;
    Tensor<float> t({2, 3});
    t[0] = 1.0f; t[1] = 2.0f; t[2] = 3.0f;
    t[3] = 4.0f; t[4] = 5.0f; t[5] = 6.0f;

    auto sum_all = reduce_sum(t);
    assert(approx_equal(sum_all[0], 21.0f));
    std::cout << "  Reduce sum: PASSED" << std::endl;
}

void test_reduce_mean() {
    std::cout << "Testing reduce_mean..." << std::endl;
    Tensor<float> t({2, 3});
    for (psi::core::usize i = 0; i < 6; ++i) t[i] = static_cast<float>(i + 1);

    auto mean_all = reduce_mean(t);
    assert(approx_equal(mean_all[0], 3.5f));
    std::cout << "  Reduce mean: PASSED" << std::endl;
}

void test_reduce_min_max() {
    std::cout << "Testing reduce_min/max..." << std::endl;
    Tensor<float> t({2, 3});
    t[0] = 5.0f; t[1] = 2.0f; t[2] = 8.0f;
    t[3] = 1.0f; t[4] = 9.0f; t[5] = 3.0f;

    auto min_val = reduce_min(t);
    assert(approx_equal(min_val[0], 1.0f));

    auto max_val = reduce_max(t);
    assert(approx_equal(max_val[0], 9.0f));
    std::cout << "  Reduce min/max: PASSED" << std::endl;
}

void test_reduce_product() {
    std::cout << "Testing reduce_product..." << std::endl;
    Tensor<float> t({3});
    t[0] = 2.0f; t[1] = 3.0f; t[2] = 4.0f;

    auto prod = reduce_product(t);
    assert(approx_equal(prod[0], 24.0f));
    std::cout << "  Reduce product: PASSED" << std::endl;
}

void test_cumsum() {
    std::cout << "Testing cumsum..." << std::endl;
    Tensor<float> t({4});
    t[0] = 1.0f; t[1] = 2.0f; t[2] = 3.0f; t[3] = 4.0f;

    auto cum = cumsum(t, 0);
    assert(approx_equal(cum[0], 1.0f));
    assert(approx_equal(cum[1], 3.0f));
    assert(approx_equal(cum[2], 6.0f));
    assert(approx_equal(cum[3], 10.0f));
    std::cout << "  Cumsum: PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PsiML++ Reduction Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        test_reduce_sum();
        test_reduce_mean();
        test_reduce_min_max();
        test_reduce_product();
        test_cumsum();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All reduction tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

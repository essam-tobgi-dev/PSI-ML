#include "../include/math/ops/arithmetic.h"
#include <iostream>
#include <cassert>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace psi::math;
using namespace psi::math::ops;

constexpr float TOL = 1e-5f;

bool approx_equal(float a, float b) {
    return std::abs(a - b) < TOL;
}

void test_add() {
    std::cout << "Testing add..." << std::endl;
    Vector<float> a({1.0f, 2.0f, 3.0f});
    Vector<float> b({4.0f, 5.0f, 6.0f});
    auto c = add(a, b);
    assert(approx_equal(c[0], 5.0f) && approx_equal(c[1], 7.0f) && approx_equal(c[2], 9.0f));

    auto d = add(a, 10.0f);
    assert(approx_equal(d[0], 11.0f));
    std::cout << "  Add: PASSED" << std::endl;
}

void test_subtract() {
    std::cout << "Testing subtract..." << std::endl;
    Vector<float> a({5.0f, 6.0f, 7.0f});
    Vector<float> b({2.0f, 3.0f, 4.0f});
    auto c = subtract(a, b);
    assert(approx_equal(c[0], 3.0f) && approx_equal(c[1], 3.0f) && approx_equal(c[2], 3.0f));
    std::cout << "  Subtract: PASSED" << std::endl;
}

void test_multiply() {
    std::cout << "Testing multiply..." << std::endl;
    Vector<float> a({2.0f, 3.0f, 4.0f});
    Vector<float> b({5.0f, 6.0f, 7.0f});
    auto c = multiply(a, b);
    assert(approx_equal(c[0], 10.0f) && approx_equal(c[1], 18.0f) && approx_equal(c[2], 28.0f));
    std::cout << "  Multiply: PASSED" << std::endl;
}

void test_divide() {
    std::cout << "Testing divide..." << std::endl;
    Vector<float> a({10.0f, 20.0f, 30.0f});
    Vector<float> b({2.0f, 4.0f, 5.0f});
    auto c = divide(a, b);
    assert(approx_equal(c[0], 5.0f) && approx_equal(c[1], 5.0f) && approx_equal(c[2], 6.0f));
    std::cout << "  Divide: PASSED" << std::endl;
}

void test_power() {
    std::cout << "Testing power..." << std::endl;
    Vector<float> a({2.0f, 3.0f, 4.0f});
    auto c = power(a, 2.0f);
    assert(approx_equal(c[0], 4.0f) && approx_equal(c[1], 9.0f) && approx_equal(c[2], 16.0f));
    std::cout << "  Power: PASSED" << std::endl;
}

void test_unary_ops() {
    std::cout << "Testing unary operations..." << std::endl;
    Vector<float> a({-2.0f, 3.0f, -4.0f});

    auto neg = negate(a);
    assert(approx_equal(neg[0], 2.0f) && approx_equal(neg[1], -3.0f));

    auto ab = abs(a);
    assert(approx_equal(ab[0], 2.0f) && approx_equal(ab[1], 3.0f) && approx_equal(ab[2], 4.0f));

    Vector<float> b({4.0f, 9.0f, 16.0f});
    auto sq = sqrt(b);
    assert(approx_equal(sq[0], 2.0f) && approx_equal(sq[1], 3.0f) && approx_equal(sq[2], 4.0f));

    std::cout << "  Unary ops: PASSED" << std::endl;
}

void test_trig() {
    std::cout << "Testing trigonometric functions..." << std::endl;
    Vector<float> a({0.0f, static_cast<float>(M_PI/2), static_cast<float>(M_PI)});

    auto s = sin(a);
    assert(approx_equal(s[0], 0.0f) && approx_equal(s[1], 1.0f) && approx_equal(s[2], 0.0f));

    auto c = cos(a);
    assert(approx_equal(c[0], 1.0f) && approx_equal(c[1], 0.0f) && approx_equal(c[2], -1.0f));

    std::cout << "  Trigonometric: PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PsiML++ Arithmetic Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        test_add();
        test_subtract();
        test_multiply();
        test_divide();
        test_power();
        test_unary_ops();
        test_trig();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All arithmetic tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

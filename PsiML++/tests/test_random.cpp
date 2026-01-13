#include "../include/math/random.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::math;
using namespace psi::core;

void test_random_generators() {
    std::cout << "Testing Random generators..." << std::endl;

    // MersenneTwister
    Random rng1(GeneratorType::MersenneTwister, 42);
    assert(rng1.generator_type() == GeneratorType::MersenneTwister);
    assert(rng1.generator_name() == "MersenneTwister");

    // XORShift
    Random rng2(GeneratorType::XORShift, 42);
    assert(rng2.generator_type() == GeneratorType::XORShift);
    assert(rng2.generator_name() == "XORShift");

    std::cout << "  Random generators: PASSED" << std::endl;
}

void test_seeding() {
    std::cout << "Testing Random seeding..." << std::endl;

    Random rng(GeneratorType::MersenneTwister, 42);

    // Generate some numbers
    float v1 = rng.uniform();
    float v2 = rng.uniform();

    // Re-seed with same value
    rng.seed(42);
    float v3 = rng.uniform();
    float v4 = rng.uniform();

    // Should produce same sequence
    assert(v1 == v3);
    assert(v2 == v4);

    // Random seed (time-based)
    rng.random_seed();
    float v5 = rng.uniform();
    // This should be different (with very high probability)
    assert(v5 != v1);

    std::cout << "  Random seeding: PASSED" << std::endl;
}

void test_uniform_distribution() {
    std::cout << "Testing Uniform distribution..." << std::endl;

    Random rng(GeneratorType::MersenneTwister, 42);

    // Default uniform [0, 1)
    for (int i = 0; i < 100; ++i) {
        float val = rng.uniform();
        assert(val >= 0.0f && val < 1.0f);
    }

    // Uniform with range
    for (int i = 0; i < 100; ++i) {
        float val = rng.uniform(5.0f, 10.0f);
        assert(val >= 5.0f && val <= 10.0f);
    }

    // Uniform integer
    for (int i = 0; i < 100; ++i) {
        int val = rng.uniform_int(1, 10);
        assert(val >= 1 && val <= 10);
    }

    std::cout << "  Uniform distribution: PASSED" << std::endl;
}

void test_normal_distribution() {
    std::cout << "Testing Normal distribution..." << std::endl;

    Random rng(GeneratorType::MersenneTwister, 42);

    // Generate many samples to test mean and stddev
    const int n_samples = 10000;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < n_samples; ++i) {
        float val = rng.normal(5.0f, 2.0f);  // mean=5, stddev=2
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / n_samples;
    float variance = (sum_sq / n_samples) - (mean * mean);
    float stddev = std::sqrt(variance);

    // Check if mean and stddev are close to expected (with tolerance)
    assert(std::abs(mean - 5.0f) < 0.1f);
    assert(std::abs(stddev - 2.0f) < 0.1f);

    std::cout << "  Normal distribution (mean: " << mean
              << ", stddev: " << stddev << "): PASSED" << std::endl;
}

void test_other_distributions() {
    std::cout << "Testing other distributions..." << std::endl;

    Random rng(GeneratorType::MersenneTwister, 42);

    // Bernoulli
    int successes = 0;
    for (int i = 0; i < 1000; ++i) {
        if (rng.bernoulli(0.7)) successes++;
    }
    // Should be around 700 (70%)
    assert(successes > 600 && successes < 800);
    std::cout << "  Bernoulli (p=0.7, successes=" << successes << "/1000)" << std::endl;

    // Exponential
    float exp_val = rng.exponential(2.0f);
    assert(exp_val >= 0.0f);

    // Gamma
    float gamma_val = rng.gamma(2.0f, 1.0f);
    assert(gamma_val >= 0.0f);

    // Beta
    float beta_val = rng.beta(2.0f, 5.0f);
    assert(beta_val >= 0.0f && beta_val <= 1.0f);

    // Poisson
    int poisson_val = rng.poisson(5.0);
    assert(poisson_val >= 0);

    std::cout << "  Other distributions: PASSED" << std::endl;
}

void test_vector_generation() {
    std::cout << "Testing Vector generation..." << std::endl;

    Random rng(GeneratorType::MersenneTwister, 42);

    // Uniform vector
    Vector<float> v1 = rng.uniform_vector<float>(10, 0.0f, 1.0f);
    assert(v1.size() == 10);
    for (usize i = 0; i < v1.size(); ++i) {
        assert(v1[i] >= 0.0f && v1[i] <= 1.0f);
    }

    // Normal vector
    Vector<float> v2 = rng.normal_vector<float>(10, 0.0f, 1.0f);
    assert(v2.size() == 10);

    std::cout << "  Vector generation: PASSED" << std::endl;
}

void test_matrix_generation() {
    std::cout << "Testing Matrix generation..." << std::endl;

    Random rng(GeneratorType::MersenneTwister, 42);

    // Uniform matrix
    Matrix<float> m1 = rng.uniform_matrix<float>(3, 4, 0.0f, 1.0f);
    assert(m1.rows() == 3);
    assert(m1.cols() == 4);
    for (usize i = 0; i < m1.size(); ++i) {
        assert(m1[i] >= 0.0f && m1[i] <= 1.0f);
    }

    // Normal matrix
    Matrix<float> m2 = rng.normal_matrix<float>(3, 3, 0.0f, 1.0f);
    assert(m2.rows() == 3);
    assert(m2.cols() == 3);

    std::cout << "  Matrix generation: PASSED" << std::endl;
}

void test_tensor_generation() {
    std::cout << "Testing Tensor generation..." << std::endl;

    Random rng(GeneratorType::MersenneTwister, 42);

    // Uniform tensor
    Tensor<float> t1 = rng.uniform_tensor<float>({2, 3, 4}, 0.0f, 1.0f);
    assert(t1.size() == 24);
    for (usize i = 0; i < t1.size(); ++i) {
        assert(t1[i] >= 0.0f && t1[i] <= 1.0f);
    }

    // Normal tensor
    Tensor<float> t2 = rng.normal_tensor<float>({3, 3}, 0.0f, 1.0f);
    assert(t2.size() == 9);

    std::cout << "  Tensor generation: PASSED" << std::endl;
}

void test_neural_network_initializers() {
    std::cout << "Testing neural network initializers..." << std::endl;

    Random rng(GeneratorType::MersenneTwister, 42);

    // Xavier uniform
    Matrix<float> m1 = rng.xavier_uniform<float>(100, 50);
    assert(m1.rows() == 100);
    assert(m1.cols() == 50);

    // Xavier normal
    Matrix<float> m2 = rng.xavier_normal<float>(100, 50);
    assert(m2.rows() == 100);
    assert(m2.cols() == 50);

    // He uniform
    Matrix<float> m3 = rng.he_uniform<float>(100, 50);
    assert(m3.rows() == 100);
    assert(m3.cols() == 50);

    // He normal
    Matrix<float> m4 = rng.he_normal<float>(100, 50);
    assert(m4.rows() == 100);
    assert(m4.cols() == 50);

    std::cout << "  Neural network initializers: PASSED" << std::endl;
}

void test_utility_methods() {
    std::cout << "Testing utility methods..." << std::endl;

    Random rng(GeneratorType::MersenneTwister, 42);

    // Shuffle
    std::vector<int> vec = {1, 2, 3, 4, 5};
    rng.shuffle(vec);
    // Check all elements still present
    assert(vec.size() == 5);
    bool all_present = true;
    for (int i = 1; i <= 5; ++i) {
        bool found = false;
        for (int v : vec) {
            if (v == i) found = true;
        }
        if (!found) all_present = false;
    }
    assert(all_present);

    // Choice
    std::vector<std::string> options = {"a", "b", "c"};
    std::string choice = rng.choice(options);
    assert(choice == "a" || choice == "b" || choice == "c");

    // Permutation
    std::vector<usize> perm = rng.permutation(10);
    assert(perm.size() == 10);
    // Check all indices 0-9 are present
    for (usize i = 0; i < 10; ++i) {
        bool found = false;
        for (usize p : perm) {
            if (p == i) found = true;
        }
        assert(found);
    }

    std::cout << "  Utility methods: PASSED" << std::endl;
}

void test_global_functions() {
    std::cout << "Testing global random functions..." << std::endl;

    // Set seed
    set_seed(42);

    // Generate random vector
    Vector<float> v = randn<float>(10, 0.0f, 1.0f);
    assert(v.size() == 10);

    // Generate random matrix
    Matrix<float> m = rand<float>(3, 4, 0.0f, 1.0f);
    assert(m.rows() == 3);
    assert(m.cols() == 4);

    // Random seed
    random_seed();

    std::cout << "  Global functions: PASSED" << std::endl;
}

int main() {
    std::cout << "\n=== Random Number Generation Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_random_generators();
        std::cout << std::endl;

        test_seeding();
        std::cout << std::endl;

        test_uniform_distribution();
        std::cout << std::endl;

        test_normal_distribution();
        std::cout << std::endl;

        test_other_distributions();
        std::cout << std::endl;

        test_vector_generation();
        std::cout << std::endl;

        test_matrix_generation();
        std::cout << std::endl;

        test_tensor_generation();
        std::cout << std::endl;

        test_neural_network_initializers();
        std::cout << std::endl;

        test_utility_methods();
        std::cout << std::endl;

        test_global_functions();
        std::cout << std::endl;

        std::cout << "=== All Random Tests PASSED ===" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

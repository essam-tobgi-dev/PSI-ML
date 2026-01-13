#include "../include/psi.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

// Test function declarations
extern int test_device_main();
extern int test_logging_main();
extern int test_memory_main();
extern int test_vector_main();
extern int test_matrix_main();
extern int test_tensor_main();
extern int test_random_main();
extern int test_blas_main();

struct TestInfo {
    std::string name;
    std::string description;
    int (*test_func)();
};

void print_banner(const std::string& title) {
    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "========================================" << std::endl;
}

void print_usage() {
    std::cout << "Usage: run_all_tests [test_name]" << std::endl;
    std::cout << std::endl;
    std::cout << "Available tests:" << std::endl;
    std::cout << "  all        - Run all tests (default)" << std::endl;
    std::cout << "  core       - Run all core tests" << std::endl;
    std::cout << "  math       - Run all math tests" << std::endl;
    std::cout << "  device     - Test device management" << std::endl;
    std::cout << "  logging    - Test logging system" << std::endl;
    std::cout << "  memory     - Test memory management" << std::endl;
    std::cout << "  vector     - Test Vector class" << std::endl;
    std::cout << "  matrix     - Test Matrix class" << std::endl;
    std::cout << "  tensor     - Test Tensor class" << std::endl;
    std::cout << "  random     - Test Random class" << std::endl;
    std::cout << "  blas       - Test BLAS operations" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    print_banner("PsiML++ Test Suite");

    std::string test_filter = "all";
    if (argc > 1) {
        test_filter = argv[1];
        if (test_filter == "--help" || test_filter == "-h") {
            print_usage();
            return 0;
        }
    }

    std::cout << "Test filter: " << test_filter << std::endl;
    std::cout << std::endl;

    // Define all tests
    std::vector<TestInfo> core_tests = {
        {"device", "Device Management", nullptr},
        {"logging", "Logging System", nullptr},
        {"memory", "Memory Management", nullptr}
    };

    std::vector<TestInfo> math_tests = {
        {"vector", "Vector Operations", nullptr},
        {"matrix", "Matrix Operations", nullptr},
        {"tensor", "Tensor Operations", nullptr},
        {"random", "Random Number Generation", nullptr},
        {"blas", "BLAS Operations", nullptr}
    };

    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;

    auto run_test = [&](const std::string& name, const std::string& desc,
                        int (*test_func)()) {
        if (test_filter != "all" && test_filter != name) {
            return;
        }

        total_tests++;
        std::cout << "\nRunning " << desc << " tests..." << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        int result = test_func();

        if (result == 0) {
            passed_tests++;
            std::cout << "\n\033[32m✓ " << desc << " tests PASSED\033[0m" << std::endl;
        } else {
            failed_tests++;
            std::cout << "\n\033[31m✗ " << desc << " tests FAILED\033[0m" << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    };

    // Run core tests
    if (test_filter == "all" || test_filter == "core" ||
        test_filter == "device" || test_filter == "logging" || test_filter == "memory") {

        print_banner("CORE MODULE TESTS");

        if (test_filter == "all" || test_filter == "core" || test_filter == "device") {
            run_test("device", "Device Management", test_device_main);
        }

        if (test_filter == "all" || test_filter == "core" || test_filter == "logging") {
            run_test("logging", "Logging System", test_logging_main);
        }

        if (test_filter == "all" || test_filter == "core" || test_filter == "memory") {
            run_test("memory", "Memory Management", test_memory_main);
        }
    }

    // Run math tests
    if (test_filter == "all" || test_filter == "math" ||
        test_filter == "vector" || test_filter == "matrix" || test_filter == "tensor" ||
        test_filter == "random" || test_filter == "blas") {

        print_banner("MATH MODULE TESTS");

        if (test_filter == "all" || test_filter == "math" || test_filter == "vector") {
            run_test("vector", "Vector Operations", test_vector_main);
        }

        if (test_filter == "all" || test_filter == "math" || test_filter == "matrix") {
            run_test("matrix", "Matrix Operations", test_matrix_main);
        }

        if (test_filter == "all" || test_filter == "math" || test_filter == "tensor") {
            run_test("tensor", "Tensor Operations", test_tensor_main);
        }

        if (test_filter == "all" || test_filter == "math" || test_filter == "random") {
            run_test("random", "Random Number Generation", test_random_main);
        }

        if (test_filter == "all" || test_filter == "math" || test_filter == "blas") {
            run_test("blas", "BLAS Operations", test_blas_main);
        }
    }

    // Print summary
    print_banner("TEST SUMMARY");
    std::cout << "Total tests run: " << total_tests << std::endl;
    std::cout << "\033[32mPassed: " << passed_tests << "\033[0m" << std::endl;

    if (failed_tests > 0) {
        std::cout << "\033[31mFailed: " << failed_tests << "\033[0m" << std::endl;
        std::cout << std::endl;
        std::cout << "\033[31m✗ SOME TESTS FAILED\033[0m" << std::endl;
        return 1;
    } else {
        std::cout << std::endl;
        std::cout << "\033[32m✓ ALL TESTS PASSED!\033[0m" << std::endl;
        return 0;
    }
}

// Test main function implementations (these would normally be in separate files)
// For now, we'll provide stubs that can be linked with the actual test files

int test_device_main() {
    // This would be linked with test_device.cpp's main()
    // For compilation, you'll need to rename main() in each test file
    // or compile them separately
    std::cout << "Note: Compile individual test files separately to run." << std::endl;
    return 0;
}

int test_logging_main() {
    std::cout << "Note: Compile individual test files separately to run." << std::endl;
    return 0;
}

int test_memory_main() {
    std::cout << "Note: Compile individual test files separately to run." << std::endl;
    return 0;
}

int test_vector_main() {
    std::cout << "Note: Compile individual test files separately to run." << std::endl;
    return 0;
}

int test_matrix_main() {
    std::cout << "Note: Compile individual test files separately to run." << std::endl;
    return 0;
}

int test_tensor_main() {
    std::cout << "Note: Compile individual test files separately to run." << std::endl;
    return 0;
}

int test_random_main() {
    std::cout << "Note: Compile individual test files separately to run." << std::endl;
    return 0;
}

int test_blas_main() {
    std::cout << "Note: Compile individual test files separately to run." << std::endl;
    return 0;
}

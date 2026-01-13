cmake_minimum_required(VERSION 3.15)
project(PsiMLTests CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(..)

# Core source files
set(CORE_SRCS
    ../src/core/device.cpp
    ../src/core/logging.cpp
    ../src/core/memory.cpp
)

# Math source files
set(MATH_SRCS
    ../src/math/linalg/blas.cpp
)

# Enable testing
enable_testing()

# Function to add a test executable
function(add_psi_test test_name)
    add_executable(${test_name} ${test_name}.cpp ${CORE_SRCS} ${ARGN})
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Core tests
add_psi_test(test_device)
add_psi_test(test_logging)
add_psi_test(test_memory)

# Math tests
add_psi_test(test_vector)
add_psi_test(test_matrix)
add_psi_test(test_tensor)
add_psi_test(test_random)
add_psi_test(test_blas ${MATH_SRCS})

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_device test_logging test_memory test_vector test_matrix test_tensor test_random test_blas
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Print message
message(STATUS "PsiML++ Tests configured")
message(STATUS "Build tests with: cmake --build .")
message(STATUS "Run tests with: ctest or make run_tests")

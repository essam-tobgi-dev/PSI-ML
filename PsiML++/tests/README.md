# PsiML++ Test Suite

This directory contains comprehensive tests for the PsiML++ library, covering both **Core** and **Math** modules.

## Test Files

### Core Module Tests
- **test_device.cpp** - Device management and detection tests
- **test_logging.cpp** - Logging system tests (formatters, handlers, loggers)
- **test_memory.cpp** - Memory management and allocation tests

### Math Module Tests
- **test_vector.cpp** - Vector class tests (operations, arithmetic, norms)
- **test_matrix.cpp** - Matrix class tests (operations, transpose, multiplication)
- **test_tensor.cpp** - Tensor class tests (multi-dimensional operations)
- **test_random.cpp** - Random number generation tests (distributions, generators)
- **test_blas.cpp** - BLAS operations tests (Level 1, 2, and 3)

### Test Runner
- **run_all_tests.cpp** - Main test runner (can run all or specific tests)

## How to Compile and Run Tests

### Option 1: Compile Individual Tests

Each test file can be compiled and run independently:

```bash
# Example: Compile and run device tests
g++ -std=c++20 -I.. test_device.cpp ../src/core/device.cpp ../src/core/logging.cpp -o test_device
./test_device

# Example: Compile and run vector tests
g++ -std=c++20 -I.. test_vector.cpp -o test_vector
./test_vector

# Example: Compile and run matrix tests
g++ -std=c++20 -I.. test_matrix.cpp -o test_matrix
./test_matrix
```

### Option 2: Using Visual Studio (Windows)

1. Add test files to your Visual Studio project
2. Set each test file as a separate build configuration
3. Build and run from Visual Studio

### Option 3: Create a Makefile (Linux/Mac)

Create a `Makefile` in the tests directory:

```makefile
CXX = g++
CXXFLAGS = -std=c++20 -I.. -Wall -Wextra
SRC_DIR = ../src

# Core module sources
CORE_SRCS = $(SRC_DIR)/core/device.cpp $(SRC_DIR)/core/logging.cpp $(SRC_DIR)/core/memory.cpp
MATH_SRCS = $(SRC_DIR)/math/linalg/blas.cpp

# Test executables
TESTS = test_device test_logging test_memory test_vector test_matrix test_tensor test_random test_blas

all: $(TESTS)

test_device: test_device.cpp
	$(CXX) $(CXXFLAGS) $< $(CORE_SRCS) -o $@

test_logging: test_logging.cpp
	$(CXX) $(CXXFLAGS) $< $(CORE_SRCS) -o $@

test_memory: test_memory.cpp
	$(CXX) $(CXXFLAGS) $< $(CORE_SRCS) -o $@

test_vector: test_vector.cpp
	$(CXX) $(CXXFLAGS) $< $(CORE_SRCS) -o $@

test_matrix: test_matrix.cpp
	$(CXX) $(CXXFLAGS) $< $(CORE_SRCS) -o $@

test_tensor: test_tensor.cpp
	$(CXX) $(CXXFLAGS) $< $(CORE_SRCS) -o $@

test_random: test_random.cpp
	$(CXX) $(CXXFLAGS) $< $(CORE_SRCS) -o $@

test_blas: test_blas.cpp
	$(CXX) $(CXXFLAGS) $< $(CORE_SRCS) $(MATH_SRCS) -o $@

run: all
	@echo "Running all tests..."
	@./test_device && echo "✓ Device tests passed"
	@./test_logging && echo "✓ Logging tests passed"
	@./test_memory && echo "✓ Memory tests passed"
	@./test_vector && echo "✓ Vector tests passed"
	@./test_matrix && echo "✓ Matrix tests passed"
	@./test_tensor && echo "✓ Tensor tests passed"
	@./test_random && echo "✓ Random tests passed"
	@./test_blas && echo "✓ BLAS tests passed"

clean:
	rm -f $(TESTS)

.PHONY: all run clean
```

Then run:
```bash
make          # Compile all tests
make run      # Compile and run all tests
make clean    # Clean up executables
```

## Test Structure

Each test file follows this structure:

1. **Test Functions** - Individual test functions for each feature
2. **Assertions** - Uses `assert()` to verify correctness
3. **Main Function** - Runs all tests and reports results
4. **Exception Handling** - Catches and reports any exceptions

## What These Tests Verify

### Device Tests
- Device detection and enumeration
- CPU device properties (cores, memory, capabilities)
- Device manager singleton
- Device context (RAII pattern)

### Logging Tests
- Log levels and formatters
- Console and file handlers
- Logger management
- Logging macros

### Memory Tests
- Memory alignment utilities
- Memory statistics tracking
- System and pool allocators
- Memory RAII wrapper
- Leak detection

### Vector Tests
- Construction and element access
- Arithmetic operations
- Mathematical operations (norm, dot product)
- Iterators and modifiers

### Matrix Tests
- Construction and element access
- Row/column operations
- Transpose and mathematical operations
- Matrix-vector and matrix-matrix multiplication
- Factory methods (zeros, ones, identity, diagonal)

### Tensor Tests
- Multi-dimensional construction
- Shape operations (reshape, squeeze, unsqueeze)
- Transpose operations
- View conversions (to vector/matrix)
- Element-wise operations

### Random Tests
- Random number generators (MT, XORShift)
- Distributions (uniform, normal, exponential, etc.)
- Vector, matrix, and tensor generation
- Neural network initializers (Xavier, He)
- Utility methods (shuffle, choice, permutation)

### BLAS Tests
- Level 1: Vector operations (dot, norm, axpy, scal)
- Level 2: Matrix-vector operations (gemv, ger)
- Level 3: Matrix-matrix operations (gemm)
- Backend detection and configuration

## Expected Output

When tests pass, you should see output like:

```
=== Device Management Tests ===

Testing DeviceInfo...
  DeviceInfo default construction: PASSED

Testing CPUDevice...
  CPU Device Name: Intel Core i7
  CPU Cores: 8
  ...

=== All Device Tests PASSED ===
```

If a test fails, you'll see an assertion error with the file and line number.

## Troubleshooting

1. **Compilation errors**: Make sure you have C++20 support and all source files are available
2. **Linking errors**: Ensure all required source files are included in compilation
3. **Assertion failures**: Check if the functionality is implemented correctly
4. **Memory errors**: Use tools like Valgrind (Linux) or Address Sanitizer to debug

## Notes

- These are basic functionality tests, not extreme edge case tests
- Tests verify that the core functionality works correctly
- All tests use simple assertions for verification
- No external testing frameworks required (uses standard `assert()`)

#pragma once

#include "types.h"
#include "config.h"
#include <exception>
#include <string>
#include <sstream>

namespace psi {
    namespace core {

        // Base exception class for all Psi exceptions
        class PsiException : public std::exception {
        public:
            explicit PsiException(const std::string& message)
                : message_(message) {
            }

            const char* what() const noexcept override {
                return message_.c_str();
            }

            const std::string& message() const noexcept {
                return message_;
            }

        protected:
            std::string message_;
        };

        // Memory-related exceptions
        class MemoryException : public PsiException {
        public:
            explicit MemoryException(const std::string& message)
                : PsiException("Memory Error: " + message) {
            }
        };

        class OutOfMemoryException : public MemoryException {
        public:
            explicit OutOfMemoryException(memory_size_t requested_size)
                : MemoryException("Out of memory, requested " + std::to_string(requested_size) + " bytes") {
            }

            OutOfMemoryException(memory_size_t requested_size, memory_size_t available_size)
                : MemoryException("Out of memory, requested " + std::to_string(requested_size) +
                    " bytes, available " + std::to_string(available_size) + " bytes") {
            }
        };

        class InvalidPointerException : public MemoryException {
        public:
            explicit InvalidPointerException(const void* ptr)
                : MemoryException("Invalid pointer: " + std::to_string(reinterpret_cast<uintptr_t>(ptr))) {
            }

            InvalidPointerException()
                : MemoryException("Invalid null pointer") {
            }
        };

        // Device-related exceptions
        class DeviceException : public PsiException {
        public:
            explicit DeviceException(const std::string& message)
                : PsiException("Device Error: " + message) {
            }
        };

        class InvalidDeviceException : public DeviceException {
        public:
            explicit InvalidDeviceException(device_id_t device_id)
                : DeviceException("Invalid device ID: " + std::to_string(device_id)) {
            }
        };

        class DeviceNotSupportedException : public DeviceException {
        public:
            explicit DeviceNotSupportedException(const std::string& operation)
                : DeviceException("Operation not supported on current device: " + operation) {
            }
        };

        // Shape and dimension exceptions
        class ShapeException : public PsiException {
        public:
            explicit ShapeException(const std::string& message)
                : PsiException("Shape Error: " + message) {
            }
        };

        class DimensionMismatchException : public ShapeException {
        public:
            DimensionMismatchException(const std::string& operation,
                const std::string& expected,
                const std::string& actual)
                : ShapeException("Dimension mismatch in " + operation +
                    ": expected " + expected + ", got " + actual) {
            }
        };

        class IndexOutOfBoundsException : public ShapeException {
        public:
            IndexOutOfBoundsException(index_t index, index_t size)
                : ShapeException("Index " + std::to_string(index) +
                    " out of bounds for size " + std::to_string(size)) {
            }

            IndexOutOfBoundsException(const std::string& dimension, index_t index, index_t size)
                : ShapeException("Index " + std::to_string(index) +
                    " out of bounds for " + dimension +
                    " dimension of size " + std::to_string(size)) {
            }
        };

        // Mathematical exceptions
        class MathException : public PsiException {
        public:
            explicit MathException(const std::string& message)
                : PsiException("Math Error: " + message) {
            }
        };

        class DivisionByZeroException : public MathException {
        public:
            DivisionByZeroException()
                : MathException("Division by zero") {
            }
        };

        class NumericalInstabilityException : public MathException {
        public:
            explicit NumericalInstabilityException(const std::string& operation)
                : MathException("Numerical instability detected in " + operation) {
            }
        };

        class ConvergenceException : public MathException {
        public:
            ConvergenceException(const std::string& algorithm, u32 iterations)
                : MathException(algorithm + " failed to converge after " +
                    std::to_string(iterations) + " iterations") {
            }
        };

        // I/O exceptions
        class IOException : public PsiException {
        public:
            explicit IOException(const std::string& message)
                : PsiException("I/O Error: " + message) {
            }
        };

        class FileNotFoundException : public IOException {
        public:
            explicit FileNotFoundException(const std::string& filename)
                : IOException("File not found: " + filename) {
            }
        };

        class InvalidFormatException : public IOException {
        public:
            explicit InvalidFormatException(const std::string& format)
                : IOException("Invalid format: " + format) {
            }

            InvalidFormatException(const std::string& format, const std::string& details)
                : IOException("Invalid format " + format + ": " + details) {
            }
        };

        // Configuration exceptions
        class ConfigException : public PsiException {
        public:
            explicit ConfigException(const std::string& message)
                : PsiException("Configuration Error: " + message) {
            }
        };

        class InvalidParameterException : public ConfigException {
        public:
            explicit InvalidParameterException(const std::string& parameter)
                : ConfigException("Invalid parameter: " + parameter) {
            }

            InvalidParameterException(const std::string& parameter, const std::string& value)
                : ConfigException("Invalid value '" + value + "' for parameter '" + parameter + "'") {
            }
        };

        // Not implemented exception
        class NotImplementedException : public PsiException {
        public:
            explicit NotImplementedException(const std::string& feature)
                : PsiException("Not implemented: " + feature) {
            }
        };

        // Assertion macros
#if PSI_ENABLE_BOUNDS_CHECKING
#define PSI_ASSERT(condition, message) \
        do { \
            if (PSI_UNLIKELY(!(condition))) { \
                throw ::psi::core::PsiException( \
                    "Assertion failed: " #condition ". " + std::string(message)); \
            } \
        } while (0)

#define PSI_BOUNDS_CHECK(index, size) \
        do { \
            if (PSI_UNLIKELY((index) < 0 || static_cast<::psi::core::index_t>(index) >= static_cast<::psi::core::index_t>(size))) { \
                throw ::psi::core::IndexOutOfBoundsException(static_cast<::psi::core::index_t>(index), static_cast<::psi::core::index_t>(size)); \
            } \
        } while (0)

#define PSI_BOUNDS_CHECK_DIM(dim, index, size) \
        do { \
            if (PSI_UNLIKELY((index) < 0 || static_cast<::psi::core::index_t>(index) >= static_cast<::psi::core::index_t>(size))) { \
                throw ::psi::core::IndexOutOfBoundsException(dim, static_cast<::psi::core::index_t>(index), static_cast<::psi::core::index_t>(size)); \
            } \
        } while (0)
#else
#define PSI_ASSERT(condition, message) ((void)0)
#define PSI_BOUNDS_CHECK(index, size) ((void)0)
#define PSI_BOUNDS_CHECK_DIM(dim, index, size) ((void)0)
#endif

// Memory checking macros
#define PSI_CHECK_POINTER(ptr) \
    do { \
        if (PSI_UNLIKELY((ptr) == nullptr)) { \
            throw ::psi::core::InvalidPointerException(); \
        } \
    } while (0)

#define PSI_CHECK_DEVICE(device_id) \
    do { \
        if (PSI_UNLIKELY(::psi::core::DeviceManager::instance().get_device(device_id) == nullptr)) { \
            throw ::psi::core::InvalidDeviceException(device_id); \
        } \
    } while (0)

// Exception throwing macros for common cases
#define PSI_THROW_MEMORY(message) \
    throw ::psi::core::MemoryException(message)

#define PSI_THROW_DEVICE(message) \
    throw ::psi::core::DeviceException(message)

#define PSI_THROW_SHAPE(message) \
    throw ::psi::core::ShapeException(message)

#define PSI_THROW_MATH(message) \
    throw ::psi::core::MathException(message)

#define PSI_THROW_IO(message) \
    throw ::psi::core::IOException(message)

#define PSI_THROW_CONFIG(message) \
    throw ::psi::core::ConfigException(message)

#define PSI_THROW_NOT_IMPLEMENTED(feature) \
    throw ::psi::core::NotImplementedException(feature)
#define PSI_THROW_ML(message) \
    throw ::psi::core::PsiException("ML Error: " + std::string(message))

// Dimension mismatch helper
#define PSI_CHECK_DIMENSIONS(op, expected, actual) \
    do { \
        if (PSI_UNLIKELY((expected) != (actual))) { \
            throw ::psi::core::DimensionMismatchException(op, \
                std::to_string(expected), std::to_string(actual)); \
        } \
    } while (0)

    } // namespace core
} // namespace psi
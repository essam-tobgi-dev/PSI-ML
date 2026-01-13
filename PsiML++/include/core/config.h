#pragma once

#include "types.h"

namespace psi {
    namespace core {

        // Version information
#define PSI_VERSION_MAJOR 1
#define PSI_VERSION_MINOR 0
#define PSI_VERSION_PATCH 0
#define PSI_VERSION_STRING "1.0.0"

// Build configuration
#ifndef PSI_DEBUG
#ifdef NDEBUG
#define PSI_DEBUG 0
#else
#define PSI_DEBUG 1
#endif
#endif

// Memory alignment configuration
#ifndef PSI_MEMORY_ALIGNMENT
#define PSI_MEMORY_ALIGNMENT 64  // 64-byte alignment for SIMD
#endif

// Cache line size
#ifndef PSI_CACHE_LINE_SIZE
#define PSI_CACHE_LINE_SIZE 64
#endif

// Maximum number of threads
#ifndef PSI_MAX_THREADS
#define PSI_MAX_THREADS 64
#endif

// Enable/disable features
#ifndef PSI_ENABLE_MULTITHREADING
#define PSI_ENABLE_MULTITHREADING 1
#endif

#ifndef PSI_ENABLE_BLAS
#define PSI_ENABLE_BLAS 1
#endif

#ifndef PSI_ENABLE_PROFILING
#define PSI_ENABLE_PROFILING PSI_DEBUG
#endif

#ifndef PSI_ENABLE_BOUNDS_CHECKING
#define PSI_ENABLE_BOUNDS_CHECKING PSI_DEBUG
#endif

// Compiler-specific optimizations
#if defined(__GNUC__) || defined(__clang__)
#define PSI_FORCE_INLINE __attribute__((always_inline)) inline
#define PSI_LIKELY(x) __builtin_expect(!!(x), 1)
#define PSI_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define PSI_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define PSI_FORCE_INLINE __forceinline
#define PSI_LIKELY(x) (x)
#define PSI_UNLIKELY(x) (x)
#define PSI_RESTRICT __restrict
#else
#define PSI_FORCE_INLINE inline
#define PSI_LIKELY(x) (x)
#define PSI_UNLIKELY(x) (x)
#define PSI_RESTRICT
#endif

// Branch prediction hints
#define PSI_EXPECT_TRUE(x) PSI_LIKELY(x)
#define PSI_EXPECT_FALSE(x) PSI_UNLIKELY(x)

// Attribute macros
#if defined(__GNUC__) || defined(__clang__)
#define PSI_PURE __attribute__((pure))
#define PSI_CONST __attribute__((const))
#define PSI_NODISCARD [[nodiscard]]
#elif defined(_MSC_VER)
#define PSI_PURE
#define PSI_CONST
#define PSI_NODISCARD [[nodiscard]]
#else
#define PSI_PURE
#define PSI_CONST
#define PSI_NODISCARD
#endif

// Disable specific warnings
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)  // Disable deprecated function warnings
#endif

// Configuration structure
        struct Config {
            // Memory configuration
            static constexpr usize memory_alignment = PSI_MEMORY_ALIGNMENT;
            static constexpr usize cache_line_size = PSI_CACHE_LINE_SIZE;

            // Threading configuration
            static constexpr bool enable_multithreading = PSI_ENABLE_MULTITHREADING;
            static constexpr u32 max_threads = PSI_MAX_THREADS;

            // Feature flags
            static constexpr bool enable_blas = PSI_ENABLE_BLAS;
            static constexpr bool enable_profiling = PSI_ENABLE_PROFILING;
            static constexpr bool enable_bounds_checking = PSI_ENABLE_BOUNDS_CHECKING;
            static constexpr bool debug_mode = PSI_DEBUG;

            // Default precision
            using default_precision = default_precision_t;
        };

        // Runtime configuration (can be modified)
        struct RuntimeConfig {
            u32 num_threads = 1;
            bool enable_profiling = Config::enable_profiling;
            bool enable_bounds_checking = Config::enable_bounds_checking;

            // Singleton access
            static RuntimeConfig& instance() {
                static RuntimeConfig config;
                return config;
            }

        private:
            RuntimeConfig() = default;
        };

        // Convenience functions
        PSI_NODISCARD PSI_FORCE_INLINE u32 get_num_threads() {
            return RuntimeConfig::instance().num_threads;
        }

        PSI_FORCE_INLINE void set_num_threads(u32 num_threads) {
            RuntimeConfig::instance().num_threads =
                (num_threads > 0 && num_threads <= Config::max_threads) ?
                num_threads : 1;
        }

        PSI_NODISCARD PSI_FORCE_INLINE bool is_profiling_enabled() {
            return RuntimeConfig::instance().enable_profiling;
        }

        PSI_FORCE_INLINE void enable_profiling(bool enable = true) {
            RuntimeConfig::instance().enable_profiling = enable;
        }

        PSI_NODISCARD PSI_FORCE_INLINE bool is_bounds_checking_enabled() {
            return RuntimeConfig::instance().enable_bounds_checking;
        }

        PSI_FORCE_INLINE void enable_bounds_checking(bool enable = true) {
            RuntimeConfig::instance().enable_bounds_checking = enable;
        }

    } // namespace core
} // namespace psi
#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace psi {
    namespace core {

        // Fundamental precision types
        using f32 = float;
        using f64 = double;
        using i8 = std::int8_t;
        using i16 = std::int16_t;
        using i32 = std::int32_t;
        using i64 = std::int64_t;
        using u8 = std::uint8_t;
        using u16 = std::uint16_t;
        using u32 = std::uint32_t;
        using u64 = std::uint64_t;
        using usize = std::size_t;

        // Index type for array/tensor indexing
        using index_t = std::ptrdiff_t;

        // Device ID type
        using device_id_t = i32;

        // Memory size type
        using memory_size_t = usize;

        // Type traits for numeric types
        template<typename T>
        struct is_floating_point : std::is_floating_point<T> {};

        template<typename T>
        struct is_integral : std::is_integral<T> {};

        template<typename T>
        struct is_arithmetic : std::is_arithmetic<T> {};

        template<typename T>
        inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

        template<typename T>
        inline constexpr bool is_integral_v = is_integral<T>::value;

        template<typename T>
        inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

        // Default precision type
#ifndef PSI_DEFAULT_PRECISION
#define PSI_DEFAULT_PRECISION f32
#endif

        using default_precision_t = PSI_DEFAULT_PRECISION;

        // Type size helpers
        template<typename T>
        constexpr usize type_size() {
            return sizeof(T);
        }

        template<typename T>
        constexpr usize type_alignment() {
            return alignof(T);
        }

        // Enum for data types (for runtime type checking)
        enum class DataType : u8 {
            Float32 = 0,
            Float64 = 1,
            Int8 = 2,
            Int16 = 3,
            Int32 = 4,
            Int64 = 5,
            UInt8 = 6,
            UInt16 = 7,
            UInt32 = 8,
            UInt64 = 9,
            Unknown = 255
        };

        // Get DataType from type
        template<typename T>
        constexpr DataType get_data_type() {
            if constexpr (std::is_same_v<T, f32>) return DataType::Float32;
            else if constexpr (std::is_same_v<T, f64>) return DataType::Float64;
            else if constexpr (std::is_same_v<T, i8>) return DataType::Int8;
            else if constexpr (std::is_same_v<T, i16>) return DataType::Int16;
            else if constexpr (std::is_same_v<T, i32>) return DataType::Int32;
            else if constexpr (std::is_same_v<T, i64>) return DataType::Int64;
            else if constexpr (std::is_same_v<T, u8>) return DataType::UInt8;
            else if constexpr (std::is_same_v<T, u16>) return DataType::UInt16;
            else if constexpr (std::is_same_v<T, u32>) return DataType::UInt32;
            else if constexpr (std::is_same_v<T, u64>) return DataType::UInt64;
            else return DataType::Unknown;
        }

        // Get size from DataType
        constexpr usize get_data_type_size(DataType type) {
            switch (type) {
            case DataType::Float32: return sizeof(f32);
            case DataType::Float64: return sizeof(f64);
            case DataType::Int8: return sizeof(i8);
            case DataType::Int16: return sizeof(i16);
            case DataType::Int32: return sizeof(i32);
            case DataType::Int64: return sizeof(i64);
            case DataType::UInt8: return sizeof(u8);
            case DataType::UInt16: return sizeof(u16);
            case DataType::UInt32: return sizeof(u32);
            case DataType::UInt64: return sizeof(u64);
            default: return 0;
            }
        }

    } // namespace core
} // namespace psi
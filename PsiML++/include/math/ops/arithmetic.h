#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/memory.h"
#include "../../core/exception.h"
#include "../../core/device.h"
#include "../vector.h"
#include "../matrix.h"
#include "../tensor.h"
#include <cmath>
#include <algorithm>
#include <functional>
#include <type_traits>

namespace psi {
    namespace math {
        namespace ops {

            // Arithmetic operation types
            enum class ArithmeticOp : core::u8 {
                Add = 0,
                Subtract = 1,
                Multiply = 2,
                Divide = 3,
                Power = 4,
                Modulo = 5,
                Min = 6,
                Max = 7
            };

            // Type trait to check if types are arithmetic compatible
            template<typename T, typename U>
            struct are_arithmetic_compatible {
                static constexpr bool value =
                    (std::is_arithmetic_v<T> && std::is_arithmetic_v<U>) ||
                    (std::is_same_v<T, U>);
            };

            template<typename T, typename U>
            inline constexpr bool are_arithmetic_compatible_v = are_arithmetic_compatible<T, U>::value;

            // Result type promotion for arithmetic operations
            template<typename T, typename U>
            struct arithmetic_result_type {
                using type = std::conditional_t<
                    std::is_floating_point_v<T> || std::is_floating_point_v<U>,
                    std::conditional_t<sizeof(T) >= sizeof(U), T, U>,
                    std::conditional_t<sizeof(T) >= sizeof(U), T, U>
                >;
            };

            template<typename T, typename U>
            using arithmetic_result_t = typename arithmetic_result_type<T, U>::type;

            // Type trait to detect if a type is a container (has value_type member)
            template<typename T, typename = void>
            struct is_container : std::false_type {};

            template<typename T>
            struct is_container<T, std::void_t<typename T::value_type>> : std::true_type {};

            template<typename T>
            inline constexpr bool is_container_v = is_container<T>::value;

            // Element-wise arithmetic function objects
            template<typename T>
            struct Add {
                PSI_FORCE_INLINE T operator()(const T& a, const T& b) const { return a + b; }
            };

            template<typename T>
            struct Subtract {
                PSI_FORCE_INLINE T operator()(const T& a, const T& b) const { return a - b; }
            };

            template<typename T>
            struct Multiply {
                PSI_FORCE_INLINE T operator()(const T& a, const T& b) const { return a * b; }
            };

            template<typename T>
            struct Divide {
                PSI_FORCE_INLINE T operator()(const T& a, const T& b) const {
                    PSI_ASSERT(b != T{}, "Division by zero");
                    return a / b;
                }
            };

            template<typename T>
            struct Power {
                PSI_FORCE_INLINE T operator()(const T& a, const T& b) const {
                    return static_cast<T>(std::pow(static_cast<double>(a), static_cast<double>(b)));
                }
            };

            template<typename T>
            struct Modulo {
                PSI_FORCE_INLINE T operator()(const T& a, const T& b) const {
                    if constexpr (std::is_floating_point_v<T>) {
                        PSI_ASSERT(b != T{}, "Modulo by zero");
                        return static_cast<T>(std::fmod(static_cast<double>(a), static_cast<double>(b)));
                    }
                    else {
                        PSI_ASSERT(b != T{}, "Modulo by zero");
                        return a % b;
                    }
                }
            };

            template<typename T>
            struct Min {
                PSI_FORCE_INLINE T operator()(const T& a, const T& b) const { return std::min(a, b); }
            };

            template<typename T>
            struct Max {
                PSI_FORCE_INLINE T operator()(const T& a, const T& b) const { return std::max(a, b); }
            };

            // Unary arithmetic operations
            template<typename T>
            struct Negate {
                PSI_FORCE_INLINE T operator()(const T& a) const { return -a; }
            };

            template<typename T>
            struct Abs {
                PSI_FORCE_INLINE T operator()(const T& a) const { return std::abs(a); }
            };

            template<typename T>
            struct Sign {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    return (a > T{}) ? T{ 1 } : ((a < T{}) ? T{ -1 } : T{});
                }
            };

            template<typename T>
            struct Square {
                PSI_FORCE_INLINE T operator()(const T& a) const { return a * a; }
            };

            template<typename T>
            struct Sqrt {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    PSI_ASSERT(a >= T{}, "Square root of negative number");
                    return static_cast<T>(std::sqrt(static_cast<double>(a)));
                }
            };

            template<typename T>
            struct Exp {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    return static_cast<T>(std::exp(static_cast<double>(a)));
                }
            };

            template<typename T>
            struct Log {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    PSI_ASSERT(a > T{}, "Logarithm of non-positive number");
                    return static_cast<T>(std::log(static_cast<double>(a)));
                }
            };

            template<typename T>
            struct Log10 {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    PSI_ASSERT(a > T{}, "Logarithm of non-positive number");
                    return static_cast<T>(std::log10(static_cast<double>(a)));
                }
            };

            template<typename T>
            struct Sin {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    return static_cast<T>(std::sin(static_cast<double>(a)));
                }
            };

            template<typename T>
            struct Cos {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    return static_cast<T>(std::cos(static_cast<double>(a)));
                }
            };

            template<typename T>
            struct Tan {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    return static_cast<T>(std::tan(static_cast<double>(a)));
                }
            };

            template<typename T>
            struct Floor {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    return static_cast<T>(std::floor(static_cast<double>(a)));
                }
            };

            template<typename T>
            struct Ceil {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    return static_cast<T>(std::ceil(static_cast<double>(a)));
                }
            };

            template<typename T>
            struct Round {
                PSI_FORCE_INLINE T operator()(const T& a) const {
                    return static_cast<T>(std::round(static_cast<double>(a)));
                }
            };

            // Generic element-wise binary operation
            template<typename Op, typename Container1, typename Container2>
            PSI_NODISCARD auto elementwise_binary(const Container1& a, const Container2& b, Op op) {
                static_assert(are_arithmetic_compatible_v<typename Container1::value_type,
                    typename Container2::value_type>,
                    "Incompatible types for arithmetic operation");

                PSI_CHECK_DIMENSIONS("elementwise operation", a.size(), b.size());

                using result_type = arithmetic_result_t<typename Container1::value_type,
                    typename Container2::value_type>;

                if constexpr (std::is_same_v<Container1, Vector<typename Container1::value_type>>) {
                    Vector<result_type> result(a.size(), a.device_id());
                    for (typename Container1::size_type i = 0; i < a.size(); ++i) {
                        result[i] = static_cast<result_type>(op(a[i], b[i]));
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container1, Matrix<typename Container1::value_type>>) {
                    Matrix<result_type> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container1::size_type i = 0; i < a.size(); ++i) {
                        result[i] = static_cast<result_type>(op(a[i], b[i]));
                    }
                    return result;
                }
                else {
                    Tensor<result_type> result(a.shape(), a.device_id());
                    for (typename Container1::size_type i = 0; i < a.size(); ++i) {
                        result[i] = static_cast<result_type>(op(a[i], b[i]));
                    }
                    return result;
                }
            }

            // Generic element-wise unary operation
            template<typename Op, typename Container>
            PSI_NODISCARD auto elementwise_unary(const Container& a, Op op) {
                using result_type = typename Container::value_type;

                if constexpr (std::is_same_v<Container, Vector<typename Container::value_type>>) {
                    Vector<result_type> result(a.size(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = op(a[i]);
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container, Matrix<typename Container::value_type>>) {
                    Matrix<result_type> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = op(a[i]);
                    }
                    return result;
                }
                else {
                    Tensor<result_type> result(a.shape(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = op(a[i]);
                    }
                    return result;
                }
            }

            // Generic scalar operation
            template<typename Op, typename Container, typename Scalar>
            PSI_NODISCARD auto elementwise_scalar(const Container& a, const Scalar& scalar, Op op) {
                using result_type = arithmetic_result_t<typename Container::value_type, Scalar>;

                if constexpr (std::is_same_v<Container, Vector<typename Container::value_type>>) {
                    Vector<result_type> result(a.size(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = static_cast<result_type>(op(a[i], scalar));
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container, Matrix<typename Container::value_type>>) {
                    Matrix<result_type> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = static_cast<result_type>(op(a[i], scalar));
                    }
                    return result;
                }
                else {
                    Tensor<result_type> result(a.shape(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = static_cast<result_type>(op(a[i], scalar));
                    }
                    return result;
                }
            }

            // Convenient arithmetic functions

            // Addition
            template<typename Container1, typename Container2,
                     typename = std::enable_if_t<is_container_v<Container1> && is_container_v<Container2>>>
            PSI_NODISCARD auto add(const Container1& a, const Container2& b) {
                return elementwise_binary(a, b, Add<typename Container1::value_type>{});
            }

            template<typename Container, typename Scalar,
                     typename = std::enable_if_t<is_container_v<Container> && !is_container_v<Scalar>>,
                     typename = void>
            PSI_NODISCARD auto add(const Container& a, const Scalar& scalar) {
                return elementwise_scalar(a, scalar, Add<typename Container::value_type>{});
            }

            // Subtraction
            template<typename Container1, typename Container2,
                     typename = std::enable_if_t<is_container_v<Container1> && is_container_v<Container2>>>
            PSI_NODISCARD auto subtract(const Container1& a, const Container2& b) {
                return elementwise_binary(a, b, Subtract<typename Container1::value_type>{});
            }

            template<typename Container, typename Scalar,
                     typename = std::enable_if_t<is_container_v<Container> && !is_container_v<Scalar>>,
                     typename = void>
            PSI_NODISCARD auto subtract(const Container& a, const Scalar& scalar) {
                return elementwise_scalar(a, scalar, Subtract<typename Container::value_type>{});
            }

            // Multiplication
            template<typename Container1, typename Container2,
                     typename = std::enable_if_t<is_container_v<Container1> && is_container_v<Container2>>>
            PSI_NODISCARD auto multiply(const Container1& a, const Container2& b) {
                return elementwise_binary(a, b, Multiply<typename Container1::value_type>{});
            }

            template<typename Container, typename Scalar,
                     typename = std::enable_if_t<is_container_v<Container> && !is_container_v<Scalar>>,
                     typename = void>
            PSI_NODISCARD auto multiply(const Container& a, const Scalar& scalar) {
                return elementwise_scalar(a, scalar, Multiply<typename Container::value_type>{});
            }

            // Division
            template<typename Container1, typename Container2,
                     typename = std::enable_if_t<is_container_v<Container1> && is_container_v<Container2>>>
            PSI_NODISCARD auto divide(const Container1& a, const Container2& b) {
                return elementwise_binary(a, b, Divide<typename Container1::value_type>{});
            }

            template<typename Container, typename Scalar,
                     typename = std::enable_if_t<is_container_v<Container> && !is_container_v<Scalar>>,
                     typename = void>
            PSI_NODISCARD auto divide(const Container& a, const Scalar& scalar) {
                return elementwise_scalar(a, scalar, Divide<typename Container::value_type>{});
            }

            // Power
            template<typename Container1, typename Container2,
                     typename = std::enable_if_t<is_container_v<Container1> && is_container_v<Container2>>>
            PSI_NODISCARD auto power(const Container1& a, const Container2& b) {
                return elementwise_binary(a, b, Power<typename Container1::value_type>{});
            }

            template<typename Container, typename Scalar,
                     typename = std::enable_if_t<is_container_v<Container> && !is_container_v<Scalar>>,
                     typename = void>
            PSI_NODISCARD auto power(const Container& a, const Scalar& scalar) {
                return elementwise_scalar(a, scalar, Power<typename Container::value_type>{});
            }

            // Modulo
            template<typename Container1, typename Container2,
                     typename = std::enable_if_t<is_container_v<Container1> && is_container_v<Container2>>>
            PSI_NODISCARD auto modulo(const Container1& a, const Container2& b) {
                return elementwise_binary(a, b, Modulo<typename Container1::value_type>{});
            }

            template<typename Container, typename Scalar,
                     typename = std::enable_if_t<is_container_v<Container> && !is_container_v<Scalar>>,
                     typename = void>
            PSI_NODISCARD auto modulo(const Container& a, const Scalar& scalar) {
                return elementwise_scalar(a, scalar, Modulo<typename Container::value_type>{});
            }

            // Min/Max
            template<typename Container1, typename Container2,
                     typename = std::enable_if_t<is_container_v<Container1> && is_container_v<Container2>>>
            PSI_NODISCARD auto minimum(const Container1& a, const Container2& b) {
                return elementwise_binary(a, b, Min<typename Container1::value_type>{});
            }

            template<typename Container, typename Scalar,
                     typename = std::enable_if_t<is_container_v<Container> && !is_container_v<Scalar>>,
                     typename = void>
            PSI_NODISCARD auto minimum(const Container& a, const Scalar& scalar) {
                return elementwise_scalar(a, scalar, Min<typename Container::value_type>{});
            }

            template<typename Container1, typename Container2,
                     typename = std::enable_if_t<is_container_v<Container1> && is_container_v<Container2>>>
            PSI_NODISCARD auto maximum(const Container1& a, const Container2& b) {
                return elementwise_binary(a, b, Max<typename Container1::value_type>{});
            }

            template<typename Container, typename Scalar,
                     typename = std::enable_if_t<is_container_v<Container> && !is_container_v<Scalar>>,
                     typename = void>
            PSI_NODISCARD auto maximum(const Container& a, const Scalar& scalar) {
                return elementwise_scalar(a, scalar, Max<typename Container::value_type>{});
            }

            // Unary operations
            template<typename Container>
            PSI_NODISCARD auto negate(const Container& a) {
                return elementwise_unary(a, Negate<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto abs(const Container& a) {
                return elementwise_unary(a, Abs<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto sign(const Container& a) {
                return elementwise_unary(a, Sign<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto square(const Container& a) {
                return elementwise_unary(a, Square<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto sqrt(const Container& a) {
                return elementwise_unary(a, Sqrt<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto exp(const Container& a) {
                return elementwise_unary(a, Exp<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto log(const Container& a) {
                return elementwise_unary(a, Log<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto log10(const Container& a) {
                return elementwise_unary(a, Log10<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto sin(const Container& a) {
                return elementwise_unary(a, Sin<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto cos(const Container& a) {
                return elementwise_unary(a, Cos<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto tan(const Container& a) {
                return elementwise_unary(a, Tan<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto floor(const Container& a) {
                return elementwise_unary(a, Floor<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto ceil(const Container& a) {
                return elementwise_unary(a, Ceil<typename Container::value_type>{});
            }

            template<typename Container>
            PSI_NODISCARD auto round(const Container& a) {
                return elementwise_unary(a, Round<typename Container::value_type>{});
            }

            // Special functions

            // Clamp values to a range
            template<typename Container, typename T>
            PSI_NODISCARD auto clamp(const Container& a, const T& min_val, const T& max_val) {
                using value_type = typename Container::value_type;

                if constexpr (std::is_same_v<Container, Vector<value_type>>) {
                    Vector<value_type> result(a.size(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = std::clamp(a[i], static_cast<value_type>(min_val),
                            static_cast<value_type>(max_val));
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container, Matrix<value_type>>) {
                    Matrix<value_type> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = std::clamp(a[i], static_cast<value_type>(min_val),
                            static_cast<value_type>(max_val));
                    }
                    return result;
                }
                else {
                    Tensor<value_type> result(a.shape(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = std::clamp(a[i], static_cast<value_type>(min_val),
                            static_cast<value_type>(max_val));
                    }
                    return result;
                }
            }

            // Linear interpolation
            template<typename Container, typename T>
            PSI_NODISCARD auto lerp(const Container& a, const Container& b, const T& t) {
                static_assert(are_arithmetic_compatible_v<typename Container::value_type, T>,
                    "Incompatible types for lerp");

                PSI_CHECK_DIMENSIONS("lerp", a.size(), b.size());

                using value_type = typename Container::value_type;

                if constexpr (std::is_same_v<Container, Vector<value_type>>) {
                    Vector<value_type> result(a.size(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = a[i] + static_cast<value_type>(t) * (b[i] - a[i]);
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container, Matrix<value_type>>) {
                    Matrix<value_type> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = a[i] + static_cast<value_type>(t) * (b[i] - a[i]);
                    }
                    return result;
                }
                else {
                    Tensor<value_type> result(a.shape(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        result[i] = a[i] + static_cast<value_type>(t) * (b[i] - a[i]);
                    }
                    return result;
                }
            }

            // Safe division (returns 0 when dividing by 0)
            template<typename Container1, typename Container2>
            PSI_NODISCARD auto safe_divide(const Container1& a, const Container2& b) {
                static_assert(are_arithmetic_compatible_v<typename Container1::value_type,
                    typename Container2::value_type>,
                    "Incompatible types for safe division");

                PSI_CHECK_DIMENSIONS("safe divide", a.size(), b.size());

                using result_type = arithmetic_result_t<typename Container1::value_type,
                    typename Container2::value_type>;

                if constexpr (std::is_same_v<Container1, Vector<typename Container1::value_type>>) {
                    Vector<result_type> result(a.size(), a.device_id());
                    for (typename Container1::size_type i = 0; i < a.size(); ++i) {
                        result[i] = (b[i] != typename Container2::value_type{}) ?
                            static_cast<result_type>(a[i] / b[i]) : result_type{};
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container1, Matrix<typename Container1::value_type>>) {
                    Matrix<result_type> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container1::size_type i = 0; i < a.size(); ++i) {
                        result[i] = (b[i] != typename Container2::value_type{}) ?
                            static_cast<result_type>(a[i] / b[i]) : result_type{};
                    }
                    return result;
                }
                else {
                    Tensor<result_type> result(a.shape(), a.device_id());
                    for (typename Container1::size_type i = 0; i < a.size(); ++i) {
                        result[i] = (b[i] != typename Container2::value_type{}) ?
                            static_cast<result_type>(a[i] / b[i]) : result_type{};
                    }
                    return result;
                }
            }

            // Fused multiply-add: a * b + c
            template<typename Container1, typename Container2, typename Container3>
            PSI_NODISCARD auto fma(const Container1& a, const Container2& b, const Container3& c) {
                PSI_CHECK_DIMENSIONS("fma a-b", a.size(), b.size());
                PSI_CHECK_DIMENSIONS("fma a-c", a.size(), c.size());

                using result_type = typename Container1::value_type;

                if constexpr (std::is_same_v<Container1, Vector<result_type>>) {
                    Vector<result_type> result(a.size(), a.device_id());
                    for (typename Container1::size_type i = 0; i < a.size(); ++i) {
                        result[i] = a[i] * b[i] + c[i];
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container1, Matrix<result_type>>) {
                    Matrix<result_type> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container1::size_type i = 0; i < a.size(); ++i) {
                        result[i] = a[i] * b[i] + c[i];
                    }
                    return result;
                }
                else {
                    Tensor<result_type> result(a.shape(), a.device_id());
                    for (typename Container1::size_type i = 0; i < a.size(); ++i) {
                        result[i] = a[i] * b[i] + c[i];
                    }
                    return result;
                }
            }

            // Check for special values
            template<typename Container>
            PSI_NODISCARD auto isnan(const Container& a) {
                using value_type = typename Container::value_type;

                if constexpr (std::is_same_v<Container, Vector<value_type>>) {
                    Vector<bool> result(a.size(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        if constexpr (std::is_floating_point_v<value_type>) {
                            result[i] = std::isnan(a[i]);
                        }
                        else {
                            result[i] = false;
                        }
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container, Matrix<value_type>>) {
                    Matrix<bool> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        if constexpr (std::is_floating_point_v<value_type>) {
                            result[i] = std::isnan(a[i]);
                        }
                        else {
                            result[i] = false;
                        }
                    }
                    return result;
                }
                else {
                    Tensor<bool> result(a.shape(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        if constexpr (std::is_floating_point_v<value_type>) {
                            result[i] = std::isnan(a[i]);
                        }
                        else {
                            result[i] = false;
                        }
                    }
                    return result;
                }
            }

            template<typename Container>
            PSI_NODISCARD auto isinf(const Container& a) {
                using value_type = typename Container::value_type;

                if constexpr (std::is_same_v<Container, Vector<value_type>>) {
                    Vector<bool> result(a.size(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        if constexpr (std::is_floating_point_v<value_type>) {
                            result[i] = std::isinf(a[i]);
                        }
                        else {
                            result[i] = false;
                        }
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container, Matrix<value_type>>) {
                    Matrix<bool> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        if constexpr (std::is_floating_point_v<value_type>) {
                            result[i] = std::isinf(a[i]);
                        }
                        else {
                            result[i] = false;
                        }
                    }
                    return result;
                }
                else {
                    Tensor<bool> result(a.shape(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        if constexpr (std::is_floating_point_v<value_type>) {
                            result[i] = std::isinf(a[i]);
                        }
                        else {
                            result[i] = false;
                        }
                    }
                    return result;
                }
            }

            template<typename Container>
            PSI_NODISCARD auto isfinite(const Container& a) {
                using value_type = typename Container::value_type;

                if constexpr (std::is_same_v<Container, Vector<value_type>>) {
                    Vector<bool> result(a.size(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        if constexpr (std::is_floating_point_v<value_type>) {
                            result[i] = std::isfinite(a[i]);
                        }
                        else {
                            result[i] = true;
                        }
                    }
                    return result;
                }
                else if constexpr (std::is_same_v<Container, Matrix<value_type>>) {
                    Matrix<bool> result(a.rows(), a.cols(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        if constexpr (std::is_floating_point_v<value_type>) {
                            result[i] = std::isfinite(a[i]);
                        }
                        else {
                            result[i] = true;
                        }
                    }
                    return result;
                }
                else {
                    Tensor<bool> result(a.shape(), a.device_id());
                    for (typename Container::size_type i = 0; i < a.size(); ++i) {
                        if constexpr (std::is_floating_point_v<value_type>) {
                            result[i] = std::isfinite(a[i]);
                        }
                        else {
                            result[i] = true;
                        }
                    }
                    return result;
                }
            }

        } // namespace ops
    } // namespace math
} // namespace psi
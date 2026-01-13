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
#include <numeric>
#include <functional>
#include <limits>

namespace psi {
    namespace math {
        namespace ops {

            // Reduction operation types
            enum class ReduceOp : core::u8 {
                Sum = 0,
                Product = 1,
                Mean = 2,
                Min = 3,
                Max = 4,
                ArgMin = 5,
                ArgMax = 6,
                Any = 7,      // Logical OR (for boolean tensors)
                All = 8,      // Logical AND (for boolean tensors)
                Norm = 9,
                Variance = 10,
                StandardDev = 11
            };

            // Keep dimensions flag
            constexpr bool KEEP_DIMS = true;
            constexpr bool REMOVE_DIMS = false;

            // All axes indicator
            constexpr core::index_t ALL_AXES = -1;

            // Helper function for incrementing multi-dimensional indices
            inline bool increment_indices(std::vector<core::index_t>& indices, const Shape& shape) {
                for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
                    if (++indices[i] < static_cast<core::index_t>(shape[i])) {
                        return true;
                    }
                    indices[i] = 0;
                }
                return false;
            }

            // Helper to compute result shape after reduction
            PSI_NODISCARD inline Shape compute_reduction_shape(const Shape& input_shape,
                const std::vector<core::index_t>& axes,
                bool keep_dims) {
                if (axes.empty() || (axes.size() == 1 && axes[0] == ALL_AXES)) {
                    // Reduce all axes
                    if (keep_dims) {
                        return Shape(input_shape.size(), 1);
                    }
                    else {
                        return Shape{ 1 };  // Scalar result
                    }
                }

                Shape result_shape;
                for (core::usize i = 0; i < input_shape.size(); ++i) {
                    bool is_reduced_axis = false;
                    for (core::index_t axis : axes) {
                        if (axis == static_cast<core::index_t>(i)) {
                            is_reduced_axis = true;
                            break;
                        }
                    }

                    if (is_reduced_axis) {
                        if (keep_dims) {
                            result_shape.push_back(1);
                        }
                        // else: omit this dimension
                    }
                    else {
                        result_shape.push_back(input_shape[i]);
                    }
                }

                if (result_shape.empty()) {
                    result_shape.push_back(1);  // Scalar result
                }

                return result_shape;
            }

            // Sum reduction
            template<typename T>
            PSI_NODISCARD Tensor<T> reduce_sum(const Tensor<T>& tensor,
                const std::vector<core::index_t>& axes = { ALL_AXES },
                bool keep_dims = REMOVE_DIMS) {

                Shape result_shape = compute_reduction_shape(tensor.shape(), axes, keep_dims);
                Tensor<T> result(result_shape, tensor.device_id());
                result.fill(T{});

                if (axes.empty() || (axes.size() == 1 && axes[0] == ALL_AXES)) {
                    // Reduce all elements to scalar
                    T sum_val{};
                    for (core::usize i = 0; i < tensor.size(); ++i) {
                        sum_val += tensor[i];
                    }
                    result[0] = sum_val;
                }
                else {
                    // Reduce along specific axes
                    std::vector<core::index_t> indices(tensor.ndim(), 0);

                    do {
                        // Compute result index by removing reduced dimensions
                        std::vector<core::index_t> result_indices;
                        for (core::usize i = 0; i < indices.size(); ++i) {
                            bool is_reduced_axis = false;
                            for (core::index_t axis : axes) {
                                if (axis == static_cast<core::index_t>(i)) {
                                    is_reduced_axis = true;
                                    break;
                                }
                            }

                            if (!is_reduced_axis) {
                                result_indices.push_back(indices[i]);
                            }
                            else if (keep_dims) {
                                result_indices.push_back(0);
                            }
                        }

                        if (result_indices.empty()) {
                            result_indices.push_back(0);  // Scalar case
                        }

                        core::usize tensor_flat = flatten_index(indices, tensor.strides());
                        core::usize result_flat = flatten_index(result_indices, result.strides());

                        result[result_flat] += tensor[tensor_flat];

                    } while (increment_indices(indices, tensor.shape()));
                }

                return result;
            }

            template<typename T>
            PSI_NODISCARD T reduce_sum(const Vector<T>& vec) {
                return vec.sum();
            }

            template<typename T>
            PSI_NODISCARD T reduce_sum(const Matrix<T>& mat) {
                return mat.sum();
            }

            // Mean reduction
            template<typename T>
            PSI_NODISCARD Tensor<T> reduce_mean(const Tensor<T>& tensor,
                const std::vector<core::index_t>& axes = { ALL_AXES },
                bool keep_dims = REMOVE_DIMS) {
                static_assert(std::is_arithmetic_v<T>, "Mean requires arithmetic type");

                auto sum_result = reduce_sum(tensor, axes, keep_dims);

                // Compute the number of elements being averaged
                core::usize count = 1;
                if (axes.empty() || (axes.size() == 1 && axes[0] == ALL_AXES)) {
                    count = tensor.size();
                }
                else {
                    for (core::index_t axis : axes) {
                        if (axis != ALL_AXES) {
                            count *= tensor.size(axis);
                        }
                    }
                }

                // Divide by count
                for (core::usize i = 0; i < sum_result.size(); ++i) {
                    sum_result[i] /= static_cast<T>(count);
                }

                return sum_result;
            }

            template<typename T>
            PSI_NODISCARD T reduce_mean(const Vector<T>& vec) {
                return vec.mean();
            }

            template<typename T>
            PSI_NODISCARD T reduce_mean(const Matrix<T>& mat) {
                return mat.mean();
            }

            // Min reduction
            template<typename T>
            PSI_NODISCARD Tensor<T> reduce_min(const Tensor<T>& tensor,
                const std::vector<core::index_t>& axes = { ALL_AXES },
                bool keep_dims = REMOVE_DIMS) {

                Shape result_shape = compute_reduction_shape(tensor.shape(), axes, keep_dims);
                Tensor<T> result(result_shape, tensor.device_id());
                result.fill(std::numeric_limits<T>::max());

                if (axes.empty() || (axes.size() == 1 && axes[0] == ALL_AXES)) {
                    // Reduce all elements to scalar
                    T min_val = std::numeric_limits<T>::max();
                    for (core::usize i = 0; i < tensor.size(); ++i) {
                        min_val = std::min(min_val, tensor[i]);
                    }
                    result[0] = min_val;
                }
                else {
                    // Reduce along specific axes
                    std::vector<core::index_t> indices(tensor.ndim(), 0);

                    do {
                        // Compute result index
                        std::vector<core::index_t> result_indices;
                        for (core::usize i = 0; i < indices.size(); ++i) {
                            bool is_reduced_axis = false;
                            for (core::index_t axis : axes) {
                                if (axis == static_cast<core::index_t>(i)) {
                                    is_reduced_axis = true;
                                    break;
                                }
                            }

                            if (!is_reduced_axis) {
                                result_indices.push_back(indices[i]);
                            }
                            else if (keep_dims) {
                                result_indices.push_back(0);
                            }
                        }

                        if (result_indices.empty()) {
                            result_indices.push_back(0);
                        }

                        core::usize tensor_flat = flatten_index(indices, tensor.strides());
                        core::usize result_flat = flatten_index(result_indices, result.strides());

                        result[result_flat] = std::min(result[result_flat], tensor[tensor_flat]);

                    } while (increment_indices(indices, tensor.shape()));
                }

                return result;
            }

            template<typename T>
            PSI_NODISCARD T reduce_min(const Vector<T>& vec) {
                return vec.min();
            }

            template<typename T>
            PSI_NODISCARD T reduce_min(const Matrix<T>& mat) {
                return mat.min();
            }

            // Max reduction
            template<typename T>
            PSI_NODISCARD Tensor<T> reduce_max(const Tensor<T>& tensor,
                const std::vector<core::index_t>& axes = { ALL_AXES },
                bool keep_dims = REMOVE_DIMS) {

                Shape result_shape = compute_reduction_shape(tensor.shape(), axes, keep_dims);
                Tensor<T> result(result_shape, tensor.device_id());
                result.fill(std::numeric_limits<T>::lowest());

                if (axes.empty() || (axes.size() == 1 && axes[0] == ALL_AXES)) {
                    // Reduce all elements to scalar
                    T max_val = std::numeric_limits<T>::lowest();
                    for (core::usize i = 0; i < tensor.size(); ++i) {
                        max_val = std::max(max_val, tensor[i]);
                    }
                    result[0] = max_val;
                }
                else {
                    // Reduce along specific axes
                    std::vector<core::index_t> indices(tensor.ndim(), 0);

                    do {
                        // Compute result index
                        std::vector<core::index_t> result_indices;
                        for (core::usize i = 0; i < indices.size(); ++i) {
                            bool is_reduced_axis = false;
                            for (core::index_t axis : axes) {
                                if (axis == static_cast<core::index_t>(i)) {
                                    is_reduced_axis = true;
                                    break;
                                }
                            }

                            if (!is_reduced_axis) {
                                result_indices.push_back(indices[i]);
                            }
                            else if (keep_dims) {
                                result_indices.push_back(0);
                            }
                        }

                        if (result_indices.empty()) {
                            result_indices.push_back(0);
                        }

                        core::usize tensor_flat = flatten_index(indices, tensor.strides());
                        core::usize result_flat = flatten_index(result_indices, result.strides());

                        result[result_flat] = std::max(result[result_flat], tensor[tensor_flat]);

                    } while (increment_indices(indices, tensor.shape()));
                }

                return result;
            }

            template<typename T>
            PSI_NODISCARD T reduce_max(const Vector<T>& vec) {
                return vec.max();
            }

            template<typename T>
            PSI_NODISCARD T reduce_max(const Matrix<T>& mat) {
                return mat.max();
            }

            // Product reduction
            template<typename T>
            PSI_NODISCARD Tensor<T> reduce_product(const Tensor<T>& tensor,
                const std::vector<core::index_t>& axes = { ALL_AXES },
                bool keep_dims = REMOVE_DIMS) {

                Shape result_shape = compute_reduction_shape(tensor.shape(), axes, keep_dims);
                Tensor<T> result(result_shape, tensor.device_id());
                result.fill(T{ 1 });

                if (axes.empty() || (axes.size() == 1 && axes[0] == ALL_AXES)) {
                    // Reduce all elements to scalar
                    T prod_val{ 1 };
                    for (core::usize i = 0; i < tensor.size(); ++i) {
                        prod_val *= tensor[i];
                    }
                    result[0] = prod_val;
                }
                else {
                    // Reduce along specific axes
                    std::vector<core::index_t> indices(tensor.ndim(), 0);

                    do {
                        // Compute result index
                        std::vector<core::index_t> result_indices;
                        for (core::usize i = 0; i < indices.size(); ++i) {
                            bool is_reduced_axis = false;
                            for (core::index_t axis : axes) {
                                if (axis == static_cast<core::index_t>(i)) {
                                    is_reduced_axis = true;
                                    break;
                                }
                            }

                            if (!is_reduced_axis) {
                                result_indices.push_back(indices[i]);
                            }
                            else if (keep_dims) {
                                result_indices.push_back(0);
                            }
                        }

                        if (result_indices.empty()) {
                            result_indices.push_back(0);
                        }

                        core::usize tensor_flat = flatten_index(indices, tensor.strides());
                        core::usize result_flat = flatten_index(result_indices, result.strides());

                        result[result_flat] *= tensor[tensor_flat];

                    } while (increment_indices(indices, tensor.shape()));
                }

                return result;
            }

            // ArgMin reduction
            template<typename T>
            PSI_NODISCARD Tensor<core::index_t> reduce_argmin(const Tensor<T>& tensor,
                core::index_t axis,
                bool keep_dims = REMOVE_DIMS) {
                PSI_BOUNDS_CHECK_DIM("argmin axis", axis, tensor.ndim());

                Shape result_shape = compute_reduction_shape(tensor.shape(), { axis }, keep_dims);
                Tensor<core::index_t> result(result_shape, tensor.device_id());
                Tensor<T> min_values(result_shape, tensor.device_id());

                result.fill(0);
                min_values.fill(std::numeric_limits<T>::max());

                std::vector<core::index_t> indices(tensor.ndim(), 0);

                do {
                    // Compute result index
                    std::vector<core::index_t> result_indices;
                    for (core::usize i = 0; i < indices.size(); ++i) {
                        if (static_cast<core::index_t>(i) != axis) {
                            result_indices.push_back(indices[i]);
                        }
                        else if (keep_dims) {
                            result_indices.push_back(0);
                        }
                    }

                    if (result_indices.empty()) {
                        result_indices.push_back(0);
                    }

                    core::usize tensor_flat = flatten_index(indices, tensor.strides());
                    core::usize result_flat = flatten_index(result_indices, result.strides());

                    if (tensor[tensor_flat] < min_values[result_flat]) {
                        min_values[result_flat] = tensor[tensor_flat];
                        result[result_flat] = indices[axis];
                    }

                } while (increment_indices(indices, tensor.shape()));

                return result;
            }

            // ArgMax reduction
            template<typename T>
            PSI_NODISCARD Tensor<core::index_t> reduce_argmax(const Tensor<T>& tensor,
                core::index_t axis,
                bool keep_dims = REMOVE_DIMS) {
                PSI_BOUNDS_CHECK_DIM("argmax axis", axis, tensor.ndim());

                Shape result_shape = compute_reduction_shape(tensor.shape(), { axis }, keep_dims);
                Tensor<core::index_t> result(result_shape, tensor.device_id());
                Tensor<T> max_values(result_shape, tensor.device_id());

                result.fill(0);
                max_values.fill(std::numeric_limits<T>::lowest());

                std::vector<core::index_t> indices(tensor.ndim(), 0);

                do {
                    // Compute result index
                    std::vector<core::index_t> result_indices;
                    for (core::usize i = 0; i < indices.size(); ++i) {
                        if (static_cast<core::index_t>(i) != axis) {
                            result_indices.push_back(indices[i]);
                        }
                        else if (keep_dims) {
                            result_indices.push_back(0);
                        }
                    }

                    if (result_indices.empty()) {
                        result_indices.push_back(0);
                    }

                    core::usize tensor_flat = flatten_index(indices, tensor.strides());
                    core::usize result_flat = flatten_index(result_indices, result.strides());

                    if (tensor[tensor_flat] > max_values[result_flat]) {
                        max_values[result_flat] = tensor[tensor_flat];
                        result[result_flat] = indices[axis];
                    }

                } while (increment_indices(indices, tensor.shape()));

                return result;
            }

            // Norm reduction (L2 norm by default)
            template<typename T>
            PSI_NODISCARD Tensor<T> reduce_norm(const Tensor<T>& tensor,
                const std::vector<core::index_t>& axes = { ALL_AXES },
                T p = T{ 2 },
                bool keep_dims = REMOVE_DIMS) {
                static_assert(std::is_floating_point_v<T>, "Norm requires floating point type");

                if (p == T{ 2 }) {
                    // L2 norm: sqrt(sum(x^2))
                    Tensor<T> squared = tensor.map([](T x) { return x * x; });
                    auto sum_sq = reduce_sum(squared, axes, keep_dims);

                    for (core::usize i = 0; i < sum_sq.size(); ++i) {
                        sum_sq[i] = std::sqrt(sum_sq[i]);
                    }

                    return sum_sq;
                }
                else if (p == T{ 1 }) {
                    // L1 norm: sum(|x|)
                    Tensor<T> abs_tensor = tensor.map([](T x) { return std::abs(x); });
                    return reduce_sum(abs_tensor, axes, keep_dims);
                }
                else {
                    // General Lp norm: (sum(|x|^p))^(1/p)
                    Tensor<T> powered = tensor.map([p](T x) {
                        return static_cast<T>(std::pow(std::abs(static_cast<double>(x)),
                            static_cast<double>(p)));
                        });
                    auto sum_p = reduce_sum(powered, axes, keep_dims);

                    for (core::usize i = 0; i < sum_p.size(); ++i) {
                        sum_p[i] = static_cast<T>(std::pow(static_cast<double>(sum_p[i]),
                            1.0 / static_cast<double>(p)));
                    }

                    return sum_p;
                }
            }

            // Cumulative reductions
            template<typename T>
            PSI_NODISCARD Tensor<T> cumsum(const Tensor<T>& tensor, core::index_t axis) {
                PSI_BOUNDS_CHECK_DIM("cumsum axis", axis, tensor.ndim());

                Tensor<T> result(tensor.shape(), tensor.device_id());
                std::vector<core::index_t> indices(tensor.ndim(), 0);

                do {
                    core::usize tensor_flat = flatten_index(indices, tensor.strides());

                    if (indices[axis] == 0) {
                        // First element along axis
                        result[tensor_flat] = tensor[tensor_flat];
                    }
                    else {
                        // Add to previous cumulative sum
                        std::vector<core::index_t> prev_indices = indices;
                        prev_indices[axis]--;
                        core::usize prev_flat = flatten_index(prev_indices, tensor.strides());

                        result[tensor_flat] = result[prev_flat] + tensor[tensor_flat];
                    }

                } while (increment_indices(indices, tensor.shape()));

                return result;
            }

            template<typename T>
            PSI_NODISCARD Tensor<T> cumprod(const Tensor<T>& tensor, core::index_t axis) {
                PSI_BOUNDS_CHECK_DIM("cumprod axis", axis, tensor.ndim());

                Tensor<T> result(tensor.shape(), tensor.device_id());
                std::vector<core::index_t> indices(tensor.ndim(), 0);

                do {
                    core::usize tensor_flat = flatten_index(indices, tensor.strides());

                    if (indices[axis] == 0) {
                        // First element along axis
                        result[tensor_flat] = tensor[tensor_flat];
                    }
                    else {
                        // Multiply with previous cumulative product
                        std::vector<core::index_t> prev_indices = indices;
                        prev_indices[axis]--;
                        core::usize prev_flat = flatten_index(prev_indices, tensor.strides());

                        result[tensor_flat] = result[prev_flat] * tensor[tensor_flat];
                    }

                } while (increment_indices(indices, tensor.shape()));

                return result;
            }

            // Convenience functions for matrices
            template<typename T>
            PSI_NODISCARD auto sum_axis(const Matrix<T>& mat, core::index_t axis, bool keep_dims = REMOVE_DIMS) {
                PSI_ASSERT(axis == 0 || axis == 1, "Matrix axis must be 0 or 1");

                if (axis == 0) {
                    // Sum along rows (result has shape [cols])
                    Vector<T> result(mat.cols(), mat.device_id());
                    for (core::usize j = 0; j < mat.cols(); ++j) {
                        T sum_val{};
                        for (core::usize i = 0; i < mat.rows(); ++i) {
                            sum_val += mat(i, j);
                        }
                        result[j] = sum_val;
                    }
                    return result;
                }
                else {
                    // Sum along columns (result has shape [rows])
                    Vector<T> result(mat.rows(), mat.device_id());
                    for (core::usize i = 0; i < mat.rows(); ++i) {
                        T sum_val{};
                        for (core::usize j = 0; j < mat.cols(); ++j) {
                            sum_val += mat(i, j);
                        }
                        result[i] = sum_val;
                    }
                    return result;
                }
            }

            template<typename T>
            PSI_NODISCARD auto mean_axis_reduce(const Matrix<T>& mat, core::index_t axis) {
                auto sum_result = sum_axis(mat, axis);
                T divisor = static_cast<T>((axis == 0) ? mat.rows() : mat.cols());

                for (core::usize i = 0; i < sum_result.size(); ++i) {
                    sum_result[i] /= divisor;
                }

                return sum_result;
            }

        } // namespace ops
    } // namespace math
} // namespace psi
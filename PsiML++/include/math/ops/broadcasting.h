#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/memory.h"
#include "../../core/exception.h"
#include "../../core/device.h"
#include "../vector.h"
#include "../matrix.h"
#include "../tensor.h"
#include <algorithm>
#include <functional>

namespace psi {
    namespace math {
        namespace ops {

            // Broadcasting rules and utilities

            // Check if two shapes are broadcastable
            PSI_NODISCARD inline bool are_broadcastable(const Shape& shape1, const Shape& shape2) {
                core::usize ndim1 = shape1.size();
                core::usize ndim2 = shape2.size();
                core::usize max_ndim = std::max(ndim1, ndim2);

                for (core::usize i = 0; i < max_ndim; ++i) {
                    core::usize dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
                    core::usize dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;

                    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                        return false;
                    }
                }

                return true;
            }

            // Compute the broadcasted shape
            PSI_NODISCARD inline Shape broadcast_shape(const Shape& shape1, const Shape& shape2) {
                if (!are_broadcastable(shape1, shape2)) {
                    PSI_THROW_SHAPE("Shapes are not broadcastable");
                }

                core::usize ndim1 = shape1.size();
                core::usize ndim2 = shape2.size();
                core::usize max_ndim = std::max(ndim1, ndim2);

                Shape result(max_ndim);

                for (core::usize i = 0; i < max_ndim; ++i) {
                    core::usize dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
                    core::usize dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;

                    result[max_ndim - 1 - i] = std::max(dim1, dim2);
                }

                return result;
            }

            // Compute multiple broadcast shapes
            PSI_NODISCARD inline Shape broadcast_shapes(const std::vector<Shape>& shapes) {
                PSI_ASSERT(!shapes.empty(), "Cannot broadcast empty list of shapes");

                Shape result = shapes[0];
                for (core::usize i = 1; i < shapes.size(); ++i) {
                    result = broadcast_shape(result, shapes[i]);
                }

                return result;
            }

            // Check if a shape needs broadcasting to target shape
            PSI_NODISCARD inline bool needs_broadcasting(const Shape& shape, const Shape& target_shape) {
                if (shape.size() != target_shape.size()) return true;

                for (core::usize i = 0; i < shape.size(); ++i) {
                    if (shape[i] != target_shape[i]) return true;
                }

                return false;
            }

            // Compute broadcast strides for efficient iteration
            PSI_NODISCARD inline Shape compute_broadcast_strides(const Shape& original_shape,
                const Shape& target_shape) {
                core::usize orig_ndim = original_shape.size();
                core::usize target_ndim = target_shape.size();

                // Align shapes from the right (trailing dimensions)
                Shape strides(target_ndim, 0);
                Shape aligned_shape(target_ndim, 1);

                // Fill aligned shape with original shape (right-aligned)
                for (core::usize i = 0; i < orig_ndim; ++i) {
                    aligned_shape[target_ndim - orig_ndim + i] = original_shape[i];
                }

                // Compute strides
                core::usize stride = 1;
                for (int i = static_cast<int>(target_ndim) - 1; i >= 0; --i) {
                    if (aligned_shape[i] == target_shape[i]) {
                        strides[i] = stride;
                        stride *= aligned_shape[i];
                    }
                    else {
                        // Broadcasting dimension (size 1 -> size n)
                        strides[i] = 0;
                    }
                }

                return strides;
            }

            // Convert flat index in broadcast space to original tensor index
            PSI_NODISCARD inline core::usize broadcast_index(core::usize flat_index,
                const Shape& target_shape,
                const Shape& broadcast_strides) {
                core::usize orig_index = 0;
                core::usize remaining = flat_index;

                for (core::usize i = 0; i < target_shape.size(); ++i) {
                    core::usize coord = remaining / compute_strides(target_shape)[i];
                    remaining %= compute_strides(target_shape)[i];

                    orig_index += coord * broadcast_strides[i];
                }

                return orig_index;
            }

            // Broadcast a tensor to a target shape
            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_to(const Tensor<T>& tensor, const Shape& target_shape) {
                if (!are_broadcastable(tensor.shape(), target_shape)) {
                    PSI_THROW_SHAPE("Cannot broadcast tensor shape to target shape");
                }

                // If already the right shape, return copy
                if (tensor.shape() == target_shape) {
                    return tensor;
                }

                Tensor<T> result(target_shape, tensor.device_id());
                Shape broadcast_strides = compute_broadcast_strides(tensor.shape(), target_shape);
                Shape target_strides = compute_strides(target_shape);

                // Fill result using broadcasting
                for (core::usize i = 0; i < result.size(); ++i) {
                    core::usize orig_index = broadcast_index(i, target_shape, broadcast_strides);
                    result[i] = tensor[orig_index];
                }

                return result;
            }

            // Broadcast multiple tensors to common shape
            template<typename T>
            PSI_NODISCARD std::vector<Tensor<T>> broadcast_tensors(const std::vector<Tensor<T>>& tensors) {
                PSI_ASSERT(!tensors.empty(), "Cannot broadcast empty list of tensors");

                // Collect all shapes
                std::vector<Shape> shapes;
                for (const auto& tensor : tensors) {
                    shapes.push_back(tensor.shape());
                }

                // Compute common broadcast shape
                Shape common_shape = broadcast_shapes(shapes);

                // Broadcast each tensor to common shape
                std::vector<Tensor<T>> result;
                result.reserve(tensors.size());

                for (const auto& tensor : tensors) {
                    result.push_back(broadcast_to(tensor, common_shape));
                }

                return result;
            }

            // Generic broadcast binary operation
            template<typename T, typename Op>
            PSI_NODISCARD Tensor<T> broadcast_binary_op(const Tensor<T>& a, const Tensor<T>& b, Op op) {
                // Check broadcastability
                if (!are_broadcastable(a.shape(), b.shape())) {
                    PSI_THROW_SHAPE("Tensors are not broadcastable");
                }

                // Special cases for efficiency
                if (a.shape() == b.shape()) {
                    // Same shape - direct element-wise operation
                    Tensor<T> result(a.shape(), a.device_id());
                    for (core::usize i = 0; i < a.size(); ++i) {
                        result[i] = op(a[i], b[i]);
                    }
                    return result;
                }

                // General broadcasting case
                Shape result_shape = broadcast_shape(a.shape(), b.shape());
                Tensor<T> result(result_shape, a.device_id());

                Shape strides_a = compute_broadcast_strides(a.shape(), result_shape);
                Shape strides_b = compute_broadcast_strides(b.shape(), result_shape);

                for (core::usize i = 0; i < result.size(); ++i) {
                    core::usize index_a = broadcast_index(i, result_shape, strides_a);
                    core::usize index_b = broadcast_index(i, result_shape, strides_b);
                    result[i] = op(a[index_a], b[index_b]);
                }

                return result;
            }

            // Generic broadcast unary operation (mostly for shape expansion)
            template<typename T, typename Op>
            PSI_NODISCARD Tensor<T> broadcast_unary_op(const Tensor<T>& a, const Shape& target_shape, Op op) {
                Tensor<T> broadcasted = broadcast_to(a, target_shape);

                for (core::usize i = 0; i < broadcasted.size(); ++i) {
                    broadcasted[i] = op(broadcasted[i]);
                }

                return broadcasted;
            }

            // Specific broadcasting arithmetic operations

            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_add(const Tensor<T>& a, const Tensor<T>& b) {
                return broadcast_binary_op(a, b, [](const T& x, const T& y) { return x + y; });
            }

            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_subtract(const Tensor<T>& a, const Tensor<T>& b) {
                return broadcast_binary_op(a, b, [](const T& x, const T& y) { return x - y; });
            }

            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_multiply(const Tensor<T>& a, const Tensor<T>& b) {
                return broadcast_binary_op(a, b, [](const T& x, const T& y) { return x * y; });
            }

            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_divide(const Tensor<T>& a, const Tensor<T>& b) {
                return broadcast_binary_op(a, b, [](const T& x, const T& y) {
                    PSI_ASSERT(y != T{}, "Division by zero in broadcast operation");
                    return x / y;
                    });
            }

            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_power(const Tensor<T>& a, const Tensor<T>& b) {
                return broadcast_binary_op(a, b, [](const T& x, const T& y) {
                    return static_cast<T>(std::pow(static_cast<double>(x), static_cast<double>(y)));
                    });
            }

            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_min(const Tensor<T>& a, const Tensor<T>& b) {
                return broadcast_binary_op(a, b, [](const T& x, const T& y) { return std::min(x, y); });
            }

            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_max(const Tensor<T>& a, const Tensor<T>& b) {
                return broadcast_binary_op(a, b, [](const T& x, const T& y) { return std::max(x, y); });
            }

            // Broadcasting with scalars

            template<typename T, typename Scalar>
            PSI_NODISCARD Tensor<T> broadcast_scalar_add(const Tensor<T>& tensor, const Scalar& scalar) {
                Tensor<T> result(tensor.shape(), tensor.device_id());
                for (core::usize i = 0; i < tensor.size(); ++i) {
                    result[i] = tensor[i] + static_cast<T>(scalar);
                }
                return result;
            }

            template<typename T, typename Scalar>
            PSI_NODISCARD Tensor<T> broadcast_scalar_multiply(const Tensor<T>& tensor, const Scalar& scalar) {
                Tensor<T> result(tensor.shape(), tensor.device_id());
                for (core::usize i = 0; i < tensor.size(); ++i) {
                    result[i] = tensor[i] * static_cast<T>(scalar);
                }
                return result;
            }

            // Broadcasting along specific axes

            // Add vector along specified axis
            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_vector_add(const Tensor<T>& tensor, const Vector<T>& vec, core::index_t axis) {
                PSI_BOUNDS_CHECK_DIM("broadcast axis", axis, tensor.ndim());
                PSI_CHECK_DIMENSIONS("broadcast vector size", tensor.size(axis), vec.size());

                Tensor<T> result = tensor;  // Copy

                // Create index vector for iteration
                std::vector<core::index_t> indices(tensor.ndim(), 0);

                do {
                    core::usize flat_index = flatten_index(indices, tensor.strides());
                    result[flat_index] += vec[indices[axis]];
                } while (increment_indices(indices, tensor.shape()));

                return result;
            }

            // Matrix broadcasting (add matrix along specified axes)
            template<typename T>
            PSI_NODISCARD Tensor<T> broadcast_matrix_add(const Tensor<T>& tensor, const Matrix<T>& mat,
                core::index_t axis1, core::index_t axis2) {
                PSI_BOUNDS_CHECK_DIM("broadcast axis1", axis1, tensor.ndim());
                PSI_BOUNDS_CHECK_DIM("broadcast axis2", axis2, tensor.ndim());
                PSI_ASSERT(axis1 != axis2, "Broadcast axes must be different");

                PSI_CHECK_DIMENSIONS("broadcast matrix dim1", tensor.size(axis1), mat.rows());
                PSI_CHECK_DIMENSIONS("broadcast matrix dim2", tensor.size(axis2), mat.cols());

                Tensor<T> result = tensor;  // Copy

                // Create index vector for iteration
                std::vector<core::index_t> indices(tensor.ndim(), 0);

                do {
                    core::usize flat_index = flatten_index(indices, tensor.strides());
                    result[flat_index] += mat(indices[axis1], indices[axis2]);
                } while (increment_indices(indices, tensor.shape()));

                return result;
            }

            // Advanced broadcasting utilities

            // Expand dimensions by adding size-1 dimensions
            template<typename T>
            PSI_NODISCARD Tensor<T> expand_dims(const Tensor<T>& tensor, core::index_t axis) {
                Shape new_shape = tensor.shape();
                PSI_ASSERT(axis >= 0 && axis <= static_cast<core::index_t>(new_shape.size()),
                    "Invalid axis for expand_dims");

                new_shape.insert(new_shape.begin() + axis, 1);
                return tensor.reshape(new_shape);
            }

            // Squeeze dimensions of size 1
            template<typename T>
            PSI_NODISCARD Tensor<T> squeeze_dims(const Tensor<T>& tensor, core::index_t axis = -1) {
                Shape new_shape;

                if (axis == -1) {
                    // Squeeze all dimensions of size 1
                    for (core::usize i = 0; i < tensor.shape().size(); ++i) {
                        if (tensor.shape()[i] != 1) {
                            new_shape.push_back(tensor.shape()[i]);
                        }
                    }
                }
                else {
                    // Squeeze specific axis
                    PSI_BOUNDS_CHECK_DIM("squeeze axis", axis, tensor.ndim());
                    PSI_ASSERT(tensor.size(axis) == 1, "Can only squeeze dimensions of size 1");

                    for (core::usize i = 0; i < tensor.shape().size(); ++i) {
                        if (static_cast<core::index_t>(i) != axis) {
                            new_shape.push_back(tensor.shape()[i]);
                        }
                    }
                }

                if (new_shape.empty()) {
                    new_shape.push_back(1);  // Scalar tensor
                }

                return tensor.reshape(new_shape);
            }

            // Tile tensor along specified axes
            template<typename T>
            PSI_NODISCARD Tensor<T> tile(const Tensor<T>& tensor, const Shape& multiples) {
                PSI_ASSERT(multiples.size() == tensor.ndim(),
                    "Number of multiples must match tensor dimensions");

                Shape new_shape(tensor.ndim());
                for (core::usize i = 0; i < tensor.ndim(); ++i) {
                    new_shape[i] = tensor.shape()[i] * multiples[i];
                }

                Tensor<T> result(new_shape, tensor.device_id());

                // Fill result by tiling
                std::vector<core::index_t> result_indices(tensor.ndim(), 0);

                do {
                    // Map result indices to original tensor indices
                    std::vector<core::index_t> orig_indices(tensor.ndim());
                    for (core::usize i = 0; i < tensor.ndim(); ++i) {
                        orig_indices[i] = result_indices[i] % tensor.shape()[i];
                    }

                    core::usize result_flat = flatten_index(result_indices, result.strides());
                    core::usize orig_flat = flatten_index(orig_indices, tensor.strides());

                    result[result_flat] = tensor[orig_flat];

                } while (increment_indices(result_indices, result.shape()));

                return result;
            }

            // Repeat tensor elements along specified axes
            template<typename T>
            PSI_NODISCARD Tensor<T> repeat(const Tensor<T>& tensor, const Shape& repeats, core::index_t axis) {
                PSI_BOUNDS_CHECK_DIM("repeat axis", axis, tensor.ndim());
                PSI_CHECK_DIMENSIONS("repeat counts", tensor.shape()[axis], repeats.size());

                // Compute new size along the specified axis
                core::usize new_axis_size = 0;
                for (core::usize count : repeats) {
                    new_axis_size += count;
                }

                Shape new_shape = tensor.shape();
                new_shape[axis] = new_axis_size;

                Tensor<T> result(new_shape, tensor.device_id());

                // Fill result by repeating elements
                std::vector<core::index_t> result_indices(tensor.ndim(), 0);

                do {
                    // Map result index along repeat axis to original index
                    std::vector<core::index_t> orig_indices = result_indices;

                    core::usize cumsum = 0;
                    core::usize orig_axis_idx = 0;

                    for (core::usize i = 0; i < repeats.size(); ++i) {
                        if (result_indices[axis] < static_cast<core::index_t>(cumsum + repeats[i])) {
                            orig_axis_idx = i;
                            break;
                        }
                        cumsum += repeats[i];
                    }

                    orig_indices[axis] = static_cast<core::index_t>(orig_axis_idx);

                    core::usize result_flat = flatten_index(result_indices, result.strides());
                    core::usize orig_flat = flatten_index(orig_indices, tensor.strides());

                    result[result_flat] = tensor[orig_flat];

                } while (increment_indices(result_indices, result.shape()));

                return result;
            }

            // Helper function to increment multi-dimensional indices
            inline bool increment_indices(std::vector<core::index_t>& indices, const Shape& shape) {
                for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
                    if (++indices[i] < static_cast<core::index_t>(shape[i])) {
                        return true;
                    }
                    indices[i] = 0;
                }
                return false;
            }

        } // namespace ops
    } // namespace math
} // namespace psi
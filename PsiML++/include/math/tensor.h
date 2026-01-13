#pragma once

#include "../core/types.h"
#include "../core/config.h"
#include "../core/memory.h"
#include "../core/exception.h"
#include "../core/device.h"
#include "vector.h"
#include "matrix.h"
#include <initializer_list>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>

namespace psi {
    namespace math {

        // Shape type for tensor dimensions
        using Shape = std::vector<core::usize>;

        // Stride calculation helper
        inline Shape compute_strides(const Shape& shape) {
            Shape strides(shape.size());
            if (!shape.empty()) {
                strides.back() = 1;
                for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
            }
            return strides;
        }

        // Compute total size from shape
        inline core::usize compute_size(const Shape& shape) {
            return std::accumulate(shape.begin(), shape.end(),
                core::usize{ 1 }, std::multiplies<core::usize>());
        }

        // Convert multi-dimensional index to flat index
        inline core::usize flatten_index(const std::vector<core::index_t>& indices,
            const Shape& strides) {
            PSI_ASSERT(indices.size() == strides.size(), "Index dimension mismatch");
            core::usize flat_index = 0;
            for (core::usize i = 0; i < indices.size(); ++i) {
                flat_index += static_cast<core::usize>(indices[i]) * strides[i];
            }
            return flat_index;
        }

        template<typename T>
        class Tensor {
        public:
            using value_type = T;
            using size_type = core::usize;
            using index_type = core::index_t;
            using pointer = T*;
            using const_pointer = const T*;
            using reference = T&;
            using const_reference = const T&;

            // Constructors
            Tensor() : data_(nullptr), shape_(), strides_(), device_id_(0) {}

            explicit Tensor(const Shape& shape, core::device_id_t device_id = 0)
                : shape_(shape)
                , strides_(compute_strides(shape))
                , device_id_(device_id) {
                size_type total_size = compute_size(shape);
                data_ = core::allocate<T>(total_size, device_id);
                std::fill(data_, data_ + total_size, T{});
            }

            Tensor(const Shape& shape, const T& value, core::device_id_t device_id = 0)
                : shape_(shape)
                , strides_(compute_strides(shape))
                , device_id_(device_id) {
                size_type total_size = compute_size(shape);
                data_ = core::allocate<T>(total_size, device_id);
                std::fill(data_, data_ + total_size, value);
            }

            // Initializer list constructor for shape (resolves ambiguity with Vector/Matrix constructors)
            explicit Tensor(std::initializer_list<core::usize> shape, core::device_id_t device_id = 0)
                : shape_(shape)
                , strides_(compute_strides(shape_))
                , device_id_(device_id) {
                size_type total_size = compute_size(shape_);
                data_ = core::allocate<T>(total_size, device_id);
                std::fill(data_, data_ + total_size, T{});
            }

            // Initializer list constructor with value
            Tensor(std::initializer_list<core::usize> shape, const T& value, core::device_id_t device_id = 0)
                : shape_(shape)
                , strides_(compute_strides(shape_))
                , device_id_(device_id) {
                size_type total_size = compute_size(shape_);
                data_ = core::allocate<T>(total_size, device_id);
                std::fill(data_, data_ + total_size, value);
            }

            // 1D tensor from vector
            explicit Tensor(const Vector<T>& vec)
                : shape_({ vec.size() })
                , strides_({ 1 })
                , device_id_(vec.device_id()) {
                data_ = core::allocate<T>(vec.size(), device_id_);
                std::copy(vec.data(), vec.data() + vec.size(), data_);
            }

            // 2D tensor from matrix
            explicit Tensor(const Matrix<T>& mat)
                : shape_({ mat.rows(), mat.cols() })
                , strides_(compute_strides(shape_))
                , device_id_(mat.device_id()) {
                data_ = core::allocate<T>(mat.size(), device_id_);
                std::copy(mat.data(), mat.data() + mat.size(), data_);
            }

            // Copy constructor
            Tensor(const Tensor& other)
                : shape_(other.shape_)
                , strides_(other.strides_)
                , device_id_(other.device_id_) {
                size_type total_size = compute_size(shape_);
                data_ = core::allocate<T>(total_size, device_id_);
                std::copy(other.data_, other.data_ + total_size, data_);
            }

            // Move constructor
            Tensor(Tensor&& other) noexcept
                : data_(other.data_)
                , shape_(std::move(other.shape_))
                , strides_(std::move(other.strides_))
                , device_id_(other.device_id_) {
                other.data_ = nullptr;
            }

            // Destructor
            ~Tensor() {
                if (data_) {
                    core::deallocate<T>(data_, compute_size(shape_), device_id_);
                }
            }

            // Assignment operators
            Tensor& operator=(const Tensor& other) {
                if (this != &other) {
                    if (data_) {
                        core::deallocate<T>(data_, compute_size(shape_), device_id_);
                    }
                    shape_ = other.shape_;
                    strides_ = other.strides_;
                    device_id_ = other.device_id_;
                    size_type total_size = compute_size(shape_);
                    data_ = core::allocate<T>(total_size, device_id_);
                    std::copy(other.data_, other.data_ + total_size, data_);
                }
                return *this;
            }

            Tensor& operator=(Tensor&& other) noexcept {
                if (this != &other) {
                    if (data_) {
                        core::deallocate<T>(data_, compute_size(shape_), device_id_);
                    }
                    data_ = other.data_;
                    shape_ = std::move(other.shape_);
                    strides_ = std::move(other.strides_);
                    device_id_ = other.device_id_;
                    other.data_ = nullptr;
                }
                return *this;
            }

            // Element access
            template<typename... Indices>
            PSI_NODISCARD reference operator()(Indices... indices) {
                std::vector<index_type> idx = { static_cast<index_type>(indices)... };
                PSI_ASSERT(idx.size() == shape_.size(), "Number of indices must match tensor dimensions");

                for (size_type i = 0; i < idx.size(); ++i) {
                    PSI_BOUNDS_CHECK_DIM("dimension " + std::to_string(i), idx[i], shape_[i]);
                }

                size_type flat_index = flatten_index(idx, strides_);
                return data_[flat_index];
            }

            template<typename... Indices>
            PSI_NODISCARD const_reference operator()(Indices... indices) const {
                std::vector<index_type> idx = { static_cast<index_type>(indices)... };
                PSI_ASSERT(idx.size() == shape_.size(), "Number of indices must match tensor dimensions");

                for (size_type i = 0; i < idx.size(); ++i) {
                    PSI_BOUNDS_CHECK_DIM("dimension " + std::to_string(i), idx[i], shape_[i]);
                }

                size_type flat_index = flatten_index(idx, strides_);
                return data_[flat_index];
            }

            PSI_NODISCARD reference at(const std::vector<index_type>& indices) {
                PSI_ASSERT(indices.size() == shape_.size(), "Number of indices must match tensor dimensions");

                for (size_type i = 0; i < indices.size(); ++i) {
                    PSI_BOUNDS_CHECK_DIM("dimension " + std::to_string(i), indices[i], shape_[i]);
                }

                size_type flat_index = flatten_index(indices, strides_);
                return data_[flat_index];
            }

            PSI_NODISCARD const_reference at(const std::vector<index_type>& indices) const {
                PSI_ASSERT(indices.size() == shape_.size(), "Number of indices must match tensor dimensions");

                for (size_type i = 0; i < indices.size(); ++i) {
                    PSI_BOUNDS_CHECK_DIM("dimension " + std::to_string(i), indices[i], shape_[i]);
                }

                size_type flat_index = flatten_index(indices, strides_);
                return data_[flat_index];
            }

            // Flat indexing
            PSI_NODISCARD reference operator[](index_type index) {
                PSI_BOUNDS_CHECK(index, compute_size(shape_));
                return data_[index];
            }

            PSI_NODISCARD const_reference operator[](index_type index) const {
                PSI_BOUNDS_CHECK(index, compute_size(shape_));
                return data_[index];
            }

            // Properties
            PSI_NODISCARD const Shape& shape() const noexcept { return shape_; }
            PSI_NODISCARD const Shape& strides() const noexcept { return strides_; }
            PSI_NODISCARD size_type ndim() const noexcept { return shape_.size(); }
            PSI_NODISCARD size_type size() const noexcept { return compute_size(shape_); }
            PSI_NODISCARD bool empty() const noexcept { return size() == 0; }
            PSI_NODISCARD size_type size(size_type dim) const {
                PSI_BOUNDS_CHECK_DIM("dimension", dim, shape_.size());
                return shape_[dim];
            }

            // Data access
            PSI_NODISCARD pointer data() noexcept { return data_; }
            PSI_NODISCARD const_pointer data() const noexcept { return data_; }

            // Device management
            PSI_NODISCARD core::device_id_t device_id() const noexcept { return device_id_; }

            Tensor to_device(core::device_id_t new_device_id) const {
                if (new_device_id == device_id_) {
                    return *this;  // Copy constructor
                }

                Tensor result(shape_, new_device_id);
                std::copy(data_, data_ + size(), result.data_);
                return result;
            }

            // Shape manipulation
            PSI_NODISCARD Tensor reshape(const Shape& new_shape) const {
                PSI_ASSERT(compute_size(new_shape) == size(),
                    "New shape must have same total size");

                Tensor result;
                result.shape_ = new_shape;
                result.strides_ = compute_strides(new_shape);
                result.device_id_ = device_id_;
                result.data_ = core::allocate<T>(size(), device_id_);
                std::copy(data_, data_ + size(), result.data_);

                return result;
            }

            void reshape_inplace(const Shape& new_shape) {
                PSI_ASSERT(compute_size(new_shape) == size(),
                    "New shape must have same total size");
                shape_ = new_shape;
                strides_ = compute_strides(new_shape);
            }

            PSI_NODISCARD Tensor squeeze() const {
                Shape new_shape;
                for (size_type dim : shape_) {
                    if (dim != 1) {
                        new_shape.push_back(dim);
                    }
                }
                if (new_shape.empty()) {
                    new_shape.push_back(1);  // Scalar tensor
                }
                return reshape(new_shape);
            }

            PSI_NODISCARD Tensor unsqueeze(size_type dim) const {
                PSI_ASSERT(dim <= shape_.size(), "Dimension index out of range");
                Shape new_shape = shape_;
                new_shape.insert(new_shape.begin() + dim, 1);
                return reshape(new_shape);
            }

            PSI_NODISCARD Tensor transpose(size_type dim1, size_type dim2) const {
                PSI_BOUNDS_CHECK_DIM("dim1", dim1, shape_.size());
                PSI_BOUNDS_CHECK_DIM("dim2", dim2, shape_.size());

                if (dim1 == dim2) {
                    return *this;  // Copy constructor
                }

                Shape new_shape = shape_;
                std::swap(new_shape[dim1], new_shape[dim2]);

                Tensor result(new_shape, device_id_);

                // Copy with transposed indices
                std::vector<index_type> indices(shape_.size(), 0);
                do {
                    std::vector<index_type> new_indices = indices;
                    std::swap(new_indices[dim1], new_indices[dim2]);

                    size_type old_flat = flatten_index(indices, strides_);
                    Shape new_strides = compute_strides(new_shape);
                    size_type new_flat = flatten_index(new_indices, new_strides);

                    result.data_[new_flat] = data_[old_flat];

                } while (increment_indices(indices, shape_));

                return result;
            }

            // View operations
            PSI_NODISCARD Vector<T> as_vector() const {
                PSI_ASSERT(ndim() == 1, "Can only convert 1D tensor to vector");
                Vector<T> result(shape_[0], device_id_);
                std::copy(data_, data_ + size(), result.data());
                return result;
            }

            PSI_NODISCARD Matrix<T> as_matrix() const {
                PSI_ASSERT(ndim() == 2, "Can only convert 2D tensor to matrix");
                Matrix<T> result(shape_[0], shape_[1], device_id_);
                std::copy(data_, data_ + size(), result.data());
                return result;
            }

            // Modifiers
            void resize(const Shape& new_shape, const T& value = T{}) {
                size_type new_size = compute_size(new_shape);
                size_type old_size = size();

                T* new_data = core::allocate<T>(new_size, device_id_);

                if (data_) {
                    size_type copy_size = std::min(old_size, new_size);
                    std::copy(data_, data_ + copy_size, new_data);

                    // Fill new elements
                    if (new_size > old_size) {
                        std::fill(new_data + copy_size, new_data + new_size, value);
                    }

                    core::deallocate<T>(data_, old_size, device_id_);
                }
                else {
                    std::fill(new_data, new_data + new_size, value);
                }

                data_ = new_data;
                shape_ = new_shape;
                strides_ = compute_strides(new_shape);
            }

            void clear() {
                if (data_) {
                    core::deallocate<T>(data_, size(), device_id_);
                    data_ = nullptr;
                }
                shape_.clear();
                strides_.clear();
            }

            void fill(const T& value) {
                std::fill(data_, data_ + size(), value);
            }

            void swap(Tensor& other) noexcept {
                std::swap(data_, other.data_);
                std::swap(shape_, other.shape_);
                std::swap(strides_, other.strides_);
                std::swap(device_id_, other.device_id_);
            }

            // Mathematical operations
            PSI_NODISCARD T sum() const {
                T result{};
                for (size_type i = 0; i < size(); ++i) {
                    result += data_[i];
                }
                return result;
            }

            PSI_NODISCARD T mean() const {
                PSI_ASSERT(size() > 0, "Cannot compute mean of empty tensor");
                return sum() / static_cast<T>(size());
            }

            PSI_NODISCARD T min() const {
                PSI_ASSERT(size() > 0, "Cannot find min of empty tensor");
                return *std::min_element(data_, data_ + size());
            }

            PSI_NODISCARD T max() const {
                PSI_ASSERT(size() > 0, "Cannot find max of empty tensor");
                return *std::max_element(data_, data_ + size());
            }

            PSI_NODISCARD T norm() const {
                T sum_sq{};
                for (size_type i = 0; i < size(); ++i) {
                    sum_sq += data_[i] * data_[i];
                }
                return std::sqrt(sum_sq);
            }

            // Element-wise operations
            Tensor& operator+=(const Tensor& other) {
                check_compatible_shapes(other, "addition");
                for (size_type i = 0; i < size(); ++i) {
                    data_[i] += other.data_[i];
                }
                return *this;
            }

            Tensor& operator-=(const Tensor& other) {
                check_compatible_shapes(other, "subtraction");
                for (size_type i = 0; i < size(); ++i) {
                    data_[i] -= other.data_[i];
                }
                return *this;
            }

            Tensor& operator*=(const Tensor& other) {
                check_compatible_shapes(other, "element-wise multiplication");
                for (size_type i = 0; i < size(); ++i) {
                    data_[i] *= other.data_[i];
                }
                return *this;
            }

            Tensor& operator/=(const Tensor& other) {
                check_compatible_shapes(other, "element-wise division");
                for (size_type i = 0; i < size(); ++i) {
                    data_[i] /= other.data_[i];
                }
                return *this;
            }

            // Scalar operations
            Tensor& operator+=(const T& scalar) {
                for (size_type i = 0; i < size(); ++i) {
                    data_[i] += scalar;
                }
                return *this;
            }

            Tensor& operator-=(const T& scalar) {
                for (size_type i = 0; i < size(); ++i) {
                    data_[i] -= scalar;
                }
                return *this;
            }

            Tensor& operator*=(const T& scalar) {
                for (size_type i = 0; i < size(); ++i) {
                    data_[i] *= scalar;
                }
                return *this;
            }

            Tensor& operator/=(const T& scalar) {
                for (size_type i = 0; i < size(); ++i) {
                    data_[i] /= scalar;
                }
                return *this;
            }

            // Unary operators
            PSI_NODISCARD Tensor operator-() const {
                Tensor result(shape_, device_id_);
                for (size_type i = 0; i < size(); ++i) {
                    result.data_[i] = -data_[i];
                }
                return result;
            }

            // Apply function
            template<typename Func>
            Tensor& apply(Func func) {
                for (size_type i = 0; i < size(); ++i) {
                    data_[i] = func(data_[i]);
                }
                return *this;
            }

            template<typename Func>
            PSI_NODISCARD Tensor map(Func func) const {
                Tensor result(shape_, device_id_);
                for (size_type i = 0; i < size(); ++i) {
                    result.data_[i] = func(data_[i]);
                }
                return result;
            }

            // Static factory methods
            static Tensor zeros(const Shape& shape, core::device_id_t device_id = 0) {
                return Tensor(shape, T{}, device_id);
            }

            static Tensor ones(const Shape& shape, core::device_id_t device_id = 0) {
                return Tensor(shape, T{ 1 }, device_id);
            }

            static Tensor full(const Shape& shape, const T& value, core::device_id_t device_id = 0) {
                return Tensor(shape, value, device_id);
            }

        private:
            T* data_;
            Shape shape_;
            Shape strides_;
            core::device_id_t device_id_;

            void check_compatible_shapes(const Tensor& other, const std::string& operation) const {
                PSI_ASSERT(shape_ == other.shape_,
                    "Shape mismatch in tensor " + operation);
            }

            // Helper for iterating through multi-dimensional indices
            bool increment_indices(std::vector<index_type>& indices, const Shape& shape) const {
                for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
                    if (++indices[i] < static_cast<index_type>(shape[i])) {
                        return true;
                    }
                    indices[i] = 0;
                }
                return false;
            }
        };

        // Non-member operators

        // Element-wise operations
        template<typename T>
        PSI_NODISCARD Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
            Tensor<T> result(lhs);
            return result += rhs;
        }

        template<typename T>
        PSI_NODISCARD Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs) {
            Tensor<T> result(lhs);
            return result -= rhs;
        }

        template<typename T>
        PSI_NODISCARD Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs) {
            Tensor<T> result(lhs);
            return result *= rhs;
        }

        template<typename T>
        PSI_NODISCARD Tensor<T> operator/(const Tensor<T>& lhs, const Tensor<T>& rhs) {
            Tensor<T> result(lhs);
            return result /= rhs;
        }

        // Scalar operations
        template<typename T>
        PSI_NODISCARD Tensor<T> operator+(const Tensor<T>& tensor, const T& scalar) {
            Tensor<T> result(tensor);
            return result += scalar;
        }

        template<typename T>
        PSI_NODISCARD Tensor<T> operator+(const T& scalar, const Tensor<T>& tensor) {
            return tensor + scalar;
        }

        template<typename T>
        PSI_NODISCARD Tensor<T> operator-(const Tensor<T>& tensor, const T& scalar) {
            Tensor<T> result(tensor);
            return result -= scalar;
        }

        template<typename T>
        PSI_NODISCARD Tensor<T> operator-(const T& scalar, const Tensor<T>& tensor) {
            Tensor<T> result(tensor.shape(), tensor.device_id());
            for (typename Tensor<T>::size_type i = 0; i < tensor.size(); ++i) {
                result[i] = scalar - tensor[i];
            }
            return result;
        }

        template<typename T>
        PSI_NODISCARD Tensor<T> operator*(const Tensor<T>& tensor, const T& scalar) {
            Tensor<T> result(tensor);
            return result *= scalar;
        }

        template<typename T>
        PSI_NODISCARD Tensor<T> operator*(const T& scalar, const Tensor<T>& tensor) {
            return tensor * scalar;
        }

        template<typename T>
        PSI_NODISCARD Tensor<T> operator/(const Tensor<T>& tensor, const T& scalar) {
            Tensor<T> result(tensor);
            return result /= scalar;
        }

        // Comparison operators
        template<typename T>
        PSI_NODISCARD bool operator==(const Tensor<T>& lhs, const Tensor<T>& rhs) {
            if (lhs.shape() != rhs.shape()) return false;
            return std::equal(lhs.data(), lhs.data() + lhs.size(), rhs.data());
        }

        template<typename T>
        PSI_NODISCARD bool operator!=(const Tensor<T>& lhs, const Tensor<T>& rhs) {
            return !(lhs == rhs);
        }

        // Stream operator
        template<typename T>
        std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
            os << "Tensor(shape=[";
            for (core::usize i = 0; i < tensor.shape().size(); ++i) {
                if (i > 0) os << ", ";
                os << tensor.shape()[i];
            }
            os << "], data=";

            if (tensor.size() <= 20) {
                os << "[";
                for (typename Tensor<T>::size_type i = 0; i < tensor.size(); ++i) {
                    if (i > 0) os << ", ";
                    os << tensor[i];
                }
                os << "]";
            }
            else {
                os << "[" << tensor[0] << ", " << tensor[1] << ", ..., "
                    << tensor[tensor.size() - 2] << ", " << tensor[tensor.size() - 1] << "]";
            }
            os << ")";
            return os;
        }

        // Type aliases
        using Tensor32 = Tensor<core::f32>;
        using Tensor64 = Tensor<core::f64>;
        using TensorI32 = Tensor<core::i32>;
        using TensorI64 = Tensor<core::i64>;

    } // namespace math
} // namespace psi
#pragma once

#include "../core/types.h"
#include "../core/config.h"
#include "../core/memory.h"
#include "../core/exception.h"
#include "../core/device.h"
#include "vector.h"
#include <initializer_list>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace psi {
    namespace math {

        template<typename T>
        class Matrix {
        public:
            using value_type = T;
            using size_type = core::usize;
            using index_type = core::index_t;
            using pointer = T*;
            using const_pointer = const T*;
            using reference = T&;
            using const_reference = const T&;

            // Constructors
            Matrix() : data_(nullptr), rows_(0), cols_(0), device_id_(0) {}

            Matrix(size_type rows, size_type cols, core::device_id_t device_id = 0)
                : data_(core::allocate<T>(rows* cols, device_id))
                , rows_(rows)
                , cols_(cols)
                , device_id_(device_id) {
                std::fill(data_, data_ + rows * cols, T{});
            }

            Matrix(size_type rows, size_type cols, const T& value, core::device_id_t device_id = 0)
                : data_(core::allocate<T>(rows* cols, device_id))
                , rows_(rows)
                , cols_(cols)
                , device_id_(device_id) {
                std::fill(data_, data_ + rows * cols, value);
            }

            Matrix(std::initializer_list<std::initializer_list<T>> init, core::device_id_t device_id = 0)
                : rows_(init.size())
                , cols_(init.size() > 0 ? init.begin()->size() : 0)
                , device_id_(device_id) {

                data_ = core::allocate<T>(rows_ * cols_, device_id_);

                size_type row = 0;
                for (const auto& row_init : init) {
                    PSI_ASSERT(row_init.size() == cols_, "All rows must have the same number of columns");
                    std::copy(row_init.begin(), row_init.end(), data_ + row * cols_);
                    ++row;
                }
            }

            // Copy constructor
            Matrix(const Matrix& other)
                : data_(core::allocate<T>(other.rows_* other.cols_, other.device_id_))
                , rows_(other.rows_)
                , cols_(other.cols_)
                , device_id_(other.device_id_) {
                std::copy(other.data_, other.data_ + rows_ * cols_, data_);
            }

            // Move constructor
            Matrix(Matrix&& other) noexcept
                : data_(other.data_)
                , rows_(other.rows_)
                , cols_(other.cols_)
                , device_id_(other.device_id_) {
                other.data_ = nullptr;
                other.rows_ = 0;
                other.cols_ = 0;
            }

            // Destructor
            ~Matrix() {
                if (data_) {
                    core::deallocate<T>(data_, rows_ * cols_, device_id_);
                }
            }

            // Assignment operators
            Matrix& operator=(const Matrix& other) {
                if (this != &other) {
                    if (data_) {
                        core::deallocate<T>(data_, rows_ * cols_, device_id_);
                    }
                    data_ = core::allocate<T>(other.rows_ * other.cols_, other.device_id_);
                    rows_ = other.rows_;
                    cols_ = other.cols_;
                    device_id_ = other.device_id_;
                    std::copy(other.data_, other.data_ + rows_ * cols_, data_);
                }
                return *this;
            }

            Matrix& operator=(Matrix&& other) noexcept {
                if (this != &other) {
                    if (data_) {
                        core::deallocate<T>(data_, rows_ * cols_, device_id_);
                    }
                    data_ = other.data_;
                    rows_ = other.rows_;
                    cols_ = other.cols_;
                    device_id_ = other.device_id_;
                    other.data_ = nullptr;
                    other.rows_ = 0;
                    other.cols_ = 0;
                }
                return *this;
            }

            // Element access
            PSI_NODISCARD reference operator()(index_type row, index_type col) {
                PSI_BOUNDS_CHECK_DIM("row", row, rows_);
                PSI_BOUNDS_CHECK_DIM("col", col, cols_);
                return data_[row * cols_ + col];
            }

            PSI_NODISCARD const_reference operator()(index_type row, index_type col) const {
                PSI_BOUNDS_CHECK_DIM("row", row, rows_);
                PSI_BOUNDS_CHECK_DIM("col", col, cols_);
                return data_[row * cols_ + col];
            }

            PSI_NODISCARD reference at(index_type row, index_type col) {
                PSI_BOUNDS_CHECK_DIM("row", row, rows_);
                PSI_BOUNDS_CHECK_DIM("col", col, cols_);
                return data_[row * cols_ + col];
            }

            PSI_NODISCARD const_reference at(index_type row, index_type col) const {
                PSI_BOUNDS_CHECK_DIM("row", row, rows_);
                PSI_BOUNDS_CHECK_DIM("col", col, cols_);
                return data_[row * cols_ + col];
            }

            // Flat indexing
            PSI_NODISCARD reference operator[](index_type index) {
                PSI_BOUNDS_CHECK(index, rows_ * cols_);
                return data_[index];
            }

            PSI_NODISCARD const_reference operator[](index_type index) const {
                PSI_BOUNDS_CHECK(index, rows_ * cols_);
                return data_[index];
            }

            // Capacity
            PSI_NODISCARD size_type rows() const noexcept { return rows_; }
            PSI_NODISCARD size_type cols() const noexcept { return cols_; }
            PSI_NODISCARD size_type size() const noexcept { return rows_ * cols_; }
            PSI_NODISCARD bool empty() const noexcept { return size() == 0; }
            PSI_NODISCARD bool is_square() const noexcept { return rows_ == cols_; }

            // Data access
            PSI_NODISCARD pointer data() noexcept { return data_; }
            PSI_NODISCARD const_pointer data() const noexcept { return data_; }

            // Device management
            PSI_NODISCARD core::device_id_t device_id() const noexcept { return device_id_; }

            Matrix to_device(core::device_id_t new_device_id) const {
                if (new_device_id == device_id_) {
                    return *this;  // Copy constructor
                }

                Matrix result(rows_, cols_, T{}, new_device_id);
                std::copy(data_, data_ + rows_ * cols_, result.data_);
                return result;
            }

            // Row and column access
            PSI_NODISCARD Vector<T> get_row(index_type row) const {
                PSI_BOUNDS_CHECK_DIM("row", row, rows_);
                Vector<T> result(cols_, device_id_);
                std::copy(data_ + row * cols_, data_ + (row + 1) * cols_, result.data());
                return result;
            }

            PSI_NODISCARD Vector<T> get_col(index_type col) const {
                PSI_BOUNDS_CHECK_DIM("col", col, cols_);
                Vector<T> result(rows_, device_id_);
                for (size_type i = 0; i < rows_; ++i) {
                    result[i] = data_[i * cols_ + col];
                }
                return result;
            }

            void set_row(index_type row, const Vector<T>& vec) {
                PSI_BOUNDS_CHECK_DIM("row", row, rows_);
                PSI_CHECK_DIMENSIONS("set_row", cols_, vec.size());
                std::copy(vec.data(), vec.data() + cols_, data_ + row * cols_);
            }

            void set_col(index_type col, const Vector<T>& vec) {
                PSI_BOUNDS_CHECK_DIM("col", col, cols_);
                PSI_CHECK_DIMENSIONS("set_col", rows_, vec.size());
                for (size_type i = 0; i < rows_; ++i) {
                    data_[i * cols_ + col] = vec[i];
                }
            }

            // Modifiers
            void resize(size_type new_rows, size_type new_cols, const T& value = T{}) {
                if (new_rows == rows_ && new_cols == cols_) return;

                T* new_data = core::allocate<T>(new_rows * new_cols, device_id_);

                // Copy existing data
                size_type min_rows = std::min(rows_, new_rows);
                size_type min_cols = std::min(cols_, new_cols);

                for (size_type i = 0; i < min_rows; ++i) {
                    for (size_type j = 0; j < min_cols; ++j) {
                        new_data[i * new_cols + j] = data_[i * cols_ + j];
                    }
                    // Fill new columns in existing rows
                    for (size_type j = min_cols; j < new_cols; ++j) {
                        new_data[i * new_cols + j] = value;
                    }
                }

                // Fill new rows
                for (size_type i = min_rows; i < new_rows; ++i) {
                    for (size_type j = 0; j < new_cols; ++j) {
                        new_data[i * new_cols + j] = value;
                    }
                }

                if (data_) {
                    core::deallocate<T>(data_, rows_ * cols_, device_id_);
                }

                data_ = new_data;
                rows_ = new_rows;
                cols_ = new_cols;
            }

            void clear() {
                if (data_) {
                    core::deallocate<T>(data_, rows_ * cols_, device_id_);
                    data_ = nullptr;
                }
                rows_ = 0;
                cols_ = 0;
            }

            void fill(const T& value) {
                std::fill(data_, data_ + rows_ * cols_, value);
            }

            void swap(Matrix& other) noexcept {
                std::swap(data_, other.data_);
                std::swap(rows_, other.rows_);
                std::swap(cols_, other.cols_);
                std::swap(device_id_, other.device_id_);
            }

            // Matrix operations
            PSI_NODISCARD Matrix transpose() const {
                Matrix result(cols_, rows_, T{}, device_id_);
                for (size_type i = 0; i < rows_; ++i) {
                    for (size_type j = 0; j < cols_; ++j) {
                        result(j, i) = (*this)(i, j);
                    }
                }
                return result;
            }

            void transpose_inplace() {
                if (!is_square()) {
                    PSI_THROW_SHAPE("In-place transpose requires square matrix");
                }

                for (size_type i = 0; i < rows_; ++i) {
                    for (size_type j = i + 1; j < cols_; ++j) {
                        std::swap((*this)(i, j), (*this)(j, i));
                    }
                }
            }

            PSI_NODISCARD T trace() const {
                PSI_ASSERT(is_square(), "Trace requires square matrix");
                T result{};
                for (size_type i = 0; i < rows_; ++i) {
                    result += (*this)(i, i);
                }
                return result;
            }

            PSI_NODISCARD T frobenius_norm() const {
                T sum_sq{};
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    sum_sq += data_[i] * data_[i];
                }
                return std::sqrt(sum_sq);
            }

            PSI_NODISCARD T sum() const {
                T result{};
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    result += data_[i];
                }
                return result;
            }

            PSI_NODISCARD T mean() const {
                PSI_ASSERT(size() > 0, "Cannot compute mean of empty matrix");
                return sum() / static_cast<T>(size());
            }

            PSI_NODISCARD T min() const {
                PSI_ASSERT(size() > 0, "Cannot find min of empty matrix");
                return *std::min_element(data_, data_ + rows_ * cols_);
            }

            PSI_NODISCARD T max() const {
                PSI_ASSERT(size() > 0, "Cannot find max of empty matrix");
                return *std::max_element(data_, data_ + rows_ * cols_);
            }

            // Matrix-vector multiplication
            PSI_NODISCARD Vector<T> operator*(const Vector<T>& vec) const {
                PSI_CHECK_DIMENSIONS("matrix-vector multiplication", cols_, vec.size());
                Vector<T> result(rows_, device_id_);

                for (size_type i = 0; i < rows_; ++i) {
                    T sum{};
                    for (size_type j = 0; j < cols_; ++j) {
                        sum += (*this)(i, j) * vec[j];
                    }
                    result[i] = sum;
                }
                return result;
            }

            // Element-wise operations with matrices
            Matrix& operator+=(const Matrix& other) {
                PSI_CHECK_DIMENSIONS("matrix addition rows", rows_, other.rows_);
                PSI_CHECK_DIMENSIONS("matrix addition cols", cols_, other.cols_);
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    data_[i] += other.data_[i];
                }
                return *this;
            }

            Matrix& operator-=(const Matrix& other) {
                PSI_CHECK_DIMENSIONS("matrix subtraction rows", rows_, other.rows_);
                PSI_CHECK_DIMENSIONS("matrix subtraction cols", cols_, other.cols_);
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    data_[i] -= other.data_[i];
                }
                return *this;
            }

            Matrix& operator*=(const Matrix& other) {
                PSI_CHECK_DIMENSIONS("element-wise multiplication rows", rows_, other.rows_);
                PSI_CHECK_DIMENSIONS("element-wise multiplication cols", cols_, other.cols_);
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    data_[i] *= other.data_[i];
                }
                return *this;
            }

            Matrix& operator/=(const Matrix& other) {
                PSI_CHECK_DIMENSIONS("element-wise division rows", rows_, other.rows_);
                PSI_CHECK_DIMENSIONS("element-wise division cols", cols_, other.cols_);
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    data_[i] /= other.data_[i];
                }
                return *this;
            }

            // Scalar operations
            Matrix& operator+=(const T& scalar) {
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    data_[i] += scalar;
                }
                return *this;
            }

            Matrix& operator-=(const T& scalar) {
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    data_[i] -= scalar;
                }
                return *this;
            }

            Matrix& operator*=(const T& scalar) {
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    data_[i] *= scalar;
                }
                return *this;
            }

            Matrix& operator/=(const T& scalar) {
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    data_[i] /= scalar;
                }
                return *this;
            }

            // Unary operators
            PSI_NODISCARD Matrix operator-() const {
                Matrix result(rows_, cols_, T{}, device_id_);
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    result.data_[i] = -data_[i];
                }
                return result;
            }

            // Apply function
            template<typename Func>
            Matrix& apply(Func func) {
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    data_[i] = func(data_[i]);
                }
                return *this;
            }

            template<typename Func>
            PSI_NODISCARD Matrix map(Func func) const {
                Matrix result(rows_, cols_, T{}, device_id_);
                for (size_type i = 0; i < rows_ * cols_; ++i) {
                    result.data_[i] = func(data_[i]);
                }
                return result;
            }

            // Static factory methods
            static Matrix zeros(size_type rows, size_type cols, core::device_id_t device_id = 0) {
                return Matrix(rows, cols, T{}, device_id);
            }

            static Matrix ones(size_type rows, size_type cols, core::device_id_t device_id = 0) {
                return Matrix(rows, cols, T{ 1 }, device_id);
            }

            static Matrix identity(size_type size, core::device_id_t device_id = 0) {
                Matrix result(size, size, T{}, device_id);
                for (size_type i = 0; i < size; ++i) {
                    result(i, i) = T{ 1 };
                }
                return result;
            }

            static Matrix diagonal(const Vector<T>& diag, core::device_id_t device_id = 0) {
                Matrix result(diag.size(), diag.size(), T{}, device_id);
                for (size_type i = 0; i < diag.size(); ++i) {
                    result(i, i) = diag[i];
                }
                return result;
            }

        private:
            T* data_;
            size_type rows_;
            size_type cols_;
            core::device_id_t device_id_;
        };

        // Non-member operators

        // Matrix-Matrix operations
        template<typename T>
        PSI_NODISCARD Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
            Matrix<T> result(lhs);
            return result += rhs;
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
            Matrix<T> result(lhs);
            return result -= rhs;
        }

        // Matrix multiplication
        template<typename T>
        PSI_NODISCARD Matrix<T> matmul(const Matrix<T>& lhs, const Matrix<T>& rhs) {
            PSI_CHECK_DIMENSIONS("matrix multiplication", lhs.cols(), rhs.rows());

            Matrix<T> result(lhs.rows(), rhs.cols(), T{}, lhs.device_id());

            for (typename Matrix<T>::size_type i = 0; i < lhs.rows(); ++i) {
                for (typename Matrix<T>::size_type j = 0; j < rhs.cols(); ++j) {
                    T sum{};
                    for (typename Matrix<T>::size_type k = 0; k < lhs.cols(); ++k) {
                        sum += lhs(i, k) * rhs(k, j);
                    }
                    result(i, j) = sum;
                }
            }
            return result;
        }

        // Element-wise multiplication
        template<typename T>
        PSI_NODISCARD Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
            Matrix<T> result(lhs);
            return result *= rhs;
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> operator/(const Matrix<T>& lhs, const Matrix<T>& rhs) {
            Matrix<T> result(lhs);
            return result /= rhs;
        }

        // Scalar operations
        template<typename T>
        PSI_NODISCARD Matrix<T> operator+(const Matrix<T>& mat, const T& scalar) {
            Matrix<T> result(mat);
            return result += scalar;
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> operator+(const T& scalar, const Matrix<T>& mat) {
            return mat + scalar;
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> operator-(const Matrix<T>& mat, const T& scalar) {
            Matrix<T> result(mat);
            return result -= scalar;
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> operator-(const T& scalar, const Matrix<T>& mat) {
            Matrix<T> result(mat.rows(), mat.cols(), T{}, mat.device_id());
            for (typename Matrix<T>::size_type i = 0; i < mat.size(); ++i) {
                result[i] = scalar - mat[i];
            }
            return result;
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> operator*(const Matrix<T>& mat, const T& scalar) {
            Matrix<T> result(mat);
            return result *= scalar;
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> operator*(const T& scalar, const Matrix<T>& mat) {
            return mat * scalar;
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> operator/(const Matrix<T>& mat, const T& scalar) {
            Matrix<T> result(mat);
            return result /= scalar;
        }

        // Comparison operators
        template<typename T>
        PSI_NODISCARD bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs) {
            if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) return false;
            return std::equal(lhs.data(), lhs.data() + lhs.size(), rhs.data());
        }

        template<typename T>
        PSI_NODISCARD bool operator!=(const Matrix<T>& lhs, const Matrix<T>& rhs) {
            return !(lhs == rhs);
        }

        // Stream operator
        template<typename T>
        std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
            os << "[";
            for (typename Matrix<T>::size_type i = 0; i < mat.rows(); ++i) {
                if (i > 0) os << ",\n ";
                os << "[";
                for (typename Matrix<T>::size_type j = 0; j < mat.cols(); ++j) {
                    if (j > 0) os << ", ";
                    os << mat(i, j);
                }
                os << "]";
            }
            os << "]";
            return os;
        }

        // Type aliases
        using Mat32 = Matrix<core::f32>;
        using Mat64 = Matrix<core::f64>;
        using MatI32 = Matrix<core::i32>;
        using MatI64 = Matrix<core::i64>;

    } // namespace math
} // namespace psi
#pragma once

#include "../core/types.h"
#include "../core/config.h"
#include "../core/memory.h"
#include "../core/exception.h"
#include "../core/device.h"
#include <initializer_list>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>

namespace psi {
    namespace math {

        template<typename T>
        class Vector {
        public:
            using value_type = T;
            using size_type = core::usize;
            using index_type = core::index_t;
            using pointer = T*;
            using const_pointer = const T*;
            using reference = T&;
            using const_reference = const T&;
            using iterator = T*;
            using const_iterator = const T*;

            // Constructors
            Vector() : data_(nullptr), size_(0), device_id_(0) {}

            explicit Vector(size_type size, core::device_id_t device_id = 0)
                : data_(core::allocate<T>(size, device_id))
                , size_(size)
                , device_id_(device_id) {
                std::fill(data_, data_ + size_, T{});
            }

            Vector(size_type size, const T& value, core::device_id_t device_id = 0)
                : data_(core::allocate<T>(size, device_id))
                , size_(size)
                , device_id_(device_id) {
                std::fill(data_, data_ + size_, value);
            }

            Vector(std::initializer_list<T> init, core::device_id_t device_id = 0)
                : data_(core::allocate<T>(init.size(), device_id))
                , size_(init.size())
                , device_id_(device_id) {
                std::copy(init.begin(), init.end(), data_);
            }

            // Copy constructor
            Vector(const Vector& other)
                : data_(core::allocate<T>(other.size_, other.device_id_))
                , size_(other.size_)
                , device_id_(other.device_id_) {
                std::copy(other.data_, other.data_ + size_, data_);
            }

            // Move constructor
            Vector(Vector&& other) noexcept
                : data_(other.data_)
                , size_(other.size_)
                , device_id_(other.device_id_) {
                other.data_ = nullptr;
                other.size_ = 0;
            }

            // Destructor
            ~Vector() {
                if (data_) {
                    core::deallocate<T>(data_, size_, device_id_);
                }
            }

            // Assignment operators
            Vector& operator=(const Vector& other) {
                if (this != &other) {
                    if (data_) {
                        core::deallocate<T>(data_, size_, device_id_);
                    }
                    data_ = core::allocate<T>(other.size_, other.device_id_);
                    size_ = other.size_;
                    device_id_ = other.device_id_;
                    std::copy(other.data_, other.data_ + size_, data_);
                }
                return *this;
            }

            Vector& operator=(Vector&& other) noexcept {
                if (this != &other) {
                    if (data_) {
                        core::deallocate<T>(data_, size_, device_id_);
                    }
                    data_ = other.data_;
                    size_ = other.size_;
                    device_id_ = other.device_id_;
                    other.data_ = nullptr;
                    other.size_ = 0;
                }
                return *this;
            }

            Vector& operator=(std::initializer_list<T> init) {
                resize(init.size());
                std::copy(init.begin(), init.end(), data_);
                return *this;
            }

            // Element access
            PSI_NODISCARD reference operator[](index_type index) {
                PSI_BOUNDS_CHECK(index, size_);
                return data_[index];
            }

            PSI_NODISCARD const_reference operator[](index_type index) const {
                PSI_BOUNDS_CHECK(index, size_);
                return data_[index];
            }

            PSI_NODISCARD reference at(index_type index) {
                PSI_BOUNDS_CHECK(index, size_);
                return data_[index];
            }

            PSI_NODISCARD const_reference at(index_type index) const {
                PSI_BOUNDS_CHECK(index, size_);
                return data_[index];
            }

            PSI_NODISCARD reference front() {
                PSI_ASSERT(size_ > 0, "Vector is empty");
                return data_[0];
            }

            PSI_NODISCARD const_reference front() const {
                PSI_ASSERT(size_ > 0, "Vector is empty");
                return data_[0];
            }

            PSI_NODISCARD reference back() {
                PSI_ASSERT(size_ > 0, "Vector is empty");
                return data_[size_ - 1];
            }

            PSI_NODISCARD const_reference back() const {
                PSI_ASSERT(size_ > 0, "Vector is empty");
                return data_[size_ - 1];
            }

            // Capacity
            PSI_NODISCARD size_type size() const noexcept { return size_; }
            PSI_NODISCARD bool empty() const noexcept { return size_ == 0; }

            // Data access
            PSI_NODISCARD pointer data() noexcept { return data_; }
            PSI_NODISCARD const_pointer data() const noexcept { return data_; }

            // Device management
            PSI_NODISCARD core::device_id_t device_id() const noexcept { return device_id_; }

            Vector to_device(core::device_id_t new_device_id) const {
                if (new_device_id == device_id_) {
                    return *this;  // Copy constructor
                }

                Vector result(size_, new_device_id);
                std::copy(data_, data_ + size_, result.data_);
                return result;
            }

            // Iterators
            PSI_NODISCARD iterator begin() noexcept { return data_; }
            PSI_NODISCARD iterator end() noexcept { return data_ + size_; }
            PSI_NODISCARD const_iterator begin() const noexcept { return data_; }
            PSI_NODISCARD const_iterator end() const noexcept { return data_ + size_; }
            PSI_NODISCARD const_iterator cbegin() const noexcept { return data_; }
            PSI_NODISCARD const_iterator cend() const noexcept { return data_ + size_; }

            // Modifiers
            void resize(size_type new_size, const T& value = T{}) {
                if (new_size == size_) return;

                T* new_data = core::allocate<T>(new_size, device_id_);
                size_type copy_size = std::min(size_, new_size);

                if (data_) {
                    std::copy(data_, data_ + copy_size, new_data);
                    core::deallocate<T>(data_, size_, device_id_);
                }

                // Fill new elements
                if (new_size > size_) {
                    std::fill(new_data + copy_size, new_data + new_size, value);
                }

                data_ = new_data;
                size_ = new_size;
            }

            void clear() {
                if (data_) {
                    core::deallocate<T>(data_, size_, device_id_);
                    data_ = nullptr;
                }
                size_ = 0;
            }

            void fill(const T& value) {
                std::fill(data_, data_ + size_, value);
            }

            void swap(Vector& other) noexcept {
                std::swap(data_, other.data_);
                std::swap(size_, other.size_);
                std::swap(device_id_, other.device_id_);
            }

            // Mathematical operations
            PSI_NODISCARD T sum() const {
                T result{};
                for (size_type i = 0; i < size_; ++i) {
                    result += data_[i];
                }
                return result;
            }

            PSI_NODISCARD T mean() const {
                PSI_ASSERT(size_ > 0, "Cannot compute mean of empty vector");
                return sum() / static_cast<T>(size_);
            }

            PSI_NODISCARD T min() const {
                PSI_ASSERT(size_ > 0, "Cannot find min of empty vector");
                return *std::min_element(data_, data_ + size_);
            }

            PSI_NODISCARD T max() const {
                PSI_ASSERT(size_ > 0, "Cannot find max of empty vector");
                return *std::max_element(data_, data_ + size_);
            }

            PSI_NODISCARD T norm() const {
                T sum_sq{};
                for (size_type i = 0; i < size_; ++i) {
                    sum_sq += data_[i] * data_[i];
                }
                return std::sqrt(sum_sq);
            }

            PSI_NODISCARD T norm_squared() const {
                T sum_sq{};
                for (size_type i = 0; i < size_; ++i) {
                    sum_sq += data_[i] * data_[i];
                }
                return sum_sq;
            }

            PSI_NODISCARD T dot(const Vector& other) const {
                PSI_CHECK_DIMENSIONS("dot product", size_, other.size_);
                T result{};
                for (size_type i = 0; i < size_; ++i) {
                    result += data_[i] * other.data_[i];
                }
                return result;
            }

            void normalize() {
                T n = norm();
                if (n > T{}) {
                    for (size_type i = 0; i < size_; ++i) {
                        data_[i] /= n;
                    }
                }
            }

            PSI_NODISCARD Vector normalized() const {
                Vector result(*this);
                result.normalize();
                return result;
            }

            // Element-wise operations
            Vector& operator+=(const Vector& other) {
                PSI_CHECK_DIMENSIONS("vector addition", size_, other.size_);
                for (size_type i = 0; i < size_; ++i) {
                    data_[i] += other.data_[i];
                }
                return *this;
            }

            Vector& operator-=(const Vector& other) {
                PSI_CHECK_DIMENSIONS("vector subtraction", size_, other.size_);
                for (size_type i = 0; i < size_; ++i) {
                    data_[i] -= other.data_[i];
                }
                return *this;
            }

            Vector& operator*=(const Vector& other) {
                PSI_CHECK_DIMENSIONS("element-wise multiplication", size_, other.size_);
                for (size_type i = 0; i < size_; ++i) {
                    data_[i] *= other.data_[i];
                }
                return *this;
            }

            Vector& operator/=(const Vector& other) {
                PSI_CHECK_DIMENSIONS("element-wise division", size_, other.size_);
                for (size_type i = 0; i < size_; ++i) {
                    data_[i] /= other.data_[i];
                }
                return *this;
            }

            // Scalar operations
            Vector& operator+=(const T& scalar) {
                for (size_type i = 0; i < size_; ++i) {
                    data_[i] += scalar;
                }
                return *this;
            }

            Vector& operator-=(const T& scalar) {
                for (size_type i = 0; i < size_; ++i) {
                    data_[i] -= scalar;
                }
                return *this;
            }

            Vector& operator*=(const T& scalar) {
                for (size_type i = 0; i < size_; ++i) {
                    data_[i] *= scalar;
                }
                return *this;
            }

            Vector& operator/=(const T& scalar) {
                for (size_type i = 0; i < size_; ++i) {
                    data_[i] /= scalar;
                }
                return *this;
            }

            // Unary operators
            PSI_NODISCARD Vector operator-() const {
                Vector result(size_, device_id_);
                for (size_type i = 0; i < size_; ++i) {
                    result.data_[i] = -data_[i];
                }
                return result;
            }

            // Apply function
            template<typename Func>
            Vector& apply(Func func) {
                for (size_type i = 0; i < size_; ++i) {
                    data_[i] = func(data_[i]);
                }
                return *this;
            }

            template<typename Func>
            PSI_NODISCARD Vector map(Func func) const {
                Vector result(size_, device_id_);
                for (size_type i = 0; i < size_; ++i) {
                    result.data_[i] = func(data_[i]);
                }
                return result;
            }

        private:
            T* data_;
            size_type size_;
            core::device_id_t device_id_;
        };

        // Non-member binary operators
        template<typename T>
        PSI_NODISCARD Vector<T> operator+(const Vector<T>& lhs, const Vector<T>& rhs) {
            Vector<T> result(lhs);
            return result += rhs;
        }

        template<typename T>
        PSI_NODISCARD Vector<T> operator-(const Vector<T>& lhs, const Vector<T>& rhs) {
            Vector<T> result(lhs);
            return result -= rhs;
        }

        template<typename T>
        PSI_NODISCARD Vector<T> operator*(const Vector<T>& lhs, const Vector<T>& rhs) {
            Vector<T> result(lhs);
            return result *= rhs;
        }

        template<typename T>
        PSI_NODISCARD Vector<T> operator/(const Vector<T>& lhs, const Vector<T>& rhs) {
            Vector<T> result(lhs);
            return result /= rhs;
        }

        // Scalar operations
        template<typename T>
        PSI_NODISCARD Vector<T> operator+(const Vector<T>& vec, const T& scalar) {
            Vector<T> result(vec);
            return result += scalar;
        }

        template<typename T>
        PSI_NODISCARD Vector<T> operator+(const T& scalar, const Vector<T>& vec) {
            return vec + scalar;
        }

        template<typename T>
        PSI_NODISCARD Vector<T> operator-(const Vector<T>& vec, const T& scalar) {
            Vector<T> result(vec);
            return result -= scalar;
        }

        template<typename T>
        PSI_NODISCARD Vector<T> operator-(const T& scalar, const Vector<T>& vec) {
            Vector<T> result(vec.size(), vec.device_id());
            for (typename Vector<T>::size_type i = 0; i < vec.size(); ++i) {
                result[i] = scalar - vec[i];
            }
            return result;
        }

        template<typename T>
        PSI_NODISCARD Vector<T> operator*(const Vector<T>& vec, const T& scalar) {
            Vector<T> result(vec);
            return result *= scalar;
        }

        template<typename T>
        PSI_NODISCARD Vector<T> operator*(const T& scalar, const Vector<T>& vec) {
            return vec * scalar;
        }

        template<typename T>
        PSI_NODISCARD Vector<T> operator/(const Vector<T>& vec, const T& scalar) {
            Vector<T> result(vec);
            return result /= scalar;
        }

        // Comparison operators
        template<typename T>
        PSI_NODISCARD bool operator==(const Vector<T>& lhs, const Vector<T>& rhs) {
            if (lhs.size() != rhs.size()) return false;
            return std::equal(lhs.begin(), lhs.end(), rhs.begin());
        }

        template<typename T>
        PSI_NODISCARD bool operator!=(const Vector<T>& lhs, const Vector<T>& rhs) {
            return !(lhs == rhs);
        }

        // Stream operators
        template<typename T>
        std::ostream& operator<<(std::ostream& os, const Vector<T>& vec) {
            os << "[";
            for (typename Vector<T>::size_type i = 0; i < vec.size(); ++i) {
                if (i > 0) os << ", ";
                os << vec[i];
            }
            os << "]";
            return os;
        }

        // Type aliases
        using Vec32 = Vector<core::f32>;
        using Vec64 = Vector<core::f64>;
        using VecI32 = Vector<core::i32>;
        using VecI64 = Vector<core::i64>;

    } // namespace math
} // namespace psi
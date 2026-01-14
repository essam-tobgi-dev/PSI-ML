#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include <cmath>
#include <limits>

namespace psi {
    namespace ml {
        namespace preprocessing {

            // Standard Scaler (z-score normalization)
            template<typename T>
            class StandardScaler {
            public:
                StandardScaler() : fitted_(false) {}

                void fit(const math::Matrix<T>& X) {
                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    mean_ = math::Vector<T>(n_features);
                    std_ = math::Vector<T>(n_features);

                    // Compute mean
                    for (core::usize j = 0; j < n_features; ++j) {
                        T sum = T{0};
                        for (core::usize i = 0; i < n_samples; ++i) {
                            sum += X(i, j);
                        }
                        mean_[j] = sum / static_cast<T>(n_samples);
                    }

                    // Compute standard deviation
                    for (core::usize j = 0; j < n_features; ++j) {
                        T sum_sq = T{0};
                        for (core::usize i = 0; i < n_samples; ++i) {
                            T diff = X(i, j) - mean_[j];
                            sum_sq += diff * diff;
                        }
                        std_[j] = std::sqrt(sum_sq / static_cast<T>(n_samples));
                        if (std_[j] < std::numeric_limits<T>::epsilon()) {
                            std_[j] = T{1};  // Avoid division by zero
                        }
                    }

                    fitted_ = true;
                }

                math::Matrix<T> transform(const math::Matrix<T>& X) const {
                    PSI_ASSERT(fitted_, "StandardScaler must be fitted before transform");
                    PSI_ASSERT(X.cols() == mean_.size(), "Feature dimension mismatch");

                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> result(n_samples, n_features);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features; ++j) {
                            result(i, j) = (X(i, j) - mean_[j]) / std_[j];
                        }
                    }

                    return result;
                }

                math::Matrix<T> fit_transform(const math::Matrix<T>& X) {
                    fit(X);
                    return transform(X);
                }

                math::Matrix<T> inverse_transform(const math::Matrix<T>& X) const {
                    PSI_ASSERT(fitted_, "StandardScaler must be fitted before inverse_transform");
                    PSI_ASSERT(X.cols() == mean_.size(), "Feature dimension mismatch");

                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> result(n_samples, n_features);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features; ++j) {
                            result(i, j) = X(i, j) * std_[j] + mean_[j];
                        }
                    }

                    return result;
                }

                PSI_NODISCARD const math::Vector<T>& mean() const { return mean_; }
                PSI_NODISCARD const math::Vector<T>& std() const { return std_; }
                PSI_NODISCARD bool is_fitted() const { return fitted_; }

            private:
                math::Vector<T> mean_;
                math::Vector<T> std_;
                bool fitted_;
            };

            // Min-Max Scaler
            template<typename T>
            class MinMaxScaler {
            public:
                MinMaxScaler(T feature_min = T{0}, T feature_max = T{1})
                    : feature_min_(feature_min), feature_max_(feature_max), fitted_(false) {}

                void fit(const math::Matrix<T>& X) {
                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    data_min_ = math::Vector<T>(n_features);
                    data_max_ = math::Vector<T>(n_features);

                    for (core::usize j = 0; j < n_features; ++j) {
                        data_min_[j] = X(0, j);
                        data_max_[j] = X(0, j);

                        for (core::usize i = 1; i < n_samples; ++i) {
                            data_min_[j] = std::min(data_min_[j], X(i, j));
                            data_max_[j] = std::max(data_max_[j], X(i, j));
                        }

                        // Avoid division by zero
                        if (std::abs(data_max_[j] - data_min_[j]) < std::numeric_limits<T>::epsilon()) {
                            data_max_[j] = data_min_[j] + T{1};
                        }
                    }

                    fitted_ = true;
                }

                math::Matrix<T> transform(const math::Matrix<T>& X) const {
                    PSI_ASSERT(fitted_, "MinMaxScaler must be fitted before transform");
                    PSI_ASSERT(X.cols() == data_min_.size(), "Feature dimension mismatch");

                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> result(n_samples, n_features);
                    T range = feature_max_ - feature_min_;

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features; ++j) {
                            T scaled = (X(i, j) - data_min_[j]) / (data_max_[j] - data_min_[j]);
                            result(i, j) = scaled * range + feature_min_;
                        }
                    }

                    return result;
                }

                math::Matrix<T> fit_transform(const math::Matrix<T>& X) {
                    fit(X);
                    return transform(X);
                }

                math::Matrix<T> inverse_transform(const math::Matrix<T>& X) const {
                    PSI_ASSERT(fitted_, "MinMaxScaler must be fitted before inverse_transform");
                    PSI_ASSERT(X.cols() == data_min_.size(), "Feature dimension mismatch");

                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> result(n_samples, n_features);
                    T range = feature_max_ - feature_min_;

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features; ++j) {
                            T unscaled = (X(i, j) - feature_min_) / range;
                            result(i, j) = unscaled * (data_max_[j] - data_min_[j]) + data_min_[j];
                        }
                    }

                    return result;
                }

                PSI_NODISCARD const math::Vector<T>& data_min() const { return data_min_; }
                PSI_NODISCARD const math::Vector<T>& data_max() const { return data_max_; }
                PSI_NODISCARD bool is_fitted() const { return fitted_; }

            private:
                T feature_min_;
                T feature_max_;
                math::Vector<T> data_min_;
                math::Vector<T> data_max_;
                bool fitted_;
            };

            // Max Abs Scaler (scales by maximum absolute value)
            template<typename T>
            class MaxAbsScaler {
            public:
                MaxAbsScaler() : fitted_(false) {}

                void fit(const math::Matrix<T>& X) {
                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    max_abs_ = math::Vector<T>(n_features);

                    for (core::usize j = 0; j < n_features; ++j) {
                        max_abs_[j] = std::abs(X(0, j));

                        for (core::usize i = 1; i < n_samples; ++i) {
                            max_abs_[j] = std::max(max_abs_[j], std::abs(X(i, j)));
                        }

                        // Avoid division by zero
                        if (max_abs_[j] < std::numeric_limits<T>::epsilon()) {
                            max_abs_[j] = T{1};
                        }
                    }

                    fitted_ = true;
                }

                math::Matrix<T> transform(const math::Matrix<T>& X) const {
                    PSI_ASSERT(fitted_, "MaxAbsScaler must be fitted before transform");
                    PSI_ASSERT(X.cols() == max_abs_.size(), "Feature dimension mismatch");

                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> result(n_samples, n_features);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features; ++j) {
                            result(i, j) = X(i, j) / max_abs_[j];
                        }
                    }

                    return result;
                }

                math::Matrix<T> fit_transform(const math::Matrix<T>& X) {
                    fit(X);
                    return transform(X);
                }

                math::Matrix<T> inverse_transform(const math::Matrix<T>& X) const {
                    PSI_ASSERT(fitted_, "MaxAbsScaler must be fitted before inverse_transform");
                    PSI_ASSERT(X.cols() == max_abs_.size(), "Feature dimension mismatch");

                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> result(n_samples, n_features);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features; ++j) {
                            result(i, j) = X(i, j) * max_abs_[j];
                        }
                    }

                    return result;
                }

                PSI_NODISCARD const math::Vector<T>& max_abs() const { return max_abs_; }
                PSI_NODISCARD bool is_fitted() const { return fitted_; }

            private:
                math::Vector<T> max_abs_;
                bool fitted_;
            };

        } // namespace preprocessing
    } // namespace ml
} // namespace psi

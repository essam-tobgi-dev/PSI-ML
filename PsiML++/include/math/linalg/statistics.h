#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/memory.h"
#include "../../core/exception.h"
#include "../../core/device.h"
#include "../vector.h"
#include "../matrix.h"
#include "../tensor.h"
#include "../ops/arithmetic.h"
#include "../ops/reduction.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <type_traits>

namespace psi {
    namespace math {
        namespace ops {

            // Statistical operation types
            enum class StatisticType : core::u8 {
                Mean = 0,
                Variance = 1,
                StandardDeviation = 2,
                Median = 3,
                Mode = 4,
                Min = 5,
                Max = 6,
                Range = 7,
                Skewness = 8,
                Kurtosis = 9,
                Percentile = 10
            };

            // Bias correction for variance calculations
            enum class BiasCorrection : core::u8 {
                None = 0,        // N denominator (population)
                Bessel = 1       // N-1 denominator (sample)
            };

            // Basic statistics for containers

            // Mean
            template<typename Container>
            PSI_NODISCARD auto mean(const Container& a) {
                using value_type = typename Container::value_type;
                static_assert(std::is_arithmetic_v<value_type>, "Mean requires arithmetic type");

                PSI_ASSERT(a.size() > 0, "Cannot compute mean of empty container");
                return reduce_sum(a) / static_cast<value_type>(a.size());
            }

            // Mean along specific axis (for matrices and tensors)
            template<typename T>
            PSI_NODISCARD Vector<T> mean_axis(const Matrix<T>& mat, core::index_t axis) {
                PSI_ASSERT(axis == 0 || axis == 1, "Matrix mean axis must be 0 (rows) or 1 (columns)");

                if (axis == 0) {
                    // Mean along rows (result has shape [cols])
                    Vector<T> result(mat.cols(), mat.device_id());
                    for (core::usize j = 0; j < mat.cols(); ++j) {
                        T sum_val{};
                        for (core::usize i = 0; i < mat.rows(); ++i) {
                            sum_val += mat(i, j);
                        }
                        result[j] = sum_val / static_cast<T>(mat.rows());
                    }
                    return result;
                }
                else {
                    // Mean along columns (result has shape [rows])
                    Vector<T> result(mat.rows(), mat.device_id());
                    for (core::usize i = 0; i < mat.rows(); ++i) {
                        T sum_val{};
                        for (core::usize j = 0; j < mat.cols(); ++j) {
                            sum_val += mat(i, j);
                        }
                        result[i] = sum_val / static_cast<T>(mat.cols());
                    }
                    return result;
                }
            }

            // Variance
            template<typename Container>
            PSI_NODISCARD auto variance(const Container& a, BiasCorrection correction = BiasCorrection::Bessel) {
                using value_type = typename Container::value_type;
                static_assert(std::is_arithmetic_v<value_type>, "Variance requires arithmetic type");

                PSI_ASSERT(a.size() > 0, "Cannot compute variance of empty container");

                if (correction == BiasCorrection::Bessel && a.size() <= 1) {
                    PSI_ASSERT(false, "Cannot compute sample variance with <= 1 sample");
                }

                auto mean_val = mean(a);

                value_type sum_sq_diff{};
                for (core::usize i = 0; i < a.size(); ++i) {
                    value_type diff = a[i] - mean_val;
                    sum_sq_diff += diff * diff;
                }

                core::usize denominator = (correction == BiasCorrection::Bessel) ?
                    (a.size() - 1) : a.size();

                return sum_sq_diff / static_cast<value_type>(denominator);
            }

            // Standard deviation
            template<typename Container>
            PSI_NODISCARD auto stddev(const Container& a, BiasCorrection correction = BiasCorrection::Bessel) {
                return std::sqrt(variance(a, correction));
            }

            // Median (requires sorting, so creates a copy)
            template<typename Container>
            PSI_NODISCARD auto median(const Container& a) {
                using value_type = typename Container::value_type;
                static_assert(std::is_arithmetic_v<value_type>, "Median requires arithmetic type");

                PSI_ASSERT(a.size() > 0, "Cannot compute median of empty container");

                // Copy data for sorting
                std::vector<value_type> sorted_data;
                sorted_data.reserve(a.size());

                for (core::usize i = 0; i < a.size(); ++i) {
                    sorted_data.push_back(a[i]);
                }

                std::sort(sorted_data.begin(), sorted_data.end());

                core::usize n = sorted_data.size();
                if (n % 2 == 0) {
                    // Even number of elements
                    return (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / value_type{ 2 };
                }
                else {
                    // Odd number of elements
                    return sorted_data[n / 2];
                }
            }

            // Percentile
            template<typename Container>
            PSI_NODISCARD auto percentile(const Container& a, double p) {
                using value_type = typename Container::value_type;
                static_assert(std::is_arithmetic_v<value_type>, "Percentile requires arithmetic type");

                PSI_ASSERT(a.size() > 0, "Cannot compute percentile of empty container");
                PSI_ASSERT(p >= 0.0 && p <= 100.0, "Percentile must be between 0 and 100");

                // Copy data for sorting
                std::vector<value_type> sorted_data;
                sorted_data.reserve(a.size());

                for (core::usize i = 0; i < a.size(); ++i) {
                    sorted_data.push_back(a[i]);
                }

                std::sort(sorted_data.begin(), sorted_data.end());

                if (p == 0.0) return static_cast<double>(sorted_data[0]);
                if (p == 100.0) return static_cast<double>(sorted_data.back());

                double index = (p / 100.0) * (sorted_data.size() - 1);
                core::usize lower_index = static_cast<core::usize>(std::floor(index));
                core::usize upper_index = static_cast<core::usize>(std::ceil(index));

                if (lower_index == upper_index) {
                    return static_cast<double>(sorted_data[lower_index]);
                }

                double weight = index - lower_index;
                return (1.0 - weight) * static_cast<double>(sorted_data[lower_index]) +
                    weight * static_cast<double>(sorted_data[upper_index]);
            }

            // Quartiles
            template<typename Container>
            PSI_NODISCARD auto quartiles(const Container& a) {
                struct Quartiles {
                    double q1, q2, q3;
                };

                return Quartiles{
                    percentile(a, 25.0),
                    percentile(a, 50.0),  // median
                    percentile(a, 75.0)
                };
            }

            // Interquartile range
            template<typename Container>
            PSI_NODISCARD auto iqr(const Container& a) {
                auto q = quartiles(a);
                return q.q3 - q.q1;
            }

            // Skewness (third moment)
            template<typename Container>
            PSI_NODISCARD auto skewness(const Container& a) {
                using value_type = typename Container::value_type;
                static_assert(std::is_floating_point_v<value_type>, "Skewness requires floating point type");

                PSI_ASSERT(a.size() > 2, "Skewness requires at least 3 data points");

                auto mean_val = mean(a);
                auto std_val = stddev(a);

                if (std_val <= value_type{}) {
                    return value_type{};  // No variance, skewness undefined
                }

                value_type sum_cubed_z{};
                for (core::usize i = 0; i < a.size(); ++i) {
                    value_type z = (a[i] - mean_val) / std_val;
                    sum_cubed_z += z * z * z;
                }

                return sum_cubed_z / static_cast<value_type>(a.size());
            }

            // Kurtosis (fourth moment)
            template<typename Container>
            PSI_NODISCARD auto kurtosis(const Container& a, bool excess = true) {
                using value_type = typename Container::value_type;
                static_assert(std::is_floating_point_v<value_type>, "Kurtosis requires floating point type");

                PSI_ASSERT(a.size() > 3, "Kurtosis requires at least 4 data points");

                auto mean_val = mean(a);
                auto std_val = stddev(a);

                if (std_val <= value_type{}) {
                    return excess ? value_type{} : value_type{ 3 };  // No variance
                }

                value_type sum_fourth_z{};
                for (core::usize i = 0; i < a.size(); ++i) {
                    value_type z = (a[i] - mean_val) / std_val;
                    value_type z_sq = z * z;
                    sum_fourth_z += z_sq * z_sq;
                }

                value_type kurt = sum_fourth_z / static_cast<value_type>(a.size());

                // Return excess kurtosis (subtract 3) or raw kurtosis
                return excess ? (kurt - value_type{ 3 }) : kurt;
            }

            // Mode (most frequent value) - simplified version
            template<typename Container>
            PSI_NODISCARD auto mode(const Container& a) {
                using value_type = typename Container::value_type;

                PSI_ASSERT(a.size() > 0, "Cannot compute mode of empty container");

                std::map<value_type, core::usize> counts;

                for (core::usize i = 0; i < a.size(); ++i) {
                    counts[a[i]]++;
                }

                auto max_it = std::max_element(counts.begin(), counts.end(),
                    [](const auto& a, const auto& b) {
                        return a.second < b.second;
                    });

                return max_it->first;
            }

            // Range (max - min)
            template<typename Container>
            PSI_NODISCARD auto range(const Container& a) {
                PSI_ASSERT(a.size() > 0, "Cannot compute range of empty container");

                auto min_max = std::minmax_element(a.data(), a.data() + a.size());
                return *min_max.second - *min_max.first;
            }

            // Covariance between two containers
            template<typename Container1, typename Container2>
            PSI_NODISCARD auto covariance(const Container1& a, const Container2& b,
                BiasCorrection correction = BiasCorrection::Bessel) {
                using value_type = typename Container1::value_type;
                static_assert(std::is_arithmetic_v<value_type>, "Covariance requires arithmetic type");

                PSI_CHECK_DIMENSIONS("covariance", a.size(), b.size());
                PSI_ASSERT(a.size() > 0, "Cannot compute covariance of empty containers");

                if (correction == BiasCorrection::Bessel && a.size() <= 1) {
                    PSI_ASSERT(false, "Cannot compute sample covariance with <= 1 sample");
                }

                auto mean_a = mean(a);
                auto mean_b = mean(b);

                value_type sum_products{};
                for (core::usize i = 0; i < a.size(); ++i) {
                    sum_products += (a[i] - mean_a) * (b[i] - mean_b);
                }

                core::usize denominator = (correction == BiasCorrection::Bessel) ?
                    (a.size() - 1) : a.size();

                return sum_products / static_cast<value_type>(denominator);
            }

            // Correlation coefficient
            template<typename Container1, typename Container2>
            PSI_NODISCARD auto correlation(const Container1& a, const Container2& b) {
                using value_type = typename Container1::value_type;
                static_assert(std::is_floating_point_v<value_type>, "Correlation requires floating point type");

                auto cov_ab = covariance(a, b);
                auto std_a = stddev(a);
                auto std_b = stddev(b);

                if (std_a <= value_type{} || std_b <= value_type{}) {
                    return value_type{};  // Undefined correlation
                }

                return cov_ab / (std_a * std_b);
            }

            // Covariance matrix for multiple variables (columns of matrix)
            template<typename T>
            PSI_NODISCARD Matrix<T> covariance_matrix(const Matrix<T>& data,
                BiasCorrection correction = BiasCorrection::Bessel) {
                static_assert(std::is_arithmetic_v<T>, "Covariance matrix requires arithmetic type");

                core::usize n_vars = data.cols();
                core::usize n_samples = data.rows();

                PSI_ASSERT(n_samples > 0, "Cannot compute covariance matrix with no samples");
                if (correction == BiasCorrection::Bessel && n_samples <= 1) {
                    PSI_ASSERT(false, "Cannot compute sample covariance matrix with <= 1 sample");
                }

                Matrix<T> cov_mat(n_vars, n_vars, data.device_id());

                // Compute means for each variable
                Vector<T> means = mean_axis(data, 0);  // Mean along rows

                // Compute covariance matrix
                T denominator = static_cast<T>((correction == BiasCorrection::Bessel) ?
                    (n_samples - 1) : n_samples);

                for (core::usize i = 0; i < n_vars; ++i) {
                    for (core::usize j = i; j < n_vars; ++j) {
                        T sum_products{};

                        for (core::usize k = 0; k < n_samples; ++k) {
                            sum_products += (data(k, i) - means[i]) * (data(k, j) - means[j]);
                        }

                        T cov_val = sum_products / denominator;
                        cov_mat(i, j) = cov_val;
                        if (i != j) {
                            cov_mat(j, i) = cov_val;  // Symmetric matrix
                        }
                    }
                }

                return cov_mat;
            }

            // Correlation matrix
            template<typename T>
            PSI_NODISCARD Matrix<T> correlation_matrix(const Matrix<T>& data) {
                static_assert(std::is_floating_point_v<T>, "Correlation matrix requires floating point type");

                Matrix<T> cov_mat = covariance_matrix(data);
                core::usize n_vars = cov_mat.rows();

                Matrix<T> corr_mat(n_vars, n_vars, data.device_id());

                // Extract standard deviations from diagonal
                Vector<T> std_devs(n_vars, data.device_id());
                for (core::usize i = 0; i < n_vars; ++i) {
                    std_devs[i] = std::sqrt(cov_mat(i, i));
                }

                // Compute correlation coefficients
                for (core::usize i = 0; i < n_vars; ++i) {
                    for (core::usize j = 0; j < n_vars; ++j) {
                        if (std_devs[i] <= T{} || std_devs[j] <= T{}) {
                            corr_mat(i, j) = (i == j) ? T{ 1 } : T{};
                        }
                        else {
                            corr_mat(i, j) = cov_mat(i, j) / (std_devs[i] * std_devs[j]);
                        }
                    }
                }

                return corr_mat;
            }

            // Z-score normalization
            template<typename Container>
            PSI_NODISCARD auto zscore(const Container& a) {
                using value_type = typename Container::value_type;
                static_assert(std::is_floating_point_v<value_type>, "Z-score requires floating point type");

                auto mean_val = mean(a);
                auto std_val = stddev(a);

                if (std_val <= value_type{}) {
                    // No variance, return zeros
                    if constexpr (std::is_same_v<Container, Vector<value_type>>) {
                        return Vector<value_type>(a.size(), a.device_id());
                    }
                    else if constexpr (std::is_same_v<Container, Matrix<value_type>>) {
                        return Matrix<value_type>(a.rows(), a.cols(), a.device_id());
                    }
                    else {
                        return Tensor<value_type>(a.shape(), a.device_id());
                    }
                }

                return divide(subtract(a, mean_val), std_val);
            }

            // Min-max normalization
            template<typename Container>
            PSI_NODISCARD auto minmax_normalize(const Container& a,
                typename Container::value_type min_val = typename Container::value_type{},
                typename Container::value_type max_val = typename Container::value_type{ 1 }) {
                using value_type = typename Container::value_type;

                auto container_min = reduce_min(a);
                auto container_max = reduce_max(a);

                if (container_max <= container_min) {
                    // No range, return constant values
                    if constexpr (std::is_same_v<Container, Vector<value_type>>) {
                        Vector<value_type> result(a.size(), a.device_id());
                        result.fill(min_val);
                        return result;
                    }
                    else if constexpr (std::is_same_v<Container, Matrix<value_type>>) {
                        Matrix<value_type> result(a.rows(), a.cols(), a.device_id());
                        result.fill(min_val);
                        return result;
                    }
                    else {
                        Tensor<value_type> result(a.shape(), a.device_id());
                        result.fill(min_val);
                        return result;
                    }
                }

                auto range_val = container_max - container_min;
                auto scale = (max_val - min_val) / range_val;

                return add(multiply(subtract(a, container_min), scale), min_val);
            }

            // Robust statistics (using median and MAD)

            // Median Absolute Deviation
            template<typename Container>
            PSI_NODISCARD auto mad(const Container& a) {
                using value_type = typename Container::value_type;
                static_assert(std::is_arithmetic_v<value_type>, "MAD requires arithmetic type");

                auto median_val = median(a);

                if constexpr (std::is_same_v<Container, Vector<value_type>>) {
                    Vector<value_type> abs_deviations(a.size(), a.device_id());
                    for (core::usize i = 0; i < a.size(); ++i) {
                        abs_deviations[i] = std::abs(a[i] - median_val);
                    }
                    return median(abs_deviations);
                }
                else if constexpr (std::is_same_v<Container, Matrix<value_type>>) {
                    Matrix<value_type> abs_deviations(a.rows(), a.cols(), a.device_id());
                    for (core::usize i = 0; i < a.size(); ++i) {
                        abs_deviations[i] = std::abs(a[i] - median_val);
                    }
                    return median(abs_deviations);
                }
                else {
                    Tensor<value_type> abs_deviations(a.shape(), a.device_id());
                    for (core::usize i = 0; i < a.size(); ++i) {
                        abs_deviations[i] = std::abs(a[i] - median_val);
                    }
                    return median(abs_deviations);
                }
            }

            // Robust z-score using median and MAD
            template<typename Container>
            PSI_NODISCARD auto robust_zscore(const Container& a) {
                using value_type = typename Container::value_type;
                static_assert(std::is_floating_point_v<value_type>, "Robust z-score requires floating point type");

                auto median_val = median(a);
                auto mad_val = mad(a);

                // MAD to standard deviation approximation factor for normal distribution
                constexpr value_type mad_to_std_factor = static_cast<value_type>(1.4826);
                auto robust_std = mad_val * mad_to_std_factor;

                if (robust_std <= value_type{}) {
                    // No variance, return zeros
                    if constexpr (std::is_same_v<Container, Vector<value_type>>) {
                        return Vector<value_type>(a.size(), a.device_id());
                    }
                    else if constexpr (std::is_same_v<Container, Matrix<value_type>>) {
                        return Matrix<value_type>(a.rows(), a.cols(), a.device_id());
                    }
                    else {
                        return Tensor<value_type>(a.shape(), a.device_id());
                    }
                }

                return divide(subtract(a, median_val), robust_std);
            }

            // Running/online statistics
            template<typename T>
            class RunningStatistics {
            public:
                RunningStatistics() : n_(0), mean_(T{}), m2_(T{}), min_val_(T{}), max_val_(T{}) {}

                void update(T value) {
                    n_++;

                    if (n_ == 1) {
                        mean_ = value;
                        min_val_ = value;
                        max_val_ = value;
                    }
                    else {
                        // Welford's online algorithm
                        T delta = value - mean_;
                        mean_ += delta / static_cast<T>(n_);
                        T delta2 = value - mean_;
                        m2_ += delta * delta2;

                        min_val_ = std::min(min_val_, value);
                        max_val_ = std::max(max_val_, value);
                    }
                }

                PSI_NODISCARD core::usize count() const { return n_; }
                PSI_NODISCARD T mean() const { return mean_; }
                PSI_NODISCARD T min() const { return min_val_; }
                PSI_NODISCARD T max() const { return max_val_; }
                PSI_NODISCARD T range() const { return max_val_ - min_val_; }

                PSI_NODISCARD T variance(BiasCorrection correction = BiasCorrection::Bessel) const {
                    if (n_ < 2) return T{};
                    core::usize denominator = (correction == BiasCorrection::Bessel) ? (n_ - 1) : n_;
                    return m2_ / static_cast<T>(denominator);
                }

                PSI_NODISCARD T stddev(BiasCorrection correction = BiasCorrection::Bessel) const {
                    return std::sqrt(variance(correction));
                }

                void reset() {
                    n_ = 0;
                    mean_ = T{};
                    m2_ = T{};
                    min_val_ = T{};
                    max_val_ = T{};
                }

            private:
                core::usize n_;
                T mean_;
                T m2_;  // Sum of squared differences from mean
                T min_val_;
                T max_val_;
            };

        } // namespace ops
    } // namespace math
} // namespace psi
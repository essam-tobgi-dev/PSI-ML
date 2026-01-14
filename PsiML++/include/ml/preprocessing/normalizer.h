#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include <cmath>

namespace psi {
    namespace ml {
        namespace preprocessing {

            // Normalization types
            enum class NormType : core::u8 {
                L1 = 0,     // L1 norm (sum of absolute values)
                L2 = 1,     // L2 norm (Euclidean)
                Max = 2     // Max norm (maximum absolute value)
            };

            // Normalizer (row-wise normalization)
            template<typename T>
            class Normalizer {
            public:
                explicit Normalizer(NormType norm = NormType::L2) : norm_(norm) {}

                math::Matrix<T> transform(const math::Matrix<T>& X) const {
                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> result(n_samples, n_features);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        T norm_val = compute_norm(X, i);

                        if (norm_val < std::numeric_limits<T>::epsilon()) {
                            // Zero norm, just copy the row
                            for (core::usize j = 0; j < n_features; ++j) {
                                result(i, j) = X(i, j);
                            }
                        } else {
                            for (core::usize j = 0; j < n_features; ++j) {
                                result(i, j) = X(i, j) / norm_val;
                            }
                        }
                    }

                    return result;
                }

                math::Matrix<T> fit_transform(const math::Matrix<T>& X) {
                    return transform(X);  // No fitting required
                }

                PSI_NODISCARD NormType norm_type() const { return norm_; }

            private:
                NormType norm_;

                T compute_norm(const math::Matrix<T>& X, core::usize row) const {
                    core::usize n_features = X.cols();
                    T result = T{0};

                    switch (norm_) {
                        case NormType::L1:
                            for (core::usize j = 0; j < n_features; ++j) {
                                result += std::abs(X(row, j));
                            }
                            break;

                        case NormType::L2:
                            for (core::usize j = 0; j < n_features; ++j) {
                                result += X(row, j) * X(row, j);
                            }
                            result = std::sqrt(result);
                            break;

                        case NormType::Max:
                            for (core::usize j = 0; j < n_features; ++j) {
                                result = std::max(result, std::abs(X(row, j)));
                            }
                            break;
                    }

                    return result;
                }
            };

            // Vector normalization utilities
            template<typename T>
            math::Vector<T> normalize_l2(const math::Vector<T>& v) {
                T norm = T{0};
                for (core::usize i = 0; i < v.size(); ++i) {
                    norm += v[i] * v[i];
                }
                norm = std::sqrt(norm);

                if (norm < std::numeric_limits<T>::epsilon()) {
                    return v;
                }

                math::Vector<T> result(v.size());
                for (core::usize i = 0; i < v.size(); ++i) {
                    result[i] = v[i] / norm;
                }
                return result;
            }

            template<typename T>
            math::Vector<T> normalize_l1(const math::Vector<T>& v) {
                T norm = T{0};
                for (core::usize i = 0; i < v.size(); ++i) {
                    norm += std::abs(v[i]);
                }

                if (norm < std::numeric_limits<T>::epsilon()) {
                    return v;
                }

                math::Vector<T> result(v.size());
                for (core::usize i = 0; i < v.size(); ++i) {
                    result[i] = v[i] / norm;
                }
                return result;
            }

            template<typename T>
            math::Vector<T> normalize_max(const math::Vector<T>& v) {
                T max_val = T{0};
                for (core::usize i = 0; i < v.size(); ++i) {
                    max_val = std::max(max_val, std::abs(v[i]));
                }

                if (max_val < std::numeric_limits<T>::epsilon()) {
                    return v;
                }

                math::Vector<T> result(v.size());
                for (core::usize i = 0; i < v.size(); ++i) {
                    result[i] = v[i] / max_val;
                }
                return result;
            }

        } // namespace preprocessing
    } // namespace ml
} // namespace psi

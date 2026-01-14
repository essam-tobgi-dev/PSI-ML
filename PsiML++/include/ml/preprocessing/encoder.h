#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include <map>
#include <vector>
#include <algorithm>

namespace psi {
    namespace ml {
        namespace preprocessing {

            // Label Encoder - encodes categorical labels to integers
            template<typename T>
            class LabelEncoder {
            public:
                LabelEncoder() : fitted_(false) {}

                void fit(const math::Vector<T>& y) {
                    classes_.clear();
                    label_to_index_.clear();
                    index_to_label_.clear();

                    // Find unique values
                    for (core::usize i = 0; i < y.size(); ++i) {
                        classes_.push_back(y[i]);
                    }

                    std::sort(classes_.begin(), classes_.end());
                    classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());

                    // Create mappings
                    for (core::usize i = 0; i < classes_.size(); ++i) {
                        label_to_index_[classes_[i]] = static_cast<core::i32>(i);
                        index_to_label_[static_cast<core::i32>(i)] = classes_[i];
                    }

                    fitted_ = true;
                }

                math::Vector<core::i32> transform(const math::Vector<T>& y) const {
                    PSI_ASSERT(fitted_, "LabelEncoder must be fitted before transform");

                    math::Vector<core::i32> result(y.size());

                    for (core::usize i = 0; i < y.size(); ++i) {
                        auto it = label_to_index_.find(y[i]);
                        PSI_ASSERT(it != label_to_index_.end(), "Unknown label encountered");
                        result[i] = it->second;
                    }

                    return result;
                }

                math::Vector<core::i32> fit_transform(const math::Vector<T>& y) {
                    fit(y);
                    return transform(y);
                }

                math::Vector<T> inverse_transform(const math::Vector<core::i32>& y) const {
                    PSI_ASSERT(fitted_, "LabelEncoder must be fitted before inverse_transform");

                    math::Vector<T> result(y.size());

                    for (core::usize i = 0; i < y.size(); ++i) {
                        auto it = index_to_label_.find(y[i]);
                        PSI_ASSERT(it != index_to_label_.end(), "Unknown index encountered");
                        result[i] = it->second;
                    }

                    return result;
                }

                PSI_NODISCARD const std::vector<T>& classes() const { return classes_; }
                PSI_NODISCARD core::usize n_classes() const { return classes_.size(); }
                PSI_NODISCARD bool is_fitted() const { return fitted_; }

            private:
                std::vector<T> classes_;
                std::map<T, core::i32> label_to_index_;
                std::map<core::i32, T> index_to_label_;
                bool fitted_;
            };

            // One-Hot Encoder
            template<typename T>
            class OneHotEncoder {
            public:
                OneHotEncoder() : fitted_(false), n_features_(0), total_categories_(0) {}

                void fit(const math::Matrix<T>& X) {
                    n_features_ = X.cols();
                    categories_.clear();
                    categories_.resize(n_features_);

                    core::usize n_samples = X.rows();

                    // Find unique values for each feature
                    for (core::usize j = 0; j < n_features_; ++j) {
                        std::vector<T> unique_vals;
                        for (core::usize i = 0; i < n_samples; ++i) {
                            unique_vals.push_back(X(i, j));
                        }
                        std::sort(unique_vals.begin(), unique_vals.end());
                        unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());
                        categories_[j] = unique_vals;
                    }

                    // Compute total number of one-hot columns
                    total_categories_ = 0;
                    for (const auto& cats : categories_) {
                        total_categories_ += cats.size();
                    }

                    fitted_ = true;
                }

                math::Matrix<T> transform(const math::Matrix<T>& X) const {
                    PSI_ASSERT(fitted_, "OneHotEncoder must be fitted before transform");
                    PSI_ASSERT(X.cols() == n_features_, "Feature dimension mismatch");

                    core::usize n_samples = X.rows();
                    math::Matrix<T> result(n_samples, total_categories_);
                    result.fill(T{0});

                    for (core::usize i = 0; i < n_samples; ++i) {
                        core::usize col_offset = 0;
                        for (core::usize j = 0; j < n_features_; ++j) {
                            T val = X(i, j);
                            const auto& cats = categories_[j];

                            for (core::usize k = 0; k < cats.size(); ++k) {
                                if (std::abs(cats[k] - val) < std::numeric_limits<T>::epsilon()) {
                                    result(i, col_offset + k) = T{1};
                                    break;
                                }
                            }
                            col_offset += cats.size();
                        }
                    }

                    return result;
                }

                math::Matrix<T> fit_transform(const math::Matrix<T>& X) {
                    fit(X);
                    return transform(X);
                }

                PSI_NODISCARD const std::vector<std::vector<T>>& categories() const { return categories_; }
                PSI_NODISCARD core::usize n_output_features() const { return total_categories_; }
                PSI_NODISCARD bool is_fitted() const { return fitted_; }

            private:
                core::usize n_features_;
                core::usize total_categories_;
                std::vector<std::vector<T>> categories_;
                bool fitted_;
            };

            // Binarizer - threshold-based binarization
            template<typename T>
            class Binarizer {
            public:
                explicit Binarizer(T threshold = T{0}) : threshold_(threshold) {}

                math::Matrix<T> transform(const math::Matrix<T>& X) const {
                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> result(n_samples, n_features);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features; ++j) {
                            result(i, j) = (X(i, j) > threshold_) ? T{1} : T{0};
                        }
                    }

                    return result;
                }

                math::Matrix<T> fit_transform(const math::Matrix<T>& X) {
                    return transform(X);  // No fitting required
                }

                PSI_NODISCARD T threshold() const { return threshold_; }

            private:
                T threshold_;
            };

        } // namespace preprocessing
    } // namespace ml
} // namespace psi

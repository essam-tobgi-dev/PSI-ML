#pragma once

#include "../core/types.h"
#include "../core/config.h"
#include "../core/exception.h"
#include "../math/vector.h"
#include "../math/matrix.h"
#include <cmath>
#include <algorithm>
#include <map>

namespace psi {
    namespace ml {
        namespace metrics {

            // ============================================================================
            // Regression Metrics
            // ============================================================================

            // Mean Squared Error
            template<typename T>
            PSI_NODISCARD T mean_squared_error(const math::Vector<T>& y_true, const math::Vector<T>& y_pred) {
                PSI_CHECK_DIMENSIONS("mean_squared_error", y_true.size(), y_pred.size());

                T sum = T{0};
                for (core::usize i = 0; i < y_true.size(); ++i) {
                    T diff = y_true[i] - y_pred[i];
                    sum += diff * diff;
                }
                return sum / static_cast<T>(y_true.size());
            }

            // Root Mean Squared Error
            template<typename T>
            PSI_NODISCARD T root_mean_squared_error(const math::Vector<T>& y_true, const math::Vector<T>& y_pred) {
                return std::sqrt(mean_squared_error(y_true, y_pred));
            }

            // Mean Absolute Error
            template<typename T>
            PSI_NODISCARD T mean_absolute_error(const math::Vector<T>& y_true, const math::Vector<T>& y_pred) {
                PSI_CHECK_DIMENSIONS("mean_absolute_error", y_true.size(), y_pred.size());

                T sum = T{0};
                for (core::usize i = 0; i < y_true.size(); ++i) {
                    sum += std::abs(y_true[i] - y_pred[i]);
                }
                return sum / static_cast<T>(y_true.size());
            }

            // R-squared (coefficient of determination)
            template<typename T>
            PSI_NODISCARD T r2_score(const math::Vector<T>& y_true, const math::Vector<T>& y_pred) {
                PSI_CHECK_DIMENSIONS("r2_score", y_true.size(), y_pred.size());

                T mean_y = y_true.mean();

                T ss_tot = T{0};
                T ss_res = T{0};

                for (core::usize i = 0; i < y_true.size(); ++i) {
                    T diff_tot = y_true[i] - mean_y;
                    T diff_res = y_true[i] - y_pred[i];
                    ss_tot += diff_tot * diff_tot;
                    ss_res += diff_res * diff_res;
                }

                if (ss_tot < std::numeric_limits<T>::epsilon()) {
                    return T{1};  // Perfect prediction if no variance in y
                }

                return T{1} - (ss_res / ss_tot);
            }

            // ============================================================================
            // Classification Metrics
            // ============================================================================

            // Accuracy
            template<typename T>
            PSI_NODISCARD T accuracy(const math::Vector<T>& y_true, const math::Vector<T>& y_pred) {
                PSI_CHECK_DIMENSIONS("accuracy", y_true.size(), y_pred.size());

                core::usize correct = 0;
                for (core::usize i = 0; i < y_true.size(); ++i) {
                    if (std::abs(y_true[i] - y_pred[i]) < T{0.5}) {
                        ++correct;
                    }
                }
                return static_cast<T>(correct) / static_cast<T>(y_true.size());
            }

            // Binary accuracy with threshold
            template<typename T>
            PSI_NODISCARD T binary_accuracy(const math::Vector<T>& y_true, const math::Vector<T>& y_pred, T threshold = T{0.5}) {
                PSI_CHECK_DIMENSIONS("binary_accuracy", y_true.size(), y_pred.size());

                core::usize correct = 0;
                for (core::usize i = 0; i < y_true.size(); ++i) {
                    T pred_class = (y_pred[i] >= threshold) ? T{1} : T{0};
                    if (std::abs(y_true[i] - pred_class) < T{0.5}) {
                        ++correct;
                    }
                }
                return static_cast<T>(correct) / static_cast<T>(y_true.size());
            }

            // Confusion matrix for binary classification
            template<typename T>
            struct ConfusionMatrix {
                core::usize true_positives = 0;
                core::usize true_negatives = 0;
                core::usize false_positives = 0;
                core::usize false_negatives = 0;
            };

            template<typename T>
            PSI_NODISCARD ConfusionMatrix<T> confusion_matrix(const math::Vector<T>& y_true, const math::Vector<T>& y_pred, T threshold = T{0.5}) {
                PSI_CHECK_DIMENSIONS("confusion_matrix", y_true.size(), y_pred.size());

                ConfusionMatrix<T> cm;

                for (core::usize i = 0; i < y_true.size(); ++i) {
                    bool actual_positive = y_true[i] >= threshold;
                    bool predicted_positive = y_pred[i] >= threshold;

                    if (actual_positive && predicted_positive) {
                        cm.true_positives++;
                    } else if (!actual_positive && !predicted_positive) {
                        cm.true_negatives++;
                    } else if (!actual_positive && predicted_positive) {
                        cm.false_positives++;
                    } else {
                        cm.false_negatives++;
                    }
                }

                return cm;
            }

            // Precision
            template<typename T>
            PSI_NODISCARD T precision(const math::Vector<T>& y_true, const math::Vector<T>& y_pred, T threshold = T{0.5}) {
                auto cm = confusion_matrix(y_true, y_pred, threshold);
                core::usize denom = cm.true_positives + cm.false_positives;
                if (denom == 0) return T{0};
                return static_cast<T>(cm.true_positives) / static_cast<T>(denom);
            }

            // Recall (Sensitivity, True Positive Rate)
            template<typename T>
            PSI_NODISCARD T recall(const math::Vector<T>& y_true, const math::Vector<T>& y_pred, T threshold = T{0.5}) {
                auto cm = confusion_matrix(y_true, y_pred, threshold);
                core::usize denom = cm.true_positives + cm.false_negatives;
                if (denom == 0) return T{0};
                return static_cast<T>(cm.true_positives) / static_cast<T>(denom);
            }

            // F1 Score
            template<typename T>
            PSI_NODISCARD T f1_score(const math::Vector<T>& y_true, const math::Vector<T>& y_pred, T threshold = T{0.5}) {
                T prec = precision(y_true, y_pred, threshold);
                T rec = recall(y_true, y_pred, threshold);

                if (prec + rec < std::numeric_limits<T>::epsilon()) {
                    return T{0};
                }

                return T{2} * (prec * rec) / (prec + rec);
            }

            // Specificity (True Negative Rate)
            template<typename T>
            PSI_NODISCARD T specificity(const math::Vector<T>& y_true, const math::Vector<T>& y_pred, T threshold = T{0.5}) {
                auto cm = confusion_matrix(y_true, y_pred, threshold);
                core::usize denom = cm.true_negatives + cm.false_positives;
                if (denom == 0) return T{0};
                return static_cast<T>(cm.true_negatives) / static_cast<T>(denom);
            }

            // ============================================================================
            // Clustering Metrics
            // ============================================================================

            // Inertia (within-cluster sum of squares)
            template<typename T>
            PSI_NODISCARD T inertia(const math::Matrix<T>& X, const math::Vector<core::i32>& labels, const math::Matrix<T>& centroids) {
                PSI_CHECK_DIMENSIONS("inertia samples", X.rows(), labels.size());

                T total_inertia = T{0};

                for (core::usize i = 0; i < X.rows(); ++i) {
                    core::i32 cluster = labels[i];
                    for (core::usize j = 0; j < X.cols(); ++j) {
                        T diff = X(i, j) - centroids(cluster, j);
                        total_inertia += diff * diff;
                    }
                }

                return total_inertia;
            }

            // Silhouette coefficient for a single sample
            template<typename T>
            PSI_NODISCARD T silhouette_sample(
                const math::Matrix<T>& X,
                const math::Vector<core::i32>& labels,
                core::usize sample_idx) {

                core::i32 sample_cluster = labels[sample_idx];
                core::usize n_samples = X.rows();
                core::usize n_features = X.cols();

                // Count samples in each cluster
                std::map<core::i32, core::usize> cluster_counts;
                for (core::usize i = 0; i < n_samples; ++i) {
                    cluster_counts[labels[i]]++;
                }

                // Compute average intra-cluster distance (a)
                T a = T{0};
                core::usize same_cluster_count = 0;
                for (core::usize i = 0; i < n_samples; ++i) {
                    if (i != sample_idx && labels[i] == sample_cluster) {
                        T dist = T{0};
                        for (core::usize j = 0; j < n_features; ++j) {
                            T diff = X(sample_idx, j) - X(i, j);
                            dist += diff * diff;
                        }
                        a += std::sqrt(dist);
                        same_cluster_count++;
                    }
                }
                if (same_cluster_count > 0) {
                    a /= static_cast<T>(same_cluster_count);
                }

                // Compute minimum average inter-cluster distance (b)
                T b = std::numeric_limits<T>::max();
                for (const auto& [cluster, count] : cluster_counts) {
                    if (cluster != sample_cluster) {
                        T avg_dist = T{0};
                        for (core::usize i = 0; i < n_samples; ++i) {
                            if (labels[i] == cluster) {
                                T dist = T{0};
                                for (core::usize j = 0; j < n_features; ++j) {
                                    T diff = X(sample_idx, j) - X(i, j);
                                    dist += diff * diff;
                                }
                                avg_dist += std::sqrt(dist);
                            }
                        }
                        avg_dist /= static_cast<T>(count);
                        b = std::min(b, avg_dist);
                    }
                }

                if (b == std::numeric_limits<T>::max()) {
                    return T{0};  // Only one cluster
                }

                T max_ab = std::max(a, b);
                if (max_ab < std::numeric_limits<T>::epsilon()) {
                    return T{0};
                }

                return (b - a) / max_ab;
            }

            // Average silhouette score
            template<typename T>
            PSI_NODISCARD T silhouette_score(const math::Matrix<T>& X, const math::Vector<core::i32>& labels) {
                PSI_CHECK_DIMENSIONS("silhouette_score", X.rows(), labels.size());

                T total = T{0};
                for (core::usize i = 0; i < X.rows(); ++i) {
                    total += silhouette_sample(X, labels, i);
                }

                return total / static_cast<T>(X.rows());
            }

            // ============================================================================
            // Loss Functions
            // ============================================================================

            // Binary cross-entropy loss
            template<typename T>
            PSI_NODISCARD T binary_cross_entropy(const math::Vector<T>& y_true, const math::Vector<T>& y_pred, T epsilon = T{1e-7}) {
                PSI_CHECK_DIMENSIONS("binary_cross_entropy", y_true.size(), y_pred.size());

                T loss = T{0};
                for (core::usize i = 0; i < y_true.size(); ++i) {
                    T pred_clipped = std::max(epsilon, std::min(T{1} - epsilon, y_pred[i]));
                    loss -= y_true[i] * std::log(pred_clipped) + (T{1} - y_true[i]) * std::log(T{1} - pred_clipped);
                }

                return loss / static_cast<T>(y_true.size());
            }

            // Hinge loss (for SVM)
            template<typename T>
            PSI_NODISCARD T hinge_loss(const math::Vector<T>& y_true, const math::Vector<T>& y_pred) {
                PSI_CHECK_DIMENSIONS("hinge_loss", y_true.size(), y_pred.size());

                T loss = T{0};
                for (core::usize i = 0; i < y_true.size(); ++i) {
                    // Assume y_true is {-1, 1} for SVM
                    T margin = y_true[i] * y_pred[i];
                    loss += std::max(T{0}, T{1} - margin);
                }

                return loss / static_cast<T>(y_true.size());
            }

        } // namespace metrics
    } // namespace ml
} // namespace psi

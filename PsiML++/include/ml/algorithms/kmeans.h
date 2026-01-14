#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include "../../math/random.h"
#include "../model.h"
#include <cmath>
#include <limits>

namespace psi {
    namespace ml {
        namespace algorithms {

            // K-Means clustering
            template<typename T>
            class KMeans : public ClusteringModel<T> {
            public:
                KMeans(
                    core::usize n_clusters = 8,
                    core::u32 max_iterations = 300,
                    T tolerance = T{1e-4},
                    core::u32 n_init = 10,
                    core::u64 seed = 0)
                    : n_clusters_(n_clusters)
                    , max_iterations_(max_iterations)
                    , tolerance_(tolerance)
                    , n_init_(n_init)
                    , seed_(seed)
                    , inertia_(T{0}) {}

                void fit(const math::Matrix<T>& X) override {
                    PSI_ASSERT(X.rows() >= n_clusters_, "Number of samples must be >= number of clusters");

                    this->state_ = ModelState::Training;

                    T best_inertia = std::numeric_limits<T>::max();
                    math::Matrix<T> best_centroids;
                    math::Vector<core::i32> best_labels;

                    math::Random rng(math::GeneratorType::MersenneTwister, seed_);

                    for (core::u32 init = 0; init < n_init_; ++init) {
                        // Initialize centroids using k-means++ style initialization
                        math::Matrix<T> centroids = initialize_centroids(X, rng);
                        math::Vector<core::i32> labels(X.rows());

                        T prev_inertia = std::numeric_limits<T>::max();

                        for (core::u32 iter = 0; iter < max_iterations_; ++iter) {
                            // Assign samples to nearest centroid
                            labels = assign_clusters(X, centroids);

                            // Update centroids
                            math::Matrix<T> new_centroids = update_centroids(X, labels);

                            // Compute inertia
                            T inertia = compute_inertia(X, labels, new_centroids);

                            // Check convergence
                            if (std::abs(prev_inertia - inertia) < tolerance_) {
                                centroids = new_centroids;
                                break;
                            }

                            centroids = new_centroids;
                            prev_inertia = inertia;
                        }

                        T final_inertia = compute_inertia(X, labels, centroids);

                        if (final_inertia < best_inertia) {
                            best_inertia = final_inertia;
                            best_centroids = centroids;
                            best_labels = labels;
                        }
                    }

                    centroids_ = best_centroids;
                    labels_ = best_labels;
                    inertia_ = best_inertia;

                    this->state_ = ModelState::Trained;
                }

                math::Matrix<T> transform(const math::Matrix<T>& X) const override {
                    PSI_ASSERT(this->is_fitted(), "Model must be fitted before transform");

                    core::usize n_samples = X.rows();
                    math::Matrix<T> distances(n_samples, n_clusters_);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize k = 0; k < n_clusters_; ++k) {
                            distances(i, k) = compute_distance(X, i, centroids_, k);
                        }
                    }

                    return distances;
                }

                math::Vector<core::i32> predict(const math::Matrix<T>& X) const override {
                    PSI_ASSERT(this->is_fitted(), "Model must be fitted before predict");
                    return assign_clusters(X, centroids_);
                }

                PSI_NODISCARD std::string name() const override { return "KMeans"; }

                PSI_NODISCARD const math::Matrix<T>& centroids() const { return centroids_; }
                PSI_NODISCARD const math::Vector<core::i32>& labels() const { return labels_; }
                PSI_NODISCARD T inertia() const { return inertia_; }
                PSI_NODISCARD core::usize n_clusters() const { return n_clusters_; }

            private:
                core::usize n_clusters_;
                core::u32 max_iterations_;
                T tolerance_;
                core::u32 n_init_;
                core::u64 seed_;

                math::Matrix<T> centroids_;
                math::Vector<core::i32> labels_;
                T inertia_;

                math::Matrix<T> initialize_centroids(const math::Matrix<T>& X, math::Random& rng) const {
                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> centroids(n_clusters_, n_features);

                    // K-means++ initialization
                    // Choose first centroid randomly
                    core::usize first_idx = rng.uniform_int(0, n_samples - 1);
                    for (core::usize j = 0; j < n_features; ++j) {
                        centroids(0, j) = X(first_idx, j);
                    }

                    // Choose remaining centroids
                    for (core::usize k = 1; k < n_clusters_; ++k) {
                        // Compute distances to nearest existing centroid
                        math::Vector<T> min_distances(n_samples);
                        T total_dist = T{0};

                        for (core::usize i = 0; i < n_samples; ++i) {
                            T min_dist = std::numeric_limits<T>::max();

                            for (core::usize c = 0; c < k; ++c) {
                                T dist = compute_distance(X, i, centroids, c);
                                min_dist = std::min(min_dist, dist);
                            }

                            min_distances[i] = min_dist * min_dist;  // Square for probability
                            total_dist += min_distances[i];
                        }

                        // Choose next centroid with probability proportional to distance squared
                        T rand_val = rng.uniform(T{0}, total_dist);
                        T cumulative = T{0};
                        core::usize chosen_idx = 0;

                        for (core::usize i = 0; i < n_samples; ++i) {
                            cumulative += min_distances[i];
                            if (cumulative >= rand_val) {
                                chosen_idx = i;
                                break;
                            }
                        }

                        for (core::usize j = 0; j < n_features; ++j) {
                            centroids(k, j) = X(chosen_idx, j);
                        }
                    }

                    return centroids;
                }

                math::Vector<core::i32> assign_clusters(const math::Matrix<T>& X, const math::Matrix<T>& centroids) const {
                    core::usize n_samples = X.rows();
                    math::Vector<core::i32> labels(n_samples);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        T min_dist = std::numeric_limits<T>::max();
                        core::i32 closest_cluster = 0;

                        for (core::usize k = 0; k < n_clusters_; ++k) {
                            T dist = compute_distance(X, i, centroids, k);
                            if (dist < min_dist) {
                                min_dist = dist;
                                closest_cluster = static_cast<core::i32>(k);
                            }
                        }

                        labels[i] = closest_cluster;
                    }

                    return labels;
                }

                math::Matrix<T> update_centroids(const math::Matrix<T>& X, const math::Vector<core::i32>& labels) const {
                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> new_centroids(n_clusters_, n_features);
                    new_centroids.fill(T{0});

                    math::Vector<core::usize> counts(n_clusters_);
                    counts.fill(0);

                    // Sum samples per cluster
                    for (core::usize i = 0; i < n_samples; ++i) {
                        core::i32 cluster = labels[i];
                        counts[cluster]++;
                        for (core::usize j = 0; j < n_features; ++j) {
                            new_centroids(cluster, j) += X(i, j);
                        }
                    }

                    // Compute mean
                    for (core::usize k = 0; k < n_clusters_; ++k) {
                        if (counts[k] > 0) {
                            for (core::usize j = 0; j < n_features; ++j) {
                                new_centroids(k, j) /= static_cast<T>(counts[k]);
                            }
                        }
                    }

                    return new_centroids;
                }

                T compute_distance(const math::Matrix<T>& X, core::usize sample_idx,
                                   const math::Matrix<T>& centroids, core::usize cluster_idx) const {
                    T dist = T{0};
                    for (core::usize j = 0; j < X.cols(); ++j) {
                        T diff = X(sample_idx, j) - centroids(cluster_idx, j);
                        dist += diff * diff;
                    }
                    return std::sqrt(dist);
                }

                T compute_inertia(const math::Matrix<T>& X, const math::Vector<core::i32>& labels,
                                  const math::Matrix<T>& centroids) const {
                    T inertia = T{0};
                    for (core::usize i = 0; i < X.rows(); ++i) {
                        core::i32 cluster = labels[i];
                        for (core::usize j = 0; j < X.cols(); ++j) {
                            T diff = X(i, j) - centroids(cluster, j);
                            inertia += diff * diff;
                        }
                    }
                    return inertia;
                }
            };

        } // namespace algorithms
    } // namespace ml
} // namespace psi

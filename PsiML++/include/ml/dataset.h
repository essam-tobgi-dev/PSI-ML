#pragma once

#include "../core/types.h"
#include "../core/config.h"
#include "../core/exception.h"
#include "../math/vector.h"
#include "../math/matrix.h"
#include "../math/random.h"
#include <vector>
#include <algorithm>
#include <numeric>

namespace psi {
    namespace ml {

        // Dataset split result
        template<typename T>
        struct TrainTestSplit {
            math::Matrix<T> X_train;
            math::Matrix<T> X_test;
            math::Vector<T> y_train;
            math::Vector<T> y_test;
        };

        // Dataset utilities
        template<typename T>
        class Dataset {
        public:
            // Train-test split
            static TrainTestSplit<T> train_test_split(
                const math::Matrix<T>& X,
                const math::Vector<T>& y,
                T test_size = T{0.2},
                bool shuffle = true,
                core::u64 seed = 0) {

                PSI_CHECK_DIMENSIONS("train_test_split", X.rows(), y.size());
                PSI_ASSERT(test_size > T{0} && test_size < T{1}, "test_size must be between 0 and 1");

                core::usize n_samples = X.rows();
                core::usize n_features = X.cols();
                core::usize n_test = static_cast<core::usize>(n_samples * test_size);
                core::usize n_train = n_samples - n_test;

                // Create indices
                std::vector<core::usize> indices(n_samples);
                std::iota(indices.begin(), indices.end(), 0);

                // Shuffle if requested
                if (shuffle) {
                    math::Random rng(math::GeneratorType::MersenneTwister, seed);
                    rng.shuffle(indices);
                }

                // Create result
                TrainTestSplit<T> result;
                result.X_train = math::Matrix<T>(n_train, n_features);
                result.X_test = math::Matrix<T>(n_test, n_features);
                result.y_train = math::Vector<T>(n_train);
                result.y_test = math::Vector<T>(n_test);

                // Fill training data
                for (core::usize i = 0; i < n_train; ++i) {
                    core::usize idx = indices[i];
                    for (core::usize j = 0; j < n_features; ++j) {
                        result.X_train(i, j) = X(idx, j);
                    }
                    result.y_train[i] = y[idx];
                }

                // Fill test data
                for (core::usize i = 0; i < n_test; ++i) {
                    core::usize idx = indices[n_train + i];
                    for (core::usize j = 0; j < n_features; ++j) {
                        result.X_test(i, j) = X(idx, j);
                    }
                    result.y_test[i] = y[idx];
                }

                return result;
            }

            // Shuffle data
            static void shuffle(math::Matrix<T>& X, math::Vector<T>& y, core::u64 seed = 0) {
                PSI_CHECK_DIMENSIONS("shuffle", X.rows(), y.size());

                core::usize n_samples = X.rows();
                core::usize n_features = X.cols();

                std::vector<core::usize> indices(n_samples);
                std::iota(indices.begin(), indices.end(), 0);

                math::Random rng(math::GeneratorType::MersenneTwister, seed);
                rng.shuffle(indices);

                // Create shuffled copies
                math::Matrix<T> X_shuffled(n_samples, n_features);
                math::Vector<T> y_shuffled(n_samples);

                for (core::usize i = 0; i < n_samples; ++i) {
                    core::usize idx = indices[i];
                    for (core::usize j = 0; j < n_features; ++j) {
                        X_shuffled(i, j) = X(idx, j);
                    }
                    y_shuffled[i] = y[idx];
                }

                X = std::move(X_shuffled);
                y = std::move(y_shuffled);
            }

            // Get batch from dataset
            static void get_batch(
                const math::Matrix<T>& X,
                const math::Vector<T>& y,
                core::usize batch_start,
                core::usize batch_size,
                math::Matrix<T>& X_batch,
                math::Vector<T>& y_batch) {

                core::usize n_samples = X.rows();
                core::usize n_features = X.cols();
                core::usize actual_batch_size = std::min(batch_size, n_samples - batch_start);

                X_batch = math::Matrix<T>(actual_batch_size, n_features);
                y_batch = math::Vector<T>(actual_batch_size);

                for (core::usize i = 0; i < actual_batch_size; ++i) {
                    for (core::usize j = 0; j < n_features; ++j) {
                        X_batch(i, j) = X(batch_start + i, j);
                    }
                    y_batch[i] = y[batch_start + i];
                }
            }

            // K-fold cross validation indices
            static std::vector<std::pair<std::vector<core::usize>, std::vector<core::usize>>>
            kfold_indices(core::usize n_samples, core::usize n_folds, bool shuffle = true, core::u64 seed = 0) {
                PSI_ASSERT(n_folds > 1, "Number of folds must be greater than 1");
                PSI_ASSERT(n_samples >= n_folds, "Number of samples must be >= number of folds");

                std::vector<core::usize> indices(n_samples);
                std::iota(indices.begin(), indices.end(), 0);

                if (shuffle) {
                    math::Random rng(math::GeneratorType::MersenneTwister, seed);
                    rng.shuffle(indices);
                }

                std::vector<std::pair<std::vector<core::usize>, std::vector<core::usize>>> folds;
                core::usize fold_size = n_samples / n_folds;

                for (core::usize fold = 0; fold < n_folds; ++fold) {
                    std::vector<core::usize> test_indices;
                    std::vector<core::usize> train_indices;

                    core::usize test_start = fold * fold_size;
                    core::usize test_end = (fold == n_folds - 1) ? n_samples : (fold + 1) * fold_size;

                    for (core::usize i = 0; i < n_samples; ++i) {
                        if (i >= test_start && i < test_end) {
                            test_indices.push_back(indices[i]);
                        } else {
                            train_indices.push_back(indices[i]);
                        }
                    }

                    folds.push_back({train_indices, test_indices});
                }

                return folds;
            }

            // Add bias column (column of ones) to feature matrix
            static math::Matrix<T> add_bias(const math::Matrix<T>& X) {
                core::usize n_samples = X.rows();
                core::usize n_features = X.cols();

                math::Matrix<T> X_bias(n_samples, n_features + 1);

                for (core::usize i = 0; i < n_samples; ++i) {
                    X_bias(i, 0) = T{1};
                    for (core::usize j = 0; j < n_features; ++j) {
                        X_bias(i, j + 1) = X(i, j);
                    }
                }

                return X_bias;
            }
        };

    } // namespace ml
} // namespace psi

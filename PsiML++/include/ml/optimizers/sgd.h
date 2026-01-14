#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include "../../math/random.h"
#include "gradient_descent.h"

namespace psi {
    namespace ml {
        namespace optimizers {

            // Stochastic Gradient Descent
            template<typename T>
            class SGD : public Optimizer<T> {
            public:
                SGD(T learning_rate = T{0.01}, T momentum = T{0}, bool nesterov = false)
                    : lr_(learning_rate)
                    , momentum_(momentum)
                    , nesterov_(nesterov)
                    , initialized_(false) {
                    PSI_ASSERT(learning_rate > T{0}, "Learning rate must be positive");
                    PSI_ASSERT(momentum >= T{0} && momentum < T{1}, "Momentum must be in [0, 1)");
                }

                void update(math::Vector<T>& weights, const math::Vector<T>& gradients) override {
                    PSI_CHECK_DIMENSIONS("SGD::update", weights.size(), gradients.size());

                    if (momentum_ > T{0}) {
                        if (!initialized_ || velocity_.size() != weights.size()) {
                            velocity_ = math::Vector<T>(weights.size());
                            velocity_.fill(T{0});
                            initialized_ = true;
                        }

                        for (core::usize i = 0; i < weights.size(); ++i) {
                            if (nesterov_) {
                                T v_prev = velocity_[i];
                                velocity_[i] = momentum_ * velocity_[i] - lr_ * gradients[i];
                                weights[i] += -momentum_ * v_prev + (T{1} + momentum_) * velocity_[i];
                            } else {
                                velocity_[i] = momentum_ * velocity_[i] - lr_ * gradients[i];
                                weights[i] += velocity_[i];
                            }
                        }
                    } else {
                        for (core::usize i = 0; i < weights.size(); ++i) {
                            weights[i] -= lr_ * gradients[i];
                        }
                    }
                }

                void reset() override {
                    if (initialized_) {
                        velocity_.fill(T{0});
                    }
                }

                PSI_NODISCARD T learning_rate() const override { return lr_; }
                void set_learning_rate(T lr) override {
                    PSI_ASSERT(lr > T{0}, "Learning rate must be positive");
                    lr_ = lr;
                }

                PSI_NODISCARD T momentum() const { return momentum_; }
                PSI_NODISCARD bool nesterov() const { return nesterov_; }

            private:
                T lr_;
                T momentum_;
                bool nesterov_;
                math::Vector<T> velocity_;
                bool initialized_;
            };

            // Mini-batch SGD helper
            template<typename T>
            class MiniBatchSGD {
            public:
                MiniBatchSGD(core::usize batch_size = 32, core::u64 seed = 0)
                    : batch_size_(batch_size)
                    , rng_(math::GeneratorType::MersenneTwister, seed) {}

                // Get random batch indices
                std::vector<core::usize> get_batch_indices(core::usize n_samples) {
                    std::vector<core::usize> indices(n_samples);
                    for (core::usize i = 0; i < n_samples; ++i) {
                        indices[i] = i;
                    }
                    rng_.shuffle(indices);

                    core::usize actual_batch_size = std::min(batch_size_, n_samples);
                    return std::vector<core::usize>(indices.begin(), indices.begin() + actual_batch_size);
                }

                // Extract batch from data
                void extract_batch(
                    const math::Matrix<T>& X,
                    const math::Vector<T>& y,
                    const std::vector<core::usize>& indices,
                    math::Matrix<T>& X_batch,
                    math::Vector<T>& y_batch) {

                    core::usize batch_size = indices.size();
                    core::usize n_features = X.cols();

                    X_batch = math::Matrix<T>(batch_size, n_features);
                    y_batch = math::Vector<T>(batch_size);

                    for (core::usize i = 0; i < batch_size; ++i) {
                        core::usize idx = indices[i];
                        for (core::usize j = 0; j < n_features; ++j) {
                            X_batch(i, j) = X(idx, j);
                        }
                        y_batch[i] = y[idx];
                    }
                }

                PSI_NODISCARD core::usize batch_size() const { return batch_size_; }
                void set_batch_size(core::usize size) { batch_size_ = size; }

            private:
                core::usize batch_size_;
                math::Random rng_;
            };

            // SGD with weight decay (L2 regularization)
            template<typename T>
            class SGDWithWeightDecay : public Optimizer<T> {
            public:
                SGDWithWeightDecay(T learning_rate = T{0.01}, T weight_decay = T{0.0001}, T momentum = T{0})
                    : lr_(learning_rate)
                    , weight_decay_(weight_decay)
                    , momentum_(momentum)
                    , initialized_(false) {
                    PSI_ASSERT(learning_rate > T{0}, "Learning rate must be positive");
                    PSI_ASSERT(weight_decay >= T{0}, "Weight decay must be non-negative");
                }

                void update(math::Vector<T>& weights, const math::Vector<T>& gradients) override {
                    PSI_CHECK_DIMENSIONS("SGDWithWeightDecay::update", weights.size(), gradients.size());

                    if (momentum_ > T{0}) {
                        if (!initialized_ || velocity_.size() != weights.size()) {
                            velocity_ = math::Vector<T>(weights.size());
                            velocity_.fill(T{0});
                            initialized_ = true;
                        }

                        for (core::usize i = 0; i < weights.size(); ++i) {
                            T grad_with_decay = gradients[i] + weight_decay_ * weights[i];
                            velocity_[i] = momentum_ * velocity_[i] - lr_ * grad_with_decay;
                            weights[i] += velocity_[i];
                        }
                    } else {
                        for (core::usize i = 0; i < weights.size(); ++i) {
                            T grad_with_decay = gradients[i] + weight_decay_ * weights[i];
                            weights[i] -= lr_ * grad_with_decay;
                        }
                    }
                }

                void reset() override {
                    if (initialized_) {
                        velocity_.fill(T{0});
                    }
                }

                PSI_NODISCARD T learning_rate() const override { return lr_; }
                void set_learning_rate(T lr) override {
                    PSI_ASSERT(lr > T{0}, "Learning rate must be positive");
                    lr_ = lr;
                }

                PSI_NODISCARD T weight_decay() const { return weight_decay_; }

            private:
                T lr_;
                T weight_decay_;
                T momentum_;
                math::Vector<T> velocity_;
                bool initialized_;
            };

        } // namespace optimizers
    } // namespace ml
} // namespace psi

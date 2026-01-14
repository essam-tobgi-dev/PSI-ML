#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include <functional>

namespace psi {
    namespace ml {
        namespace optimizers {

            // Base optimizer interface
            template<typename T>
            class Optimizer {
            public:
                virtual ~Optimizer() = default;

                // Update weights given gradients
                virtual void update(math::Vector<T>& weights, const math::Vector<T>& gradients) = 0;

                // Reset optimizer state
                virtual void reset() = 0;

                // Get learning rate
                PSI_NODISCARD virtual T learning_rate() const = 0;

                // Set learning rate
                virtual void set_learning_rate(T lr) = 0;
            };

            // Batch Gradient Descent
            template<typename T>
            class GradientDescent : public Optimizer<T> {
            public:
                explicit GradientDescent(T learning_rate = T{0.01})
                    : lr_(learning_rate) {
                    PSI_ASSERT(learning_rate > T{0}, "Learning rate must be positive");
                }

                void update(math::Vector<T>& weights, const math::Vector<T>& gradients) override {
                    PSI_CHECK_DIMENSIONS("GradientDescent::update", weights.size(), gradients.size());

                    for (core::usize i = 0; i < weights.size(); ++i) {
                        weights[i] -= lr_ * gradients[i];
                    }
                }

                void reset() override {
                    // No state to reset
                }

                PSI_NODISCARD T learning_rate() const override { return lr_; }
                void set_learning_rate(T lr) override {
                    PSI_ASSERT(lr > T{0}, "Learning rate must be positive");
                    lr_ = lr;
                }

            private:
                T lr_;
            };

            // Gradient Descent with learning rate decay
            template<typename T>
            class GradientDescentDecay : public Optimizer<T> {
            public:
                GradientDescentDecay(T initial_lr = T{0.01}, T decay_rate = T{0.99}, core::u32 decay_steps = 100)
                    : initial_lr_(initial_lr)
                    , current_lr_(initial_lr)
                    , decay_rate_(decay_rate)
                    , decay_steps_(decay_steps)
                    , step_(0) {
                    PSI_ASSERT(initial_lr > T{0}, "Learning rate must be positive");
                    PSI_ASSERT(decay_rate > T{0} && decay_rate <= T{1}, "Decay rate must be in (0, 1]");
                }

                void update(math::Vector<T>& weights, const math::Vector<T>& gradients) override {
                    PSI_CHECK_DIMENSIONS("GradientDescentDecay::update", weights.size(), gradients.size());

                    for (core::usize i = 0; i < weights.size(); ++i) {
                        weights[i] -= current_lr_ * gradients[i];
                    }

                    ++step_;
                    if (step_ % decay_steps_ == 0) {
                        current_lr_ *= decay_rate_;
                    }
                }

                void reset() override {
                    current_lr_ = initial_lr_;
                    step_ = 0;
                }

                PSI_NODISCARD T learning_rate() const override { return current_lr_; }
                void set_learning_rate(T lr) override {
                    PSI_ASSERT(lr > T{0}, "Learning rate must be positive");
                    current_lr_ = lr;
                    initial_lr_ = lr;
                }

                PSI_NODISCARD core::u32 step() const { return step_; }

            private:
                T initial_lr_;
                T current_lr_;
                T decay_rate_;
                core::u32 decay_steps_;
                core::u32 step_;
            };

            // Gradient computation helper for linear models
            template<typename T>
            math::Vector<T> compute_gradient_mse(
                const math::Matrix<T>& X,
                const math::Vector<T>& y,
                const math::Vector<T>& weights) {

                core::usize n_samples = X.rows();
                core::usize n_features = X.cols();

                PSI_CHECK_DIMENSIONS("compute_gradient_mse X-y", n_samples, y.size());
                PSI_CHECK_DIMENSIONS("compute_gradient_mse X-w", n_features, weights.size());

                // predictions = X * weights
                math::Vector<T> predictions(n_samples);
                for (core::usize i = 0; i < n_samples; ++i) {
                    T pred = T{0};
                    for (core::usize j = 0; j < n_features; ++j) {
                        pred += X(i, j) * weights[j];
                    }
                    predictions[i] = pred;
                }

                // errors = predictions - y
                // gradient = (2/n) * X^T * errors
                math::Vector<T> gradient(n_features);
                T scale = T{2} / static_cast<T>(n_samples);

                for (core::usize j = 0; j < n_features; ++j) {
                    T sum = T{0};
                    for (core::usize i = 0; i < n_samples; ++i) {
                        sum += X(i, j) * (predictions[i] - y[i]);
                    }
                    gradient[j] = scale * sum;
                }

                return gradient;
            }

        } // namespace optimizers
    } // namespace ml
} // namespace psi

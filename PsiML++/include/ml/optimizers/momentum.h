#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "gradient_descent.h"

namespace psi {
    namespace ml {
        namespace optimizers {

            // Momentum optimizer
            template<typename T>
            class Momentum : public Optimizer<T> {
            public:
                Momentum(T learning_rate = T{0.01}, T momentum = T{0.9})
                    : lr_(learning_rate)
                    , momentum_(momentum)
                    , initialized_(false) {
                    PSI_ASSERT(learning_rate > T{0}, "Learning rate must be positive");
                    PSI_ASSERT(momentum >= T{0} && momentum < T{1}, "Momentum must be in [0, 1)");
                }

                void update(math::Vector<T>& weights, const math::Vector<T>& gradients) override {
                    PSI_CHECK_DIMENSIONS("Momentum::update", weights.size(), gradients.size());

                    if (!initialized_ || velocity_.size() != weights.size()) {
                        velocity_ = math::Vector<T>(weights.size());
                        velocity_.fill(T{0});
                        initialized_ = true;
                    }

                    for (core::usize i = 0; i < weights.size(); ++i) {
                        // v = momentum * v - lr * gradient
                        velocity_[i] = momentum_ * velocity_[i] - lr_ * gradients[i];
                        // weights = weights + v
                        weights[i] += velocity_[i];
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
                void set_momentum(T m) {
                    PSI_ASSERT(m >= T{0} && m < T{1}, "Momentum must be in [0, 1)");
                    momentum_ = m;
                }

            private:
                T lr_;
                T momentum_;
                math::Vector<T> velocity_;
                bool initialized_;
            };

            // Nesterov Accelerated Gradient (NAG)
            template<typename T>
            class NesterovMomentum : public Optimizer<T> {
            public:
                NesterovMomentum(T learning_rate = T{0.01}, T momentum = T{0.9})
                    : lr_(learning_rate)
                    , momentum_(momentum)
                    , initialized_(false) {
                    PSI_ASSERT(learning_rate > T{0}, "Learning rate must be positive");
                    PSI_ASSERT(momentum >= T{0} && momentum < T{1}, "Momentum must be in [0, 1)");
                }

                void update(math::Vector<T>& weights, const math::Vector<T>& gradients) override {
                    PSI_CHECK_DIMENSIONS("NesterovMomentum::update", weights.size(), gradients.size());

                    if (!initialized_ || velocity_.size() != weights.size()) {
                        velocity_ = math::Vector<T>(weights.size());
                        velocity_.fill(T{0});
                        initialized_ = true;
                    }

                    for (core::usize i = 0; i < weights.size(); ++i) {
                        T v_prev = velocity_[i];
                        // v = momentum * v - lr * gradient
                        velocity_[i] = momentum_ * velocity_[i] - lr_ * gradients[i];
                        // weights = weights - momentum * v_prev + (1 + momentum) * v
                        weights[i] += -momentum_ * v_prev + (T{1} + momentum_) * velocity_[i];
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
                void set_momentum(T m) {
                    PSI_ASSERT(m >= T{0} && m < T{1}, "Momentum must be in [0, 1)");
                    momentum_ = m;
                }

            private:
                T lr_;
                T momentum_;
                math::Vector<T> velocity_;
                bool initialized_;
            };

            // AdaGrad optimizer
            template<typename T>
            class AdaGrad : public Optimizer<T> {
            public:
                AdaGrad(T learning_rate = T{0.01}, T epsilon = T{1e-8})
                    : lr_(learning_rate)
                    , epsilon_(epsilon)
                    , initialized_(false) {
                    PSI_ASSERT(learning_rate > T{0}, "Learning rate must be positive");
                }

                void update(math::Vector<T>& weights, const math::Vector<T>& gradients) override {
                    PSI_CHECK_DIMENSIONS("AdaGrad::update", weights.size(), gradients.size());

                    if (!initialized_ || accumulated_.size() != weights.size()) {
                        accumulated_ = math::Vector<T>(weights.size());
                        accumulated_.fill(T{0});
                        initialized_ = true;
                    }

                    for (core::usize i = 0; i < weights.size(); ++i) {
                        accumulated_[i] += gradients[i] * gradients[i];
                        weights[i] -= lr_ * gradients[i] / (std::sqrt(accumulated_[i]) + epsilon_);
                    }
                }

                void reset() override {
                    if (initialized_) {
                        accumulated_.fill(T{0});
                    }
                }

                PSI_NODISCARD T learning_rate() const override { return lr_; }
                void set_learning_rate(T lr) override {
                    PSI_ASSERT(lr > T{0}, "Learning rate must be positive");
                    lr_ = lr;
                }

            private:
                T lr_;
                T epsilon_;
                math::Vector<T> accumulated_;
                bool initialized_;
            };

            // RMSprop optimizer
            template<typename T>
            class RMSprop : public Optimizer<T> {
            public:
                RMSprop(T learning_rate = T{0.001}, T decay = T{0.9}, T epsilon = T{1e-8})
                    : lr_(learning_rate)
                    , decay_(decay)
                    , epsilon_(epsilon)
                    , initialized_(false) {
                    PSI_ASSERT(learning_rate > T{0}, "Learning rate must be positive");
                    PSI_ASSERT(decay > T{0} && decay < T{1}, "Decay must be in (0, 1)");
                }

                void update(math::Vector<T>& weights, const math::Vector<T>& gradients) override {
                    PSI_CHECK_DIMENSIONS("RMSprop::update", weights.size(), gradients.size());

                    if (!initialized_ || mean_square_.size() != weights.size()) {
                        mean_square_ = math::Vector<T>(weights.size());
                        mean_square_.fill(T{0});
                        initialized_ = true;
                    }

                    for (core::usize i = 0; i < weights.size(); ++i) {
                        mean_square_[i] = decay_ * mean_square_[i] + (T{1} - decay_) * gradients[i] * gradients[i];
                        weights[i] -= lr_ * gradients[i] / (std::sqrt(mean_square_[i]) + epsilon_);
                    }
                }

                void reset() override {
                    if (initialized_) {
                        mean_square_.fill(T{0});
                    }
                }

                PSI_NODISCARD T learning_rate() const override { return lr_; }
                void set_learning_rate(T lr) override {
                    PSI_ASSERT(lr > T{0}, "Learning rate must be positive");
                    lr_ = lr;
                }

            private:
                T lr_;
                T decay_;
                T epsilon_;
                math::Vector<T> mean_square_;
                bool initialized_;
            };

            // Adam optimizer
            template<typename T>
            class Adam : public Optimizer<T> {
            public:
                Adam(T learning_rate = T{0.001}, T beta1 = T{0.9}, T beta2 = T{0.999}, T epsilon = T{1e-8})
                    : lr_(learning_rate)
                    , beta1_(beta1)
                    , beta2_(beta2)
                    , epsilon_(epsilon)
                    , t_(0)
                    , initialized_(false) {
                    PSI_ASSERT(learning_rate > T{0}, "Learning rate must be positive");
                    PSI_ASSERT(beta1 >= T{0} && beta1 < T{1}, "Beta1 must be in [0, 1)");
                    PSI_ASSERT(beta2 >= T{0} && beta2 < T{1}, "Beta2 must be in [0, 1)");
                }

                void update(math::Vector<T>& weights, const math::Vector<T>& gradients) override {
                    PSI_CHECK_DIMENSIONS("Adam::update", weights.size(), gradients.size());

                    if (!initialized_ || m_.size() != weights.size()) {
                        m_ = math::Vector<T>(weights.size());
                        v_ = math::Vector<T>(weights.size());
                        m_.fill(T{0});
                        v_.fill(T{0});
                        initialized_ = true;
                    }

                    ++t_;

                    for (core::usize i = 0; i < weights.size(); ++i) {
                        // Update biased first moment estimate
                        m_[i] = beta1_ * m_[i] + (T{1} - beta1_) * gradients[i];
                        // Update biased second raw moment estimate
                        v_[i] = beta2_ * v_[i] + (T{1} - beta2_) * gradients[i] * gradients[i];

                        // Bias-corrected estimates
                        T m_hat = m_[i] / (T{1} - std::pow(beta1_, static_cast<T>(t_)));
                        T v_hat = v_[i] / (T{1} - std::pow(beta2_, static_cast<T>(t_)));

                        weights[i] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                    }
                }

                void reset() override {
                    t_ = 0;
                    if (initialized_) {
                        m_.fill(T{0});
                        v_.fill(T{0});
                    }
                }

                PSI_NODISCARD T learning_rate() const override { return lr_; }
                void set_learning_rate(T lr) override {
                    PSI_ASSERT(lr > T{0}, "Learning rate must be positive");
                    lr_ = lr;
                }

            private:
                T lr_;
                T beta1_;
                T beta2_;
                T epsilon_;
                core::u32 t_;
                math::Vector<T> m_;
                math::Vector<T> v_;
                bool initialized_;
            };

        } // namespace optimizers
    } // namespace ml
} // namespace psi

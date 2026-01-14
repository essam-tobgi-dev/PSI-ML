#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include "../model.h"
#include "../metrics.h"
#include <cmath>
#include <algorithm>

namespace psi {
    namespace ml {
        namespace algorithms {

            // Linear Support Vector Machine using sub-gradient descent
            template<typename T>
            class LinearSVM : public SupervisedModel<T> {
            public:
                LinearSVM(
                    T C = T{1.0},
                    T learning_rate = T{0.001},
                    core::u32 max_iterations = 1000,
                    T tolerance = T{1e-6},
                    bool fit_intercept = true)
                    : C_(C)
                    , learning_rate_(learning_rate)
                    , max_iterations_(max_iterations)
                    , tolerance_(tolerance)
                    , fit_intercept_(fit_intercept) {
                    PSI_ASSERT(C > T{0}, "C must be positive");
                }

                void fit(const math::Matrix<T>& X, const math::Vector<T>& y) override {
                    PSI_CHECK_DIMENSIONS("LinearSVM::fit", X.rows(), y.size());

                    this->state_ = ModelState::Training;

                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    // Convert labels to {-1, 1}
                    math::Vector<T> y_converted(n_samples);
                    for (core::usize i = 0; i < n_samples; ++i) {
                        y_converted[i] = (y[i] > T{0.5}) ? T{1} : T{-1};
                    }

                    // Initialize weights and bias
                    weights_ = math::Vector<T>(n_features);
                    weights_.fill(T{0});
                    bias_ = T{0};

                    T prev_loss = std::numeric_limits<T>::max();

                    for (core::u32 iter = 0; iter < max_iterations_; ++iter) {
                        // Compute sub-gradient
                        math::Vector<T> grad_w(n_features);
                        grad_w.fill(T{0});
                        T grad_b = T{0};

                        T hinge_loss = T{0};

                        for (core::usize i = 0; i < n_samples; ++i) {
                            // Compute decision function: w^T x + b
                            T decision = T{0};
                            for (core::usize j = 0; j < n_features; ++j) {
                                decision += weights_[j] * X(i, j);
                            }
                            if (fit_intercept_) {
                                decision += bias_;
                            }

                            T margin = y_converted[i] * decision;

                            if (margin < T{1}) {
                                // Misclassified or within margin
                                hinge_loss += T{1} - margin;
                                for (core::usize j = 0; j < n_features; ++j) {
                                    grad_w[j] -= y_converted[i] * X(i, j);
                                }
                                if (fit_intercept_) {
                                    grad_b -= y_converted[i];
                                }
                            }
                        }

                        // Add regularization gradient
                        for (core::usize j = 0; j < n_features; ++j) {
                            grad_w[j] = weights_[j] + C_ * grad_w[j] / static_cast<T>(n_samples);
                        }
                        grad_b = C_ * grad_b / static_cast<T>(n_samples);

                        // Update weights
                        for (core::usize j = 0; j < n_features; ++j) {
                            weights_[j] -= learning_rate_ * grad_w[j];
                        }
                        if (fit_intercept_) {
                            bias_ -= learning_rate_ * grad_b;
                        }

                        // Compute loss
                        T reg_loss = T{0};
                        for (core::usize j = 0; j < n_features; ++j) {
                            reg_loss += weights_[j] * weights_[j];
                        }
                        T loss = T{0.5} * reg_loss + C_ * hinge_loss / static_cast<T>(n_samples);

                        // Check convergence
                        if (std::abs(prev_loss - loss) < tolerance_) {
                            break;
                        }
                        prev_loss = loss;
                    }

                    this->state_ = ModelState::Trained;
                }

                math::Vector<T> predict(const math::Matrix<T>& X) const override {
                    PSI_ASSERT(this->is_trained(), "Model must be trained before prediction");

                    core::usize n_samples = X.rows();
                    math::Vector<T> predictions(n_samples);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        T decision = decision_function_single(X, i);
                        predictions[i] = (decision >= T{0}) ? T{1} : T{0};
                    }

                    return predictions;
                }

                math::Vector<T> decision_function(const math::Matrix<T>& X) const {
                    PSI_ASSERT(this->is_trained(), "Model must be trained before prediction");

                    core::usize n_samples = X.rows();
                    math::Vector<T> decisions(n_samples);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        decisions[i] = decision_function_single(X, i);
                    }

                    return decisions;
                }

                T score(const math::Matrix<T>& X, const math::Vector<T>& y) const override {
                    math::Vector<T> predictions = predict(X);
                    return metrics::accuracy(y, predictions);
                }

                PSI_NODISCARD std::string name() const override { return "LinearSVM"; }

                PSI_NODISCARD const math::Vector<T>& weights() const { return weights_; }
                PSI_NODISCARD T bias() const { return bias_; }
                PSI_NODISCARD T C() const { return C_; }

            private:
                T C_;
                T learning_rate_;
                core::u32 max_iterations_;
                T tolerance_;
                bool fit_intercept_;

                math::Vector<T> weights_;
                T bias_;

                T decision_function_single(const math::Matrix<T>& X, core::usize sample_idx) const {
                    T decision = fit_intercept_ ? bias_ : T{0};
                    for (core::usize j = 0; j < weights_.size(); ++j) {
                        decision += weights_[j] * X(sample_idx, j);
                    }
                    return decision;
                }
            };

            // Kernel types
            enum class KernelType : core::u8 {
                Linear = 0,
                RBF = 1,
                Polynomial = 2
            };

            // Kernel SVM using simplified SMO
            template<typename T>
            class KernelSVM : public SupervisedModel<T> {
            public:
                KernelSVM(
                    KernelType kernel = KernelType::RBF,
                    T C = T{1.0},
                    T gamma = T{1.0},
                    core::u32 degree = 3,
                    core::u32 max_iterations = 1000,
                    T tolerance = T{1e-3})
                    : kernel_(kernel)
                    , C_(C)
                    , gamma_(gamma)
                    , degree_(degree)
                    , max_iterations_(max_iterations)
                    , tolerance_(tolerance)
                    , bias_(T{0}) {
                    PSI_ASSERT(C > T{0}, "C must be positive");
                }

                void fit(const math::Matrix<T>& X, const math::Vector<T>& y) override {
                    PSI_CHECK_DIMENSIONS("KernelSVM::fit", X.rows(), y.size());

                    this->state_ = ModelState::Training;

                    n_samples_ = X.rows();
                    n_features_ = X.cols();

                    // Store training data for kernel computation
                    X_train_ = X;

                    // Convert labels to {-1, 1}
                    y_train_ = math::Vector<T>(n_samples_);
                    for (core::usize i = 0; i < n_samples_; ++i) {
                        y_train_[i] = (y[i] > T{0.5}) ? T{1} : T{-1};
                    }

                    // Initialize alphas
                    alphas_ = math::Vector<T>(n_samples_);
                    alphas_.fill(T{0});
                    bias_ = T{0};

                    // Simplified SMO
                    for (core::u32 iter = 0; iter < max_iterations_; ++iter) {
                        core::usize num_changed = 0;

                        for (core::usize i = 0; i < n_samples_; ++i) {
                            T Ei = compute_error(i);

                            if ((y_train_[i] * Ei < -tolerance_ && alphas_[i] < C_) ||
                                (y_train_[i] * Ei > tolerance_ && alphas_[i] > T{0})) {

                                // Select j != i randomly
                                core::usize j = (i + 1) % n_samples_;

                                T Ej = compute_error(j);

                                T alpha_i_old = alphas_[i];
                                T alpha_j_old = alphas_[j];

                                // Compute bounds
                                T L, H;
                                if (y_train_[i] != y_train_[j]) {
                                    L = std::max(T{0}, alphas_[j] - alphas_[i]);
                                    H = std::min(C_, C_ + alphas_[j] - alphas_[i]);
                                } else {
                                    L = std::max(T{0}, alphas_[i] + alphas_[j] - C_);
                                    H = std::min(C_, alphas_[i] + alphas_[j]);
                                }

                                if (L >= H) continue;

                                // Compute eta
                                T Kii = kernel_func(i, i);
                                T Kjj = kernel_func(j, j);
                                T Kij = kernel_func(i, j);
                                T eta = T{2} * Kij - Kii - Kjj;

                                if (eta >= T{0}) continue;

                                // Update alpha_j
                                alphas_[j] = alpha_j_old - y_train_[j] * (Ei - Ej) / eta;
                                alphas_[j] = std::max(L, std::min(H, alphas_[j]));

                                if (std::abs(alphas_[j] - alpha_j_old) < T{1e-5}) continue;

                                // Update alpha_i
                                alphas_[i] = alpha_i_old + y_train_[i] * y_train_[j] * (alpha_j_old - alphas_[j]);

                                // Update bias
                                T b1 = bias_ - Ei - y_train_[i] * (alphas_[i] - alpha_i_old) * Kii
                                             - y_train_[j] * (alphas_[j] - alpha_j_old) * Kij;
                                T b2 = bias_ - Ej - y_train_[i] * (alphas_[i] - alpha_i_old) * Kij
                                             - y_train_[j] * (alphas_[j] - alpha_j_old) * Kjj;

                                if (alphas_[i] > T{0} && alphas_[i] < C_) {
                                    bias_ = b1;
                                } else if (alphas_[j] > T{0} && alphas_[j] < C_) {
                                    bias_ = b2;
                                } else {
                                    bias_ = (b1 + b2) / T{2};
                                }

                                num_changed++;
                            }
                        }

                        if (num_changed == 0) break;
                    }

                    this->state_ = ModelState::Trained;
                }

                math::Vector<T> predict(const math::Matrix<T>& X) const override {
                    PSI_ASSERT(this->is_trained(), "Model must be trained before prediction");

                    core::usize n_samples = X.rows();
                    math::Vector<T> predictions(n_samples);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        T decision = decision_function_single(X, i);
                        predictions[i] = (decision >= T{0}) ? T{1} : T{0};
                    }

                    return predictions;
                }

                T score(const math::Matrix<T>& X, const math::Vector<T>& y) const override {
                    math::Vector<T> predictions = predict(X);
                    return metrics::accuracy(y, predictions);
                }

                PSI_NODISCARD std::string name() const override { return "KernelSVM"; }

                PSI_NODISCARD const math::Vector<T>& alphas() const { return alphas_; }
                PSI_NODISCARD T bias() const { return bias_; }

            private:
                KernelType kernel_;
                T C_;
                T gamma_;
                core::u32 degree_;
                core::u32 max_iterations_;
                T tolerance_;

                math::Matrix<T> X_train_;
                math::Vector<T> y_train_;
                math::Vector<T> alphas_;
                T bias_;
                core::usize n_samples_;
                core::usize n_features_;

                T kernel_func(core::usize i, core::usize j) const {
                    switch (kernel_) {
                        case KernelType::Linear:
                            return linear_kernel(i, j);
                        case KernelType::RBF:
                            return rbf_kernel(i, j);
                        case KernelType::Polynomial:
                            return poly_kernel(i, j);
                        default:
                            return linear_kernel(i, j);
                    }
                }

                T linear_kernel(core::usize i, core::usize j) const {
                    T dot = T{0};
                    for (core::usize k = 0; k < n_features_; ++k) {
                        dot += X_train_(i, k) * X_train_(j, k);
                    }
                    return dot;
                }

                T rbf_kernel(core::usize i, core::usize j) const {
                    T dist_sq = T{0};
                    for (core::usize k = 0; k < n_features_; ++k) {
                        T diff = X_train_(i, k) - X_train_(j, k);
                        dist_sq += diff * diff;
                    }
                    return std::exp(-gamma_ * dist_sq);
                }

                T poly_kernel(core::usize i, core::usize j) const {
                    T dot = linear_kernel(i, j);
                    return std::pow(gamma_ * dot + T{1}, static_cast<T>(degree_));
                }

                T kernel_func_test(const math::Matrix<T>& X, core::usize test_idx, core::usize train_idx) const {
                    switch (kernel_) {
                        case KernelType::Linear: {
                            T dot = T{0};
                            for (core::usize k = 0; k < n_features_; ++k) {
                                dot += X(test_idx, k) * X_train_(train_idx, k);
                            }
                            return dot;
                        }
                        case KernelType::RBF: {
                            T dist_sq = T{0};
                            for (core::usize k = 0; k < n_features_; ++k) {
                                T diff = X(test_idx, k) - X_train_(train_idx, k);
                                dist_sq += diff * diff;
                            }
                            return std::exp(-gamma_ * dist_sq);
                        }
                        case KernelType::Polynomial: {
                            T dot = T{0};
                            for (core::usize k = 0; k < n_features_; ++k) {
                                dot += X(test_idx, k) * X_train_(train_idx, k);
                            }
                            return std::pow(gamma_ * dot + T{1}, static_cast<T>(degree_));
                        }
                        default:
                            return T{0};
                    }
                }

                T compute_error(core::usize i) const {
                    T f = bias_;
                    for (core::usize j = 0; j < n_samples_; ++j) {
                        if (alphas_[j] > T{0}) {
                            f += alphas_[j] * y_train_[j] * kernel_func(j, i);
                        }
                    }
                    return f - y_train_[i];
                }

                T decision_function_single(const math::Matrix<T>& X, core::usize test_idx) const {
                    T decision = bias_;
                    for (core::usize i = 0; i < n_samples_; ++i) {
                        if (alphas_[i] > T{0}) {
                            decision += alphas_[i] * y_train_[i] * kernel_func_test(X, test_idx, i);
                        }
                    }
                    return decision;
                }
            };

        } // namespace algorithms
    } // namespace ml
} // namespace psi

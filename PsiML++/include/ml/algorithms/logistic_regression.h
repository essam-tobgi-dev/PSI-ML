#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include "../model.h"
#include "../metrics.h"
#include "../optimizers/gradient_descent.h"
#include <cmath>

namespace psi {
    namespace ml {
        namespace algorithms {

            // Sigmoid function
            template<typename T>
            T sigmoid(T x) {
                if (x >= T{0}) {
                    return T{1} / (T{1} + std::exp(-x));
                } else {
                    T exp_x = std::exp(x);
                    return exp_x / (T{1} + exp_x);
                }
            }

            // Logistic Regression for binary classification
            template<typename T>
            class LogisticRegression : public SupervisedModel<T> {
            public:
                LogisticRegression(
                    T learning_rate = T{0.01},
                    core::u32 max_iterations = 1000,
                    T tolerance = T{1e-6},
                    T regularization = T{0},
                    bool fit_intercept = true)
                    : learning_rate_(learning_rate)
                    , max_iterations_(max_iterations)
                    , tolerance_(tolerance)
                    , regularization_(regularization)
                    , fit_intercept_(fit_intercept) {}

                void fit(const math::Matrix<T>& X, const math::Vector<T>& y) override {
                    PSI_CHECK_DIMENSIONS("LogisticRegression::fit", X.rows(), y.size());

                    this->state_ = ModelState::Training;

                    math::Matrix<T> X_design = fit_intercept_ ? add_bias_column(X) : X;
                    core::usize n_samples = X_design.rows();
                    core::usize n_features = X_design.cols();

                    // Initialize weights to zero
                    weights_ = math::Vector<T>(n_features);
                    weights_.fill(T{0});

                    T prev_loss = std::numeric_limits<T>::max();

                    for (core::u32 iter = 0; iter < max_iterations_; ++iter) {
                        // Compute predictions
                        math::Vector<T> predictions(n_samples);
                        for (core::usize i = 0; i < n_samples; ++i) {
                            T logit = T{0};
                            for (core::usize j = 0; j < n_features; ++j) {
                                logit += X_design(i, j) * weights_[j];
                            }
                            predictions[i] = sigmoid(logit);
                        }

                        // Compute gradient
                        math::Vector<T> gradient(n_features);
                        for (core::usize j = 0; j < n_features; ++j) {
                            T sum = T{0};
                            for (core::usize i = 0; i < n_samples; ++i) {
                                sum += X_design(i, j) * (predictions[i] - y[i]);
                            }
                            gradient[j] = sum / static_cast<T>(n_samples);

                            // Add regularization (don't regularize intercept)
                            if (regularization_ > T{0} && (j > 0 || !fit_intercept_)) {
                                gradient[j] += regularization_ * weights_[j];
                            }
                        }

                        // Update weights
                        for (core::usize j = 0; j < n_features; ++j) {
                            weights_[j] -= learning_rate_ * gradient[j];
                        }

                        // Check convergence (using binary cross-entropy)
                        T loss = metrics::binary_cross_entropy(y, predictions);

                        if (std::abs(prev_loss - loss) < tolerance_) {
                            break;
                        }
                        prev_loss = loss;
                    }

                    this->state_ = ModelState::Trained;
                }

                math::Vector<T> predict(const math::Matrix<T>& X) const override {
                    math::Vector<T> proba = predict_proba(X);
                    math::Vector<T> predictions(proba.size());

                    for (core::usize i = 0; i < proba.size(); ++i) {
                        predictions[i] = (proba[i] >= T{0.5}) ? T{1} : T{0};
                    }

                    return predictions;
                }

                math::Vector<T> predict_proba(const math::Matrix<T>& X) const {
                    PSI_ASSERT(this->is_trained(), "Model must be trained before prediction");

                    math::Matrix<T> X_design = fit_intercept_ ? add_bias_column(X) : X;
                    PSI_ASSERT(X_design.cols() == weights_.size(), "Feature dimension mismatch");

                    core::usize n_samples = X_design.rows();
                    math::Vector<T> predictions(n_samples);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        T logit = T{0};
                        for (core::usize j = 0; j < weights_.size(); ++j) {
                            logit += X_design(i, j) * weights_[j];
                        }
                        predictions[i] = sigmoid(logit);
                    }

                    return predictions;
                }

                T score(const math::Matrix<T>& X, const math::Vector<T>& y) const override {
                    math::Vector<T> predictions = predict(X);
                    return metrics::accuracy(y, predictions);
                }

                PSI_NODISCARD std::string name() const override { return "LogisticRegression"; }

                PSI_NODISCARD const math::Vector<T>& weights() const { return weights_; }

                PSI_NODISCARD T intercept() const {
                    return fit_intercept_ ? weights_[0] : T{0};
                }

                PSI_NODISCARD math::Vector<T> coefficients() const {
                    if (!fit_intercept_) return weights_;

                    math::Vector<T> coef(weights_.size() - 1);
                    for (core::usize i = 1; i < weights_.size(); ++i) {
                        coef[i - 1] = weights_[i];
                    }
                    return coef;
                }

            private:
                T learning_rate_;
                core::u32 max_iterations_;
                T tolerance_;
                T regularization_;
                bool fit_intercept_;
                math::Vector<T> weights_;

                math::Matrix<T> add_bias_column(const math::Matrix<T>& X) const {
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

        } // namespace algorithms
    } // namespace ml
} // namespace psi

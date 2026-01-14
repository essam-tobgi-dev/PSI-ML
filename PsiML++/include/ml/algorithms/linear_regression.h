#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include "../../math/linalg/solvers.h"
#include "../../math/linalg/decomposition.h"
#include "../model.h"
#include "../metrics.h"
#include "../optimizers/gradient_descent.h"
#include <cmath>

namespace psi {
    namespace ml {
        namespace algorithms {

            // Linear Regression using normal equation or gradient descent
            template<typename T>
            class LinearRegression : public SupervisedModel<T> {
            public:
                enum class Solver {
                    NormalEquation,   // Closed-form solution using (X^T X)^-1 X^T y
                    GradientDescent   // Iterative optimization
                };

                LinearRegression(
                    Solver solver = Solver::NormalEquation,
                    T learning_rate = T{0.01},
                    core::u32 max_iterations = 1000,
                    T tolerance = T{1e-6},
                    bool fit_intercept = true)
                    : solver_(solver)
                    , learning_rate_(learning_rate)
                    , max_iterations_(max_iterations)
                    , tolerance_(tolerance)
                    , fit_intercept_(fit_intercept) {}

                void fit(const math::Matrix<T>& X, const math::Vector<T>& y) override {
                    PSI_CHECK_DIMENSIONS("LinearRegression::fit", X.rows(), y.size());

                    this->state_ = ModelState::Training;

                    // Add bias column if fitting intercept
                    math::Matrix<T> X_design = fit_intercept_ ? add_bias_column(X) : X;

                    [[maybe_unused]] core::usize n_features = X_design.cols();

                    if (solver_ == Solver::NormalEquation) {
                        fit_normal_equation(X_design, y);
                    } else {
                        fit_gradient_descent(X_design, y);
                    }

                    this->state_ = ModelState::Trained;
                }

                math::Vector<T> predict(const math::Matrix<T>& X) const override {
                    PSI_ASSERT(this->is_trained(), "Model must be trained before prediction");

                    math::Matrix<T> X_design = fit_intercept_ ? add_bias_column(X) : X;
                    PSI_ASSERT(X_design.cols() == weights_.size(), "Feature dimension mismatch");

                    core::usize n_samples = X_design.rows();
                    math::Vector<T> predictions(n_samples);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        T pred = T{0};
                        for (core::usize j = 0; j < weights_.size(); ++j) {
                            pred += X_design(i, j) * weights_[j];
                        }
                        predictions[i] = pred;
                    }

                    return predictions;
                }

                T score(const math::Matrix<T>& X, const math::Vector<T>& y) const override {
                    math::Vector<T> predictions = predict(X);
                    return metrics::r2_score(y, predictions);
                }

                PSI_NODISCARD std::string name() const override { return "LinearRegression"; }

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
                Solver solver_;
                T learning_rate_;
                core::u32 max_iterations_;
                T tolerance_;
                bool fit_intercept_;
                math::Vector<T> weights_;

                math::Matrix<T> add_bias_column(const math::Matrix<T>& X) const {
                    core::usize n_samples = X.rows();
                    core::usize n_features = X.cols();

                    math::Matrix<T> X_bias(n_samples, n_features + 1);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        X_bias(i, 0) = T{1};  // Bias column
                        for (core::usize j = 0; j < n_features; ++j) {
                            X_bias(i, j + 1) = X(i, j);
                        }
                    }

                    return X_bias;
                }

                void fit_normal_equation(const math::Matrix<T>& X, const math::Vector<T>& y) {
                    // weights = (X^T X)^-1 X^T y
                    core::usize n_features = X.cols();

                    // Compute X^T X
                    math::Matrix<T> XtX(n_features, n_features);
                    for (core::usize i = 0; i < n_features; ++i) {
                        for (core::usize j = 0; j < n_features; ++j) {
                            T sum = T{0};
                            for (core::usize k = 0; k < X.rows(); ++k) {
                                sum += X(k, i) * X(k, j);
                            }
                            XtX(i, j) = sum;
                        }
                    }

                    // Compute X^T y
                    math::Vector<T> Xty(n_features);
                    for (core::usize i = 0; i < n_features; ++i) {
                        T sum = T{0};
                        for (core::usize k = 0; k < X.rows(); ++k) {
                            sum += X(k, i) * y[k];
                        }
                        Xty[i] = sum;
                    }

                    // Solve (X^T X) w = X^T y
                    weights_ = math::linalg::solve(XtX, Xty);
                }

                void fit_gradient_descent(const math::Matrix<T>& X, const math::Vector<T>& y) {
                    core::usize n_features = X.cols();

                    // Initialize weights to zero
                    weights_ = math::Vector<T>(n_features);
                    weights_.fill(T{0});

                    optimizers::GradientDescent<T> optimizer(learning_rate_);

                    T prev_loss = std::numeric_limits<T>::max();

                    for (core::u32 iter = 0; iter < max_iterations_; ++iter) {
                        // Compute gradient
                        math::Vector<T> gradient = optimizers::compute_gradient_mse(X, y, weights_);

                        // Update weights
                        optimizer.update(weights_, gradient);

                        // Check convergence
                        math::Vector<T> predictions(X.rows());
                        for (core::usize i = 0; i < X.rows(); ++i) {
                            T pred = T{0};
                            for (core::usize j = 0; j < n_features; ++j) {
                                pred += X(i, j) * weights_[j];
                            }
                            predictions[i] = pred;
                        }

                        T loss = metrics::mean_squared_error(y, predictions);

                        if (std::abs(prev_loss - loss) < tolerance_) {
                            break;
                        }
                        prev_loss = loss;
                    }
                }
            };

            // Ridge Regression (L2 regularization)
            template<typename T>
            class RidgeRegression : public SupervisedModel<T> {
            public:
                RidgeRegression(T alpha = T{1.0}, bool fit_intercept = true)
                    : alpha_(alpha), fit_intercept_(fit_intercept) {
                    PSI_ASSERT(alpha >= T{0}, "Regularization parameter must be non-negative");
                }

                void fit(const math::Matrix<T>& X, const math::Vector<T>& y) override {
                    PSI_CHECK_DIMENSIONS("RidgeRegression::fit", X.rows(), y.size());

                    this->state_ = ModelState::Training;

                    math::Matrix<T> X_design = fit_intercept_ ? add_bias_column(X) : X;
                    core::usize n_features = X_design.cols();

                    // Compute X^T X + alpha * I
                    math::Matrix<T> XtX(n_features, n_features);
                    for (core::usize i = 0; i < n_features; ++i) {
                        for (core::usize j = 0; j < n_features; ++j) {
                            T sum = T{0};
                            for (core::usize k = 0; k < X_design.rows(); ++k) {
                                sum += X_design(k, i) * X_design(k, j);
                            }
                            XtX(i, j) = sum;
                            if (i == j && (i > 0 || !fit_intercept_)) {
                                XtX(i, j) += alpha_;  // Don't regularize intercept
                            }
                        }
                    }

                    // Compute X^T y
                    math::Vector<T> Xty(n_features);
                    for (core::usize i = 0; i < n_features; ++i) {
                        T sum = T{0};
                        for (core::usize k = 0; k < X_design.rows(); ++k) {
                            sum += X_design(k, i) * y[k];
                        }
                        Xty[i] = sum;
                    }

                    weights_ = math::linalg::solve(XtX, Xty);

                    this->state_ = ModelState::Trained;
                }

                math::Vector<T> predict(const math::Matrix<T>& X) const override {
                    PSI_ASSERT(this->is_trained(), "Model must be trained before prediction");

                    math::Matrix<T> X_design = fit_intercept_ ? add_bias_column(X) : X;

                    core::usize n_samples = X_design.rows();
                    math::Vector<T> predictions(n_samples);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        T pred = T{0};
                        for (core::usize j = 0; j < weights_.size(); ++j) {
                            pred += X_design(i, j) * weights_[j];
                        }
                        predictions[i] = pred;
                    }

                    return predictions;
                }

                T score(const math::Matrix<T>& X, const math::Vector<T>& y) const override {
                    math::Vector<T> predictions = predict(X);
                    return metrics::r2_score(y, predictions);
                }

                PSI_NODISCARD std::string name() const override { return "RidgeRegression"; }
                PSI_NODISCARD const math::Vector<T>& weights() const { return weights_; }
                PSI_NODISCARD T alpha() const { return alpha_; }

            private:
                T alpha_;
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

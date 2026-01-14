#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../../math/vector.h"
#include "../../math/matrix.h"
#include "../../math/linalg/eigen.h"
#include "../../math/linalg/statistics.h"
#include "../model.h"
#include <cmath>
#include <algorithm>

namespace psi {
    namespace ml {
        namespace algorithms {

            // Principal Component Analysis
            template<typename T>
            class PCA : public UnsupervisedModel<T> {
            public:
                PCA(core::usize n_components = 0, bool center = true)
                    : n_components_(n_components)
                    , center_(center)
                    , n_features_(0) {}

                void fit(const math::Matrix<T>& X) override {
                    this->state_ = ModelState::Training;

                    core::usize n_samples = X.rows();
                    n_features_ = X.cols();

                    // Determine number of components
                    core::usize actual_components = (n_components_ == 0) ?
                        std::min(n_samples, n_features_) : std::min(n_components_, std::min(n_samples, n_features_));

                    // Center the data
                    if (center_) {
                        mean_ = math::Vector<T>(n_features_);
                        for (core::usize j = 0; j < n_features_; ++j) {
                            T sum = T{0};
                            for (core::usize i = 0; i < n_samples; ++i) {
                                sum += X(i, j);
                            }
                            mean_[j] = sum / static_cast<T>(n_samples);
                        }
                    }

                    // Create centered data matrix
                    math::Matrix<T> X_centered(n_samples, n_features_);
                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features_; ++j) {
                            X_centered(i, j) = center_ ? (X(i, j) - mean_[j]) : X(i, j);
                        }
                    }

                    // Compute covariance matrix: (1/(n-1)) * X^T * X
                    math::Matrix<T> cov_matrix(n_features_, n_features_);
                    T scale = T{1} / static_cast<T>(n_samples - 1);

                    for (core::usize i = 0; i < n_features_; ++i) {
                        for (core::usize j = i; j < n_features_; ++j) {
                            T sum = T{0};
                            for (core::usize k = 0; k < n_samples; ++k) {
                                sum += X_centered(k, i) * X_centered(k, j);
                            }
                            cov_matrix(i, j) = sum * scale;
                            cov_matrix(j, i) = cov_matrix(i, j);  // Symmetric
                        }
                    }

                    // Eigendecomposition
                    auto eigen_result = math::linalg::jacobi_eigenvalue(cov_matrix);

                    if (!eigen_result.converged) {
                        PSI_THROW_ML("PCA eigendecomposition did not converge");
                    }

                    // Store results (eigenvalues are already sorted in descending order)
                    explained_variance_ = math::Vector<T>(actual_components);
                    explained_variance_ratio_ = math::Vector<T>(actual_components);
                    components_ = math::Matrix<T>(actual_components, n_features_);

                    T total_variance = T{0};
                    for (core::usize i = 0; i < eigen_result.eigenvalues.size(); ++i) {
                        total_variance += eigen_result.eigenvalues[i];
                    }

                    for (core::usize i = 0; i < actual_components; ++i) {
                        explained_variance_[i] = eigen_result.eigenvalues[i];
                        explained_variance_ratio_[i] = eigen_result.eigenvalues[i] / total_variance;

                        for (core::usize j = 0; j < n_features_; ++j) {
                            components_(i, j) = eigen_result.eigenvectors(j, i);  // Transpose
                        }
                    }

                    this->state_ = ModelState::Trained;
                }

                math::Matrix<T> transform(const math::Matrix<T>& X) const override {
                    PSI_ASSERT(this->is_fitted(), "PCA must be fitted before transform");
                    PSI_ASSERT(X.cols() == n_features_, "Feature dimension mismatch");

                    core::usize n_samples = X.rows();
                    core::usize n_components = components_.rows();

                    // Center the data
                    math::Matrix<T> X_centered(n_samples, n_features_);
                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features_; ++j) {
                            X_centered(i, j) = center_ ? (X(i, j) - mean_[j]) : X(i, j);
                        }
                    }

                    // Project onto principal components: X_transformed = X_centered * components^T
                    math::Matrix<T> X_transformed(n_samples, n_components);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize k = 0; k < n_components; ++k) {
                            T sum = T{0};
                            for (core::usize j = 0; j < n_features_; ++j) {
                                sum += X_centered(i, j) * components_(k, j);
                            }
                            X_transformed(i, k) = sum;
                        }
                    }

                    return X_transformed;
                }

                math::Matrix<T> inverse_transform(const math::Matrix<T>& X_transformed) const {
                    PSI_ASSERT(this->is_fitted(), "PCA must be fitted before inverse_transform");
                    PSI_ASSERT(X_transformed.cols() == components_.rows(), "Component dimension mismatch");

                    core::usize n_samples = X_transformed.rows();
                    core::usize n_components = X_transformed.cols();

                    // Reconstruct: X_reconstructed = X_transformed * components + mean
                    math::Matrix<T> X_reconstructed(n_samples, n_features_);

                    for (core::usize i = 0; i < n_samples; ++i) {
                        for (core::usize j = 0; j < n_features_; ++j) {
                            T sum = center_ ? mean_[j] : T{0};
                            for (core::usize k = 0; k < n_components; ++k) {
                                sum += X_transformed(i, k) * components_(k, j);
                            }
                            X_reconstructed(i, j) = sum;
                        }
                    }

                    return X_reconstructed;
                }

                PSI_NODISCARD std::string name() const override { return "PCA"; }

                PSI_NODISCARD const math::Matrix<T>& components() const { return components_; }
                PSI_NODISCARD const math::Vector<T>& explained_variance() const { return explained_variance_; }
                PSI_NODISCARD const math::Vector<T>& explained_variance_ratio() const { return explained_variance_ratio_; }
                PSI_NODISCARD const math::Vector<T>& mean() const { return mean_; }

                PSI_NODISCARD T total_explained_variance_ratio() const {
                    T total = T{0};
                    for (core::usize i = 0; i < explained_variance_ratio_.size(); ++i) {
                        total += explained_variance_ratio_[i];
                    }
                    return total;
                }

            private:
                core::usize n_components_;
                bool center_;
                core::usize n_features_;

                math::Matrix<T> components_;
                math::Vector<T> explained_variance_;
                math::Vector<T> explained_variance_ratio_;
                math::Vector<T> mean_;
            };

        } // namespace algorithms
    } // namespace ml
} // namespace psi

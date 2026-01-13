#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/memory.h"
#include "../../core/exception.h"
#include "../vector.h"
#include "../matrix.h"
#include "blas.h"
#include "decomposition.h"
#include "solvers.h"
#include <cmath>
#include <complex>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace psi {
    namespace math {
        namespace linalg {

            // Forward declarations
            template<typename T>
            void hessenberg_reduction(Matrix<T>& A, Matrix<T>& Q);

            // Complex number support for eigenvalues
            template<typename T>
            using Complex = std::complex<T>;

            // Eigenvalue/eigenvector result types
            template<typename T>
            struct EigenResult {
                Vector<Complex<T>> eigenvalues;
                Matrix<Complex<T>> eigenvectors;
                bool converged;
                core::u32 iterations;

                EigenResult(core::usize n, core::device_id_t device_id = 0)
                    : eigenvalues(n, device_id)
                    , eigenvectors(n, n, device_id)
                    , converged(false)
                    , iterations(0) {
                }
            };

            template<typename T>
            struct RealEigenResult {
                Vector<T> eigenvalues;
                Matrix<T> eigenvectors;
                bool converged;
                core::u32 iterations;

                RealEigenResult(core::usize n, core::device_id_t device_id = 0)
                    : eigenvalues(n, device_id)
                    , eigenvectors(n, n, device_id)
                    , converged(false)
                    , iterations(0) {
                }
            };

            // Power iteration for dominant eigenvalue
            template<typename T>
            PSI_NODISCARD std::pair<T, Vector<T>> power_iteration(const Matrix<T>& A,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square(), "Power iteration requires square matrix");

                core::usize n = A.rows();

                // Initialize vector
                Vector<T> v(n, A.device_id());
                for (core::usize i = 0; i < n; ++i) {
                    v[i] = static_cast<T>(1.0 + 0.1 * i);
                }

                T eigenvalue = T{};

                for (core::u32 iter = 0; iter < max_iterations; ++iter) {
                    // Normalize vector
                    T norm = nrm2(v);
                    if (norm < tolerance) {
                        PSI_THROW_MATH("Power iteration: zero vector encountered");
                    }
                    v = scal(T{ 1 } / norm, v);

                    // Apply matrix
                    Vector<T> Av = matvec(A, v);

                    // Compute Rayleigh quotient
                    T new_eigenvalue = dot(v, Av);

                    // Check convergence
                    if (iter > 0 && std::abs(new_eigenvalue - eigenvalue) < tolerance) {
                        eigenvalue = new_eigenvalue;
                        break;
                    }

                    eigenvalue = new_eigenvalue;
                    v = std::move(Av);
                }

                // Final normalization
                T norm = nrm2(v);
                if (norm > tolerance) {
                    v = scal(T{ 1 } / norm, v);
                }

                return { eigenvalue, v };
            }

            // Inverse power iteration for smallest eigenvalue
            template<typename T>
            PSI_NODISCARD std::pair<T, Vector<T>> inverse_power_iteration(const Matrix<T>& A,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square(), "Inverse power iteration requires square matrix");

                core::usize n = A.rows();

                // Compute LU decomposition of A
                LUDecomposition<T> lu = lu_decomposition(A);
                if (lu.is_singular) {
                    PSI_THROW_MATH("Inverse power iteration: singular matrix");
                }

                // Initialize vector
                Vector<T> v(n, A.device_id());
                for (core::usize i = 0; i < n; ++i) {
                    v[i] = static_cast<T>(1.0 + 0.1 * i);
                }

                T eigenvalue = T{};

                for (core::u32 iter = 0; iter < max_iterations; ++iter) {
                    // Normalize vector
                    T norm = nrm2(v);
                    if (norm < tolerance) {
                        PSI_THROW_MATH("Inverse power iteration: zero vector encountered");
                    }
                    v = scal(T{ 1 } / norm, v);

                    // Solve A * w = v using LU decomposition
                    Vector<T> w = lu_solve(lu, v);

                    // Compute Rayleigh quotient
                    T new_eigenvalue = dot(v, w) / dot(w, w);

                    // Check convergence
                    if (iter > 0 && std::abs(new_eigenvalue - eigenvalue) < tolerance) {
                        eigenvalue = new_eigenvalue;
                        v = w;
                        break;
                    }

                    eigenvalue = new_eigenvalue;
                    v = std::move(w);
                }

                // Final normalization
                T norm = nrm2(v);
                if (norm > tolerance) {
                    v = scal(T{ 1 } / norm, v);
                }

                return { T{1} / eigenvalue, v };
            }

            // QR algorithm for all eigenvalues
            template<typename T>
            PSI_NODISCARD EigenResult<T> qr_algorithm(const Matrix<T>& A,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square(), "QR algorithm requires square matrix");

                core::usize n = A.rows();
                EigenResult<T> result(n, A.device_id());

                // Make a copy to work with
                Matrix<T> H = A;

                // Initialize Q as identity for eigenvector accumulation
                Matrix<T> Q_total = Matrix<T>::identity(n, A.device_id());

                // Apply Hessenberg reduction first
                hessenberg_reduction(H, Q_total);

                // QR iterations
                for (core::u32 iter = 0; iter < max_iterations; ++iter) {
                    bool converged_all = true;

                    // Check for convergence (subdiagonal elements should be small)
                    for (core::usize i = 1; i < n; ++i) {
                        if (std::abs(H(i, i - 1)) > tolerance) {
                            converged_all = false;
                            break;
                        }
                    }

                    if (converged_all) {
                        result.converged = true;
                        result.iterations = iter;
                        break;
                    }

                    // Perform QR decomposition
                    QRDecomposition<T> qr = qr_decomposition(H);

                    // Update H = R * Q
                    H = linalg::matmul(qr.R, qr.Q);

                    // Accumulate transformation for eigenvectors
                    Q_total = linalg::matmul(Q_total, qr.Q);
                }

                // Extract eigenvalues from diagonal
                for (core::usize i = 0; i < n; ++i) {
                    result.eigenvalues[i] = Complex<T>(H(i, i), T{});
                }

                // Extract eigenvectors
                for (core::usize j = 0; j < n; ++j) {
                    for (core::usize i = 0; i < n; ++i) {
                        result.eigenvectors(i, j) = Complex<T>(Q_total(i, j), T{});
                    }
                }

                return result;
            }

            // Jacobi method for symmetric matrices
            template<typename T>
            PSI_NODISCARD RealEigenResult<T> jacobi_eigenvalue(const Matrix<T>& A,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square(), "Jacobi method requires square matrix");

                core::usize n = A.rows();
                RealEigenResult<T> result(n, A.device_id());

                // Check if matrix is symmetric
                for (core::usize i = 0; i < n; ++i) {
                    for (core::usize j = i + 1; j < n; ++j) {
                        if (std::abs(A(i, j) - A(j, i)) > tolerance) {
                            PSI_THROW_MATH("Jacobi method requires symmetric matrix");
                        }
                    }
                }

                // Initialize
                Matrix<T> D = A;
                result.eigenvectors = Matrix<T>::identity(n, A.device_id());

                for (core::u32 iter = 0; iter < max_iterations; ++iter) {
                    // Find largest off-diagonal element
                    T max_off_diag = T{};
                    core::usize max_i = 0, max_j = 1;

                    for (core::usize i = 0; i < n - 1; ++i) {
                        for (core::usize j = i + 1; j < n; ++j) {
                            T val = std::abs(D(i, j));
                            if (val > max_off_diag) {
                                max_off_diag = val;
                                max_i = i;
                                max_j = j;
                            }
                        }
                    }

                    // Check convergence
                    if (max_off_diag < tolerance) {
                        result.converged = true;
                        result.iterations = iter;
                        break;
                    }

                    // Compute rotation angle
                    T theta;
                    T d_ii = D(max_i, max_i);
                    T d_jj = D(max_j, max_j);
                    T d_ij = D(max_i, max_j);

                    if (std::abs(d_ii - d_jj) < tolerance) {
                        theta = M_PI / 4;
                    }
                    else {
                        theta = 0.5 * std::atan(2 * d_ij / (d_ii - d_jj));
                    }

                    T c = std::cos(theta);
                    T s = std::sin(theta);

                    // Apply Givens rotation to D
                    for (core::usize k = 0; k < n; ++k) {
                        if (k != max_i && k != max_j) {
                            T d_ik = D(max_i, k);
                            T d_jk = D(max_j, k);
                            D(max_i, k) = c * d_ik - s * d_jk;
                            D(k, max_i) = D(max_i, k);
                            D(max_j, k) = s * d_ik + c * d_jk;
                            D(k, max_j) = D(max_j, k);
                        }
                    }

                    // Update diagonal elements
                    T new_dii = c * c * d_ii + s * s * d_jj - 2 * s * c * d_ij;
                    T new_djj = s * s * d_ii + c * c * d_jj + 2 * s * c * d_ij;
                    D(max_i, max_i) = new_dii;
                    D(max_j, max_j) = new_djj;
                    D(max_i, max_j) = T{};
                    D(max_j, max_i) = T{};

                    // Update eigenvectors
                    for (core::usize k = 0; k < n; ++k) {
                        T v_ki = result.eigenvectors(k, max_i);
                        T v_kj = result.eigenvectors(k, max_j);
                        result.eigenvectors(k, max_i) = c * v_ki - s * v_kj;
                        result.eigenvectors(k, max_j) = s * v_ki + c * v_kj;
                    }
                }

                // Extract eigenvalues from diagonal
                for (core::usize i = 0; i < n; ++i) {
                    result.eigenvalues[i] = D(i, i);
                }

                // Sort eigenvalues and eigenvectors in descending order
                for (core::usize i = 0; i < n - 1; ++i) {
                    for (core::usize j = i + 1; j < n; ++j) {
                        if (result.eigenvalues[j] > result.eigenvalues[i]) {
                            std::swap(result.eigenvalues[i], result.eigenvalues[j]);

                            // Swap corresponding eigenvectors (columns)
                            for (core::usize k = 0; k < n; ++k) {
                                std::swap(result.eigenvectors(k, i), result.eigenvectors(k, j));
                            }
                        }
                    }
                }

                return result;
            }

            // Hessenberg reduction (helper for QR algorithm)
            template<typename T>
            void hessenberg_reduction(Matrix<T>& A, Matrix<T>& Q) {
                core::usize n = A.rows();

                for (core::usize k = 0; k < n - 2; ++k) {
                    // Find Householder vector for column k
                    Vector<T> x(n - k - 1, A.device_id());
                    for (core::usize i = 0; i < n - k - 1; ++i) {
                        x[i] = A(k + 1 + i, k);
                    }

                    T norm_x = nrm2(x);
                    if (norm_x < std::numeric_limits<T>::epsilon()) continue;

                    // Compute Householder vector
                    Vector<T> v = x;
                    v[0] += (x[0] >= T{}) ? norm_x : -norm_x;
                    T norm_v = nrm2(v);

                    if (norm_v < std::numeric_limits<T>::epsilon()) continue;

                    v = scal(T{ 1 } / norm_v, v);

                    // Apply Householder transformation H = I - 2*v*v^T

                    // Left multiplication: A = H * A
                    for (core::usize j = k; j < n; ++j) {
                        T dot_product = T{};
                        for (core::usize i = 0; i < n - k - 1; ++i) {
                            dot_product += v[i] * A(k + 1 + i, j);
                        }

                        for (core::usize i = 0; i < n - k - 1; ++i) {
                            A(k + 1 + i, j) -= 2 * v[i] * dot_product;
                        }
                    }

                    // Right multiplication: A = A * H
                    for (core::usize i = 0; i < n; ++i) {
                        T dot_product = T{};
                        for (core::usize j = 0; j < n - k - 1; ++j) {
                            dot_product += A(i, k + 1 + j) * v[j];
                        }

                        for (core::usize j = 0; j < n - k - 1; ++j) {
                            A(i, k + 1 + j) -= 2 * dot_product * v[j];
                        }
                    }

                    // Update Q matrix
                    for (core::usize i = 0; i < n; ++i) {
                        T dot_product = T{};
                        for (core::usize j = 0; j < n - k - 1; ++j) {
                            dot_product += Q(i, k + 1 + j) * v[j];
                        }

                        for (core::usize j = 0; j < n - k - 1; ++j) {
                            Q(i, k + 1 + j) -= 2 * dot_product * v[j];
                        }
                    }
                }
            }

            // Rayleigh quotient
            template<typename T>
            PSI_NODISCARD T rayleigh_quotient(const Matrix<T>& A, const Vector<T>& x) {
                PSI_ASSERT(A.is_square(), "Rayleigh quotient requires square matrix");
                PSI_CHECK_DIMENSIONS("Rayleigh quotient", A.rows(), x.size());

                Vector<T> Ax = matvec(A, x);
                return dot(x, Ax) / dot(x, x);
            }

            // Check if matrix is positive definite using eigenvalues
            template<typename T>
            PSI_NODISCARD bool is_positive_definite(const Matrix<T>& A, T tolerance = T{ 1e-10 }) {
                RealEigenResult<T> eigen_result = jacobi_eigenvalue(A);

                if (!eigen_result.converged) {
                    PSI_THROW_MATH("Failed to compute eigenvalues for positive definite test");
                }

                for (core::usize i = 0; i < eigen_result.eigenvalues.size(); ++i) {
                    if (eigen_result.eigenvalues[i] <= tolerance) {
                        return false;
                    }
                }

                return true;
            }

            // Principal Component Analysis using eigendecomposition
            template<typename T>
            struct PCAResult {
                Matrix<T> principal_components;
                Vector<T> explained_variance;
                Vector<T> explained_variance_ratio;

                PCAResult(core::usize n, core::device_id_t device_id = 0)
                    : principal_components(n, n, device_id)
                    , explained_variance(n, device_id)
                    , explained_variance_ratio(n, device_id) {
                }
            };

            template<typename T>
            PSI_NODISCARD PCAResult<T> pca(const Matrix<T>& data, bool center = true) {
                core::usize n_samples = data.rows();
                core::usize n_features = data.cols();

                PCAResult<T> result(n_features, data.device_id());

                // Center the data if requested
                Matrix<T> centered_data = data;
                if (center) {
                    for (core::usize j = 0; j < n_features; ++j) {
                        Vector<T> col = data.get_col(j);
                        T mean_val = col.mean();

                        for (core::usize i = 0; i < n_samples; ++i) {
                            centered_data(i, j) -= mean_val;
                        }
                    }
                }

                // Compute covariance matrix: cov = (1/(n-1)) * X^T * X
                Matrix<T> cov_matrix(n_features, n_features, data.device_id());
                T scale_factor = T{ 1 } / static_cast<T>(n_samples - 1);

                cov_matrix = gemm(true, false, scale_factor,
                    centered_data, centered_data, T{ 0 }, cov_matrix);

                // Compute eigendecomposition
                RealEigenResult<T> eigen_result = jacobi_eigenvalue(cov_matrix);

                if (!eigen_result.converged) {
                    PSI_THROW_MATH("PCA: Failed to compute eigendecomposition of covariance matrix");
                }

                // Copy results
                result.principal_components = eigen_result.eigenvectors;
                result.explained_variance = eigen_result.eigenvalues;

                // Compute explained variance ratios
                T total_variance = result.explained_variance.sum();
                for (core::usize i = 0; i < n_features; ++i) {
                    result.explained_variance_ratio[i] = result.explained_variance[i] / total_variance;
                }

                return result;
            }

            // Generalized eigenvalue problem: A*x = lambda*B*x
            template<typename T>
            PSI_NODISCARD RealEigenResult<T> generalized_eigenvalue(const Matrix<T>& A, const Matrix<T>& B,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square() && B.is_square(), "Both matrices must be square");
                PSI_CHECK_DIMENSIONS("generalized eigenvalue", A.rows(), B.rows());

                // Use Cholesky decomposition of B (assumes B is positive definite)
                CholeskyDecomposition<T> chol = cholesky_decomposition(B);
                if (!chol.is_positive_definite) {
                    PSI_THROW_MATH("Generalized eigenvalue: B matrix is not positive definite");
                }

                // Transform to standard eigenvalue problem
                // A*x = lambda*B*x  =>  L^(-1)*A*L^(-T)*y = lambda*y, where x = L^(-T)*y

                // Compute L^(-1) using forward substitution approach
                core::usize n = A.rows();
                Matrix<T> L_inv = Matrix<T>::identity(n, A.device_id());

                for (core::usize j = 0; j < n; ++j) {
                    Vector<T> e_j = L_inv.get_col(j);
                    Vector<T> x_j = cholesky_solve(chol, e_j);
                    L_inv.set_col(j, x_j);
                }

                // Compute C = L^(-1) * A * L^(-T)
                Matrix<T> temp = linalg::matmul(L_inv, A);
                Matrix<T> C = linalg::matmul(temp, L_inv.transpose());

                // Solve standard eigenvalue problem
                RealEigenResult<T> standard_result = jacobi_eigenvalue(C, max_iterations, tolerance);

                // Transform eigenvectors back: x = L^(-T) * y
                RealEigenResult<T> result(A.rows(), A.device_id());
                result.eigenvalues = standard_result.eigenvalues;
                result.converged = standard_result.converged;
                result.iterations = standard_result.iterations;
                result.eigenvectors = linalg::matmul(L_inv.transpose(), standard_result.eigenvectors);

                return result;
            }

        } // namespace linalg
    } // namespace math
} // namespace psi

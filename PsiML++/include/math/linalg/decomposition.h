#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/memory.h"
#include "../../core/exception.h"
#include "../vector.h"
#include "../matrix.h"
#include "blas.h"
#include <cmath>
#include <algorithm>
#include <tuple>

namespace psi {
    namespace math {
        namespace linalg {

            // Decomposition result types
            template<typename T>
            struct LUDecomposition {
                Matrix<T> L;  // Lower triangular
                Matrix<T> U;  // Upper triangular
                Vector<core::index_t> P;  // Permutation vector
                bool is_singular;

                LUDecomposition(core::usize n, core::device_id_t device_id = 0)
                    : L(n, n, device_id), U(n, n, device_id), P(n, device_id), is_singular(false) {
                }
            };

            template<typename T>
            struct QRDecomposition {
                Matrix<T> Q;  // Orthogonal matrix
                Matrix<T> R;  // Upper triangular

                QRDecomposition(core::usize m, core::usize n, core::device_id_t device_id = 0)
                    : Q(m, m, device_id), R(m, n, device_id) {
                }
            };

            template<typename T>
            struct CholeskyDecomposition {
                Matrix<T> L;  // Lower triangular
                bool is_positive_definite;

                CholeskyDecomposition(core::usize n, core::device_id_t device_id = 0)
                    : L(n, n, device_id), is_positive_definite(false) {
                }
            };

            template<typename T>
            struct SVDDecomposition {
                Matrix<T> U;  // Left singular vectors
                Vector<T> S;  // Singular values
                Matrix<T> Vt; // Right singular vectors (transposed)

                SVDDecomposition(core::usize m, core::usize n, core::device_id_t device_id = 0)
                    : U(m, m, device_id), S(std::min(m, n), device_id), Vt(n, n, device_id) {
                }
            };

            template<typename T>
            struct SchurDecomposition {
                Matrix<T> T_matrix;  // Schur form
                Matrix<T> Q;         // Orthogonal transformation

                SchurDecomposition(core::usize n, core::device_id_t device_id = 0)
                    : T_matrix(n, n, device_id), Q(n, n, device_id) {
                }
            };

            // LU Decomposition with partial pivoting
            template<typename T>
            PSI_NODISCARD LUDecomposition<T> lu_decomposition(const Matrix<T>& A) {
                PSI_ASSERT(A.is_square(), "LU decomposition requires square matrix");

                core::usize n = A.rows();
                LUDecomposition<T> result(n, A.device_id());

                // Copy A to U
                result.U = A;

                // Initialize L as identity
                result.L = Matrix<T>::identity(n, A.device_id());

                // Initialize permutation vector
                for (core::usize i = 0; i < n; ++i) {
                    result.P[i] = static_cast<core::index_t>(i);
                }

                const T eps = std::numeric_limits<T>::epsilon() * static_cast<T>(100);

                for (core::usize k = 0; k < n - 1; ++k) {
                    // Find pivot
                    core::usize pivot_row = k;
                    T max_val = std::abs(result.U(k, k));

                    for (core::usize i = k + 1; i < n; ++i) {
                        T val = std::abs(result.U(i, k));
                        if (val > max_val) {
                            max_val = val;
                            pivot_row = i;
                        }
                    }

                    // Check for singularity
                    if (max_val < eps) {
                        result.is_singular = true;
                        return result;
                    }

                    // Swap rows in U and P
                    if (pivot_row != k) {
                        for (core::usize j = 0; j < n; ++j) {
                            std::swap(result.U(k, j), result.U(pivot_row, j));
                        }
                        std::swap(result.P[k], result.P[pivot_row]);

                        // Swap corresponding rows in L (for columns < k)
                        for (core::usize j = 0; j < k; ++j) {
                            std::swap(result.L(k, j), result.L(pivot_row, j));
                        }
                    }

                    // Eliminate column k
                    for (core::usize i = k + 1; i < n; ++i) {
                        if (std::abs(result.U(k, k)) < eps) {
                            result.is_singular = true;
                            return result;
                        }

                        T factor = result.U(i, k) / result.U(k, k);
                        result.L(i, k) = factor;

                        for (core::usize j = k; j < n; ++j) {
                            result.U(i, j) -= factor * result.U(k, j);
                        }
                    }
                }

                return result;
            }

            // QR Decomposition using Gram-Schmidt process
            template<typename T>
            PSI_NODISCARD QRDecomposition<T> qr_decomposition(const Matrix<T>& A) {
                core::usize m = A.rows();
                core::usize n = A.cols();

                QRDecomposition<T> result(m, n, A.device_id());
                result.Q.fill(T{});
                result.R.fill(T{});

                // Copy columns of A
                std::vector<Vector<T>> columns;
                for (core::usize j = 0; j < n; ++j) {
                    columns.push_back(A.get_col(j));
                }

                // Modified Gram-Schmidt process
                for (core::usize j = 0; j < n; ++j) {
                    Vector<T> v = columns[j];

                    // Orthogonalize against previous columns
                    for (core::usize i = 0; i < j; ++i) {
                        Vector<T> q_i = result.Q.get_col(i);
                        T r_ij = dot(q_i, v);
                        result.R(i, j) = r_ij;

                        // v = v - r_ij * q_i
                        axpy(-r_ij, q_i, v);
                    }

                    // Normalize
                    T norm = nrm2(v);
                    if (norm > std::numeric_limits<T>::epsilon()) {
                        result.R(j, j) = norm;
                        scal(T{ 1 } / norm, v);
                        result.Q.set_col(j, v);
                    }
                    else {
                        PSI_THROW_MATH("QR decomposition: linearly dependent columns");
                    }
                }

                // Fill remaining columns of Q with orthonormal vectors
                for (core::usize j = n; j < m; ++j) {
                    Vector<T> v(m, A.device_id());
                    v[j] = T{ 1 };  // Start with unit vector

                    // Orthogonalize against all previous columns
                    for (core::usize i = 0; i < j; ++i) {
                        Vector<T> q_i = result.Q.get_col(i);
                        T proj = dot(q_i, v);
                        axpy(-proj, q_i, v);
                    }

                    T norm = nrm2(v);
                    if (norm > std::numeric_limits<T>::epsilon()) {
                        scal(T{ 1 } / norm, v);
                        result.Q.set_col(j, v);
                    }
                }

                return result;
            }

            // Cholesky Decomposition for positive definite matrices
            template<typename T>
            PSI_NODISCARD CholeskyDecomposition<T> cholesky_decomposition(const Matrix<T>& A) {
                PSI_ASSERT(A.is_square(), "Cholesky decomposition requires square matrix");

                core::usize n = A.rows();
                CholeskyDecomposition<T> result(n, A.device_id());
                result.L.fill(T{});

                const T eps = std::numeric_limits<T>::epsilon() * static_cast<T>(100);

                for (core::usize i = 0; i < n; ++i) {
                    for (core::usize j = 0; j <= i; ++j) {
                        if (i == j) {
                            // Diagonal element
                            T sum = T{};
                            for (core::usize k = 0; k < j; ++k) {
                                sum += result.L(j, k) * result.L(j, k);
                            }

                            T val = A(j, j) - sum;
                            if (val <= eps) {
                                result.is_positive_definite = false;
                                return result;
                            }

                            result.L(j, j) = std::sqrt(val);
                        }
                        else {
                            // Lower triangular element
                            T sum = T{};
                            for (core::usize k = 0; k < j; ++k) {
                                sum += result.L(i, k) * result.L(j, k);
                            }

                            if (std::abs(result.L(j, j)) < eps) {
                                result.is_positive_definite = false;
                                return result;
                            }

                            result.L(i, j) = (A(i, j) - sum) / result.L(j, j);
                        }
                    }
                }

                result.is_positive_definite = true;
                return result;
            }

            // SVD using Jacobi method (for educational purposes - not optimal for large matrices)
            template<typename T>
            PSI_NODISCARD SVDDecomposition<T> svd_decomposition(const Matrix<T>& A,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                core::usize m = A.rows();
                core::usize n = A.cols();

                SVDDecomposition<T> result(m, n, A.device_id());

                // For thin SVD, work with A^T A if m > n, otherwise A A^T
                Matrix<T> work_matrix(std::min(m, n), std::min(m, n), A.device_id());
                bool use_aat = (m >= n);

                if (use_aat) {
                    // Compute A^T A for thin SVD
                    work_matrix = linalg::matmul(A.transpose(), A);
                    result.Vt = Matrix<T>::identity(n, A.device_id());
                }
                else {
                    // Compute A A^T
                    work_matrix = linalg::matmul(A, A.transpose());
                    result.U = Matrix<T>::identity(m, A.device_id());
                }

                // Jacobi iterations to diagonalize
                for (core::u32 iter = 0; iter < max_iterations; ++iter) {
                    bool converged = true;

                    for (core::usize i = 0; i < work_matrix.rows() - 1; ++i) {
                        for (core::usize j = i + 1; j < work_matrix.cols(); ++j) {
                            T off_diag = work_matrix(i, j);

                            if (std::abs(off_diag) > tolerance) {
                                converged = false;

                                T aii = work_matrix(i, i);
                                T ajj = work_matrix(j, j);

                                // Compute rotation angle
                                T theta;
                                if (std::abs(aii - ajj) < tolerance) {
                                    theta = M_PI / 4;
                                }
                                else {
                                    theta = 0.5 * std::atan(2 * off_diag / (aii - ajj));
                                }

                                T c = std::cos(theta);
                                T s = std::sin(theta);

                                // Apply Givens rotation
                                for (core::usize k = 0; k < work_matrix.rows(); ++k) {
                                    T temp_ik = work_matrix(i, k);
                                    T temp_jk = work_matrix(j, k);
                                    work_matrix(i, k) = c * temp_ik - s * temp_jk;
                                    work_matrix(j, k) = s * temp_ik + c * temp_jk;
                                }

                                for (core::usize k = 0; k < work_matrix.cols(); ++k) {
                                    T temp_ki = work_matrix(k, i);
                                    T temp_kj = work_matrix(k, j);
                                    work_matrix(k, i) = c * temp_ki - s * temp_kj;
                                    work_matrix(k, j) = s * temp_ki + c * temp_kj;
                                }

                                // Update eigenvector matrix
                                if (use_aat) {
                                    for (core::usize k = 0; k < n; ++k) {
                                        T temp_ki = result.Vt(i, k);
                                        T temp_kj = result.Vt(j, k);
                                        result.Vt(i, k) = c * temp_ki - s * temp_kj;
                                        result.Vt(j, k) = s * temp_ki + c * temp_kj;
                                    }
                                }
                                else {
                                    for (core::usize k = 0; k < m; ++k) {
                                        T temp_ki = result.U(k, i);
                                        T temp_kj = result.U(k, j);
                                        result.U(k, i) = c * temp_ki - s * temp_kj;
                                        result.U(k, j) = s * temp_ki + c * temp_kj;
                                    }
                                }
                            }
                        }
                    }

                    if (converged) break;
                }

                // Extract singular values
                for (core::usize i = 0; i < result.S.size(); ++i) {
                    T val = work_matrix(i, i);
                    result.S[i] = (val > T{}) ? std::sqrt(val) : T{};
                }

                // Sort singular values in descending order
                for (core::usize i = 0; i < result.S.size() - 1; ++i) {
                    for (core::usize j = i + 1; j < result.S.size(); ++j) {
                        if (result.S[j] > result.S[i]) {
                            std::swap(result.S[i], result.S[j]);

                            if (use_aat) {
                                // Swap columns of V^T (rows of V)
                                for (core::usize k = 0; k < n; ++k) {
                                    std::swap(result.Vt(i, k), result.Vt(j, k));
                                }
                            }
                            else {
                                // Swap columns of U
                                for (core::usize k = 0; k < m; ++k) {
                                    std::swap(result.U(k, i), result.U(k, j));
                                }
                            }
                        }
                    }
                }

                // Compute the other factor
                if (use_aat) {
                    // Compute U = A * V * S^(-1)
                    result.U.fill(T{});
                    for (core::usize i = 0; i < std::min(m, n); ++i) {
                        if (result.S[i] > tolerance) {
                            Vector<T> v_i = result.Vt.get_row(i);  // i-th column of V
                            Vector<T> u_i = matvec(A, v_i);
                            scal(T{ 1 } / result.S[i], u_i);
                            result.U.set_col(i, u_i);
                        }
                    }

                    // Complete U to orthonormal basis
                    QRDecomposition<T> qr = qr_decomposition(result.U);
                    result.U = qr.Q;
                }
                else {
                    // Compute V^T = S^(-1) * U^T * A
                    result.Vt.fill(T{});
                    for (core::usize i = 0; i < std::min(m, n); ++i) {
                        if (result.S[i] > tolerance) {
                            Vector<T> u_i = result.U.get_col(i);
                            Vector<T> v_i(n, A.device_id());

                            for (core::usize j = 0; j < n; ++j) {
                                Vector<T> a_j = A.get_col(j);
                                v_i[j] = dot(u_i, a_j) / result.S[i];
                            }

                            result.Vt.set_row(i, v_i);
                        }
                    }
                }

                return result;
            }

            // Simplified SVD for rank-k approximation
            template<typename T>
            PSI_NODISCARD SVDDecomposition<T> truncated_svd(const Matrix<T>& A, core::usize k) {
                SVDDecomposition<T> full_svd = svd_decomposition(A);

                core::usize min_dim = std::min(A.rows(), A.cols());
                k = std::min(k, min_dim);

                // Create truncated decomposition
                SVDDecomposition<T> result(A.rows(), A.cols(), A.device_id());

                // Copy first k components
                for (core::usize i = 0; i < k; ++i) {
                    result.S[i] = full_svd.S[i];

                    Vector<T> u_col = full_svd.U.get_col(i);
                    result.U.set_col(i, u_col);

                    Vector<T> vt_row = full_svd.Vt.get_row(i);
                    result.Vt.set_row(i, vt_row);
                }

                // Zero out remaining components
                for (core::usize i = k; i < min_dim; ++i) {
                    result.S[i] = T{};
                }

                return result;
            }

            // Matrix rank using SVD
            template<typename T>
            PSI_NODISCARD core::usize matrix_rank(const Matrix<T>& A, T tolerance = T{ 1e-10 }) {
                SVDDecomposition<T> svd = svd_decomposition(A);

                core::usize rank = 0;
                for (core::usize i = 0; i < svd.S.size(); ++i) {
                    if (svd.S[i] > tolerance) {
                        ++rank;
                    }
                }

                return rank;
            }

            // Condition number using SVD
            template<typename T>
            PSI_NODISCARD T condition_number(const Matrix<T>& A) {
                SVDDecomposition<T> svd = svd_decomposition(A);

                if (svd.S.size() == 0) return T{};

                T max_sv = svd.S[0];
                T min_sv = svd.S[svd.S.size() - 1];

                if (min_sv <= T{}) {
                    return std::numeric_limits<T>::infinity();
                }

                return max_sv / min_sv;
            }

            // Pseudoinverse using SVD
            template<typename T>
            PSI_NODISCARD Matrix<T> pseudoinverse(const Matrix<T>& A, T tolerance = T{ 1e-10 }) {
                SVDDecomposition<T> svd = svd_decomposition(A);

                // Create pseudoinverse of diagonal matrix
                Vector<T> s_inv(svd.S.size(), A.device_id());
                for (core::usize i = 0; i < svd.S.size(); ++i) {
                    s_inv[i] = (svd.S[i] > tolerance) ? (T{ 1 } / svd.S[i]) : T{};
                }

                // A+ = V * S+ * U^T
                Matrix<T> s_inv_diag = Matrix<T>::diagonal(s_inv, A.device_id());
                Matrix<T> temp = linalg::matmul(svd.Vt.transpose(), s_inv_diag);
                return linalg::matmul(temp, svd.U.transpose());
            }

        } // namespace linalg
    } // namespace math
} // namespace psi
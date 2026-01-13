#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/memory.h"
#include "../../core/exception.h"
#include "../vector.h"
#include "../matrix.h"
#include "blas.h"
#include "decomposition.h"
#include <cmath>
#include <algorithm>

namespace psi {
    namespace math {
        namespace linalg {

            // Solver result types
            template<typename T>
            struct SolverResult {
                Vector<T> solution;
                bool converged;
                core::u32 iterations;
                T residual_norm;

                SolverResult(core::usize n, core::device_id_t device_id = 0)
                    : solution(n, device_id)
                    , converged(false)
                    , iterations(0)
                    , residual_norm(T{}) {
                }
            };

            // Linear system solving using LU decomposition
            template<typename T>
            PSI_NODISCARD Vector<T> lu_solve(const LUDecomposition<T>& lu, const Vector<T>& b) {
                PSI_CHECK_DIMENSIONS("LU solve", lu.L.rows(), b.size());

                if (lu.is_singular) {
                    PSI_THROW_MATH("Cannot solve with singular matrix");
                }

                core::usize n = b.size();
                Vector<T> x(n, b.device_id());

                // Apply permutation to b
                Vector<T> pb(n, b.device_id());
                for (core::usize i = 0; i < n; ++i) {
                    pb[i] = b[lu.P[i]];
                }

                // Forward substitution: L * y = P * b
                Vector<T> y(n, b.device_id());
                for (core::usize i = 0; i < n; ++i) {
                    T sum = T{};
                    for (core::usize j = 0; j < i; ++j) {
                        sum += lu.L(i, j) * y[j];
                    }
                    y[i] = pb[i] - sum;
                }

                // Backward substitution: U * x = y
                for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
                    T sum = T{};
                    for (core::usize j = i + 1; j < n; ++j) {
                        sum += lu.U(i, j) * x[j];
                    }

                    if (std::abs(lu.U(i, i)) < std::numeric_limits<T>::epsilon()) {
                        PSI_THROW_MATH("Division by zero in backward substitution");
                    }

                    x[i] = (y[i] - sum) / lu.U(i, i);
                }

                return x;
            }

            // Direct solve using LU decomposition
            template<typename T>
            PSI_NODISCARD Vector<T> solve(const Matrix<T>& A, const Vector<T>& b) {
                PSI_ASSERT(A.is_square(), "Matrix must be square for direct solve");
                PSI_CHECK_DIMENSIONS("solve", A.rows(), b.size());

                LUDecomposition<T> lu = lu_decomposition(A);
                return lu_solve(lu, b);
            }

            // Linear system solving using Cholesky decomposition (for positive definite matrices)
            template<typename T>
            PSI_NODISCARD Vector<T> cholesky_solve(const CholeskyDecomposition<T>& chol, const Vector<T>& b) {
                PSI_CHECK_DIMENSIONS("Cholesky solve", chol.L.rows(), b.size());

                if (!chol.is_positive_definite) {
                    PSI_THROW_MATH("Cannot solve with non-positive-definite matrix");
                }

                core::usize n = b.size();
                Vector<T> x(n, b.device_id());

                // Forward substitution: L * y = b
                Vector<T> y(n, b.device_id());
                for (core::usize i = 0; i < n; ++i) {
                    T sum = T{};
                    for (core::usize j = 0; j < i; ++j) {
                        sum += chol.L(i, j) * y[j];
                    }
                    y[i] = (b[i] - sum) / chol.L(i, i);
                }

                // Backward substitution: L^T * x = y
                for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
                    T sum = T{};
                    for (core::usize j = i + 1; j < n; ++j) {
                        sum += chol.L(j, i) * x[j];  // L^T(i,j) = L(j,i)
                    }
                    x[i] = (y[i] - sum) / chol.L(i, i);
                }

                return x;
            }

            // Solve using QR decomposition
            template<typename T>
            PSI_NODISCARD Vector<T> qr_solve(const QRDecomposition<T>& qr, const Vector<T>& b) {
                PSI_CHECK_DIMENSIONS("QR solve", qr.Q.rows(), b.size());

                core::usize m = qr.Q.rows();
                core::usize n = qr.R.cols();

                // Compute Q^T * b
                Vector<T> qtb = matvec(qr.Q.transpose(), b);

                // Solve R * x = Q^T * b using backward substitution
                Vector<T> x(n, b.device_id());

                for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
                    T sum = T{};
                    for (core::usize j = i + 1; j < n; ++j) {
                        sum += qr.R(i, j) * x[j];
                    }

                    if (std::abs(qr.R(i, i)) < std::numeric_limits<T>::epsilon()) {
                        PSI_THROW_MATH("Division by zero in QR backward substitution");
                    }

                    x[i] = (qtb[i] - sum) / qr.R(i, i);
                }

                return x;
            }

            // Least squares solve using QR decomposition
            template<typename T>
            PSI_NODISCARD Vector<T> least_squares_solve(const Matrix<T>& A, const Vector<T>& b) {
                PSI_CHECK_DIMENSIONS("least squares solve", A.rows(), b.size());

                QRDecomposition<T> qr = qr_decomposition(A);
                return qr_solve(qr, b);
            }

            // Conjugate Gradient method for symmetric positive definite systems
            template<typename T>
            PSI_NODISCARD SolverResult<T> conjugate_gradient(const Matrix<T>& A, const Vector<T>& b,
                const Vector<T>& x0,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square(), "CG requires square matrix");
                PSI_CHECK_DIMENSIONS("CG A-b", A.rows(), b.size());
                PSI_CHECK_DIMENSIONS("CG A-x0", A.rows(), x0.size());

                core::usize n = A.rows();
                SolverResult<T> result(n, A.device_id());

                // Initialize
                result.solution = x0;
                Vector<T> r = b - matvec(A, result.solution);  // residual
                Vector<T> p = r;  // search direction
                T rsold = dot(r, r);

                for (core::u32 iter = 0; iter < max_iterations; ++iter) {
                    Vector<T> Ap = matvec(A, p);
                    T pAp = dot(p, Ap);

                    if (std::abs(pAp) < std::numeric_limits<T>::epsilon()) {
                        PSI_THROW_MATH("CG: Division by zero in step size computation");
                    }

                    T alpha = rsold / pAp;

                    // Update solution and residual
                    axpy(alpha, p, result.solution);   // x = x + alpha * p
                    axpy(-alpha, Ap, r);              // r = r - alpha * A * p

                    T rsnew = dot(r, r);
                    result.residual_norm = std::sqrt(rsnew);

                    // Check convergence
                    if (result.residual_norm < tolerance) {
                        result.converged = true;
                        result.iterations = iter + 1;
                        return result;
                    }

                    T beta = rsnew / rsold;

                    // Update search direction
                    scal(beta, p);
                    axpy(T{ 1 }, r, p);  // p = r + beta * p

                    rsold = rsnew;
                }

                result.iterations = max_iterations;
                return result;
            }

            // BiCGSTAB method for general non-symmetric systems
            template<typename T>
            PSI_NODISCARD SolverResult<T> bicgstab(const Matrix<T>& A, const Vector<T>& b,
                const Vector<T>& x0,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square(), "BiCGSTAB requires square matrix");
                PSI_CHECK_DIMENSIONS("BiCGSTAB A-b", A.rows(), b.size());
                PSI_CHECK_DIMENSIONS("BiCGSTAB A-x0", A.rows(), x0.size());

                core::usize n = A.rows();
                SolverResult<T> result(n, A.device_id());

                // Initialize
                result.solution = x0;
                Vector<T> r = b - matvec(A, result.solution);
                Vector<T> r_hat = r;  // Arbitrary choice
                Vector<T> p(n, A.device_id());
                Vector<T> v(n, A.device_id());
                Vector<T> s(n, A.device_id());
                Vector<T> t(n, A.device_id());

                T rho = T{ 1 };
                T alpha = T{ 1 };
                T omega = T{ 1 };

                for (core::u32 iter = 0; iter < max_iterations; ++iter) {
                    T rho_new = dot(r_hat, r);

                    if (std::abs(rho_new) < std::numeric_limits<T>::epsilon()) {
                        PSI_THROW_MATH("BiCGSTAB: breakdown in rho computation");
                    }

                    T beta = (rho_new / rho) * (alpha / omega);

                    // p = r + beta * (p - omega * v)
                    axpy(-omega, v, p);
                    scal(beta, p);
                    axpy(T{ 1 }, r, p);

                    v = matvec(A, p);
                    T r_hat_v = dot(r_hat, v);

                    if (std::abs(r_hat_v) < std::numeric_limits<T>::epsilon()) {
                        PSI_THROW_MATH("BiCGSTAB: breakdown in alpha computation");
                    }

                    alpha = rho_new / r_hat_v;

                    // s = r - alpha * v
                    s = r;
                    axpy(-alpha, v, s);

                    // Check if s is small enough
                    T s_norm = nrm2(s);
                    if (s_norm < tolerance) {
                        axpy(alpha, p, result.solution);
                        result.residual_norm = s_norm;
                        result.converged = true;
                        result.iterations = iter + 1;
                        return result;
                    }

                    t = matvec(A, s);
                    T t_norm_sq = dot(t, t);

                    if (t_norm_sq < std::numeric_limits<T>::epsilon()) {
                        PSI_THROW_MATH("BiCGSTAB: breakdown in omega computation");
                    }

                    omega = dot(t, s) / t_norm_sq;

                    // Update solution and residual
                    axpy(alpha, p, result.solution);
                    axpy(omega, s, result.solution);

                    r = s;
                    axpy(-omega, t, r);

                    result.residual_norm = nrm2(r);

                    // Check convergence
                    if (result.residual_norm < tolerance) {
                        result.converged = true;
                        result.iterations = iter + 1;
                        return result;
                    }

                    if (std::abs(omega) < std::numeric_limits<T>::epsilon()) {
                        PSI_THROW_MATH("BiCGSTAB: omega breakdown");
                    }

                    rho = rho_new;
                }

                result.iterations = max_iterations;
                return result;
            }

            // GMRES method (Generalized Minimal Residual)
            template<typename T>
            PSI_NODISCARD SolverResult<T> gmres(const Matrix<T>& A, const Vector<T>& b,
                const Vector<T>& x0,
                core::u32 restart = 30,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square(), "GMRES requires square matrix");
                PSI_CHECK_DIMENSIONS("GMRES A-b", A.rows(), b.size());
                PSI_CHECK_DIMENSIONS("GMRES A-x0", A.rows(), x0.size());

                core::usize n = A.rows();
                SolverResult<T> result(n, A.device_id());
                result.solution = x0;

                core::u32 m = std::min(restart, static_cast<core::u32>(n));

                for (core::u32 outer_iter = 0; outer_iter < max_iterations / m; ++outer_iter) {
                    // Compute initial residual
                    Vector<T> r = b - matvec(A, result.solution);
                    T beta = nrm2(r);

                    result.residual_norm = beta;
                    if (beta < tolerance) {
                        result.converged = true;
                        result.iterations = outer_iter * m;
                        return result;
                    }

                    // Initialize Krylov basis
                    Matrix<T> V(n, m + 1, A.device_id());
                    Matrix<T> H(m + 1, m, A.device_id());
                    H.fill(T{});

                    // v_1 = r / ||r||
                    Vector<T> v1 = r;
                    scal(T{ 1 } / beta, v1);
                    V.set_col(0, v1);

                    // Arnoldi process
                    for (core::u32 j = 0; j < m; ++j) {
                        Vector<T> v_j = V.get_col(j);
                        Vector<T> w = matvec(A, v_j);

                        // Modified Gram-Schmidt orthogonalization
                        for (core::u32 i = 0; i <= j; ++i) {
                            Vector<T> v_i = V.get_col(i);
                            T h_ij = dot(v_i, w);
                            H(i, j) = h_ij;
                            axpy(-h_ij, v_i, w);
                        }

                        T h_jp1_j = nrm2(w);
                        H(j + 1, j) = h_jp1_j;

                        if (h_jp1_j < tolerance) {
                            // Lucky breakdown
                            m = j + 1;
                            break;
                        }

                        if (j + 1 < V.cols()) {
                            scal(T{ 1 } / h_jp1_j, w);
                            V.set_col(j + 1, w);
                        }
                    }

                    // Solve least squares problem: min ||beta * e_1 - H * y||
                    Vector<T> e1(m + 1, A.device_id());
                    e1[0] = beta;

                    // Extract the m x m upper part of H
                    Matrix<T> H_m(m, m, A.device_id());
                    for (core::u32 i = 0; i < m; ++i) {
                        for (core::u32 j = 0; j < m; ++j) {
                            H_m(i, j) = H(i, j);
                        }
                    }

                    // Solve using QR decomposition of H_m
                    QRDecomposition<T> qr_h = qr_decomposition(H_m);

                    // Apply Q^T to the first m components of e1
                    Vector<T> e1_m(m, A.device_id());
                    for (core::u32 i = 0; i < m; ++i) {
                        e1_m[i] = e1[i];
                    }

                    Vector<T> y = qr_solve(qr_h, e1_m);

                    // Update solution: x = x0 + V_m * y
                    for (core::u32 j = 0; j < m; ++j) {
                        Vector<T> v_j = V.get_col(j);
                        axpy(y[j], v_j, result.solution);
                    }

                    // Check convergence
                    Vector<T> r_new = b - matvec(A, result.solution);
                    result.residual_norm = nrm2(r_new);

                    if (result.residual_norm < tolerance) {
                        result.converged = true;
                        result.iterations = outer_iter * restart + m;
                        return result;
                    }
                }

                result.iterations = max_iterations;
                return result;
            }

            // Gauss-Seidel method
            template<typename T>
            PSI_NODISCARD SolverResult<T> gauss_seidel(const Matrix<T>& A, const Vector<T>& b,
                const Vector<T>& x0,
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square(), "Gauss-Seidel requires square matrix");
                PSI_CHECK_DIMENSIONS("Gauss-Seidel A-b", A.rows(), b.size());
                PSI_CHECK_DIMENSIONS("Gauss-Seidel A-x0", A.rows(), x0.size());

                core::usize n = A.rows();
                SolverResult<T> result(n, A.device_id());
                result.solution = x0;

                for (core::u32 iter = 0; iter < max_iterations; ++iter) {
                    Vector<T> x_old = result.solution;

                    for (core::usize i = 0; i < n; ++i) {
                        if (std::abs(A(i, i)) < std::numeric_limits<T>::epsilon()) {
                            PSI_THROW_MATH("Gauss-Seidel: zero diagonal element");
                        }

                        T sum = b[i];
                        for (core::usize j = 0; j < n; ++j) {
                            if (j != i) {
                                sum -= A(i, j) * result.solution[j];
                            }
                        }

                        result.solution[i] = sum / A(i, i);
                    }

                    // Compute residual norm
                    Vector<T> residual = b - matvec(A, result.solution);
                    result.residual_norm = nrm2(residual);

                    // Check convergence
                    if (result.residual_norm < tolerance) {
                        result.converged = true;
                        result.iterations = iter + 1;
                        return result;
                    }
                }

                result.iterations = max_iterations;
                return result;
            }

            // Successive Over-Relaxation (SOR) method
            template<typename T>
            PSI_NODISCARD SolverResult<T> sor(const Matrix<T>& A, const Vector<T>& b,
                const Vector<T>& x0,
                T omega = T{ 1.2 },
                core::u32 max_iterations = 1000,
                T tolerance = T{ 1e-10 }) {
                PSI_ASSERT(A.is_square(), "SOR requires square matrix");
                PSI_CHECK_DIMENSIONS("SOR A-b", A.rows(), b.size());
                PSI_CHECK_DIMENSIONS("SOR A-x0", A.rows(), x0.size());
                PSI_ASSERT(omega > T{} && omega < T{ 2 }, "SOR relaxation parameter must be in (0, 2)");

                core::usize n = A.rows();
                SolverResult<T> result(n, A.device_id());
                result.solution = x0;

                for (core::u32 iter = 0; iter < max_iterations; ++iter) {
                    for (core::usize i = 0; i < n; ++i) {
                        if (std::abs(A(i, i)) < std::numeric_limits<T>::epsilon()) {
                            PSI_THROW_MATH("SOR: zero diagonal element");
                        }

                        T sum = b[i];
                        for (core::usize j = 0; j < n; ++j) {
                            if (j != i) {
                                sum -= A(i, j) * result.solution[j];
                            }
                        }

                        T x_new = sum / A(i, i);
                        result.solution[i] = (T{ 1 } - omega) * result.solution[i] + omega * x_new;
                    }

                    // Compute residual norm
                    Vector<T> residual = b - matvec(A, result.solution);
                    result.residual_norm = nrm2(residual);

                    // Check convergence
                    if (result.residual_norm < tolerance) {
                        result.converged = true;
                        result.iterations = iter + 1;
                        return result;
                    }
                }

                result.iterations = max_iterations;
                return result;
            }

            // Matrix inversion using LU decomposition
            template<typename T>
            PSI_NODISCARD Matrix<T> invert(const Matrix<T>& A) {
                PSI_ASSERT(A.is_square(), "Matrix inversion requires square matrix");

                core::usize n = A.rows();
                Matrix<T> inv(n, n, A.device_id());
                Matrix<T> I = Matrix<T>::identity(n, A.device_id());

                LUDecomposition<T> lu = lu_decomposition(A);

                if (lu.is_singular) {
                    PSI_THROW_MATH("Cannot invert singular matrix");
                }

                // Solve A * X = I column by column
                for (core::usize j = 0; j < n; ++j) {
                    Vector<T> e_j = I.get_col(j);
                    Vector<T> x_j = lu_solve(lu, e_j);
                    inv.set_col(j, x_j);
                }

                return inv;
            }

            // Matrix determinant using LU decomposition
            template<typename T>
            PSI_NODISCARD T determinant(const Matrix<T>& A) {
                PSI_ASSERT(A.is_square(), "Determinant requires square matrix");

                LUDecomposition<T> lu = lu_decomposition(A);

                if (lu.is_singular) {
                    return T{};
                }

                // det(A) = det(P) * det(L) * det(U) = det(P) * 1 * prod(U_ii)
                T det = T{ 1 };

                // Compute determinant of U (product of diagonal elements)
                for (core::usize i = 0; i < A.rows(); ++i) {
                    det *= lu.U(i, i);
                }

                // Account for permutation sign
                core::usize permutation_inversions = 0;
                for (core::usize i = 0; i < A.rows(); ++i) {
                    if (static_cast<core::usize>(lu.P[i]) != i) {
                        ++permutation_inversions;
                    }
                }

                if (permutation_inversions % 2 == 1) {
                    det = -det;
                }

                return det;
            }

        } // namespace linalg
    } // namespace math
} // namespace psi
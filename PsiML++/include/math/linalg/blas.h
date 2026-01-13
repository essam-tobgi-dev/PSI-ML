#pragma once

#include "../../core/types.h"
#include "../../core/config.h"
#include "../../core/exception.h"
#include "../vector.h"
#include "../matrix.h"
#include <cmath>

namespace psi {
    namespace math {
        namespace linalg {

            // ============================================================================
            // BLAS Level 1: Vector-Vector Operations
            // ============================================================================

            // Dot product: result = x^T * y
            template<typename T>
            T dot(const Vector<T>& x, const Vector<T>& y);

            // Euclidean norm: result = ||x||_2
            template<typename T>
            T norm(const Vector<T>& x);

            // Alias for norm (BLAS naming convention)
            template<typename T>
            inline T nrm2(const Vector<T>& x) { return norm(x); }

            // Sum of absolute values: result = ||x||_1
            template<typename T>
            T asum(const Vector<T>& x);

            // Index of maximum absolute value
            template<typename T>
            core::index_t iamax(const Vector<T>& x);

            // Scalar-vector multiplication: y = alpha * x
            template<typename T>
            Vector<T> scal(T alpha, const Vector<T>& x);

            // Vector addition: result = x + y
            template<typename T>
            Vector<T> add(const Vector<T>& x, const Vector<T>& y);

            // Vector subtraction: result = x - y
            template<typename T>
            Vector<T> sub(const Vector<T>& x, const Vector<T>& y);

            // AXPY: y = alpha * x + y (in-place)
            template<typename T>
            void axpy(T alpha, const Vector<T>& x, Vector<T>& y);

            // Copy vector: y = x
            template<typename T>
            void copy(const Vector<T>& x, Vector<T>& y);

            // Swap vectors: x <-> y
            template<typename T>
            void swap(Vector<T>& x, Vector<T>& y);

            // ============================================================================
            // BLAS Level 2: Matrix-Vector Operations
            // ============================================================================

            // General matrix-vector multiplication: y = alpha * A * x + beta * y
            // trans: false for A, true for A^T
            template<typename T>
            Vector<T> gemv(bool trans, T alpha, const Matrix<T>& A, const Vector<T>& x,
                          T beta, const Vector<T>& y);

            // General matrix-vector multiplication (simplified): result = A * x
            template<typename T>
            Vector<T> matvec(const Matrix<T>& A, const Vector<T>& x);

            // Matrix-vector multiplication with transpose: result = A^T * x
            template<typename T>
            Vector<T> matvec_trans(const Matrix<T>& A, const Vector<T>& x);

            // Rank-1 update: A = alpha * x * y^T + A (in-place)
            template<typename T>
            void ger(T alpha, const Vector<T>& x, const Vector<T>& y, Matrix<T>& A);

            // ============================================================================
            // BLAS Level 3: Matrix-Matrix Operations
            // ============================================================================

            // General matrix-matrix multiplication: C = alpha * A * B + beta * C
            // transA: false for A, true for A^T
            // transB: false for B, true for B^T
            template<typename T>
            Matrix<T> gemm(bool transA, bool transB, T alpha, const Matrix<T>& A,
                          const Matrix<T>& B, T beta, const Matrix<T>& C);

            // Matrix multiplication (simplified): result = A * B
            template<typename T>
            Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B);

            // Matrix addition: result = A + B
            template<typename T>
            Matrix<T> matadd(const Matrix<T>& A, const Matrix<T>& B);

            // Matrix subtraction: result = A - B
            template<typename T>
            Matrix<T> matsub(const Matrix<T>& A, const Matrix<T>& B);

            // Matrix transpose: result = A^T
            template<typename T>
            Matrix<T> transpose(const Matrix<T>& A);

            // Matrix scalar multiplication: result = alpha * A
            template<typename T>
            Matrix<T> matscal(T alpha, const Matrix<T>& A);

            // ============================================================================
            // Utility Functions
            // ============================================================================

            // Outer product: result = x * y^T (returns matrix)
            template<typename T>
            Matrix<T> outer(const Vector<T>& x, const Vector<T>& y);

            // Trace of a matrix: result = sum of diagonal elements
            template<typename T>
            T trace(const Matrix<T>& A);

            // Frobenius norm of a matrix: result = ||A||_F
            template<typename T>
            T frobenius_norm(const Matrix<T>& A);

        } // namespace linalg
    } // namespace math
} // namespace psi

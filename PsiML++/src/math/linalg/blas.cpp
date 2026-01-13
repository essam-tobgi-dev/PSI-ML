#include "../../../include/math/linalg/blas.h"
#include <algorithm>

namespace psi {
    namespace math {
        namespace linalg {

            // ============================================================================
            // BLAS Level 1: Vector-Vector Operations
            // ============================================================================

            template<typename T>
            T dot(const Vector<T>& x, const Vector<T>& y) {
                PSI_CHECK_DIMENSIONS("dot", x.size(), y.size());

                T result = T{};
                for (core::usize i = 0; i < x.size(); ++i) {
                    result += x[i] * y[i];
                }
                return result;
            }

            template<typename T>
            T norm(const Vector<T>& x) {
                T sum_sq = T{};
                for (core::usize i = 0; i < x.size(); ++i) {
                    sum_sq += x[i] * x[i];
                }
                return std::sqrt(sum_sq);
            }

            template<typename T>
            T asum(const Vector<T>& x) {
                T sum = T{};
                for (core::usize i = 0; i < x.size(); ++i) {
                    sum += std::abs(x[i]);
                }
                return sum;
            }

            template<typename T>
            core::index_t iamax(const Vector<T>& x) {
                PSI_ASSERT(x.size() > 0, "Vector must not be empty");

                core::index_t max_idx = 0;
                T max_val = std::abs(x[0]);

                for (core::usize i = 1; i < x.size(); ++i) {
                    T abs_val = std::abs(x[i]);
                    if (abs_val > max_val) {
                        max_val = abs_val;
                        max_idx = static_cast<core::index_t>(i);
                    }
                }
                return max_idx;
            }

            template<typename T>
            Vector<T> scal(T alpha, const Vector<T>& x) {
                Vector<T> result(x.size(), x.device_id());
                for (core::usize i = 0; i < x.size(); ++i) {
                    result[i] = alpha * x[i];
                }
                return result;
            }

            template<typename T>
            Vector<T> add(const Vector<T>& x, const Vector<T>& y) {
                PSI_CHECK_DIMENSIONS("add", x.size(), y.size());

                Vector<T> result(x.size(), x.device_id());
                for (core::usize i = 0; i < x.size(); ++i) {
                    result[i] = x[i] + y[i];
                }
                return result;
            }

            template<typename T>
            Vector<T> sub(const Vector<T>& x, const Vector<T>& y) {
                PSI_CHECK_DIMENSIONS("sub", x.size(), y.size());

                Vector<T> result(x.size(), x.device_id());
                for (core::usize i = 0; i < x.size(); ++i) {
                    result[i] = x[i] - y[i];
                }
                return result;
            }

            template<typename T>
            void axpy(T alpha, const Vector<T>& x, Vector<T>& y) {
                PSI_CHECK_DIMENSIONS("axpy", x.size(), y.size());

                for (core::usize i = 0; i < x.size(); ++i) {
                    y[i] += alpha * x[i];
                }
            }

            template<typename T>
            void copy(const Vector<T>& x, Vector<T>& y) {
                PSI_CHECK_DIMENSIONS("copy", x.size(), y.size());

                for (core::usize i = 0; i < x.size(); ++i) {
                    y[i] = x[i];
                }
            }

            template<typename T>
            void swap(Vector<T>& x, Vector<T>& y) {
                PSI_CHECK_DIMENSIONS("swap", x.size(), y.size());

                for (core::usize i = 0; i < x.size(); ++i) {
                    T temp = x[i];
                    x[i] = y[i];
                    y[i] = temp;
                }
            }

            // ============================================================================
            // BLAS Level 2: Matrix-Vector Operations
            // ============================================================================

            template<typename T>
            Vector<T> gemv(bool trans, T alpha, const Matrix<T>& A, const Vector<T>& x,
                          T beta, const Vector<T>& y) {
                if (!trans) {
                    // y = alpha * A * x + beta * y
                    PSI_CHECK_DIMENSIONS("gemv", A.cols(), x.size());
                    PSI_CHECK_DIMENSIONS("gemv", A.rows(), y.size());

                    Vector<T> result(A.rows(), A.device_id());

                    for (core::usize i = 0; i < A.rows(); ++i) {
                        T sum = T{};
                        for (core::usize j = 0; j < A.cols(); ++j) {
                            sum += A(i, j) * x[j];
                        }
                        result[i] = alpha * sum + beta * y[i];
                    }
                    return result;
                } else {
                    // y = alpha * A^T * x + beta * y
                    PSI_CHECK_DIMENSIONS("gemv", A.rows(), x.size());
                    PSI_CHECK_DIMENSIONS("gemv", A.cols(), y.size());

                    Vector<T> result(A.cols(), A.device_id());

                    for (core::usize j = 0; j < A.cols(); ++j) {
                        T sum = T{};
                        for (core::usize i = 0; i < A.rows(); ++i) {
                            sum += A(i, j) * x[i];
                        }
                        result[j] = alpha * sum + beta * y[j];
                    }
                    return result;
                }
            }

            template<typename T>
            Vector<T> matvec(const Matrix<T>& A, const Vector<T>& x) {
                PSI_CHECK_DIMENSIONS("matvec", A.cols(), x.size());

                Vector<T> result(A.rows(), A.device_id());

                for (core::usize i = 0; i < A.rows(); ++i) {
                    T sum = T{};
                    for (core::usize j = 0; j < A.cols(); ++j) {
                        sum += A(i, j) * x[j];
                    }
                    result[i] = sum;
                }
                return result;
            }

            template<typename T>
            Vector<T> matvec_trans(const Matrix<T>& A, const Vector<T>& x) {
                PSI_CHECK_DIMENSIONS("matvec_trans", A.rows(), x.size());

                Vector<T> result(A.cols(), A.device_id());

                for (core::usize j = 0; j < A.cols(); ++j) {
                    T sum = T{};
                    for (core::usize i = 0; i < A.rows(); ++i) {
                        sum += A(i, j) * x[i];
                    }
                    result[j] = sum;
                }
                return result;
            }

            template<typename T>
            void ger(T alpha, const Vector<T>& x, const Vector<T>& y, Matrix<T>& A) {
                PSI_CHECK_DIMENSIONS("ger", A.rows(), x.size());
                PSI_CHECK_DIMENSIONS("ger", A.cols(), y.size());

                for (core::usize i = 0; i < x.size(); ++i) {
                    for (core::usize j = 0; j < y.size(); ++j) {
                        A(i, j) += alpha * x[i] * y[j];
                    }
                }
            }

            // ============================================================================
            // BLAS Level 3: Matrix-Matrix Operations
            // ============================================================================

            template<typename T>
            Matrix<T> gemm(bool transA, bool transB, T alpha, const Matrix<T>& A,
                          const Matrix<T>& B, T beta, const Matrix<T>& C) {
                core::usize m = transA ? A.cols() : A.rows();
                core::usize n = transB ? B.rows() : B.cols();
                core::usize kA = transA ? A.rows() : A.cols();
                core::usize kB = transB ? B.cols() : B.rows();

                PSI_CHECK_DIMENSIONS("gemm", kA, kB);
                PSI_CHECK_DIMENSIONS("gemm C rows", m, C.rows());
                PSI_CHECK_DIMENSIONS("gemm C cols", n, C.cols());

                Matrix<T> result(m, n, A.device_id());

                for (core::usize i = 0; i < m; ++i) {
                    for (core::usize j = 0; j < n; ++j) {
                        T sum = T{};
                        for (core::usize k = 0; k < kA; ++k) {
                            T a_val = transA ? A(k, i) : A(i, k);
                            T b_val = transB ? B(j, k) : B(k, j);
                            sum += a_val * b_val;
                        }
                        result(i, j) = alpha * sum + beta * C(i, j);
                    }
                }
                return result;
            }

            template<typename T>
            Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B) {
                PSI_CHECK_DIMENSIONS("matmul", A.cols(), B.rows());

                Matrix<T> result(A.rows(), B.cols(), A.device_id());

                for (core::usize i = 0; i < A.rows(); ++i) {
                    for (core::usize j = 0; j < B.cols(); ++j) {
                        T sum = T{};
                        for (core::usize k = 0; k < A.cols(); ++k) {
                            sum += A(i, k) * B(k, j);
                        }
                        result(i, j) = sum;
                    }
                }
                return result;
            }

            template<typename T>
            Matrix<T> matadd(const Matrix<T>& A, const Matrix<T>& B) {
                PSI_CHECK_DIMENSIONS("matadd rows", A.rows(), B.rows());
                PSI_CHECK_DIMENSIONS("matadd cols", A.cols(), B.cols());

                Matrix<T> result(A.rows(), A.cols(), A.device_id());

                for (core::usize i = 0; i < A.rows(); ++i) {
                    for (core::usize j = 0; j < A.cols(); ++j) {
                        result(i, j) = A(i, j) + B(i, j);
                    }
                }
                return result;
            }

            template<typename T>
            Matrix<T> matsub(const Matrix<T>& A, const Matrix<T>& B) {
                PSI_CHECK_DIMENSIONS("matsub rows", A.rows(), B.rows());
                PSI_CHECK_DIMENSIONS("matsub cols", A.cols(), B.cols());

                Matrix<T> result(A.rows(), A.cols(), A.device_id());

                for (core::usize i = 0; i < A.rows(); ++i) {
                    for (core::usize j = 0; j < A.cols(); ++j) {
                        result(i, j) = A(i, j) - B(i, j);
                    }
                }
                return result;
            }

            template<typename T>
            Matrix<T> transpose(const Matrix<T>& A) {
                Matrix<T> result(A.cols(), A.rows(), A.device_id());

                for (core::usize i = 0; i < A.rows(); ++i) {
                    for (core::usize j = 0; j < A.cols(); ++j) {
                        result(j, i) = A(i, j);
                    }
                }
                return result;
            }

            template<typename T>
            Matrix<T> matscal(T alpha, const Matrix<T>& A) {
                Matrix<T> result(A.rows(), A.cols(), A.device_id());

                for (core::usize i = 0; i < A.rows(); ++i) {
                    for (core::usize j = 0; j < A.cols(); ++j) {
                        result(i, j) = alpha * A(i, j);
                    }
                }
                return result;
            }

            // ============================================================================
            // Utility Functions
            // ============================================================================

            template<typename T>
            Matrix<T> outer(const Vector<T>& x, const Vector<T>& y) {
                Matrix<T> result(x.size(), y.size(), x.device_id());

                for (core::usize i = 0; i < x.size(); ++i) {
                    for (core::usize j = 0; j < y.size(); ++j) {
                        result(i, j) = x[i] * y[j];
                    }
                }
                return result;
            }

            template<typename T>
            T trace(const Matrix<T>& A) {
                PSI_ASSERT(A.is_square(), "Matrix must be square for trace");

                T sum = T{};
                for (core::usize i = 0; i < A.rows(); ++i) {
                    sum += A(i, i);
                }
                return sum;
            }

            template<typename T>
            T frobenius_norm(const Matrix<T>& A) {
                T sum_sq = T{};
                for (core::usize i = 0; i < A.rows(); ++i) {
                    for (core::usize j = 0; j < A.cols(); ++j) {
                        T val = A(i, j);
                        sum_sq += val * val;
                    }
                }
                return std::sqrt(sum_sq);
            }

            // ============================================================================
            // Explicit template instantiations
            // ============================================================================

            // Level 1
            template float dot(const Vector<float>&, const Vector<float>&);
            template double dot(const Vector<double>&, const Vector<double>&);

            template float norm(const Vector<float>&);
            template double norm(const Vector<double>&);

            template float asum(const Vector<float>&);
            template double asum(const Vector<double>&);

            template core::index_t iamax(const Vector<float>&);
            template core::index_t iamax(const Vector<double>&);

            template Vector<float> scal(float, const Vector<float>&);
            template Vector<double> scal(double, const Vector<double>&);

            template Vector<float> add(const Vector<float>&, const Vector<float>&);
            template Vector<double> add(const Vector<double>&, const Vector<double>&);

            template Vector<float> sub(const Vector<float>&, const Vector<float>&);
            template Vector<double> sub(const Vector<double>&, const Vector<double>&);

            template void axpy(float, const Vector<float>&, Vector<float>&);
            template void axpy(double, const Vector<double>&, Vector<double>&);

            template void copy(const Vector<float>&, Vector<float>&);
            template void copy(const Vector<double>&, Vector<double>&);

            template void swap(Vector<float>&, Vector<float>&);
            template void swap(Vector<double>&, Vector<double>&);

            // Level 2
            template Vector<float> gemv(bool, float, const Matrix<float>&, const Vector<float>&, float, const Vector<float>&);
            template Vector<double> gemv(bool, double, const Matrix<double>&, const Vector<double>&, double, const Vector<double>&);

            template Vector<float> matvec(const Matrix<float>&, const Vector<float>&);
            template Vector<double> matvec(const Matrix<double>&, const Vector<double>&);

            template Vector<float> matvec_trans(const Matrix<float>&, const Vector<float>&);
            template Vector<double> matvec_trans(const Matrix<double>&, const Vector<double>&);

            template void ger(float, const Vector<float>&, const Vector<float>&, Matrix<float>&);
            template void ger(double, const Vector<double>&, const Vector<double>&, Matrix<double>&);

            // Level 3
            template Matrix<float> gemm(bool, bool, float, const Matrix<float>&, const Matrix<float>&, float, const Matrix<float>&);
            template Matrix<double> gemm(bool, bool, double, const Matrix<double>&, const Matrix<double>&, double, const Matrix<double>&);

            template Matrix<float> matmul(const Matrix<float>&, const Matrix<float>&);
            template Matrix<double> matmul(const Matrix<double>&, const Matrix<double>&);

            template Matrix<float> matadd(const Matrix<float>&, const Matrix<float>&);
            template Matrix<double> matadd(const Matrix<double>&, const Matrix<double>&);

            template Matrix<float> matsub(const Matrix<float>&, const Matrix<float>&);
            template Matrix<double> matsub(const Matrix<double>&, const Matrix<double>&);

            template Matrix<float> transpose(const Matrix<float>&);
            template Matrix<double> transpose(const Matrix<double>&);

            template Matrix<float> matscal(float, const Matrix<float>&);
            template Matrix<double> matscal(double, const Matrix<double>&);

            // Utilities
            template Matrix<float> outer(const Vector<float>&, const Vector<float>&);
            template Matrix<double> outer(const Vector<double>&, const Vector<double>&);

            template float trace(const Matrix<float>&);
            template double trace(const Matrix<double>&);

            template float frobenius_norm(const Matrix<float>&);
            template double frobenius_norm(const Matrix<double>&);

        } // namespace linalg
    } // namespace math
} // namespace psi

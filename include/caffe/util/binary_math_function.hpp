#ifndef CAFFE_UTIL_BINARY_MATH_FUNCTION_HPP_
#define CAFFE_UTIL_BINARY_MATH_FUNCTION_HPP_
#include <cstdint>
#include <vector>

namespace caffe {
using std::vector;
#ifndef BINARY_32_BIT
typedef uint64_t binary_t;
const int BINARY_SIZE = 8 * sizeof(binary_t);
#define bitcount __builtin_popcountll
#else
typedef uint32_t binary_t;
const int BINARY_SIZE = 8 * sizeof(binary_t);
#define bitcount __builtin_popcount
#endif
/**
 * \brief Computes a matrix-matrix product with general compressed binary
 *        matrices.
 *
 * \param M Specifies the number of rows of the matrix A and of the matrix C.
 *        The value of m must be at least zero.
 *
 * \param N Specifies the number of columns of the matrix B and the number of
 *        columns of the matrix C. The value of n must be at least zero.
 *
 * \param K Specifies the number of columns of the matrix A and the number of
 *        rows of the matrix B. The value of K must be at least zero.
 *
 * \param A compressed binary matrix, size M-by-K
 *
 * \param B compressed binary matrix, size K-by-N
 *
 * \param C result, is a M-by-N matrix, C = alpha * A x B + beta * C
 *
 */
template<typename Dtype>
void caffe_cpu_binary_gemm_and(const bool transposeA, const bool transposeB,
    const int M, const int N, const int K, const Dtype alpha, const binary_t* A,
    const binary_t *B, const Dtype* scaleA, const Dtype* scaleB,
    Dtype beta, Dtype* C);

/**
 * \brief Computes a matrix-matrix product. The input matrix \f$ A, B \in
 *        \{ -1, +1 \} ^ {m \times n} \f$ , and must be compressed as binary_t
 *        (uint32_t / uint64_4) array.
 */
template<typename Dtype>
void caffe_cpu_binary_gemm_xor(const bool transposeA, const bool transposeB,
    const int M, const int N, const int K, const binary_t* A,
    const binary_t *B, const Dtype* scaleA, const Dtype* scaleB, Dtype* C);

/**
 * \brief Binary and compress matrix to a binary_t matrix.
 *        First, \f$ Out = \mathrm{sign}(In) \f$ and get scale along given axis.
 *        Second, compress the sign matrix along given axis.
 *
 * \param axis if is true, the compress along the first dimension, otherwise
 *        along the second dimension.
 * \param M the row number of matrix
 * \param N the column number of matrix
 * \param In the input M-by-N matrix
 * \param Out the binarized and compressed matrix. If axis is true, its shape
 *        is \f$ \lceil \frac{M}{ \mathrm{BINARY_SIZE}} \rceil \f$-by-N; otherwise
 *        its shape is M-by-\f$ \lceil \frac{N}{\mathrm{BINARY_SIZE}} \rceil \f$ ;
 * \param scale the scale with shape N (axis is true) or M (axis is false).
 */
template<typename Dtype>
void caffe_cpu_binary(const int axis, const int M, const int N,
    const Dtype* In, vector<binary_t>& Out, vector<Dtype> &scale);

template<typename Dtype>
void caffe_cpu_binary_approx(const int axis, const int M, const int N,
    const Dtype* In, const vector<Dtype> &scale, vector<Dtype> &Out);

template<typename Dtype>
void caffe_cpu_binary_scale(const int axis, const int M, const int N,
    const Dtype* In, vector<Dtype> &scale);

template<typename Dtype>
void caffe_cpu_binary_gradient(const int axis, const int M, const int N,
    const Dtype* In, const vector<Dtype> &scale, Dtype *grad);

template<typename Dtype>
void caffe_cpu_ternary(const int axis, const int M, const int N, const Dtype* In,
    vector<binary_t> &pos, vector<binary_t> &neg, Dtype &delta,
    vector<Dtype> &scale) ;
}
#endif // CAFFE_UTIL_BINARY_MATH_FUNCTION_HPP_

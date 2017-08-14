#ifndef CAFFE_UTIL_BINARY_MATH_FUNCTION_HPP_
#define CAFFE_UTIL_BINARY_MATH_FUNCTION_HPP_
#include <cstdint>
#include <vector>

namespace caffe {
using std::vector;
using std::max;
#ifndef BINARY_32_BIT
typedef uint64_t Btype;
const int BINARY_SIZE = 8 * sizeof(Btype);
#define bitcount __builtin_popcountll
#else
typedef uint32_t Btype;
const int BINARY_SIZE = 8 * sizeof(Btype);
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
void caffe_cpu_binary_gemm_and(
  const bool transposeA, const bool transposeB,
  const int M, const int N, const int K, const Dtype alpha, const Btype *A,
  const Btype *B, const Dtype *scaleA, const Dtype *scaleB,
  Dtype beta, Dtype *C);

/**
 * \brief Binary and compress matrix to a Btype matrix.
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
void caffe_cpu_binary(const int N, Dtype *in, Btype *code);

template<typename Dtype>
void caffe_cpu_binary_approx(const int axis, const int M, const int N,
                             const Dtype *In, const Dtype *scale, Dtype *Out);

template<typename Dtype>
void caffe_cpu_binary_scale(const int axis, const int M, const int N,
                            const Dtype *In, Dtype *scale);

template<typename Dtype>
void caffe_cpu_binary_gradient(
  const int axis, const int M, const int N,
  const Dtype *In, const Dtype *scale, Dtype *grad);

template<typename Dtype>
void caffe_cpu_ternary(
  const int axis, const int M, const int N, const Dtype *in,
  Btype *code, Btype *mask, Dtype &delta, Dtype *scale, Dtype *sum2);

template<typename Dtype>
void caffe_cpu_binary_norm(
  const int axis, const int M, const int N, const Dtype *in,
  Btype *code, Dtype *scale, Dtype *bias, Dtype *sum, const bool use_bias);

template<typename Dtype>
void caffe_cpu_ternary_norm(
  const int axis, const int M, const int N, const Dtype *in,
  Btype *code, Btype *mask, Dtype *delta, Dtype *scale,
  Dtype *bias, Dtype *sum,  Dtype *sum2, const bool use_bias);

template<typename Dtype>
void caffe_cpu_binary_norm_gradient(
  const int axis, const int M, const int N, const Dtype *in,
  const Dtype *scale, const Dtype *bias, Dtype *grad);

template<typename Dtype>
void caffe_cpu_ternary_norm_gradient(
  const int axis, const int M, const int N, const Dtype *in,
  const Dtype *delta, const Dtype *scale, const Dtype *bias, Dtype *grad);

template<typename Dtype>
void caffe_cpu_binary_gemm(
  const bool transA, const bool transB,
  const int M, const int N, const int K,
  const Btype *A, const Dtype *A_scale,
  const Btype *B, const Dtype *B_scale,
  Dtype *C,
  const bool bias,
  const Dtype *A_bias, const Dtype *A_sum,
  const Dtype *B_bias, const Dtype *B_sum);

template<typename Dtype>
void caffe_cpu_tb_gemm(
  const bool transA, const bool transB,
  const int M, const int N, const int K,
  const Btype *A,      const Btype *A_mask, const Dtype *A_scale,
  const Dtype *A_sum2, const Btype *B,      const Dtype *B_scale,
  Dtype *C,
  const bool  bias,
  const Dtype *A_bias, const Dtype *A_sum,
  const Dtype *B_bias, const Dtype *B_sum);

template<typename Dtype>
void caffe_cpu_bt_gemm(
  const bool transA, const bool transB,
  const int M, const int N, const int K,
  const Btype *A,       const Dtype *A_scale,
  const Btype *B,       const Btype *B_mask,
  const Dtype *B_scale, const Dtype *B_sum2,
  Dtype *C,
  const bool bias,
  const Dtype *A_bias, const Dtype *A_sum,
  const Dtype *B_bias, const Dtype *B_sum);

template<typename Dtype>
void caffe_cpu_binary_restore(
  const int axis, const int M, const int N,
  const Btype *code, const Dtype *scale,
  const Dtype *bias, Dtype *out);

template<typename Dtype>
void caffe_cpu_ternary_restore(
  const int axis, const int M, const int N,
  const Btype *code, const Btype *mask,
  const Dtype *scale, const Dtype *bias, Dtype *out);
}
#endif // CAFFE_UTIL_BINARY_MATH_FUNCTION_HPP_

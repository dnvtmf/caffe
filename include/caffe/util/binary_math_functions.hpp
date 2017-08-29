#ifndef CAFFE_UTIL_BINARY_MATH_FUNCTIONS_HPP_
#define CAFFE_UTIL_BINARY_MATH_FUNCTIONS_HPP_
#include <cstdint>
#include <vector>

namespace caffe {
using std::vector;
using std::max;
using std::min;
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

template<typename Dtype>
void caffe_cpu_binary(const int axis, const int M, const int N,
                      const Dtype *in, Btype *code, Dtype *scale);

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
  const Dtype *bias, const bool use_bias, Dtype *out);

template<typename Dtype>
void caffe_cpu_ternary_restore(
  const int axis, const int M, const int N,
  const Btype *code, const Btype *mask,
  const Dtype *scale, const Dtype *bias, const bool use_bias, Dtype *out);

template<typename Dtype>
void caffe_cpu_clip(const int N, Dtype min_value, Dtype max_value, Dtype *X);

}
#endif // CAFFE_UTIL_BINARY_MATH_FUNCTIONS_HPP_

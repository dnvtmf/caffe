#ifndef CAFFE_UTIL_BINARY_MATH_FUNCTION_HPP_
#define CAFFE_UTIL_BINARY_MATH_FUNCTION_HPP_
#include <cstdint>

namespace caffe {
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
 * \param A compressed binary matrix, size M-by-(BINARY_SIZE * K)
 *
 * \param B compressed binary matrix, size (BINARY_SIZE * K)-by-N
 *
 * \param C result, is a M-by-N matrix
 *
 */

template<typename Dtype>
void caffe_cpu_bgemm(const int M, const int N, const int K, const binary_t* A,
                     const binary_t *B, Dtype* C);
}
#endif // CAFFE_UTIL_BINARY_MATH_FUNCTION_HPP_

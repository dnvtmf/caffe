#include "caffe/util/bb_gemm.hpp"

// C[BLOCK_M][BLOCK_N] = A[BLOCK_M][BLOCK_N] x B[BLOCK_K][BLOCK_N]
// C[i][j] <= BLOCK_K * BINARY_SIZE < 2^(sizeof(result_type))
const int BLOCK_M = 16;
const int BLOCK_N = 64;
const int BLOCK_K = 16;
typedef unsigned short result_type;

template <typename Dtype>
void block_gemm(const int index, const int M, const int N, const int K,
    const int BK, const Btype *A, const Dtype *A_scale, const Btype *B,
    const Dtype *B_scale, Dtype *C) {
  for (int j = 0; j < BLOCK_N; ++j) {
    const int dim_n   = N / BLOCK_N;
    const int m_index = index / dim_n * BLOCK_M;
    const int n_index = index % dim_n * BLOCK_N + j;
    static Btype A_sub[BLOCK_M][BLOCK_K];
    static Btype B_col[BLOCK_K];
    static result_type C_col[BLOCK_M];
    for (int i = 0; i < BLOCK_M; ++i) C_col[i] = 0;

    // compute postion of A, B, C
    const Btype *A_ptr = A + m_index * BK;
    const Btype *B_ptr = B + n_index;
    Dtype *C_ptr       = C + m_index * N + n_index;
    const Btype *B_end = B + BK * N;

    while (B_ptr < B_end) {
      for (int i = 0; i < BLOCK_M; ++i) {
        for (int k = 0; k < BLOCK_K; ++k) {
          A_sub[i][k] = A_ptr[i * BK + k];
        }
      }
      A_ptr += BLOCK_K;

      for (int k = 0; k < BLOCK_K; ++k) B_col[k] = B_ptr[k * N];
      B_ptr += BLOCK_K * N;

      for (int i = 0; i < BLOCK_M; ++i) {
        for (int k = 0; k < BLOCK_K; ++k) {
          // C_col[i] += A_sub[i][k] * B_col[k];
          C_col[i] += bitcount(A_sub[i][k] ^ B_col[k]);
        }
      }
    }

    for (int i = 0; i < BLOCK_M; ++i) {
      C_ptr[i * N] =
          (K - 2 * C_col[i]) * A_scale[m_index + i] * B_scale[n_index];
    }
  }
}
template <typename Dtype>
void bb_gemm_cpu(const int M, const int N, const int K, const Btype *A,
    const Dtype *A_scale, const Btype *B, const Dtype *B_scale, Dtype *C) {
  const int BK = (K - 1) / BINARY_SIZE + 1;
  assert(M % BLOCK_M == 0);
  assert(N % BLOCK_N == 0);
  assert(BK % BLOCK_K == 0);
  const int num_kernel = (M / BLOCK_M) * (N / BLOCK_N);
  for (int i = 0; i < num_kernel; ++i)
    block_gemm<Dtype>(i, M, N, K, BK, A, A_scale, B, B_scale, C);
}

template void bb_gemm_cpu<float>(const int M, const int N, const int K,
    const Btype *A, const float *A_scale, const Btype *B, const float *B_scale,
    float *C);
template void bb_gemm_cpu<double>(const int M, const int N, const int K,
    const Btype *A, const double *A_scale, const Btype *B,
    const double *B_scale, double *C);

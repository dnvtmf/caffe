#include "caffe/util/bb_gemm.hpp"

// C[BLOCK_M][BLOCK_N] = A[BLOCK_M][BLOCK_N] x B[BLOCK_K][BLOCK_N]
// C[i][j] <= BLOCK_K * BINARY_SIZE < 2^(sizeof(result_type))
#define BLOCK_M 16
#define BLOCK_N 64
#define BLOCK_K 16
typedef unsigned short result_type;

// assume M = m * BLOCK_M; N = n * BLOCK_N; K = BLOCK_K
// each block compute a sub-matrex of C (BLOCK_M x BLOCK_N)
template <typename Dtype>
__global__ void block_gemm(
    const int M, const int N, const int K, const Btype *A, const Dtype *A_scale,
    const Btype *B, const Dtype *B_scale, Dtype *C) {
  const int j       = threadIdx.x;
  const int dim_n   = N / BLOCK_N;
  const int m_index = blockIdx.x / dim_n * BLOCK_M;
  const int n_index = blockIdx.x % dim_n * BLOCK_N + j;

  __shared__ Btype A_sub[BLOCK_M][BLOCK_K];
  Btype B_col[BLOCK_K];        // register array
  result_type C_col[BLOCK_M];  // register array
  for (int i = 0; i < BLOCK_M; ++i) C_col[i] = 0;

  A += m_index * K;
  B += n_index;
  C += m_index * N + n_index;

  const Btype *B_end = B + K * N;
  while (B < B_end) {
    // copy a sub-matrix of A to A_sub
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += blockDim.x) {
      const int i = idx / BLOCK_K;
      const int k = idx % BLOCK_K;
      A_sub[i][k] = A[i * K + k];
    }
    A += BLOCK_K;  // update the pointer A
    __syncthreads();

// copy a col of B to B_col
#pragma unroll
    for (int k = 0; k < BLOCK_K; ++k) B_col[k] = B[k * N];
    B += BLOCK_K * N;

    for (int k = 0; k < BLOCK_K; ++k) {
#pragma unroll
      for (int i = 0; i < BLOCK_M; ++i) {
        // C_col[i] += A_sub[i][k] * B_col[k];
        C_col[i] += cuda_bitcount(A_sub[i][k] ^ B_col[k]);
      }
    }
  }
#pragma unrool
  for (int i = 0; i < BLOCK_K; ++i)
    C[i * N] = (K - 2 * C_col[i]) * A_scale[i] * B_scale[j];
}

template <typename Dtype>
void bb_gemm_gpu(
    const int M, const int N, const int K, const Btype *A, const Dtype *A_scale,
    const Btype *B, const Dtype *B_scale, Dtype *C) {
  CHECK(M % BLOCK_M == 0) << "M % BLOCK_M = " << M << " % " << BLOCK_M << " != 0";
  CHECK(N % BLOCK_N == 0) << "N % BLOCK_N = " << N << " % " << BLOCK_N << " != 0";
  CHECK(K % BLOCK_K == 0) << "K % BLOCK_K = " << K << " % " << BLOCK_K << " != 0";
  const int num_kernel = M / BLOCK_M * N / BLOCK_N;
  block_gemm<Dtype>
      <<<num_kernel, BLOCK_N>>>(M, N, K, A, A_scale, B, B_scale, C);
}

template void bb_gemm_gpu<float>(
    const int M, const int N, const int K, const Btype *A, const float *A_scale,
    const Btype *B, const float *B_scale, float *C);
template void bb_gemm_gpu<double>(
    const int M, const int N, const int K, const Btype *A,
    const double *A_scale, const Btype *B, const double *B_scale, double *C);

#include "caffe/util/bb_gemm.hpp"

/*
based on "GeForce GTX 1080 Ti"
Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536),
3D=(16384, 16384, 16384)
Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
Total amount of constant memory:               65536 bytes
Total amount of shared memory per block:       49152 bytes
Total number of registers available per block: 65536
Warp size:                                     32
Maximum number of threads per multiprocessor:  2048
Maximum number of threads per block:           1024
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
Maximum memory pitch:                          2147483647 bytes
Texture alignment:                             512 bytes
Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
*/
// C[BLOCK_M][BLOCK_N] = A[BLOCK_M][K] x B[K][BLOCK_N]
// C[i][j] <= BLOCK_K * BINARY_SIZE < 2^(sizeof(result_type))
#define BLOCK_M 16
#define BLOCK_N 64
#define BLOCK_K 16
typedef unsigned short result_type;

// assume M = m * BLOCK_M; N = n * BLOCK_N; K = BLOCK_K
// each block compute a sub-matrex of C (BLOCK_M x BLOCK_N)
template <typename Dtype>
__global__ void block_gemm(const int M, const int N, const int K, const int BK,
    const Btype *A, const Dtype *A_scale, const Btype *B, const Dtype *B_scale,
    Dtype *C) {
  const int j       = threadIdx.x;
  const int dim_n   = N / BLOCK_N;
  const int m_index = blockIdx.x / dim_n * BLOCK_M;
  const int n_index = blockIdx.x % dim_n * BLOCK_N + j;

  __shared__ Btype A_sub[BLOCK_M][BLOCK_K];
  Btype B_col[BLOCK_K];        // register array
  result_type C_col[BLOCK_M];  // register array
  for (int i = 0; i < BLOCK_M; ++i) C_col[i] = 0;

  A += m_index * BK;
  B += n_index;
  C += m_index * N + n_index;

  const Btype *B_end = B + BK * N;
  while (B < B_end) {
    // copy a sub-matrix of A to A_sub
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += blockDim.x) {
      const int i = idx / BLOCK_K;
      const int k = idx % BLOCK_K;
      A_sub[i][k] = A[i * BK + k];
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
#pragma unroll
  for (int i = 0; i < BLOCK_K; ++i)
    C[i * N] = (K - 2 * C_col[i]) * A_scale[m_index + i] * B_scale[n_index];
}

template <typename Dtype>
void bb_gemm_gpu(const int M, const int N, const int K, const Btype *A,
    const Dtype *A_scale, const Btype *B, const Dtype *B_scale, Dtype *C) {
  const int BK = (K - 1) / BINARY_SIZE + 1;
  assert(M % BLOCK_M == 0);
  assert(N % BLOCK_N == 0);
  assert(BK % BLOCK_K == 0);
  const int num_kernel = (M / BLOCK_M) * (N / BLOCK_N);
  block_gemm<Dtype>
      <<<num_kernel, BLOCK_N>>>(M, N, K, BK, A, A_scale, B, B_scale, C);
}

template void bb_gemm_gpu<float>(const int M, const int N, const int K,
    const Btype *A, const float *A_scale, const Btype *B, const float *B_scale,
    float *C);
template void bb_gemm_gpu<double>(const int M, const int N, const int K,
    const Btype *A, const double *A_scale, const Btype *B,
    const double *B_scale, double *C);

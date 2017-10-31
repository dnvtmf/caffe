#ifndef BB_GEMM_HPP_
#define BB_GEMM_HPP_

#include <bits/stdc++.h>
#include "caffe/common.hpp"

#ifndef BINARY_32_BIT
typedef uint64_t Btype;
const int BINARY_SIZE = 8 * sizeof(Btype);
#define bitcount __builtin_popcountll
#define cuda_bitcount __popc
#else
typedef uint32_t Btype;
const int BINARY_SIZE = 8 * sizeof(Btype);
#define bitcount __builtin_popcount
#define cuda_bitcount __popcll
#endif

template <typename Dtype>
void bb_gemm_cpu(
    const int M, const int N, const int K, const Btype *A, const Dtype *A_scale,
    const Btype *B, const Dtype *B_scale, Dtype *C);

#ifndef CPU_ONLY
template <typename Dtype>
void bb_gemm_gpu(
    const int M, const int N, const int K, const Btype *A, const Dtype *A_scale,
    const Btype *B, const Dtype *B_scale, Dtype *C);
#endif  // CPU_ONLY

#endif  // BB_GEMM_HPP_

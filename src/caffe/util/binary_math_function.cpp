#include <algorithm>

#include "caffe/util/binary_math_function.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template<typename Dtype>
void caffe_cpu_binary_gemm(const int M, const int N, const int K, const binary_t* A,
    const binary_t* B, const Dtype* scaleA, const Dtype* scaleB, Dtype *C) {
  caffe_set(M * N, 0, C);
  binary_t *pA = A, *pB;
  Dtype *pC;
  for(int i = 0; i < M; ++i) {
    pB = B;
    for(int k = 0; k < K; ++k) {
      pC = C + i * N;
      binary_t vA = *(pA++);
      for(int j = 0; j < N; ++j) {
        *(pC++) += bitcount(vA & *(pB++));
      }
    }
  }
  pC = C;
  for(int i = 0; i < M; ++i) {
    for(int j = 0; j < N; ++j) {
      *pC++ *= scaleA[i] * scaleB[j];
    }
  }
}
template<typename Dtype>
void caffe_cpu_binary_comprees_row(const int M, const int N, const Dtype* In,
    binary_t* Out, Dtype* scale) {
  const int cM = (M + BINARY_SIZE - 1) / BINARY_SIZE;
  for(int i = 0; i < cM * N; ++i) {
    Out[i] = 0;
  }
  caffe_set<Dtype>(N, 0, scale);
  auto p = In;
  binary_t *q = Out;
  for(int i = 0; i < M;) {
    for(binary_t k = 0; i < M && k < BINARY_SIZE; ++k, ++i) {
      for(int j = 0; j < N; ++j) {
        scale[j] += std::abs(*p);
        *q++ |= (static_cast<binary_t>(*p++ >= Dtype(0))) << k;
      }
      q -= N;
    }
    q += N;
  }
  caffe_scal<Dtype>(N, 1. / M, scale);
}
template<typename Dtype>
void caffe_cpu_binary_comprees_col(const int M, const int N, const Dtype* In,
    binary_t* Out, Dtype* scale) {
  const int cN = (N + BINARY_SIZE - 1) / BINARY_SIZE;
  for(int i = 0; i < M * cN; ++i) {
    Out[i] = 0;
  }
  caffe_set<Dtype>(M, 0, scale);
  auto p = In;
  for(int i = 0; i < M; ++i) {
    binary_t *q = Out + i * cN;
    for(int j = 0; j < N; ++q) {
      for(binary_t k = 0; j < N && k < BINARY_SIZE; ++k, ++j) {
          scale[i] += std::abs(*p);
        *q |= (static_cast<binary_t>(*p++ >= Dtype(0))) << k;
      }
    }
  }
  caffe_scal<Dtype>(M, 1. / N, scale);
}

template<>
void caffe_cpu_binary_gemm <float> (const int M, const int N, const int K, const binary_t* A,
    const binary_t* B, const float* scaleA, const float* scaleB, float *C); 

template<>
void caffe_cpu_binary_gemm <double> (const int M, const int N, const int K, const binary_t* A,
    const binary_t* B, const double* scaleA, const double* scaleB, double *C); 

template<>
void caffe_cpu_binary_comprees_row<float>(const int M, const int N, const float* In,
    binary_t* Out, float* scale);
template<>
void caffe_cpu_binary_comprees_row<double>(const int M, const int N, const double* In,
    binary_t* Out, double* scale);

template<>
void caffe_cpu_binary_comprees_col<float>(const int M, const int N, const float* In,
    binary_t* Out, float* scale);
template<>
void caffe_cpu_binary_comprees_col<double>(const int M, const int N, const double* In,
    binary_t* Out, double *scale);
}

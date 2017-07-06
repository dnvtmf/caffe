#include <algorithm>

#include "caffe/util/binary_math_function.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template<typename Dtype>
void caffe_cpu_binary_gemm_and(const bool transposeA, const bool transposeB,
    const int M, const int N, const int K, const binary_t* A,
    const binary_t* B, const Dtype* scaleA, const Dtype* scaleB, Dtype *C) {
  const int sz = M * N;
  const int KK = (K + BINARY_SIZE - 1) / BINARY_SIZE;
  caffe_set(sz, Dtype(0), C);
  const binary_t *pA, *pB;
  Dtype *pC;
  // through adjust the order of i, j, k to implement matrix multiplication.
  if(!transposeA && !transposeB) {
      pA = A;
      for(int i = 0; i < M; ++i) {
      pB = B;
      for(int k = 0; k < KK; ++k) {
        pC = C + i * N;
        const binary_t vA = *pA++;
        for(int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA & (*pB++));
          // C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  } else if(transposeA && !transposeB) {
    pA = A;
    for(int k = 0; k < KK; ++k) {
      pC = C;
      for(int i = 0; i < M; ++i) {
        pB = B + k * N;
        const binary_t vA = *pA++;
        for(int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA & (*pB++));
          //C[i][j] += A[k][i] * B[k][j];
        }
      }
    }
  } else if(!transposeA && transposeB) {
      pC = C;
      for(int i = 0; i < M; ++i) {
        pB = B;
        for(int j = 0; j < N; ++j) {
          pA = A + i * KK;
          auto &result = *(pC++);
          for(int k = 0; k < KK; ++k) {
            result += bitcount((*pA++) & (*pB++));
            // C[i][j] += A[i][k] * B[j][k];
        }
      }
    }
  } else {
    pA = A;
    for(int k = 0; k < KK; ++k) {
      pC = C;
      for(int i = 0; i < M; ++i) {
        auto vA = *pA++;
        for(int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA & *(B + j * KK + k));
          // C[i][j] += A[k][i] * B[j][k];
        }
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
void caffe_cpu_binary_gemm_xor(const bool transposeA, const bool transposeB,
    const int M, const int N, const int K, const binary_t* A,
    const binary_t* B, const Dtype* scaleA, const Dtype* scaleB, Dtype *C) {
  const int sz = M * N;
  const int KK = (K + BINARY_SIZE - 1) / BINARY_SIZE;
  caffe_set(sz, Dtype(0), C);
  const binary_t *pA, *pB;
  Dtype *pC;
  // through adjust the order of i, j, k to implement matrix multiplication.
  if(!transposeA && !transposeB) {
      pA = A;
      for(int i = 0; i < M; ++i) {
      pB = B;
      for(int k = 0; k < KK; ++k) {
        pC = C + i * N;
        const binary_t vA = *pA++;
        for(int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA ^ (*pB++));
          // C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  } else if(transposeA && !transposeB) {
    pA = A;
    for(int k = 0; k < KK; ++k) {
      pC = C;
      for(int i = 0; i < M; ++i) {
        pB = B + k * N;
        const binary_t vA = *pA++;
        for(int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA ^ (*pB++));
          //C[i][j] += A[k][i] * B[k][j];
        }
      }
    }
  } else if(!transposeA && transposeB) {
      pC = C;
      for(int i = 0; i < M; ++i) {
        pB = B;
        for(int j = 0; j < N; ++j) {
          pA = A + i * KK;
          auto &result = *(pC++);
          for(int k = 0; k < KK; ++k) {
            result += bitcount((*pA++) ^ (*pB++));
            // C[i][j] += A[i][k] * B[j][k];
        }
      }
    }
  } else {
    pA = A;
    for(int k = 0; k < KK; ++k) {
      pC = C;
      for(int i = 0; i < M; ++i) {
        auto vA = *pA++;
        for(int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA ^ *(B + j * KK + k));
          // C[i][j] += A[k][i] * B[j][k];
        }
      }
    }
  }

  pC = C;
  for(int i = 0; i < M; ++i) {
    for(int j = 0; j < N; ++j) {
      *pC = (K - (Dtype)2 * *pC) * scaleA[i] * scaleB[j];
      ++pC;
    }
  }
}

template<typename Dtype>
void caffe_cpu_binary_compress(const bool axis, const int M, const int N,
  const Dtype* In, binary_t* Out, Dtype* scale) {
  if(axis) {
    const int cM = (M + BINARY_SIZE - 1) / BINARY_SIZE;
    for(int i = 0; i < cM * N; ++i) {
      Out[i] = 0;
    }
    const int sz = N;
    caffe_set<Dtype>(sz, Dtype(0), scale);
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
    caffe_scal<Dtype>(sz, Dtype(1. / M), scale);
  } else {
    const int cN = (N + BINARY_SIZE - 1) / BINARY_SIZE;
    for(int i = 0; i < M * cN; ++i) {
      Out[i] = 0;
    }
    const int sz = M;
    caffe_set<Dtype>(sz, Dtype(0), scale);
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
    caffe_scal<Dtype>(sz, Dtype(1. / N), scale);
  }
}

template void caffe_cpu_binary_gemm_and <float> (const bool transposeA,
    const bool transposeB,const int M, const int N, const int K,
    const binary_t* A, const binary_t* B, const float* scaleA,
    const float* scaleB, float *C);
template void caffe_cpu_binary_gemm_and <double> (const bool transposeA,
    const bool transposeB, const int M, const int N, const int K,
    const binary_t* A, const binary_t* B, const double* scaleA,
    const double* scaleB, double *C);

template void caffe_cpu_binary_gemm_xor <float> (const bool transposeA,
    const bool transposeB,const int M, const int N, const int K,
    const binary_t* A, const binary_t* B, const float* scaleA,
    const float* scaleB, float *C);
template void caffe_cpu_binary_gemm_xor <double> (const bool transposeA,
    const bool transposeB, const int M, const int N, const int K,
    const binary_t* A, const binary_t* B, const double* scaleA,
    const double* scaleB, double *C);

template void caffe_cpu_binary_compress<float>(const bool axis, const int M,
    const int N, const float* In, binary_t* Out, float* scale);
template void caffe_cpu_binary_compress<double>(const bool axis, const int M,
    const int N, const double* In, binary_t* Out, double* scale);
}

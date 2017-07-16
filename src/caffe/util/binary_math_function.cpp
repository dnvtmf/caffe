#include <algorithm>
#include <iostream>

#include "caffe/util/binary_math_function.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template<typename Dtype>
void caffe_cpu_binary_gemm_and(const bool transposeA, const bool transposeB,
    const int M, const int N, const int K, const Dtype alpha, const binary_t* A,
    const binary_t *B, const Dtype* scaleA, const Dtype* scaleB,
    Dtype beta, Dtype* C) {
  const int KK = (K + BINARY_SIZE - 1) / BINARY_SIZE;
  const binary_t *pA, *pB;
  Dtype *pC;
  if (caffe_sign(alpha) != 0)
    beta /= alpha;
  pC = C;
  for (int i = 0; i < M * N; ++i)
    *pC++ *= beta;
  if (caffe_sign(alpha) == 0)
    return ;
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
      *pC++ *= alpha * scaleA[i] * scaleB[j];
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
void caffe_cpu_binary(const int axis, const int M, const int N,
  const Dtype* In, vector<binary_t>& Out, vector<Dtype> &scale) {
  if(axis == 1) {
    const int cM = (M + BINARY_SIZE - 1) / BINARY_SIZE;
    Out.resize(cM * N);
    scale.resize(N);
    fill(Out.begin(), Out.end(), binary_t(0));
    fill(scale.begin(), scale.end(), Dtype(0));
    auto p = In;
    auto q = Out.begin();
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
    for (int j = 0; j < N; ++j) {
      scale[j] /= M;
    }
  } else if(axis == 0) {
    const int cN = (N + BINARY_SIZE - 1) / BINARY_SIZE;
    Out.resize(M * cN);
    scale.resize(M);
    fill(Out.begin(), Out.end(), binary_t(0));
    fill(scale.begin(), scale.end(), Dtype(0));
    auto p = In;
    for(int i = 0; i < M; ++i) {
      auto q = Out.begin() + i * cN;
      for(int j = 0; j < N; ++q) {
        for(binary_t k = 0; j < N && k < BINARY_SIZE; ++k, ++j) {
            scale[i] += std::abs(*p);
          *q |= (static_cast<binary_t>(*p++ >= Dtype(0))) << k;
        }
      }
    }
    for (int i = 0; i < M; ++i) {
      scale[i] /= N;
    }
  } else
    CHECK(false) << "Error axis!";
}

template<typename Dtype>
void caffe_cpu_binary_approx(const int axis, const int M, const int N,
    const Dtype* In, const vector<Dtype> &scale, vector<Dtype> &Out) {
  auto p = Out.begin();
  const Dtype* q = In;
//  CHECK((int)Out.size() == M * N) << "Error appros out size!";
  if (axis == 0) {
//    CHECK((int)scale.size() == M) << "Error approx 0!";
    for(int i = 0; i < M; ++i) {
      for(int j = 0; j < N; ++j) {
        *p++ = *q++ >= Dtype(0) ? scale[i] : -scale[i];
      }
    }
  } else if (axis == 1) {
//    CHECK((int)scale.size() == N) << "Error approx 1!";
    for(int i = 0; i < M; ++i) {
      for(int j = 0; j < N; ++j) {
        *p++ = *q++ >= Dtype(0) ? scale[j] : -scale[j];
      }
    }
  } else
    CHECK(false) << "Error axis!";
}

template<typename Dtype>
void caffe_cpu_binary_scale(const int axis, const int M, const int N,
    const Dtype* In, vector<Dtype> &scale) {
  const Dtype* q = In;
  if (axis == 0) {
//    CHECK(scale.size() == M) << "scale size ERROR! 0";
    for(int i = 0; i < M; ++i) {
      scale[i] = 0;
      for(int j = 0; j < N; ++j) {
        scale[i] += *q++;
      }
      scale[i] /= (double) N;
    }
  } else if (axis == 1) {
//    CHECK_EQ(scale.size(), N) << "scale size ERROR";
    for (int i = 0; i < N; ++i) {
      scale[i] = 0;
    }
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        scale[j] += *q++;
      }
    }
    for(int i = 0; i < N; ++i) {
      scale[i] /= (double) M;
    }
  } else
    CHECK(false) << "Error axis!";
}

template<typename Dtype>
void caffe_cpu_binary_gradient(const int axis, const int M, const int N,
    const Dtype* In, const vector<Dtype> &scale, Dtype *grad) {
  auto p = In;
  auto q = grad;
  if(axis == 0) {
//    CHECK_EQ(scale.size(), M) << "gradient scale size Error";
    double co = 1. / N;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *q++ *= co + Dtype(std::abs(*p++) <= Dtype(1) ? 1 : 0) * scale[i];
      }
    }
  } else if (axis == 1) {
//    CHECK_EQ(scale.size(), N) << "gradient scale size Error";
    double co = 1. / M;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *q++ *= co + Dtype(std::abs(*p++) <= Dtype(1) ? 1 : 0) * scale[j];
      }
    }
  } else {
    CHECK(false) << "Error axis!";
  }
}


template<typename Dtype>
void caffe_cpu_ternary(const int axis, const int M, const int N, const Dtype* In,
    vector<binary_t> &pos, vector<binary_t> &neg, Dtype &delta,
    vector<Dtype> &scale) {
  if (axis == 0) {
    const int BN = (N + BINARY_SIZE - 1) / BINARY_SIZE;
    pos.resize(M * BN);
    neg.resize(M * BN);
    scale.resize(M);
    fill(pos.begin(), pos.end(), binary_t(0));
    fill(neg.begin(), neg.end(), binary_t(0));
    fill(scale.begin(), scale.end(), Dtype(0));
    delta = 0.7 * caffe_cpu_asum<Dtype>(M * N, In) / (1. * M * N);
    auto p = In;
    auto it1 = pos.begin(), it2 = neg.begin();
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++it1, ++it2) {
        for (int k = 0; k < BINARY_SIZE && j < N; ++j, ++k) {
          if (*p > delta) {
            *it1 |= binary_t(1) << k;
            scale[i] += *p;
          } else if (*p < -delta) {
            *it2 |= binary_t(1) << k;
            scale[i] -= *p;
          }
          ++p;
        }
      }
    }
    for (int i = 0; i < M; ++i) {
      scale[i] /= N;
    }
  } else {
    const int BM = (M + BINARY_SIZE - 1) / BINARY_SIZE;
    pos.resize(BM * N);
    neg.resize(BM * N);
    scale.resize(N);
    fill(pos.begin(), pos.end(), binary_t(0));
    fill(neg.begin(), neg.end(), binary_t(0));
    fill(scale.begin(), scale.end(), Dtype(0));
    delta = 0.7 * caffe_cpu_asum<Dtype>(M * N, In) / (1. * M * N);
    auto p = In;
    auto it1 = pos.begin(), it2 = neg.begin();
    for (int i = 0; i < M;) {
      for (int k = 0; k < BINARY_SIZE && i < M; ++i, ++k) {
        for (int j = 0; j < N; ++j, ++p) {
          if (*p > delta) {
            *it1 |= binary_t(1) << k;
            scale[j] += *p;
          } else if (*p < -delta) {
            *it2 |= binary_t(1) << k;
            scale[j] -= *p;
          }
          ++it1;
          ++it2;
        }
        it1 -= N;
        it2 -= N;
      }
    }
    for (int j = 0; j < N; ++j) {
      scale[j] /= M;
    }
  }
}

template void caffe_cpu_binary_gemm_xor <float> (const bool transposeA,
    const bool transposeB,const int M, const int N, const int K,
    const binary_t* A, const binary_t* B, const float* scaleA,
    const float* scaleB, float *C);
template void caffe_cpu_binary_gemm_xor <double> (const bool transposeA,
    const bool transposeB, const int M, const int N, const int K,
    const binary_t* A, const binary_t* B, const double* scaleA,
    const double* scaleB, double *C);

template void caffe_cpu_binary_approx<float>(const int axis, const int M,
    const int N, const float* In, const vector<float> &scale, vector<float> &Out);
template void caffe_cpu_binary_approx<double>(const int axis, const int M,
    const int N, const double* In, const vector<double> &scale, vector<double> &Out);

template void caffe_cpu_binary_scale<float>(const int axis, const int M,
    const int N, const float* In, vector<float> &scale);
template void caffe_cpu_binary_scale<double>(const int axis, const int M,
    const int N, const double* In, vector<double> &scale);

template void caffe_cpu_binary_gradient<float>(const int axis, const int M,
    const int N, const float* In, const vector<float> &scale, float *grad);
template void caffe_cpu_binary_gradient<double>(const int axis, const int M,
    const int N, const double* In, const vector<double> &scale, double *grad);

#define INSTANTIATE_BINARY_MATH(Dtype) \
template void caffe_cpu_binary_gemm_and<Dtype>(\
    const bool transposeA, const bool transposeB, const int M, const int N, \
    const int K, const Dtype alpha, const binary_t* A, const binary_t *B, \
    const Dtype* scaleA, const Dtype* scaleB, Dtype beta, Dtype* C);\
template void caffe_cpu_ternary<Dtype>(const int axis, const int M, \
    const int N, const Dtype* In, vector<binary_t> &pos, vector<binary_t> &neg, \
    Dtype &delta, vector<Dtype> &scale);\
template void caffe_cpu_binary<Dtype>(const int axis, const int M, const int N, \
    const Dtype* In, vector<binary_t>& Out, vector<Dtype> &scale);
INSTANTIATE_BINARY_MATH(float);
INSTANTIATE_BINARY_MATH(double);

}

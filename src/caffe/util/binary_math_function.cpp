#include <algorithm>
#include <iostream>

#include "caffe/util/binary_math_function.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template<typename Dtype>
void caffe_cpu_binary_gemm_and(
  const bool transposeA, const bool transposeB,
  const int M, const int N, const int K, const Dtype alpha, const binary_t* A,
  const binary_t *B, const Dtype* scaleA, const Dtype* scaleB,
  Dtype beta, Dtype* C) {
  const int KK = (K + BINARY_SIZE - 1) / BINARY_SIZE;
  const binary_t *pA, *pB;
  Dtype *pC;
  if (caffe_sign(alpha) != 0)
  { beta /= alpha; }
  pC = C;
  for (int i = 0; i < M * N; ++i)
  { *pC++ *= beta; }
  if (caffe_sign(alpha) == 0)
  { return ; }
  // through adjust the order of i, j, k to implement matrix multiplication.
  if (!transposeA && !transposeB) {
    pA = A;
    for (int i = 0; i < M; ++i) {
      pB = B;
      for (int k = 0; k < KK; ++k) {
        pC = C + i * N;
        const binary_t vA = *pA++;
        for (int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA & (*pB++));
          // C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }
  else if (transposeA && !transposeB) {
    pA = A;
    for (int k = 0; k < KK; ++k) {
      pC = C;
      for (int i = 0; i < M; ++i) {
        pB = B + k * N;
        const binary_t vA = *pA++;
        for (int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA & (*pB++));
          //C[i][j] += A[k][i] * B[k][j];
        }
      }
    }
  }
  else if (!transposeA && transposeB) {
    pC = C;
    for (int i = 0; i < M; ++i) {
      pB = B;
      for (int j = 0; j < N; ++j) {
        pA = A + i * KK;
        auto &result = *(pC++);
        for (int k = 0; k < KK; ++k) {
          result += bitcount((*pA++) & (*pB++));
          // C[i][j] += A[i][k] * B[j][k];
        }
      }
    }
  }
  else {
    pA = A;
    for (int k = 0; k < KK; ++k) {
      pC = C;
      for (int i = 0; i < M; ++i) {
        auto vA = *pA++;
        for (int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA & *(B + j * KK + k));
          // C[i][j] += A[k][i] * B[j][k];
        }
      }
    }
  }
  pC = C;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      *pC++ *= alpha * scaleA[i] * scaleB[j];
    }
  }
}

template<typename Dtype>
void caffe_cpu_binary_gemm_xor(
  const bool transposeA, const bool transposeB,
  const int M, const int N, const int K, const binary_t* A,
  const binary_t* B, const Dtype* scaleA, const Dtype* scaleB, Dtype *C) {
  const int sz = M * N;
  const int KK = (K + BINARY_SIZE - 1) / BINARY_SIZE;
  caffe_set(sz, Dtype(0), C);
  const binary_t *pA, *pB;
  Dtype *pC;
  // through adjust the order of i, j, k to implement matrix multiplication.
  if (!transposeA && !transposeB) {
    pA = A;
    for (int i = 0; i < M; ++i) {
      pB = B;
      for (int k = 0; k < KK; ++k) {
        pC = C + i * N;
        const binary_t vA = *pA++;
        for (int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA ^ (*pB++));
          // C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }
  else if (transposeA && !transposeB) {
    pA = A;
    for (int k = 0; k < KK; ++k) {
      pC = C;
      for (int i = 0; i < M; ++i) {
        pB = B + k * N;
        const binary_t vA = *pA++;
        for (int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA ^ (*pB++));
          //C[i][j] += A[k][i] * B[k][j];
        }
      }
    }
  }
  else if (!transposeA && transposeB) {
    pC = C;
    for (int i = 0; i < M; ++i) {
      pB = B;
      for (int j = 0; j < N; ++j) {
        pA = A + i * KK;
        auto &result = *(pC++);
        for (int k = 0; k < KK; ++k) {
          result += bitcount((*pA++) ^ (*pB++));
          // C[i][j] += A[i][k] * B[j][k];
        }
      }
    }
  }
  else {
    pA = A;
    for (int k = 0; k < KK; ++k) {
      pC = C;
      for (int i = 0; i < M; ++i) {
        auto vA = *pA++;
        for (int j = 0; j < N; ++j) {
          *pC++ += bitcount(vA ^ * (B + j * KK + k));
          // C[i][j] += A[k][i] * B[j][k];
        }
      }
    }
  }
  pC = C;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      *pC = (K - (Dtype)2 * *pC) * scaleA[i] * scaleB[j];
      ++pC;
    }
  }
}

template<typename Dtype>
void caffe_cpu_binary(
  const int axis, const int M, const int N,
  const Dtype* In, vector<binary_t>& Out, vector<Dtype> &scale) {
  if (axis == 1) {
    const int cM = (M + BINARY_SIZE - 1) / BINARY_SIZE;
    Out.resize(cM * N);
    scale.resize(N);
    fill(Out.begin(), Out.end(), binary_t(0));
    fill(scale.begin(), scale.end(), Dtype(0));
    auto p = In;
    auto q = Out.begin();
    for (int i = 0; i < M;) {
      for (binary_t k = 0; i < M && k < BINARY_SIZE; ++k, ++i) {
        for (int j = 0; j < N; ++j) {
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
  }
  else if (axis == 0) {
    const int cN = (N + BINARY_SIZE - 1) / BINARY_SIZE;
    Out.resize(M * cN);
    scale.resize(M);
    fill(Out.begin(), Out.end(), binary_t(0));
    fill(scale.begin(), scale.end(), Dtype(0));
    auto p = In;
    for (int i = 0; i < M; ++i) {
      auto q = Out.begin() + i * cN;
      for (int j = 0; j < N; ++q) {
        for (binary_t k = 0; j < N && k < BINARY_SIZE; ++k, ++j) {
          scale[i] += std::abs(*p);
          *q |= (static_cast<binary_t>(*p++ >= Dtype(0))) << k;
        }
      }
    }
    for (int i = 0; i < M; ++i) {
      scale[i] /= N;
    }
  }
  else
  { CHECK(false) << "Error axis!"; }
}

template<typename Dtype>
void caffe_cpu_binary_approx(
  const int axis, const int M, const int N,
  const Dtype* In, const vector<Dtype> &scale, vector<Dtype> &Out) {
  auto p = Out.begin();
  const Dtype* q = In;
//  CHECK((int)Out.size() == M * N) << "Error appros out size!";
  if (axis == 0) {
//    CHECK((int)scale.size() == M) << "Error approx 0!";
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *p++ = *q++ >= Dtype(0) ? scale[i] : -scale[i];
      }
    }
  }
  else if (axis == 1) {
//    CHECK((int)scale.size() == N) << "Error approx 1!";
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *p++ = *q++ >= Dtype(0) ? scale[j] : -scale[j];
      }
    }
  }
  else
  { CHECK(false) << "Error axis!"; }
}

template<typename Dtype>
void caffe_cpu_binary_scale(
  const int axis, const int M, const int N,
  const Dtype* In, vector<Dtype> &scale) {
  const Dtype* q = In;
  if (axis == 0) {
//    CHECK(scale.size() == M) << "scale size ERROR! 0";
    for (int i = 0; i < M; ++i) {
      scale[i] = 0;
      for (int j = 0; j < N; ++j) {
        scale[i] += *q++;
      }
      scale[i] /= (double) N;
    }
  }
  else if (axis == 1) {
//    CHECK_EQ(scale.size(), N) << "scale size ERROR";
    for (int i = 0; i < N; ++i) {
      scale[i] = 0;
    }
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        scale[j] += *q++;
      }
    }
    for (int i = 0; i < N; ++i) {
      scale[i] /= (double) M;
    }
  }
  else
  { CHECK(false) << "Error axis!"; }
}

template<typename Dtype>
void caffe_cpu_binary_gradient(
  const int axis, const int M, const int N,
  const Dtype* In, const vector<Dtype> &scale, Dtype *grad) {
  auto p = In;
  auto q = grad;
  if (axis == 0) {
//    CHECK_EQ(scale.size(), M) << "gradient scale size Error";
    double co = 1. / N;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *q++ *= co + Dtype(std::abs(*p++) <= Dtype(1) ? 1 : 0) * scale[i];
      }
    }
  }
  else if (axis == 1) {
//    CHECK_EQ(scale.size(), N) << "gradient scale size Error";
    double co = 1. / M;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *q++ *= co + Dtype(std::abs(*p++) <= Dtype(1) ? 1 : 0) * scale[j];
      }
    }
  }
  else {
    CHECK(false) << "Error axis!";
  }
}


template<typename Dtype>
void caffe_cpu_ternary(
  const int axis, const int M, const int N, const Dtype* In,
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
          }
          else if (*p < -delta) {
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
  }
  else {
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
          }
          else if (*p < -delta) {
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

template<typename Dtype>
void caffe_cpu_binary_norm(
  const int axis, const int n_row, const int n_col, const Dtype *in,
  vector<binary_t> &code, vector<Dtype> &scale,
  vector<Dtype> &bias, vector<Dtype> &sum) {
  if (axis == 0) {
    int b_col = (n_col + BINARY_SIZE - 1) / BINARY_SIZE;
    code.resize(n_row * b_col);
    scale.resize(n_row);
    bias.resize(n_row);
    sum.resize(n_row);
    fill(code.begin(), code.end(), 0);
    fill(scale.begin(), scale.end(), 0);
    fill(bias.begin(), bias.end(), 0);
    fill(sum.begin(), sum.end(), 0);
    const Dtype* p = in;
    // sum
    for (int r = 0; r < n_row; ++r) {
      for (int c = 0; c < n_col; ++c) {
        bias[r] += *p++;
      }
    }
    // mean
    for (int r = 0; r < n_row; ++r) {
      bias[r] /= n_col;
    }
    // mean normalization
    // compress
    // sum
    p = in;
    auto it_code = code.begin();
    for (int r = 0; r < n_row; ++r) {
      for (int c = 0; c < n_col; ++it_code) {
        for (int k = 0; k < BINARY_SIZE && c < n_col; ++k, ++c, ++p) {
          if (*p > bias[r]) {
            *it_code |= binary_t(1) << k;
            scale[r] += *p - bias[r];
            sum[r]++;
          }
          else {
            scale[r] += bias[r] - *p;
            sum[r]--;
          }
        }
      }
    }
    // scale
    for (int r = 0; r < n_row; ++r) {
      scale[r] /= n_col;
    }
  }
  else {
    int b_row = (n_row + BINARY_SIZE - 1) / BINARY_SIZE;
    code.resize(b_row * n_col);
    scale.resize(n_col);
    bias.resize(n_col);
    sum.resize(n_col);
    fill(code.begin(), code.end(), 0);
    fill(scale.begin(), scale.end(), 0);
    fill(bias.begin(), bias.end(), 0);
    fill(sum.begin(), sum.end(), 0);
    // sum
    const Dtype* p = in;
    for (int r = 0; r < n_row; ++r) {
      for (int c = 0; c < n_col; ++c) {
        bias[c] += *p++;
      }
    }
    // mean
    for (int c = 0; c < n_col; ++c) {
      bias[c] /= n_row;
    }
    // binary, compress, sum
    p = in;
    auto it_code = code.begin();
    for (int r = 0; r < n_row;) {
      for (int k = 0; k < BINARY_SIZE && r < n_row; ++k, ++r) {
        for (int c = 0; c < n_col; ++c, ++p, ++it_code) {
          if (*p > bias[c]) {
            *it_code |= binary_t(1) << k;
            scale[c] += *p - bias[c];
            sum[c]++;
          }
          else {
            scale[c] += bias[c] - *p;
            sum[c]--;
          }
        }
        it_code -= n_col;
      }
    }
    // scale
    for (int c = 0; c < n_col; ++c) {
      scale[c] /= n_row;
    }
  }
}


template<typename Dtype>
void caffe_cpu_binary_norm_gradient(
  const int axis, const int M, const int N, const Dtype* In,
  const vector<Dtype> &scale, const vector<Dtype> &bias, Dtype *grad) {
  auto p = grad;
  auto q = In;
  const double beta = 0.1;
  const double alpha = 0;
  if (axis == 0) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j, ++p, ++q) {
        if (fabs(*q - bias[i]) <= 2. * scale[i])
        { *p *= (scale[i] * beta + 2. / N); }
        else
        { *p *= (alpha * scale[i] + 2. / N); }
      }
    }
  }
  else {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j, ++p, ++q) {
        if (fabs(*q - bias[j]) <= 2. * scale[j])
        { *p *= (beta * scale[j] + 2. / M); }
        else
        { *p *= (alpha * scale[j] + 2. / M); }
      }
    }
  }
}

template<typename Dtype>
void caffe_cpu_binary_gemm(
  const bool transA, const bool transB,
  const int M, const int N, const int K,
  const vector<binary_t> &A, const vector<Dtype> &A_scale,
  const vector<Dtype> &A_bias, const vector<Dtype> &A_sum,
  const vector<binary_t> &B, const vector<Dtype> &B_scale,
  const vector<Dtype> &B_bias, const vector<Dtype> &B_sum,
  Dtype *C) {
  const int bK = (K + BINARY_SIZE - 1) / BINARY_SIZE;
  Dtype *ptr_c = C;
  // Check
  CHECK_EQ(A.size(), M * bK);
  CHECK_EQ(A_scale.size(), M);
  CHECK_EQ(A_bias.size(), M);
  CHECK_EQ(A_sum.size(), M);
  CHECK_EQ(B.size(), bK * N);
  CHECK_EQ(B_scale.size(), N);
  CHECK_EQ(B_bias.size(), N);
  CHECK_EQ(B_sum.size(), N);
  // C <-- 0
  for (int r = 0; r < M * N; ++r) {
    *ptr_c++ = 0;
  }
  // matrix multiplication
  auto it_a = A.begin();
  auto it_b = B.begin();
  if (!transA && !transB) {
    for (int r = 0; r < M; ++r) {
      it_b = B.begin();
      for (int k = 0; k < bK; ++k) {
        auto temp = *it_a++;
        ptr_c = C + r * N;
        for (int c = 0; c < N; ++c) {
          // c[r][c] += a[r][k] * b[k][c]
          *ptr_c++ += bitcount(temp ^ (*it_b++));
        }
      }
    }
  }
  else if (transA) {
    for (int k = 0; k < bK; ++k) {
      ptr_c = C;
      for (int r = 0; r < M; ++r) {
        auto temp = *it_a++;
        it_b = B.begin() + k * N;
        for (int c = 0; c < N; ++c) {
          // c[r][c] = a[k][r] * b[k][c]
          *ptr_c++ += bitcount(temp ^ (*it_b++));
        }
      }
    }
  }
  else if (transB) {
    ptr_c = C;
    for (int r = 0; r < M; ++r) {
      it_b = B.begin();
      for (int c = 0; c < N; ++c, ++ptr_c) {
        it_a = A.begin() + r * bK;
        for (int k = 0; k < bK; ++k) {
          // c[r][c] = a[r][k] * b[c][k]
          *ptr_c += bitcount((*it_a++) ^ (*it_b++));
        }
      }
    }
  }
  else {
    ptr_c = C;
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c, ++ptr_c) {
        for (int k = 0; k < bK; ++k) {
          // c[r][c] = a[k][r] * b[c][k]
          *ptr_c += bitcount(A[k * M + r] ^ B[c * bK + k]);
        }
      }
    }
  }
  ptr_c = C;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c, ++ptr_c) {
      *ptr_c = A_scale[r] * B_scale[c] * (K - 2 * *ptr_c) +
               A_scale[r] * A_sum[r] * B_bias[c] +
               B_scale[c] * B_sum[c] * A_bias[r] +
               A_bias[r] * B_bias[c] * K;
    }
  }
}

template void caffe_cpu_binary_gemm_xor <float> (const bool transposeA,
    const bool transposeB, const int M, const int N, const int K,
    const binary_t* A, const binary_t* B, const float* scaleA,
    const float* scaleB, float *C);
template void caffe_cpu_binary_gemm_xor <double> (const bool transposeA,
    const bool transposeB, const int M, const int N, const int K,
    const binary_t* A, const binary_t* B, const double* scaleA,
    const double* scaleB, double *C);

template void caffe_cpu_binary_approx<float>(const int axis, const int M,
    const int N, const float* In, const vector<float> &scale, vector<float> &Out);
template void caffe_cpu_binary_approx<double>(const int axis, const int M,
    const int N, const double* In, const vector<double> &scale,
    vector<double> &Out);

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
    const Dtype* In, vector<binary_t>& Out, vector<Dtype> &scale); \
template void caffe_cpu_binary_norm<Dtype>(const int axis, \
    const int n_row, const int n_col, \
    const Dtype *in, vector<binary_t> &code, vector<Dtype> &scale, \
    vector<Dtype> &bias, vector<Dtype> &sum);\
template void caffe_cpu_binary_gemm<Dtype>( \
    const bool transA, const bool transB,\
    const int M, const int N, const int K, \
    const vector<binary_t> &A, const vector<Dtype> &A_scale, \
    const vector<Dtype> &A_bias, const vector<Dtype> &A_sum, \
    const vector<binary_t> &B, const vector<Dtype> &B_scale, \
    const vector<Dtype> &B_bias, const vector<Dtype> &B_sum, \
    Dtype *C);\
template void caffe_cpu_binary_norm_gradient<Dtype>(\
    const int axis, const int M, const int N, const Dtype* In,\
    const vector<Dtype> &scale, const vector<Dtype> &bias, Dtype *grad);
INSTANTIATE_BINARY_MATH(float);
INSTANTIATE_BINARY_MATH(double);

}

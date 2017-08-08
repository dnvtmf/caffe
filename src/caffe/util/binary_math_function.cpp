#include <algorithm>
#include <iostream>

#include "caffe/util/binary_math_function.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template<typename Dtype>
void caffe_cpu_binary_gemm_and(
  const bool transposeA, const bool transposeB,
  const int M, const int N, const int K, const Dtype alpha, const Btype *A,
  const Btype *B, const Dtype *scaleA, const Dtype *scaleB,
  Dtype beta, Dtype *C) {
  const int KK = (K + BINARY_SIZE - 1) / BINARY_SIZE;
  const Btype *pA, *pB;
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
        const Btype vA = *pA++;
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
        const Btype vA = *pA++;
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
void caffe_cpu_binary(const int axis, const int M, const int N,
                      const Dtype *in, Btype *code, Dtype *scale) {
  if (axis == 0) {
    const int BN = (N + BINARY_SIZE - 1) / BINARY_SIZE;
    memset(code,  0, sizeof(Btype) * M * BN);
    memset(scale, 0, sizeof(Dtype) * M);
    auto p_in = in;
    auto p_code = code;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++p_code) {
        for (Btype k = 0; j < N && k < BINARY_SIZE; ++k, ++j) {
          scale[i] += std::abs(*p_in);
          *p_code |= (static_cast<Btype>(*p_in++ >= 0)) << k;
        }
      }
    }
    for (int i = 0; i < M; ++i) {
      scale[i] /= N;
    }
  }
  else {
    const int BM = (M - 1) / BINARY_SIZE + 1;
    memset(code,  0, sizeof(Btype) * BM * N);
    memset(scale, 0, sizeof(Dtype) * N);
    auto p = in;
    auto q = code;
    for (int i = 0; i < M;) {
      for (Btype k = 0; i < M && k < BINARY_SIZE; ++k, ++i) {
        for (int j = 0; j < N; ++j) {
          scale[j] += std::abs(*p);
          *q++ |= (static_cast<Btype>(*p++ >= 0)) << k;
        }
        q -= N;
      }
      q += N;
    }
    for (int j = 0; j < N; ++j) {
      scale[j] /= M;
    }
  }
}

template<typename Dtype>
void caffe_cpu_binary_approx(const int axis, const int M, const int N,
                             const Dtype *In, const Dtype *scale, Dtype *Out) {
  auto p = Out;
  const Dtype *q = In;
  if (axis == 0) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *p++ = *q++ >= Dtype(0) ? scale[i] : -scale[i];
      }
    }
  }
  else {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *p++ = *q++ >= Dtype(0) ? scale[j] : -scale[j];
      }
    }
  }
}

template<typename Dtype>
void caffe_cpu_binary_scale(const int axis, const int M, const int N,
                            const Dtype *In, Dtype *scale) {
  const Dtype *q = In;
  if (axis == 0) {
    for (int i = 0; i < M; ++i) {
      scale[i] = 0;
      for (int j = 0; j < N; ++j) {
        scale[i] += *q++;
      }
      scale[i] /= (double) N;
    }
  }
  else {
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
}

template<typename Dtype>
void caffe_cpu_binary_gradient(
  const int axis, const int M, const int N,
  const Dtype *In, const Dtype *scale, Dtype *grad) {
  auto p = In;
  auto q = grad;
  if (axis == 0) {
    double co = 1. / N;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *q++ *= co + Dtype(std::abs(*p++) <= Dtype(1) ? 1 : 0) * scale[i];
      }
    }
  }
  else {
    double co = 1. / M;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        *q++ *= co + Dtype(std::abs(*p++) <= Dtype(1) ? 1 : 0) * scale[j];
      }
    }
  }
}


template<typename Dtype>
void caffe_cpu_ternary(
  const int axis, const int M, const int N, const Dtype *in,
  Btype *code, Btype *mask, Dtype &delta, Dtype *scale, Dtype *sum2) {
  if (axis == 0) {
    const int BN = (N - 1) / BINARY_SIZE + 1;
    memset(code,  0, sizeof(Btype) * M * BN);
    memset(mask,  0, sizeof(Btype) * M * BN);
    memset(scale, 0, sizeof(Dtype) * M);
    memset(sum2,  0, sizeof(Dtype) * M);
    delta = 0.7 * caffe_cpu_asum<Dtype>(M * N, in) / (1. * M * N);
    auto p = in;
    auto it1 = code, it2 = mask;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N;) {
        for (int k = 0; k < BINARY_SIZE && j < N; ++j, ++k) {
          if (*p > delta) {
            *it1 |= Btype(1) << k;
            *it2 |= Btype(1) << k;
            scale[i] += *p;
          }
          else if (*p < -delta) {
            *it2 |= Btype(1) << k;
            scale[i] -= *p;
          }
          ++p;
        }
        sum2[i] += bitcount(*it2);
        ++it1;
        ++it2;
      }
    }
    for (int i = 0; i < M; ++i) {
      scale[i] /= N;
    }
  }
  else {
    const int BM = (M - 1) / BINARY_SIZE + 1;
    memset(code,  0, sizeof(Btype) * BM * N);
    memset(mask,  0, sizeof(Btype) * BM * N);
    memset(scale, 0, sizeof(Dtype) * N);
    memset(sum2,  0, sizeof(Dtype) * N);
    delta = 0.7 * caffe_cpu_asum<Dtype>(M * N, in) / (1. * M * N);
    auto p = in;
    auto it1 = code, it2 = mask;
    for (int i = 0; i < M;) {
      for (int k = 0; k < BINARY_SIZE && i < M; ++i, ++k) {
        for (int j = 0; j < N; ++j, ++p) {
          if (*p > delta) {
            *it1 |= Btype(1) << k;
            *it2 |= Btype(1) << k;
            scale[j] += *p;
          }
          else if (*p < -delta) {
            *it2 |= Btype(1) << k;
            scale[j] -= *p;
          }
          ++it1;
          ++it2;
        }
        it1 -= N;
        it2 -= N;
      }
      for (int j = 0; j < N; ++j) {
        sum2[j] += bitcount(*it2++);
      }
      it1 += N;
    }
    for (int j = 0; j < N; ++j) {
      scale[j] /= M;
    }
  }
}

template<typename Dtype>
void caffe_cpu_binary_norm(
  const int axis, const int M, const int N, const Dtype *in,
  Btype *code, Dtype *scale, Dtype *bias, Dtype *sum, const bool use_bias) {
  if (!use_bias) {
    caffe_cpu_binary<Dtype>(axis, M, N, in, code, scale);
    return ;
  }
  if (axis == 0) {
    int BN = (N - 1) / BINARY_SIZE + 1;
    memset(code,  0, sizeof(Btype) * M * BN);
    memset(scale, 0, sizeof(Dtype) * M);
    memset(bias,  0, sizeof(Dtype) * M);
    memset(sum,   0, sizeof(Dtype) * M);
    const Dtype *p = in;
    // sum
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c) {
        bias[r] += *p++;
      }
    }
    // mean
    for (int r = 0; r < M; ++r) {
      bias[r] /= N;
    }
    // mean normalization
    // compress
    // sum
    p = in;
    auto it_code = code;
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++it_code) {
        for (int k = 0; k < BINARY_SIZE && c < N; ++k, ++c, ++p) {
//          assert(it_code == code + r * b_col + c / BINARY_SIZE);
//          assert(p == in + r * n_col + c);
          if (*p > bias[r]) {
            *it_code |= Btype(1) << k;
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
    for (int r = 0; r < M; ++r) {
      scale[r] /= N;
      sum[r] *= scale[r];
    }
  }
  else {
    int BM = (M - 1) / BINARY_SIZE + 1;
    memset(code,  0, sizeof(Btype) * BM * N);
    memset(scale, 0, sizeof(Dtype) * N);
    memset(bias,  0, sizeof(Dtype) * N);
    memset(sum,   0, sizeof(Dtype) * N);
    // sum
    const Dtype *p = in;
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c) {
        bias[c] += *p++;
      }
    }
    // mean
    for (int c = 0; c < N; ++c) {
      bias[c] /= M;
    }
    // binary, compress, sum
    p = in;
    auto it_code = code;
    for (int r = 0; r < M;) {
      for (int k = 0; k < BINARY_SIZE && r < M; ++k, ++r) {
        for (int c = 0; c < N; ++c, ++p, ++it_code) {
          if (*p > bias[c]) {
            *it_code |= Btype(1) << k;
            scale[c] += *p - bias[c];
            sum[c]++;
          }
          else {
            scale[c] += bias[c] - *p;
            sum[c]--;
          }
        }
        it_code -= N;
      }
      it_code += N;
    }
    // scale
    for (int c = 0; c < N; ++c) {
      scale[c] /= M;
      sum[c] *= scale[c];
    }
  }
}

template<typename Dtype>
void caffe_cpu_ternary_norm(
  const int axis, const int M, const int N, const Dtype *in,
  Btype *code, Btype *mask, Dtype *delta, Dtype *scale,
  Dtype *bias, Dtype *sum,  Dtype *sum2, const bool use_bias)  {
  if (!use_bias) {
    caffe_cpu_ternary<Dtype>(axis, M, N, in, code, mask, delta[0], scale, sum2);
    return ;
  }
  if (axis == 0) {
    const int BN = (N - 1) / BINARY_SIZE + 1;
    memset(code,  0, sizeof(Btype) * M * BN);
    memset(mask,  0, sizeof(Btype) * M * BN);
    memset(delta, 0, sizeof(Dtype) * M);
    memset(scale, 0, sizeof(Dtype) * M);
    memset(bias,  0, sizeof(Dtype) * M);
    memset(sum,   0, sizeof(Dtype) * M);
    memset(sum2,  0, sizeof(Dtype) * M);
    auto p = in;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        bias[i] += *p++;
      }
    }
    for (int i = 0; i < M; ++i) {
      bias[i] /= N;
    }
    p = in;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j, ++p) {
        delta[i] += std::abs(*p - bias[i]);
      }
    }
    for (int i = 0; i < M; ++i) {
      delta[i] *= 0.7 / N;
    }
    p = in;
    auto code_it = code;
    auto mask_it = mask;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N;) {
        for (int k = 0; k < BINARY_SIZE && j < N; ++k, ++j, ++p) {
//          assert(code_it == code + i * BN + j / BINARY_SIZE);
//          assert(mask_it == mask + i * BN + j / BINARY_SIZE);
          if (*p - bias[i] > delta[i]) {
            *code_it |= Btype(1) << k;
            *mask_it |= Btype(1) << k;
            scale[i] += *p - bias[i];
            sum[i]++;
          }
          else if (*p - bias[i] < -delta[i]) {
            *mask_it |= Btype(1) << k;
            scale[i] -= *p - bias[i];
            sum[i]--;
          }
          else {
          }
        }
        sum2[i] += bitcount(*mask_it);
        ++code_it;
        ++mask_it;
      }
    }
    for (int i = 0; i < M; ++i)  {
      if (sum2[i] > 0.)
      { scale[i] /= sum2[i]; }
      sum[i] *= scale[i];
    }
  }
  else {
    const int BM = (M - 1) / BINARY_SIZE + 1;
    memset(code,  0, sizeof(Btype) * BM * N);
    memset(mask,  0, sizeof(Btype) * BM * N);
    memset(delta, 0, sizeof(Dtype) * N);
    memset(scale, 0, sizeof(Dtype) * N);
    memset(bias,  0, sizeof(Dtype) * N);
    memset(sum,   0, sizeof(Dtype) * N);
    memset(sum2,  0, sizeof(Dtype) * N);
    auto p = in;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        bias[j] += *p++;
      }
    }
    for (int j = 0; j < N; ++j) {
      bias[j] /= M;
    }
    p = in;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        delta[j] += std::abs(*p++ - bias[j]);
      }
    }
    for (int j = 0; j < N; ++j) {
      delta[j] *= 0.7 / M;
    }
    p = in;
    auto code_it = code;
    auto mask_it = mask;
    for (int i = 0; i < M;) {
      for (int k = 0; k < BINARY_SIZE && i < M; ++i, ++k) {
        for (int j = 0; j < N; ++j, ++p) {
          if (*p - bias[j] > delta[j]) {
            *code_it |= Btype(1) << k;
            *mask_it |= Btype(1) << k;
            scale[j] += *p - bias[j];
            sum[j]++;
          }
          else if (*p - bias[j] < -delta[j]) {
            *mask_it |= Btype(1) << k;
            scale[j] -= *p - bias[j];
            sum[j]--;
          }
          ++code_it;
          ++mask_it;
        }
        code_it -= N;
        mask_it -= N;
      }
      for (int j = 0; j < N; ++j) {
        sum2[j] += bitcount(*mask_it++);
      }
      code_it += N;
    }
    for (int j = 0; j < N; ++j) {
      if (sum2[j] > 0.)
      { scale[j] /= sum2[j]; }
      sum[j] *= scale[j];
    }
  }
}

template<typename Dtype>
void caffe_cpu_binary_norm_gradient(
  const int axis, const int M, const int N, const Dtype *in,
  const Dtype *scale, const Dtype *bias, Dtype *grad) {
  if (axis == 0) {
    Dtype mul = 1. / N;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
//        *grad++ *= std::abs(*in++ - bias[i]) / scale[i];
        *grad++ *= (mul + scale[i]) * (1 + mul);
      }
    }
  }
  else {
    Dtype mul = 1. / M;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
//        *grad++ *= std::abs(*in++ - bias[j]) / scale[j];
        *grad++ *= (mul + scale[j]) * (1 + mul);
      }
    }
  }
}

template<typename Dtype>
void caffe_cpu_ternary_norm_gradient(
  const int axis, const int M, const int N, const Dtype *in,
  const Dtype *delta, const Dtype *scale, const Dtype *bias, Dtype *grad) {
  if (axis == 0) {
    Dtype mul = 1. / N;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
//        *grad++ *= std::abs(*in++ - bias[i]) / scale[i];
        *grad++ *= (mul + scale[i]) * (1 + mul);
      }
    }
  }
  else {
    Dtype mul = 1. / M;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
//        *grad++ *= std::abs(*in++ - bias[j]) / scale[j];
        *grad++ *= (mul + scale[j]) * (1 + mul);
      }
    }
  }
}
template<typename Dtype>
void caffe_cpu_binary_gemm(
  const bool transA, const bool transB,
  const int M, const int N, const int K,
  const Btype *A, const Dtype *A_scale,
  const Btype *B, const Dtype *B_scale,
  Dtype *C,
  const bool bias,
  const Dtype *A_bias, const Dtype *A_sum,
  const Dtype *B_bias, const Dtype *B_sum) {
  const int BK = (K - 1) / BINARY_SIZE + 1;
  Dtype *ptr_c = C;
  // C <-- 0
  caffe_set<Dtype>(M * N, 0, C);
  // matrix multiplication
  auto it_a = A;
  auto it_b = B;
  if (!transA && !transB) {
    for (int r = 0; r < M; ++r) {
      it_b = B;
      for (int k = 0; k < BK; ++k) {
        auto temp = *it_a++;
        for (int c = 0; c < N; ++c) {
          // c[r][c] += a[r][k] * b[k][c]
          *ptr_c++ += bitcount(temp ^ (*it_b++));
        }
        ptr_c -= N;
      }
      ptr_c += N;
    }
  }
  else if (transA) {
    for (int k = 0; k < BK; ++k) {
      ptr_c = C;
      for (int r = 0; r < M; ++r) {
        auto temp = *it_a++;
        for (int c = 0; c < N; ++c) {
          // c[r][c] = a[k][r] * b[k][c]
          *ptr_c++ += bitcount(temp ^ (*it_b++));
        }
        it_b -= N;
      }
      it_b += N;
    }
  }
  else if (transB) {
    ptr_c = C;
    for (int r = 0; r < M; ++r) {
      it_b = B;
      for (int c = 0; c < N; ++c, ++ptr_c) {
        for (int k = 0; k < BK; ++k) {
          // c[r][c] = a[r][k] * b[c][k]
          *ptr_c += bitcount((*it_a++) ^ (*it_b++));
        }
        it_a -= BK;
      }
      it_a += BK;
    }
  }
  else {
    ptr_c = C;
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c, ++ptr_c) {
        for (int k = 0; k < BK; ++k) {
          // c[r][c] = a[k][r] * b[c][k]
          *ptr_c += bitcount(A[k * M + r] ^ B[c * BK + k]);
        }
      }
    }
  }
  ptr_c = C;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c, ++ptr_c) {
      *ptr_c = A_scale[r] * B_scale[c] * (K - 2 * *ptr_c);
    }
  }
  if (!bias) return ;
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(1.),
                        A_sum, B_bias, Dtype(1.), C);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(1.),
                        A_bias, B_sum, Dtype(1.), C);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(K),
                        A_bias, B_bias, Dtype(1.), C);
}

template<typename Dtype>
void caffe_cpu_tb_gemm(
  const bool transA, const bool transB,
  const int M, const int N, const int K,
  const Btype *A,      const Btype *A_mask, const Dtype *A_scale,
  const Dtype *A_sum2, const Btype *B,      const Dtype *B_scale,
  Dtype *C,
  const bool  bias,
  const Dtype *A_bias, const Dtype *A_sum,
  const Dtype *B_bias, const Dtype *B_sum) {
  const int BK = (K - 1) / BINARY_SIZE + 1;
  Dtype *ptr_c = C;
  // C <-- 0
  caffe_set<Dtype>(M * N, 0, C);
  // matrix multiplication
  auto a_it      = A;
  auto a_mask_it = A_mask;
  auto b_it      = B;
  if (!transA && !transB) {
    for (int r = 0; r < M; ++r) {
      b_it = B;
      for (int k = 0; k < BK; ++k) {
        auto temp = *a_it++;
        auto mask_temp = *a_mask_it++;
        ptr_c = C + r * N;
        for (int c = 0; c < N; ++c) {
          // c[r][c] += a[r][k] * b[k][c]
          *ptr_c++ += bitcount((temp ^ (*b_it++)) & mask_temp);
        }
      }
    }
  }
  else if (transA && !transB) {
    for (int k = 0; k < BK; ++k) {
      ptr_c = C;
      for (int r = 0; r < M; ++r) {
        auto temp = *a_it++;
        auto mask_temp = *a_mask_it++;
        for (int c = 0; c < N; ++c) {
          // c[r][c] = a[k][r] * b[k][c]
          *ptr_c++ += bitcount((temp ^ (*b_it++)) & mask_temp);
        }
        b_it -= N;
      }
      b_it += N;
    }
  }
  else if (!transA && transB) {
    ptr_c = C;
    for (int r = 0; r < M; ++r) {
      b_it = B;
      for (int c = 0; c < N; ++c, ++ptr_c) {
        for (int k = 0; k < BK; ++k) {
          // c[r][c] = a[r][k] * b[c][k]
          *ptr_c += bitcount(((*a_it++) ^ (*b_it++)) & (*a_mask_it++));
        }
        a_it -= BK;
        a_mask_it -= BK;
      }
      a_it += BK;
      a_mask_it += BK;
    }
  }
  else {
    ptr_c = C;
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c, ++ptr_c) {
        for (int k = 0; k < BK; ++k) {
          // c[r][c] = a[k][r] * b[c][k]
          *ptr_c += bitcount((A[k * M + r] ^ B[c * BK + k]) & A_mask[k * M + r]);
        }
      }
    }
  }
//  ptr_c = C;
//  for (int r = 0; r < M; ++r) {
//    for (int c = 0; c < N; ++c, ++ptr_c) {
//      *ptr_c = A_scale[r] * B_scale[c] * (A_sum2[r] - 2 * *ptr_c) +
//               A_scale[r] * A_sum[r]   * B_bias[c] +
//               B_scale[c] * B_sum[c]   * A_bias[r] +
//               A_bias[r]  * B_bias[c]  * K;
//    }
//  }
  ptr_c = C;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      *ptr_c = A_scale[r] * B_scale[c] * (A_sum2[r] - 2 * *ptr_c);
      ptr_c++;
    }
  }
//  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(1.),
//                        A_scale, B_scale, Dtype(1.), C);
  if (!bias) return ;
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(1.),
                        A_sum, B_bias, Dtype(1.), C);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(1.),
                        A_bias, B_sum, Dtype(1.), C);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(K),
                        A_bias, B_bias, Dtype(1.), C);
}

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
  const Dtype *B_bias, const Dtype *B_sum) {
  const int bK = (K - 1) / BINARY_SIZE + 1;
  Dtype *ptr_c = C;
  // C <-- 0
  caffe_set<Dtype>(M * N, 0, C);
  // matrix multiplication
  auto a_it = A;
  auto b_it = B;
  auto b_mask_it = B_mask;
  if (!transA && !transB) {
    for (int r = 0; r < M; ++r) {
      b_it = B;
      b_mask_it = B_mask;
      for (int k = 0; k < bK; ++k) {
        auto temp = *a_it++;
        ptr_c = C + r * N;
        for (int c = 0; c < N; ++c) {
          // c[r][c] += a[r][k] * b[k][c]
          *ptr_c++ += bitcount((temp ^ (*b_it++)) & (*b_mask_it++));
        }
      }
    }
  }
  else if (transA && !transB) {
    for (int k = 0; k < bK; ++k) {
      ptr_c = C;
      for (int r = 0; r < M; ++r) {
        auto temp = *a_it++;
        for (int c = 0; c < N; ++c) {
          // c[r][c] = a[k][r] * b[k][c]
          *ptr_c++ += bitcount((temp ^ (*b_it++)) & (*b_mask_it++));
        }
        b_it -= N;
        b_mask_it -= N;
      }
      b_it += N;
      b_mask_it += N;
    }
  }
  else if (!transA && transB) {
    ptr_c = C;
    for (int r = 0; r < M; ++r) {
      b_it = B;
      b_mask_it = B_mask;
      for (int c = 0; c < N; ++c, ++ptr_c) {
        for (int k = 0; k < bK; ++k) {
          // c[r][c] = a[r][k] * b[c][k]
          *ptr_c += bitcount(((*a_it++) ^ (*b_it++)) & (*b_mask_it++));
        }
        a_it -= bK;
      }
      a_it += bK;
    }
  }
  else {
    ptr_c = C;
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c, ++ptr_c) {
        for (int k = 0; k < bK; ++k) {
          // c[r][c] = a[k][r] * b[c][k]
          *ptr_c += bitcount((A[k * M + r] ^ B[c * bK + k]) & B_mask[c * bK + k]);
        }
      }
    }
  }
//  ptr_c = C;
//  for (int r = 0; r < M; ++r) {
//    for (int c = 0; c < N; ++c, ++ptr_c) {
//      *ptr_c = A_scale[r] * B_scale[c] * (B_sum2[c] - 2 * *ptr_c) +
//               A_sum[r]   * B_bias[c] +
//               B_sum[c]   * A_bias[r] +
//               A_bias[r]  * B_bias[c]  * K;
//    }
//  }
  ptr_c = C;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      *ptr_c = A_scale[r] * B_scale[c] * (B_sum2[c] - 2 * *ptr_c);
      ptr_c++;
    }
  }
//  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(1.),
//                        A_scale, B_scale, Dtype(1.), C);
  if (!bias) return;
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(1.),
                        A_sum, B_bias, Dtype(1.), C);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(1.),
                        A_bias, B_sum, Dtype(1.), C);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1, Dtype(K),
                        A_bias, B_bias, Dtype(1.), C);
}


template<typename Dtype>
void caffe_cpu_binary_restore(
  const int axis, const int M, const int N,
  const Btype *code, const Dtype *scale,
  const Dtype *bias, Dtype *out) {
  if (axis == 0) {
    // const int BN = (N - 1) / BINARY_SIZE + 1;
    auto it = code;
    Dtype *p = out;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++it) {
        for (int k = 0; k < BINARY_SIZE && j < N; ++j, ++k, ++p) {
          if (*it & (1 << k))
            *p = scale[i] + bias[i];
          else
            *p = -scale[i] + bias[i];
        }
      }
    }
  }
  else {
    // const int BM = (M - 1) / BINARY_SIZE + 1;
    auto it = code;
    Dtype *p = out;
    for (int i = 0; i < M;) {
      for (int k = 0; k < BINARY_SIZE && i < M; ++i, ++k) {
        for (int j = 0; j < N; ++j, ++p, ++it) {
          if (*it & (1 << k))
            *p = scale[j] + bias[j];
          else
            *p = -scale[j] + bias[j];
        }
        it -= N;
      }
      it += N;
    }
  }
}

template<typename Dtype>
void caffe_cpu_ternary_restore(
  const int axis, const int M, const int N,
  const Btype *code, const Btype *mask,
  const Dtype *scale, const Dtype *bias, Dtype *out) {
  if (axis == 0) {
    // const int BN = (N - 1) / BINARY_SIZE + 1;
    auto it_code = code;
    auto it_mask = mask;
    auto p = out;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N;) {
        for (int k = 0; k < BINARY_SIZE && j < N; ++j, ++k, ++p) {
          if (*it_mask & (1 << k)) {
            if (*it_code & (1 << k))
              *p = scale[i] + bias[i];
            else
              *p = -scale[i] + bias[i];
          }
          else
            *p = bias[i];
        }
        ++it_code;
        ++it_mask;
      }
    }
  }
  else {
    // const int BM = (M - 1) / BINARY_SIZE + 1;
    auto it_code = code;
    auto it_mask = mask;
    auto p = out;
    for (int i = 0; i < M;) {
      for (int k = 0; k < BINARY_SIZE && i < M; ++i, ++k) {
        for (int j = 0; j < N; ++j, ++p) {
          if (*it_mask & (1 << k)) {
            if (*it_code & (1 << k))
              *p = scale[j] + bias[j];
            else
              *p = -scale[j] + bias[j];
          }
          else
            *p = bias[j];
          ++it_code;
          ++it_mask;
        }
        it_code -= N;
        it_mask -= N;
      }
      it_code += N;
      it_mask += N;
    }
  }
}

#define INSTANTIATE_BINARY_MATH(Dtype) \
  \
template void caffe_cpu_binary_gemm_and<Dtype>( \
  const bool transposeA, const bool transposeB, \
  const int M, const int N, const int K, const Dtype alpha, const Btype *A, \
  const Btype *B, const Dtype *scaleA, const Dtype *scaleB, \
  Dtype beta, Dtype *C); \
   \
template void caffe_cpu_binary<Dtype>( \
  const int axis, const int M, const int N, \
  const Dtype *in, Btype *code, Dtype *scale); \
  \
template void caffe_cpu_binary_approx<Dtype>( \
  const int axis, const int M, const int N, \
  const Dtype *In, const Dtype *scale, Dtype *Out); \
  \
template void caffe_cpu_binary_scale<Dtype>( \
  const int axis, const int M, const int N, \
  const Dtype *In, Dtype *scale); \
  \
template void caffe_cpu_binary_gradient<Dtype>( \
  const int axis, const int M, const int N, \
  const Dtype *In, const Dtype *scale, Dtype *grad); \
  \
template void caffe_cpu_ternary<Dtype>( \
  const int axis, const int M, const int N, const Dtype *in, \
  Btype *code, Btype *mask, Dtype &delta, Dtype *scale, Dtype *sum2); \
  \
template void caffe_cpu_binary_norm<Dtype>( \
  const int axis, const int M, const int N, const Dtype *in, \
  Btype *code, Dtype *scale, Dtype *bias, Dtype *sum, const bool use_bias); \
  \
template void caffe_cpu_ternary_norm<Dtype>( \
  const int axis, const int M, const int N, const Dtype *in, \
  Btype *code, Btype *mask, Dtype *delta, Dtype *scale, \
  Dtype *bias, Dtype *sum,  Dtype *sum2, const bool use_bias); \
  \
template void caffe_cpu_binary_norm_gradient<Dtype>( \
  const int axis, const int M, const int N, const Dtype *in, \
  const Dtype *scale, const Dtype *bias, Dtype *grad); \
  \
template void caffe_cpu_ternary_norm_gradient<Dtype>( \
  const int axis, const int M, const int N, const Dtype *in, \
  const Dtype *delta, const Dtype *scale, const Dtype *bias, Dtype *grad); \
  \
template void caffe_cpu_binary_gemm<Dtype>( \
  const bool transA, const bool transB, \
  const int M, const int N, const int K, \
  const Btype *A, const Dtype *A_scale, \
  const Btype *B, const Dtype *B_scale, \
  Dtype *C, \
  const bool bias, \
  const Dtype *A_bias, const Dtype *A_sum, \
  const Dtype *B_bias, const Dtype *B_sum); \
  \
template void caffe_cpu_tb_gemm<Dtype>( \
  const bool transA, const bool transB, \
  const int M, const int N, const int K, \
  const Btype *A,      const Btype *A_mask, const Dtype *A_scale, \
  const Dtype *A_sum2, const Btype *B,      const Dtype *B_scale, \
  Dtype *C, \
  const bool  bias, \
  const Dtype *A_bias, const Dtype *A_sum, \
  const Dtype *B_bias, const Dtype *B_sum); \
  \
template void caffe_cpu_bt_gemm<Dtype>( \
  const bool transA, const bool transB, \
  const int M, const int N, const int K, \
  const Btype *A,       const Dtype *A_scale, \
  const Btype *B,       const Btype *B_mask, \
  const Dtype *B_scale, const Dtype *B_sum2, \
  Dtype *C, \
  const bool bias, \
  const Dtype *A_bias, const Dtype *A_sum, \
  const Dtype *B_bias, const Dtype *B_sum); \
  \
template void caffe_cpu_binary_restore<Dtype>( \
  const int axis, const int M, const int N, \
  const Btype *code, const Dtype *scale, \
  const Dtype *bias, Dtype *out); \
  \
template void caffe_cpu_ternary_restore<Dtype>( \
  const int axis, const int M, const int N, \
  const Btype *code, const Btype *mask, \
  const Dtype *scale, const Dtype *bias, Dtype *out);
INSTANTIATE_BINARY_MATH(float);
INSTANTIATE_BINARY_MATH(double);
}

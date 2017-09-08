#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#ifndef CPU_ONLY

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }

template <typename Dtype>
__global__ void binary_gradient_kernel_0(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, Dtype *grad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  Dtype mul = 1. / N;
  for (int j = 0; j < N; ++j) {
    grad[i * N + j] *=
        mul + (gpu_abs(in[i * N + j]) <= Dtype(1) ? scale[i] : Dtype(0));
  }
  if (use_bias) {
    mul = 1. - mul;
    for (int j = 0; j < N; ++j) grad[i * N + j] *= mul;
  }
}

template <typename Dtype>
__global__ void binary_gradient_kernel_1(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, Dtype *grad) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  Dtype mul = 1. / M;
  for (int i = 0; i < M; ++i) {
    grad[i * N + j] *=
        mul + (gpu_abs(in[i * N + j]) <= Dtype(1) ? scale[j] : Dtype(0));
  }
  if (use_bias) {
    mul = 1. - mul;
    for (int i = 0; i < M; ++i) grad[i * N + j] *= mul;
  }
}

template <typename Dtype>
void caffe_gpu_binary_gradient(
    const int axis, const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, Dtype *grad) {
  if (axis == 0) {
    binary_gradient_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, grad);
  } else {
    binary_gradient_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, grad);
  }
}

template <typename Dtype>
__global__ void ternary_gradient_kernel_0(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *delta, Dtype *grad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  Dtype mul = 1. / N;
  for (int j = 0; j < N; ++j) {
    register Dtype val = gpu_abs(in[i * N + j]);
    grad[i * N + j] *=
        mul + Dtype(val <= delta[i] ? 1 : val > Dtype(1) ? 0 : scale[i]);
  }
  if (use_bias) {
    mul = 1. - mul;
    for (int j = 0; j < N; ++j) grad[i * N + j] *= mul;
  }
}

template <typename Dtype>
__global__ void ternary_gradient_kernel_1(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *delta, Dtype *grad) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  Dtype mul = 1. / M;
  for (int i = 0; i < M; ++i) {
    register Dtype val = gpu_abs(in[i * N + j]);
    grad[i * N + j] *=
        mul + Dtype(val <= delta[j] ? 1 : val > Dtype(1) ? 0 : scale[j]);
  }
  if (use_bias) {
    mul = 1. - mul;
    for (int i = 0; i < M; ++i) grad[i * N + j] *= mul;
  }
}

template <typename Dtype>
void caffe_gpu_ternary_gradient(
    const int axis, const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *delta, Dtype *grad) {
  if (axis == 0) {
    ternary_gradient_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, delta, grad);
  } else {
    ternary_gradient_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, delta, grad);
  }
}

template <typename Dtype>
__global__ void clip_kernel(
    const int N, const Dtype min_value, const Dtype max_value, Dtype *X);

template <>
__global__ void clip_kernel<float>(
    const int N, const float min_value, const float max_value, float *X) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    X[x] = fmaxf(fminf(X[x], max_value), min_value);
  }
}

template <>
__global__ void clip_kernel<double>(
    const int N, const double min_value, const double max_value, double *X) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    X[x] = fmax(fmin(X[x], max_value), min_value);
  }
}

template <typename Dtype>
void caffe_gpu_clip(const int N, Dtype min_value, Dtype max_value, Dtype *X) {
  clip_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, min_value, max_value, X);
}

template <typename Dtype>
__global__ void binary_approx_kernel_0(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  Dtype bias = 0;
  if (use_bias) {
    for (int j = 0; j < N; ++j) bias += in[i * N + j];
    bias /= N;
    for (int j = 0; j < N; ++j) in[i * N + j] -= bias;
  }
  scale[i] = 0;
  for (int j = 0; j < N; ++j) scale[i] += gpu_abs(in[i * N + j]);
  scale[i] /= Dtype(N);
  for (int j = 0; j < N; ++j)
    out[i * N + j] = in[i * N + j] >= (Dtype) 0 ? scale[i] : -scale[i];
  if (use_bias) {
    for (int j = 0; j < N; ++j) in[i * N + j] -= bias;
  }
}

template <typename Dtype>
__global__ void binary_approx_kernel_1(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  Dtype bias = 0;
  if (use_bias) {
    for (int i = 0; i < M; ++i) bias += in[i * N + j];
    bias /= M;
    for (int i = 0; i < M; ++i) in[i * N + j] -= bias;
  }
  scale[j] = 0;
  for (int i = 0; i < M; ++i) scale[j] += gpu_abs(in[i * N + j]);
  scale[j] /= Dtype(M);
  for (int i = 0; i < M; ++i)
    out[i * N + j] = in[i * N + j] >= (Dtype) 0 ? scale[j] : -scale[j];
  if (use_bias) {
    for (int i = 0; i < M; ++i) in[i * N + j] += bias;
  }
}

template <typename Dtype>
void caffe_gpu_binary_approx(
    const int axis, const int M, const int N, bool use_bias, Dtype *in,
    Dtype *out, Dtype *scale) {
  if (axis == 0) {
    binary_approx_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, out, scale);
  } else {
    binary_approx_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, out, scale);
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_0(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *delta, Dtype *sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  Dtype bias = 0;
  if (use_bias) {
    for (int j = 0; j < N; ++j) bias += in[i * N + j];
    bias /= N;
    for (int j = 0; j < N; ++j) in[i * N + j] -= bias;
  }
  scale[i] = 0;
  delta[i] = 0;
  sum[i] = 0;
  for (int j = 0; j < N; ++j) delta[i] += gpu_abs(in[i * N + j]);
  delta[i] *= 0.7 / N;
  for (int j = 0; j < N; ++j) {
    Dtype val = gpu_abs(in[i * N + j]);
    if (val > delta[i]) {
      scale[i] += val;
      ++sum[i];
    }
  }
  if (sum[i] > 0) scale[i] /= (Dtype) sum[i];
  for (int j = 0; j < N; ++j) {
    if (in[i * N + j] > delta[i])
      out[i * N + j] = scale[i];
    else if (in[i * N + j] < -delta[i])
      out[i * N + j] = -scale[i];
    else
      out[i * N + j] = 0;
  }
  if (use_bias) {
    for (int j = 0; j < N; ++j) in[i * N + j] += bias;
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_1(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *delta, Dtype *sum) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  Dtype bias = 0;
  if (use_bias) {
    for (int i = 0; i < M; ++i) bias += in[i * N + j];
    bias /= M;
    for (int i = 0; i < M; ++i) in[i * N + j] -= bias;
  }
  delta[j] = 0;
  scale[j] = 0;
  sum[j] = 0;
  for (int i = 0; i < M; ++i) delta[j] += gpu_abs(in[i * N + j]);
  delta[j] *= 0.7 / M;
  for (int i = 0; i < M; ++i) {
    Dtype val = gpu_abs(in[i * N + j]);
    if (val > delta[j]) {
      scale[j] += val;
      ++sum[j];
    }
  }
  if (sum[j] > 0) scale[j] /= (Dtype) sum[j];
  for (int i = 0; i < M; ++i) {
    if (in[i * N + j] > delta[j])
      out[i * N + j] = scale[j];
    else if (in[i * N + j] < -delta[j])
      out[i * N + j] = -scale[j];
    else
      out[i * N + j] = 0;
  }
  if (use_bias) {
    for (int i = 0; i < M; ++i) in[i * N + j] += bias;
  }
}

template <typename Dtype>
void caffe_gpu_ternary_approx(
    const int axis, const int M, const int N, bool use_bias, Dtype *in,
    Dtype *out, Dtype *scale, Dtype *delta, Dtype *sum) {
  if (axis == 0) {
    ternary_approx_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, out, scale, delta, sum);
  } else {
    ternary_approx_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, out, scale, delta, sum);
  }
}
#define INSTANTIATE_BINARY_MATH(Dtype)                                       \
  template void caffe_gpu_binary_gradient<Dtype>(                            \
      const int axis, const int M, const int N, bool use_bias,               \
      const Dtype *in, const Dtype *scale, Dtype *grad);                     \
                                                                             \
  template void caffe_gpu_ternary_gradient<Dtype>(                           \
      const int axis, const int M, const int N, bool use_bias,               \
      const Dtype *in, const Dtype *scale, const Dtype *delta, Dtype *grad); \
                                                                             \
  template void caffe_gpu_clip<Dtype>(                                       \
      const int N, Dtype min_value, Dtype max_value, Dtype *X);              \
                                                                             \
  template void caffe_gpu_binary_approx<Dtype>(                              \
      const int axis, const int M, const int N, bool use_bias, Dtype *in,    \
      Dtype *out, Dtype *scale);                                             \
                                                                             \
  template void caffe_gpu_ternary_approx<Dtype>(                             \
      const int axis, const int M, const int N, bool use_bias, Dtype *in,    \
      Dtype *out, Dtype *scale, Dtype *delta, Dtype *sum);

INSTANTIATE_BINARY_MATH(float);
INSTANTIATE_BINARY_MATH(double);
#endif  // CPU_ONLY
}

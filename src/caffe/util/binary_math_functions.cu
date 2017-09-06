#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include "binary_math_functions.hpp"
namespace caffe {
#ifndef CPU_ONLY

#define KERNEL_FUNC(NAME, OP, PARAMS...) \
template <typename Dtype> \
__global__ void NAME(const int M, const int N, PARAMS)\
{ \
  int i = blockIdx.x * blockDim.x + threadIdx.x; \
  int j = blockIdx.y * blockDim.y + threadIdx.y; \
  if (i < M && j < N) { \
    OP; \
  } \
}
KERNEL_FUNC(
  binary_gradient_kernel_0,
  grad[i * N + j] *=
    mul + (fabs(in[i * N + j]) <= Dtype(1) ? scale[i] : Dtype(0)),
  const Dtype *in, const Dtype *scale, Dtype *grad, const Dtype mul);
KERNEL_FUNC(
  binary_gradient_kernel_1,
  grad[i * N + j] *=
    mul + (fabs(in[i * N + j]) <= Dtype(1) ? scale[j] : Dtype(0)),
  const Dtype *in, const Dtype *scale, Dtype *grad, const Dtype mul);

template<typename Dtype>
void caffe_gpu_binary_gradient(
  const int axis, const int M, const int N,
  const Dtype *in, const Dtype *scale, Dtype *grad) {
  dim2 blocks((M - 1) / 32 + 1, (N - 1) / 16 + 1);
  dim2 threads(32, 16);
  if (axis == 0) {
    const Dtype mul = 1. / N;
    binary_gradient_kernel_0<Dtype> <<< blocks, threads>>> (
      M, N, in, scale, grad, mul);
  }
  else {
    const Dtype mul = 1. / M;
    binary_gradient_kernel_1<Dtype> <<< blocks, threads>>> (
      M, N, in, scale, grad, mul);
  }
}

template<typename Dtype>
__global__ void ternary_gradient_kernel_0(
  const int M, const int N, const Dtype *in, const Dtype *scale,
  const Dtype *delta, Dtype *grad, const Dtype mul) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blcokIdx.y * blockDim.y + threadIdx.y;
  if (x < M && y < N) {
    register Dtype val = fabs(in[x * N + y]);
    grad[x * N + y] *= mul + Dtype(val <= delta[x] ? 1 : val > Dtype(1) ? 0 :
                                   scale[x]);
  }
}

template<typename Dtype>
__global__ void ternary_gradient_kernel_1(
  const int M, const int N, const Dtype *in, const Dtype *scale,
  const Dtype *delta, Dtype *grad, const Dtype mul) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blcokIdx.y * blockDim.y + threadIdx.y;
  if (x < M && y < N) {
    register Dtype val = fabs(in[x * N + y]);
    grad[x * N + y] *= mul + Dtype(val <= delta[y] ? 1 : val > Dtype(1) ? 0 :
                                   scale[y]);
  }
}

template<typename Dtype>
void caffe_gpu_ternary_gradient(
  const int axis, const int M, const int N,
  const Dtype *in, const Dtype *scale, const Dtype *delta, Dtype *grad) {
  dim2 blocks((M - 1) / 32 + 1, (N - 1) / 16 + 1);
  dim2 threads(32, 16);
  if (axis == 0) {
    const Dtype mul = 1. / N;
    ternary_gradient_kernel_0<Dtype> <<< blocks, threads>>> (
      M, N, in, scale, delta, grad, mul);
  }
  else {
    const Dtype mul = 1. / M;
    ternary_gradient_kernel_1<Dtype> <<< blocks, threads>>> (
      M, N, in, scale, delta, grad, mul);
  }
}

template <Dtype>
__global__ void clip_kernel(
  const int N, const Dtype min_value, const Dtype max_value, Dtype *X) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    X[x] = max(min(X[x], max_value), min_value);
  }
}

template<typename Dtype>
void caffe_gpu_clip(const int N, Dtype min_value, Dtype max_value, Dtype *X) {
  clip_kernel<Dtype> <<< CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
    N, min_value, max_value, X);
}

KERNEL_FUNC(
  asum_kernel_0,
  asum[i] += fabs(in[i * N + j]),
  const Dtype *in, Dtype *asum);
KERNEL_FUNC(
  asum_kernel_1,
  asum[j] += fabs(in[i * N + j]),
  const Dtype *in, Dtype *asum);

KERNEL_FUNC(
  set_kernel_0,
  out[i * N + j] = in[i * N + j] >= 0 ? asum[j] : -asum[j],
  const Dtype *in, const Dtype *asum, Dtype *out);
KERNEL_FUNC(
  set_kernel_1,
  out[i * N + j] = in[i * N + j] >= 0 ? asum[i] : -asum[i],
  const Dtype *in, const Dtype *asum, Dtype *out);

template <typename Dtype>
void caffe_gpu_binary_approx(
  const int axis, const int M, const int N, const Dtype* in,
  Dtype* out, Dtype *scale) {
  dim2 blocks((M - 1) / 32 + 1, (N - 1) / 16 + 1);
  dim2 threads(32, 16);
  if (axis == 0) {
    caffe_gpu_set<Dtype>(M, Dtype(0), scale);
    asum_kernel_0<Dtype> <<< blocks, threads>>> (M, N, in, scale);
    caffe_gpu_scal<Dtype>(M, Dtype(1. / N), scale);
    set_kernel_0<Dtype> <<< blocks, threads > (M, N, in, scale, out);
  }
  else {
    caffe_gpu_set<Dtype>(N, Dtype(0), scale);
    asum_kernel_1<Dtype> <<<blocks, threads>>> (M, N, in, scale);
    caffe_gpu_scal<Dtype>(N, Dtype(1. / M), scale);
    set_kernel_1<Dtype> <<< blocks, threads > (M, N, in, scale, out);
  }
}

KERNEL_FUNC(
ternary_kernel_0, {
  if (in[i * N + j] > delta[i])
    out[i * N + j] = Dtype(1), scale[i] += in[i * N + j], ++sum[i];
  else if (in[i * N + j] < -delta[i])
    out[i * N + j] = Dtype(-1), scale[i] -= in[i * N + j], ++sum[i];
  else
    out[i * N + j] = 0;
},
const Dtype *in, const Dtype *delta, const Dtype *scale,
const Dtype *sum, Dtype *out);

KERNEL_FUNC(
ternary_kernel_1, {
  if (in[i * N + j] > delta[j])
    out[i * N + j] = Dtype(1), scale[j] += in[i * N + j], ++sum[j];
  else if (in[i * N + j] < -delta[i])
    out[i * N + j] = Dtype(-1), scale[j] -= in[i * N + j], ++sum[j];
  else
    out[i * N + j] = 0;
},
const Dtype *in, const Dtype *delta, const Dtype *scale,
const Dtype *sum, Dtype *out);

KERNEL_FUNC(
  mul_eq_kernel_0,
  out[i * N + j] *= scale[i],
  const Dtype *scale, Dtype *out);

KERNEL_FUNC(
  mul_eq_kernel_1,
  out[i * N + j] *= scale[j],
  const Dtype *scale, Dtype *out);

template <typename Dtype>
__global__ void div_eq_kernel(const int N, const Dtype *X, Dtype* Y) {
  int i = threadIdx.x;
  if (i < N) {
    if (X[i] > 0)
      Y[i] /= X[i];
  }
}
template <typename Dtype>
void caffe_gpu_ternary_approx(
  const int axis, const int M, const int N, const Dtype *in,
  Dtype* out, Dtype *scale, Dtype *delta) {
  dim2 blocks((M - 1) / 32 + 1, (N - 1) / 16 + 1);
  dim2 threads(32, 16);
  if (axis == 0) {
    caffe_gpu_set<Dtype>(M, Dtype(0), scale);
    caffe_gpu_set<Dtype>(M, Dtype(0), delta);
    thrust::device_vector<Dtype> sum(M, 0);
    asum_kernel_0<Dtype> <<< blocks, threads>>> (M, N, in, delta);
    caffe_gpu_scal<Dtype>(M, Dtype(0.7 / N), delta);
    ternary_kernel_0<Dtype> <<< blocks, threads>>>(
      M, N, in, scale, delta, sum, out);
    div_eq_kernel<Dtype> <<< CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(
      M, sum.data(), scale);
    mul_eq_kernel_0<Dtype> <<< blocks, threads>>> (M, N, scale, out);
  }
  else {
    caffe_gpu_set<Dtype>(N, Dtype(0), scale);
    caffe_gpu_set<Dtype>(N, Dtype(0), delta);
    thrust::device_vector<Dtype> sum(M, 0);
    asum_kernel_1<Dtype> <<< blocks, threads>>> (M, N, in, delta);
    caffe_gpu_scal<Dtype>(M, Dtype(0.7 / M), delta);
    ternary_kernel_1<Dtype> <<< blocks, threads>>>(
      M, N, in, scale, delta, sum, out);
    div_eq_kernel<Dtype> <<< CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, sum.data(), scale);
    mul_eq_kernel_0<Dtype> <<< blocks, threads>>> (M, N, scale, out);
  }
}

#define INSTANTIATE_BINARY_MATH(Dtype) \
template void caffe_gpu_binary_gradient<Dtype>( \
  const int axis, const int M, const int N, \
  const Dtype *in, const Dtype *scale, Dtype *grad);  \
  \
template void caffe_gpu_ternary_gradient<Dtype>(  \
  const int axis, const int M, const int N, \
  const Dtype *in, const Dtype *scale, const Dtype *delta, Dtype *grad);  \
  \
template void caffe_gpu_clip<Dtype>(  \
  const int N, Dtype min_value, Dtype max_value, Dtype *X); \
  \
template void caffe_gpu_binary_approx<Dtype>( \
  const int axis, const int M, const int N, const Dtype* in,  \
  Dtype* out, Dtype *scale);  \
  \
template void caffe_gpu_ternary_approx<Dtype>(  \
  const int axis, const int M, const int N, const Dtype *in,  \
  Dtype* out, Dtype *scale, Dtype *delta);

INSTANTIATE_BINARY_MATH(float);
INSTANTIATE_BINARY_MATH(double);
#endif // CPU_ONLY
}

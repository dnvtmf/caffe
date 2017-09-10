#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#ifndef CPU_ONLY
#define THREADS_NUM_ROW 16
#define THREADS_NUM_COL 32
const dim3 dim_threads(THREADS_NUM_ROW, THREADS_NUM_COL, 1);

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }

template <typename Dtype>
__global__ void binary_gradient_kernel_0(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, Dtype *grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int i = idx / N;
  int j = idx % N;
  const Dtype mul = (Dtype) 1. / N;
  grad[i * N + j] *=
      mul + (gpu_abs(in[i * N + j]) <= Dtype(1) ? scale[i] : Dtype(0));
  if (use_bias) grad[i * N + j] *= (Dtype) 1. - mul;
}

template <typename Dtype>
__global__ void binary_gradient_kernel_1(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, Dtype *grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int i = idx / N;
  int j = idx % N;
  const Dtype mul = (Dtype) 1. / M;
  grad[i * N + j] *=
      mul + (gpu_abs(in[i * N + j]) <= Dtype(1) ? scale[j] : Dtype(0));
  if (use_bias) grad[i * N + j] *= (Dtype) 1. - mul;
}

template <typename Dtype>
void caffe_gpu_binary_gradient(
    const int axis, const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, Dtype *grad) {
  if (axis == 0) {
    binary_gradient_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, grad);
  } else {
    binary_gradient_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, grad);
  }
}

template <typename Dtype>
__global__ void ternary_gradient_kernel_0(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *delta, Dtype *grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int i = idx / N;
  int j = idx % N;
  Dtype mul = 1. / N;
  Dtype val = gpu_abs(in[i * N + j]);
  grad[i * N + j] *=
      mul + Dtype(val <= delta[j] ? 1 : val > Dtype(1) ? 0 : scale[j]);
  if (use_bias) grad[i * N + j] *= (Dtype) 1. - mul;
}

template <typename Dtype>
__global__ void ternary_gradient_kernel_1(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *delta, Dtype *grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int i = idx / N;
  int j = idx % N;
  Dtype mul = 1. / M;
  Dtype val = gpu_abs(in[i * N + j]);
  grad[i * N + j] *=
      mul + Dtype(val <= delta[j] ? 1 : val > Dtype(1) ? 0 : scale[j]);
  if (use_bias) grad[i * N + j] *= (Dtype) 1. - mul;
}

template <typename Dtype>
void caffe_gpu_ternary_gradient(
    const int axis, const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *delta, Dtype *grad) {
  if (axis == 0) {
    ternary_gradient_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, delta, grad);
  } else {
    ternary_gradient_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
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
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  const int id = threadIdx.y;
  const int j_start = id * N / THREADS_NUM_COL;
  const int j_end = (id + 1) * N / THREADS_NUM_COL;
  if (j_start == j_end) return;
  volatile __shared__ Dtype temp[THREADS_NUM_COL];
  volatile __shared__ Dtype bias;
  if (use_bias) {
    temp[id] = 0;
    for (int j = j_start; j < j_end; ++j) temp[id] += in[i * N + j];
    __syncthreads();
    if (id == 0) {
      bias = 0;
      for (int idx = 0; idx < THREADS_NUM_COL; ++idx) bias += temp[idx];
      bias /= N;
    }
    __syncthreads();
    for (int j = j_start; j < j_end; ++j) in[i * N + j] -= bias;
  }
  temp[id] = 0;
  for (int j = j_start; j < j_end; ++j) temp[id] += gpu_abs(in[i * N + j]);
  __syncthreads();
  if (id == 0) {
    scale[i] = 0;
    for (int idx = 0; idx < THREADS_NUM_COL; ++idx) scale[i] += temp[idx];
    scale[i] /= Dtype(N);
  }
  __syncthreads();
  for (int j = j_start; j < j_end; ++j)
    out[i * N + j] = in[i * N + j] >= (Dtype) 0 ? scale[i] : -scale[i];
  if (use_bias) {
    for (int j = j_start; j < j_end; ++j) in[i * N + j] += bias;
  }
}

template <typename Dtype>
__global__ void binary_approx_kernel_1(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  const int id = threadIdx.y;
  const int i_start = id * M / THREADS_NUM_COL;
  const int i_end = (id + 1) * M / THREADS_NUM_COL;
  volatile __shared__ Dtype temp[THREADS_NUM_COL];
  volatile __shared__ Dtype bias;
  if (use_bias) {
    temp[id] = 0;
    for (int i = i_start; i < i_end; ++i) temp[id] += in[i * N + j];
    __syncthreads();
    if (id == 0) {
      bias = 0;
      for (int idx = 0; idx < THREADS_NUM_COL; ++idx) bias += temp[idx];
      bias /= M;
    }
    __syncthreads();
    for (int i = i_start; i < i_end; ++i) in[i * N + j] -= bias;
  }
  temp[id] = 0;
  for (int i = i_start; i < i_end; ++i) temp[id] += gpu_abs(in[i * N + j]);
  __syncthreads();
  if (id == 0) {
    scale[j] = 0;
    for (int idx = 0; idx < THREADS_NUM_COL; ++idx) scale[j] += temp[idx];
    scale[j] /= (Dtype) M;
  }
  __syncthreads();
  for (int i = i_start; i < i_end; ++i)
    out[i * N + j] = in[i * N + j] >= (Dtype) 0 ? scale[j] : -scale[j];
  if (use_bias) {
    for (int i = i_start; i < i_end; ++i) in[i * N + j] += bias;
  }
}

template <typename Dtype>
void caffe_gpu_binary_approx(
    const int axis, const int M, const int N, bool use_bias, Dtype *in,
    Dtype *out, Dtype *scale) {
  if (axis == 0) {
    dim3 dim_blocks((M - 1) / THREADS_NUM_ROW + 1, 1, 1);
    binary_approx_kernel_0<Dtype>
        <<<dim_blocks, dim_threads>>>(M, N, use_bias, in, out, scale);
  } else {
    dim3 dim_blocks((N - 1) / THREADS_NUM_ROW + 1, 1, 1);
    binary_approx_kernel_1<Dtype>
        <<<dim_blocks, dim_threads>>>(M, N, use_bias, in, out, scale);
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_0(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *delta, Dtype *sum) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  const int id = threadIdx.y;
  const int j_start = id * N / THREADS_NUM_COL;
  const int j_end = (id + 1) * N / THREADS_NUM_COL;
  if (j_start == j_end) return;
  volatile __shared__ Dtype temp[THREADS_NUM_COL];
  volatile __shared__ Dtype temp2[THREADS_NUM_COL];
  volatile __shared__ Dtype bias;
  if (use_bias) {
    temp[id] = 0;
    for (int j = j_start; j < j_end; ++j) temp[id] += in[i * N + j];
    __syncthreads();
    if (id == 0) {
      bias = 0;
      for (int idx = 0; idx < THREADS_NUM_COL; ++idx) bias += temp[idx];
      bias /= N;
    }
    __syncthreads();
    for (int j = j_start; j < j_end; ++j) in[i * N + j] -= bias;
  }
  temp[id] = 0;
  for (int j = j_start; j < j_end; ++j) temp[id] += gpu_abs(in[i * N + j]);
  __syncthreads();
  if (id == 0) {
    delta[i] = 0;
    for (int idx = 0; idx < THREADS_NUM_COL; ++idx) delta[i] += temp[idx];
    delta[i] *= 0.7 / N;
  }
  temp[id] = 0;
  temp2[id] = 0;
  for (int j = j_start; j < j_end; ++j) {
    const Dtype val = gpu_abs(in[i * N + j]);
    if (val > delta[i]) {
      temp[id] += val;
      ++temp2[id];
    }
  }
  __syncthreads();
  if (id == 0) {
    scale[i] = 0;
    sum[i] = 0;
    for (int idx = 0; idx < THREADS_NUM_COL; ++idx) {
      scale[i] += temp[idx];
      sum[i] += temp2[idx];
    }
    if (sum[i] > 0) scale[i] /= (Dtype) sum[i];
  }
  __syncthreads();
  for (int j = j_start; j < j_end; ++j) {
    if (in[i * N + j] > delta[i])
      out[i * N + j] = scale[i];
    else if (in[i * N + j] < -delta[i])
      out[i * N + j] = -scale[i];
    else
      out[i * N + j] = 0;
  }
  if (use_bias) {
    for (int j = j_start; j < j_end; ++j) in[i * N + j] += bias;
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_1(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *delta, Dtype *sum) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  const int id = threadIdx.y;
  const int i_start = id * M / THREADS_NUM_COL;
  const int i_end = (id + 1) * M / THREADS_NUM_COL;
  if (i_start == i_end) return;
  volatile __shared__ Dtype temp[THREADS_NUM_COL];
  volatile __shared__ Dtype temp2[THREADS_NUM_COL];
  volatile __shared__ Dtype bias;
  if (use_bias) {
    temp[id] = 0;
    for (int i = i_start; i < i_end; ++i) temp[id] += in[i * N + j];
    __syncthreads();
    if (id == 0) {
      bias = 0;
      for (int idx = 0; idx < THREADS_NUM_COL; ++idx) bias += temp[idx];
      bias /= M;
    }
    for (int i = i_start; i < i_end; ++i) in[i * N + j] -= bias;
  }

  scale[j] = 0;
  sum[j] = 0;
  temp[id] = 0;
  for (int i = i_start; i < i_end; ++i) temp[id] += gpu_abs(in[i * N + j]);
  __syncthreads();
  if (id == 0) {
    delta[j] = 0;
    for (int idx = 0; idx < THREADS_NUM_COL; ++idx) delta[j] += temp[idx];
    delta[j] *= 0.7 / M;
  }
  __syncthreads();
  temp[id] = 0;
  temp2[id] = 0;
  for (int i = i_start; i < i_end; ++i) {
    Dtype val = gpu_abs(in[i * N + j]);
    if (val > delta[j]) {
      scale[j] += val;
      ++sum[j];
    }
  }
  if (sum[j] > 0) scale[j] /= (Dtype) sum[j];
  for (int i = i_start; i < i_end; ++i) {
    if (in[i * N + j] > delta[j])
      out[i * N + j] = scale[j];
    else if (in[i * N + j] < -delta[j])
      out[i * N + j] = -scale[j];
    else
      out[i * N + j] = 0;
  }
  if (use_bias) {
    for (int i = i_start; i < i_end; ++i) in[i * N + j] += bias;
  }
}

template <typename Dtype>
void caffe_gpu_ternary_approx(
    const int axis, const int M, const int N, bool use_bias, Dtype *in,
    Dtype *out, Dtype *scale, Dtype *delta, Dtype *sum) {
  if (axis == 0) {
    dim3 dim_blocks((M - 1) / THREADS_NUM_ROW + 1, 1, 1);
    ternary_approx_kernel_0<Dtype><<<dim_blocks, dim_threads>>>(
        M, N, use_bias, in, out, scale, delta, sum);
  } else {
    dim3 dim_blocks((N - 1) / THREADS_NUM_ROW + 1, 1, 1);
    ternary_approx_kernel_1<Dtype><<<dim_blocks, dim_threads>>>(
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

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#define WARP_SIZE 32

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }
inline __device__ float sign(float val, float s) { return copysignf(val, s); }
inline __device__ double sign(double val, double s) { return copysign(val, s); }

template <typename Dtype>
__global__ void binary_gradient_kernel_0(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *bias, Dtype *grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int i = idx / N;
  const Dtype mul = (Dtype) 1. / (Dtype) N;
  const Dtype val = gpu_abs(in[idx] - Dtype(use_bias) * bias[i]);
  grad[idx] *= mul + Dtype(val < Dtype(1)) * scale[i];
  if (use_bias) grad[idx] *= (Dtype) 1. - mul;
}

template <typename Dtype>
__global__ void binary_gradient_kernel_1(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *bias, Dtype *grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int j = idx % N;
  const Dtype mul = (Dtype) 1. / (Dtype) M;
  const Dtype val = gpu_abs(in[idx] - Dtype(use_bias) * bias[j]);
  grad[idx] *= mul + Dtype(val < Dtype(1)) * scale[j];
  if (use_bias) grad[idx] *= (Dtype) 1. - mul;
}

template <typename Dtype>
void caffe_gpu_binary_gradient(
    const int axis, const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *bias, Dtype *grad) {
  if (axis == 0) {
    binary_gradient_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, bias, grad);
  } else {
    binary_gradient_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, bias, grad);
  }
}

template <typename Dtype>
__global__ void ternary_gradient_kernel_0(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *bias, const Dtype *delta, Dtype *grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int i = idx / N;
  Dtype mul = (Dtype) 1. / (Dtype) N;
  Dtype val = gpu_abs(in[idx] - Dtype(use_bias) * bias[i]);
  grad[idx] *=
      mul + Dtype(val < Dtype(1)) * Dtype(val <= delta[i] ? 1 : scale[i]);
  if (use_bias) grad[idx] *= (Dtype) 1. - mul;
}

template <typename Dtype>
__global__ void ternary_gradient_kernel_1(
    const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *bias, const Dtype *delta, Dtype *grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int j = idx % N;
  Dtype mul = (Dtype) 1. / (Dtype) M;
  Dtype val = gpu_abs(in[idx] - Dtype(use_bias) * bias[j]);
  grad[idx] *=
      mul + Dtype(val < Dtype(1)) * Dtype(val <= delta[j] ? 1 : scale[j]);
  if (use_bias) grad[idx] *= (Dtype) 1. - mul;
}

template <typename Dtype>
void caffe_gpu_ternary_gradient(
    const int axis, const int M, const int N, bool use_bias, const Dtype *in,
    const Dtype *scale, const Dtype *bias, const Dtype *delta, Dtype *grad) {
  if (axis == 0) {
    ternary_gradient_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, bias, delta, grad);
  } else {
    ternary_gradient_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, bias, delta, grad);
  }
}

template <typename Dtype>
__global__ void clip_kernel(
    const int N, const Dtype min_value, const Dtype max_value, Dtype *X);

template <>
__global__ void clip_kernel<float>(
    const int N, const float min_value, const float max_value, float *X) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= N) return;
  X[x] = fmaxf(fminf(X[x], max_value), min_value);
}

template <>
__global__ void clip_kernel<double>(
    const int N, const double min_value, const double max_value, double *X) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= N) return;
  X[x] = fmax(fmin(X[x], max_value), min_value);
}

template <typename Dtype>
void caffe_gpu_clip(const int N, Dtype min_value, Dtype max_value, Dtype *X) {
  clip_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, min_value, max_value, X);
}

template <typename Dtype>
__global__ void binary_approx_kernel_0(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *bias) {
  const int i = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val;
  if (use_bias) {
    // bias[i] = \frac{1}{N} \sum_{j=0}^{N-1}{in[i * N + j}
    val = 0;
    for (int j = id; j < N; j += blockDim.x) val += in[i * N + j];
    if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
    __syncthreads();
    if (id < WARP_SIZE) {
#pragma unroll
      for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
        val += temp[k];
      temp[id] = val;
    }
    if (id == 0) {
      for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
      bias[i] = val / Dtype(N);
    }
    __syncthreads();
    // in[i * N + j] -= bias[i];
    for (int j = id; j < N; j += blockDim.x) in[i * N + j] -= bias[i];
  }
  // scale[i] = \frac{1}{N} \sum_{j=0}^{N-1}{|in[i * N + j|}
  val = 0;
  for (int j = id; j < N; j += blockDim.x) val += gpu_abs(in[i * N + j]);
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k];
    temp[id] = val;
  }
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    scale[i] = val / Dtype(N);
  }
  __syncthreads();
  // out
  for (int j = id; j < N; j += blockDim.x)
    out[i * N + j] = sign(scale[i], in[i * N + j]);
  // add bias
  if (use_bias) {
    for (int j = id; j < N; j += blockDim.x) {
      in[i * N + j] += bias[i];
      out[i * N + j] += bias[i];
    }
  }
}

template <typename Dtype>
__global__ void binary_approx_kernel_1(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *bias) {
  const int j = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val;
  if (use_bias) {
    val = 0;
    for (int i = id; i < M; i += blockDim.x) val += in[i * N + j];
    if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
    __syncthreads();
    if (id < WARP_SIZE) {
#pragma unroll
      for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
        val += temp[k];
      temp[id] = val;
    }
    if (id == 0) {
      for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
      bias[j] = val / Dtype(M);
    }
    __syncthreads();
    for (int i = id; i < M; i += blockDim.x) in[i * N + j] -= bias[j];
  }
  val = 0;
  for (int i = id; i < M; i += blockDim.x) val += gpu_abs(in[i * N + j]);
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k];
    temp[id] = val;
  }
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    scale[j] = val / Dtype(M);
  }
  __syncthreads();
  for (int i = id; i < M; i += blockDim.x)
    out[i * N + j] = sign(scale[j], in[i * N + j]);
  if (use_bias) {
    for (int i = id; i < M; i += blockDim.x) {
      in[i * N + j] += bias[j];
      out[i * N + j] += bias[j];
    }
  }
}

template <typename Dtype>
void caffe_gpu_binary_approx(
    const int axis, const int M, const int N, bool use_bias, Dtype *in,
    Dtype *out, Dtype *scale, Dtype *bias) {
  if (axis == 0) {
    binary_approx_kernel_0<Dtype>
        <<<M, CAFFE_CUDA_NUM_THREADS>>>(M, N, use_bias, in, out, scale, bias);
  } else {
    binary_approx_kernel_1<Dtype>
        <<<N, CAFFE_CUDA_NUM_THREADS>>>(M, N, use_bias, in, out, scale, bias);
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_0(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *bias, Dtype *delta) {
  const int i = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  volatile __shared__ Dtype temp2[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val, val2;
  if (use_bias) {
    val = 0;
    for (int j = id; j < N; j += blockDim.x) val += in[i * N + j];
    if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
    __syncthreads();
    if (id < WARP_SIZE) {
#pragma unroll
      for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
        val += temp[k];
      temp[id] = val;
    }
    if (id == 0) {
      for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
      bias[i] = val / Dtype(N);
    }
    __syncthreads();
    for (int j = id; j < N; j += blockDim.x) in[i * N + j] -= bias[i];
  }
  // delta[i]
  val = 0;
  for (int j = id; j < N; j += blockDim.x) val += gpu_abs(in[i * N + j]);
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k];
    temp[id] = val;
  }
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    delta[i] = val * Dtype(0.7) / Dtype(N);
  }
  __syncthreads();
  // scale
  val = 0;
  val2 = 0;
  for (int j = id; j < N; j += blockDim.x) {
    const Dtype v = gpu_abs(in[i * N + j]);
    val += Dtype(v > delta[i]) * v;
    val2 += Dtype(v > delta[i]);
  }
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val, temp2[id - WARP_SIZE] = val2;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k], val2 += temp2[k];
    temp[id] = val, temp2[id] = val2;
  }
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k], val2 += temp2[k];
    scale[i] = val / val2;
  }
  __syncthreads();
  // out
  for (int j = id; j < N; j += blockDim.x) {
    out[i * N + j] = Dtype(in[i * N + j] > delta[i]) * scale[i] +
                     Dtype(in[i * N + j] < -delta[i]) * -scale[i];
  }
  if (use_bias) {
    for (int j = id; j < N; j += blockDim.x) {
      out[i * N + j] += bias[i];
      in[i * N + j] += bias[i];
    }
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_1(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *bias, Dtype *delta) {
  const int j = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  volatile __shared__ Dtype temp2[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val, val2;
  if (use_bias) {
    val = 0;
    for (int i = id; i < M; i += blockDim.x) val += in[i * N + j];
    if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
    __syncthreads();
    if (id < WARP_SIZE) {
#pragma unroll
      for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
        val += temp[k];
      temp[id] = val;
    }
    if (id == 0) {
      for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
      bias[j] = val / Dtype(M);
    }
    __syncthreads();
    for (int i = id; i < M; i += blockDim.x) in[i * N + j] -= bias[j];
  }
  // delta
  val = 0;
  for (int i = id; i < M; i += blockDim.x) val += gpu_abs(in[i * N + j]);
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k];
    temp[id] = val;
  }
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    delta[j] = val * Dtype(0.7) / Dtype(M);
  }
  __syncthreads();
  // scale
  val = 0;
  val2 = 0;
  for (int i = id; i < M; i += blockDim.x) {
    const Dtype v = gpu_abs(in[i * N + j]);
    val += Dtype(v > delta[j]) * v;
    val2 += Dtype(v > delta[j]);
  }
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val, temp2[id - WARP_SIZE] = val2;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k], val2 += temp2[k];
    temp[id] = val, temp2[id] = val2;
  }
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k], val2 += temp2[k];
    scale[j] = val / val2;
  }
  __syncthreads();
  // out
  for (int i = id; i < M; i += blockDim.x) {
    out[i * N + j] = Dtype(in[i * N + j] > delta[j]) * scale[j] +
                     Dtype(in[i * N + j] < -delta[j]) * -scale[j];
  }
  if (use_bias) {
    for (int i = id; i < M; i += blockDim.x) {
      out[i * N + j] += bias[j];
      in[i * N + j] += bias[j];
    }
  }
}

template <typename Dtype>
void caffe_gpu_ternary_approx(
    const int axis, const int M, const int N, bool use_bias, Dtype *in,
    Dtype *out, Dtype *scale, Dtype *bias, Dtype *delta) {
  if (axis == 0) {
    ternary_approx_kernel_0<Dtype><<<M, CAFFE_CUDA_NUM_THREADS>>>(
        M, N, use_bias, in, out, scale, bias, delta);
  } else {
    ternary_approx_kernel_1<Dtype><<<N, CAFFE_CUDA_NUM_THREADS>>>(
        M, N, use_bias, in, out, scale, bias, delta);
  }
}
#define INSTANTIATE_BINARY_MATH(Dtype)                                      \
  template void caffe_gpu_binary_gradient<Dtype>(                           \
      const int axis, const int M, const int N, bool use_bias,              \
      const Dtype *in, const Dtype *scale, const Dtype *bias, Dtype *grad); \
                                                                            \
  template void caffe_gpu_ternary_gradient<Dtype>(                          \
      const int axis, const int M, const int N, bool use_bias,              \
      const Dtype *in, const Dtype *scale, const Dtype *bias,               \
      const Dtype *delta, Dtype *grad);                                     \
                                                                            \
  template void caffe_gpu_clip<Dtype>(                                      \
      const int N, Dtype min_value, Dtype max_value, Dtype *X);             \
                                                                            \
  template void caffe_gpu_binary_approx<Dtype>(                             \
      const int axis, const int M, const int N, bool use_bias, Dtype *in,   \
      Dtype *out, Dtype *scale, Dtype *bias);                               \
                                                                            \
  template void caffe_gpu_ternary_approx<Dtype>(                            \
      const int axis, const int M, const int N, bool use_bias, Dtype *in,   \
      Dtype *out, Dtype *scale, Dtype *bias, Dtype *delta);

INSTANTIATE_BINARY_MATH(float);
INSTANTIATE_BINARY_MATH(double);
}

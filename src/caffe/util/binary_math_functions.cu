#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#define WARP_SIZE 32

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }
inline __device__ float sign(float val, float s) { return copysignf(val, s); }
inline __device__ double sign(double val, double s) { return copysign(val, s); }

template <typename Dtype>
__global__ void binary_gradient_kernel_0(const int M, const int N,
    bool use_bias, const Dtype *in, const Dtype *scale, const Dtype *bias,
    Dtype *grad, const Dtype mul) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int i = idx / N;
  grad[idx] *= mul + Dtype(gpu_abs(in[idx]) < Dtype(1)) * scale[i];
  /*
  const Dtype val = gpu_abs(in[idx] - Dtype(use_bias) * bias[i]);
  grad[idx] *= mul + Dtype(val < Dtype(1)) * scale[i];
  if (use_bias) grad[idx] *= (Dtype) 1. - mul;
  */
}

template <typename Dtype>
__global__ void binary_gradient_kernel_1(const int M, const int N,
    bool use_bias, const Dtype *in, const Dtype *scale, const Dtype *bias,
    Dtype *grad, const Dtype mul) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int j = idx % N;
  grad[idx] *= mul + Dtype(gpu_abs(in[idx]) < Dtype(1)) * scale[j];
  /*
  const Dtype val = gpu_abs(in[idx] - Dtype(use_bias) * bias[j]);
  grad[idx] *= mul + Dtype(val < Dtype(1)) * scale[j];
  if (use_bias) grad[idx] *= (Dtype) 1. - mul;
  */
}

template <typename Dtype>
void caffe_gpu_binary_gradient(const int axis, const int M, const int N,
    bool use_bias, const Dtype *in, const Dtype *scale, const Dtype *bias,
    Dtype *grad) {
  if (axis == 0) {
    binary_gradient_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, bias, grad, 1. / N);
  } else {
    binary_gradient_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, bias, grad, 1. / M);
  }
}

template <typename Dtype>
__global__ void ternary_gradient_kernel_0(const int M, const int N,
    bool use_bias, const Dtype *in, const Dtype *scale, const Dtype *bias,
    const Dtype *delta, Dtype *grad, const Dtype mul) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int i = idx / N;
  // Dtype val = gpu_abs(in[idx] - Dtype(use_bias) * bias[i]);
  Dtype val = gpu_abs(in[idx]);
  grad[idx] *=
      mul + Dtype(val < Dtype(1)) * Dtype(val <= delta[i] ? 1 : scale[i]);
  // if (use_bias) grad[idx] *= (Dtype) 1. - mul;
}

template <typename Dtype>
__global__ void ternary_gradient_kernel_1(const int M, const int N,
    bool use_bias, const Dtype *in, const Dtype *scale, const Dtype *bias,
    const Dtype *delta, Dtype *grad, const Dtype mul) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int j = idx % N;
  // Dtype val = gpu_abs(in[idx] - Dtype(use_bias) * bias[j]);
  Dtype val = gpu_abs(in[idx]);
  grad[idx] *=
      mul + Dtype(val < Dtype(1)) * Dtype(val <= delta[j] ? 1 : scale[j]);
  // if (use_bias) grad[idx] *= (Dtype) 1. - mul;
}

template <typename Dtype>
void caffe_gpu_ternary_gradient(const int axis, const int M, const int N,
    bool use_bias, const Dtype *in, const Dtype *scale, const Dtype *bias,
    const Dtype *delta, Dtype *grad) {
  if (axis == 0) {
    ternary_gradient_kernel_0<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, bias, delta, grad, 1. / N);
  } else {
    ternary_gradient_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(
            M, N, use_bias, in, scale, bias, delta, grad, 1. / M);
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
__global__ void binary_approx_kernel_0(const int M, const int N, bool use_bias,
    Dtype *in, Dtype *out, Dtype *scale, Dtype *bias) {
  const int i  = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val;
  if (use_bias) {
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
    scale[i]   = val / Dtype(N);
  }
  __syncthreads();
  // out
  for (int j       = id; j < N; j += blockDim.x)
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
__global__ void binary_approx_kernel_1(const int M, const int N, bool use_bias,
    Dtype *in, Dtype *out, Dtype *scale, Dtype *bias) {
  const int j  = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val;
  if (use_bias) {
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
  __syncthreads();
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    scale[j]   = val / Dtype(M);
  }
  __syncthreads();
  for (int i       = id; i < M; i += blockDim.x)
    out[i * N + j] = sign(scale[j], in[i * N + j]);
  if (use_bias) {
    for (int i = id; i < M; i += blockDim.x) {
      in[i * N + j] += bias[j];
      out[i * N + j] += bias[j];
    }
  }
}

template <typename Dtype>
void caffe_gpu_binary_approx(const int axis, const int M, const int N,
    bool use_bias, Dtype *in, Dtype *out, Dtype *scale, Dtype *bias) {
  if (axis == 0) {
    binary_approx_kernel_0<Dtype>
        <<<M, CAFFE_CUDA_NUM_THREADS>>>(M, N, use_bias, in, out, scale, bias);
  } else {
    binary_approx_kernel_1<Dtype>
        <<<N, CAFFE_CUDA_NUM_THREADS>>>(M, N, use_bias, in, out, scale, bias);
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_0(const int M, const int N, bool use_bias,
    Dtype *in, Dtype *out, Dtype *scale, Dtype *bias, Dtype *delta) {
  const int i  = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  volatile __shared__ Dtype temp2[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val, val2;
  if (use_bias) {
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
  __syncthreads();
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    delta[i]   = val * Dtype(0.5) / Dtype(N);
  }
  __syncthreads();
  // scale
  val  = 0;
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
  __syncthreads();
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k], val2 += temp2[k];
    scale[i]   = val / val2;
  }
  __syncthreads();
  // out
  for (int j = id; j < N; j += blockDim.x) {
    out[i * N + j] =
        (Dtype(in[i * N + j] > delta[i]) - Dtype(in[i * N + j] < -delta[i])) *
        scale[i];
  }
  if (use_bias) {
    for (int j = id; j < N; j += blockDim.x) {
      out[i * N + j] += bias[i];
      in[i * N + j] += bias[i];
    }
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_1(const int M, const int N, bool use_bias,
    Dtype *in, Dtype *out, Dtype *scale, Dtype *bias, Dtype *delta) {
  const int j  = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  volatile __shared__ Dtype temp2[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val, val2;
  if (use_bias) {
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
  __syncthreads();
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    delta[j]   = val * Dtype(0.5) / Dtype(M);
  }
  __syncthreads();
  // scale
  val  = 0;
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
  __syncthreads();
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k], val2 += temp2[k];
    scale[j]   = val / val2;
  }
  __syncthreads();
  // out
  for (int i = id; i < M; i += blockDim.x) {
    out[i * N + j] =
        (Dtype(in[i * N + j] > delta[j]) - Dtype(in[i * N + j] < -delta[j])) *
        scale[j];
  }
  if (use_bias) {
    for (int i = id; i < M; i += blockDim.x) {
      out[i * N + j] += bias[j];
      in[i * N + j] += bias[j];
    }
  }
}

template <typename Dtype>
void caffe_gpu_ternary_approx(const int axis, const int M, const int N,
    bool use_bias, Dtype *in, Dtype *out, Dtype *scale, Dtype *bias,
    Dtype *delta) {
  if (axis == 0) {
    ternary_approx_kernel_0<Dtype><<<M, CAFFE_CUDA_NUM_THREADS>>>(
        M, N, use_bias, in, out, scale, bias, delta);
  } else {
    ternary_approx_kernel_1<Dtype><<<N, CAFFE_CUDA_NUM_THREADS>>>(
        M, N, use_bias, in, out, scale, bias, delta);
  }
}

template <typename Dtype>
__global__ void mean_center_kernel(
    const int c_out, const int c_in, const int wh, Dtype *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int x = i / wh;
  const int y = i % wh;
  if (x >= c_out) return;
  Dtype mean  = 0;
  Dtype *p_in = in + x * c_in * wh + y;
  for (int i = 0; i < c_in; ++i, p_in += wh) mean += *p_in;
  mean /= c_in;
  p_in = in + x * c_in * wh + y;
  for (int i = 0; i < c_in; ++i, p_in += wh) *p_in -= mean;
}

template <typename Dtype>
void mean_center(const int c_out, const int c_in, const int wh, Dtype *in) {
  mean_center_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(c_out * wh), CAFFE_CUDA_NUM_THREADS>>>(
          c_out, c_in, wh, in);
}

template <>
void caffe_gpu_swap<float>(const int N, float *X, float *Y) {
  CUBLAS_CHECK(cublasSswap(Caffe::cublas_handle(), N, X, 1, Y, 1));
}

template <>
void caffe_gpu_swap<double>(const int N, double *X, double *Y) {
  CUBLAS_CHECK(cublasDswap(Caffe::cublas_handle(), N, X, 1, Y, 1));
}

template <typename Dtype>
void __global__ axis_asum_kernel_0(
    const int M, const int N, const Dtype *in, Dtype *out) {
  const int i  = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val = 0;
  for (int j = id; j < N; j += blockDim.x) val += gpu_abs(in[i * N + j]);
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k];
    temp[id] = val;
  }
  __syncthreads();
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    out[i]     = val;
  }
}

template <typename Dtype>
void __global__ axis_asum_kernel_1(
    const int M, const int N, const Dtype *in, Dtype *out) {
  const int j  = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val = 0;
  for (int i = id; i < M; i += blockDim.x) val += gpu_abs(in[i * N + j]);
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k];
    temp[id] = val;
  }
  __syncthreads();
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    out[j]     = val;
  }
}

template <typename Dtype>
void caffe_gpu_axis_asum(
    const int axis, const int M, const int N, const Dtype *in, Dtype *out) {
  if (axis == 0) {
    axis_asum_kernel_0<Dtype><<<M, CAFFE_CUDA_NUM_THREADS>>>(M, N, in, out);
  } else {
    axis_asum_kernel_1<Dtype><<<N, CAFFE_CUDA_NUM_THREADS>>>(M, N, in, out);
  }
}

/* in \in \mathbb{R}^n
 * out \in \{-1, 0, 1\}^n or out \in \{-1, +1\}^n
 */
template <typename Dtype>
void __global__ input_scale_kernel(const int channels, const int dim,
    const Dtype *in, const Dtype *out, Dtype *beta, Dtype *sum) {
  const int idx = blockIdx.x;
  const int id  = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  volatile __shared__ Dtype temp2[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val  = 0;
  Dtype val2 = 0;
  int offset = idx / dim * channels * dim + idx % dim;
  in += offset;
  out += offset;
  for (int c = id; c < channels; c += blockDim.x) {
    offset = c * dim;
    val += out[offset] * in[offset];
    val2 += gpu_abs(out[offset]);
  }
  if (id >= WARP_SIZE) {
    temp[id - WARP_SIZE]  = val;
    temp2[id - WARP_SIZE] = val2;
  }
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k], val2 += temp2[k];
    temp[id]  = val;
    temp2[id] = val2;
  }
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k], val2 += temp2[k];
    beta[idx]  = val;
    sum[idx]   = val2;
  }
}
template <typename Dtype>
void caffe_gpu_input_scale(const int num, const int channels, const int dim,
    const Dtype *in, const Dtype *out, Dtype *beta, Dtype *sum) {
  input_scale_kernel<Dtype><<<num * dim, CAFFE_CUDA_NUM_THREADS>>>(
      channels, dim, in, out, beta, sum);
}
template <typename Dtype>
void __global__ clip_grad_kernel(const int n, const Dtype *in, Dtype *diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if (gpu_abs(in[index]) >= 1) diff[index] = 0;
  }
}
template <typename Dtype>
void caffe_gpu_clip_grad(const int n, const Dtype *in, Dtype *diff) {
  clip_grad_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, in, diff);
}
#define INSTANTIATE_BINARY_MATH(Dtype)                                         \
  template void caffe_gpu_binary_gradient<Dtype>(const int axis, const int M,  \
      const int N, bool use_bias, const Dtype *in, const Dtype *scale,         \
      const Dtype *bias, Dtype *grad);                                         \
                                                                               \
  template void caffe_gpu_ternary_gradient<Dtype>(const int axis, const int M, \
      const int N, bool use_bias, const Dtype *in, const Dtype *scale,         \
      const Dtype *bias, const Dtype *delta, Dtype *grad);                     \
                                                                               \
  template void caffe_gpu_clip<Dtype>(                                         \
      const int N, Dtype min_value, Dtype max_value, Dtype *X);                \
                                                                               \
  template void caffe_gpu_binary_approx<Dtype>(const int axis, const int M,    \
      const int N, bool use_bias, Dtype *in, Dtype *out, Dtype *scale,         \
      Dtype *bias);                                                            \
                                                                               \
  template void caffe_gpu_ternary_approx<Dtype>(const int axis, const int M,   \
      const int N, bool use_bias, Dtype *in, Dtype *out, Dtype *scale,         \
      Dtype *bias, Dtype *delta);                                              \
                                                                               \
  template void mean_center<Dtype>(                                            \
      const int c_out, const int c_in, const int wh, Dtype *in);               \
                                                                               \
  template void caffe_gpu_axis_asum<Dtype>(                                    \
      const int axis, const int M, const int N, const Dtype *in, Dtype *out);  \
  template void caffe_gpu_input_scale<Dtype>(const int num,                    \
      const int channels, const int dim, const Dtype *in, const Dtype *out,    \
      Dtype *beta, Dtype *sum);                                                \
  template void caffe_gpu_clip_grad<Dtype>(                                    \
      const int n, const Dtype *in, Dtype *diff);

INSTANTIATE_BINARY_MATH(float);
INSTANTIATE_BINARY_MATH(double);
}

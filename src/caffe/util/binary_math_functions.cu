#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#ifndef CPU_ONLY
#define THREADS_NUM_ROW 1
#define THREADS_NUM_COL 512
#define WARP_SIZE 32
const dim3 dim_threads(THREADS_NUM_ROW, THREADS_NUM_COL, 1);

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }

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
  const int j_start = id * N / CAFFE_CUDA_NUM_THREADS;
  const int j_end = (id + 1) * N / CAFFE_CUDA_NUM_THREADS;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  if (use_bias) {
    temp[id] = 0;
    for (int j = j_start; j < j_end; ++j) temp[id] += in[i * N + j];
    __syncthreads();
    if (id == 0) {
      bias[i] = 0;
      for (int idx = 0; idx < THREADS_NUM_COL; ++idx) bias[i] += temp[idx];
      bias[i] /= (Dtype) N;
    }
    __syncthreads();
    for (int j = j_start; j < j_end; ++j) in[i * N + j] -= bias[i];
  }
  Dtype val = 0;
  for (int j = j_start; j < j_end; ++j) val += gpu_abs(in[i * N + j]);
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
  for (int j = j_start; j < j_end; ++j)
    out[i * N + j] = in[i * N + j] >= (Dtype) 0 ? scale[i] : -scale[i];
  if (use_bias) {
    for (int j = j_start; j < j_end; ++j) {
      in[i * N + j] += bias[i];
      out[i * N + j] += bias[i];
    }
  }
}

template <typename Dtype>
__global__ void binary_approx_kernel_1(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *bias) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = threadIdx.y;
  const int i_start = id * M / THREADS_NUM_COL;
  const int i_end = (id + 1) * M / THREADS_NUM_COL;
  volatile __shared__ Dtype temp[THREADS_NUM_COL];
  temp[id] = 0;
  if (j >= N) return;
  if (use_bias) {
    temp[id] = 0;
    for (int i = i_start; i < i_end; ++i) temp[id] += in[i * N + j];
    __syncthreads();
    if (id == 0) {
      bias[j] = 0;
      for (int idx = 0; idx < THREADS_NUM_COL; ++idx) bias[j] += temp[idx];
      bias[j] /= (Dtype) M;
    }
    __syncthreads();
    for (int i = i_start; i < i_end; ++i) in[i * N + j] -= bias[j];
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
    for (int i = i_start; i < i_end; ++i) {
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
    dim3 dim_blocks((N - 1) / THREADS_NUM_ROW + 1, 1, 1);
    binary_approx_kernel_1<Dtype>
        <<<dim_blocks, dim_threads>>>(M, N, use_bias, in, out, scale, bias);
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_0(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *bias, Dtype *delta) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = threadIdx.y;
  const int j_start = id * N / THREADS_NUM_COL;
  const int j_end = (id + 1) * N / THREADS_NUM_COL;
  volatile __shared__ Dtype temp[THREADS_NUM_COL];
  volatile __shared__ Dtype temp2[THREADS_NUM_COL];
  temp[id] = 0;
  temp2[id] = 0;
  if (i >= M) return;
  if (use_bias) {
    temp[id] = 0;
    for (int j = j_start; j < j_end; ++j) temp[id] += in[i * N + j];
    __syncthreads();
    if (id == 0) {
      bias[i] = 0;
      for (int idx = 0; idx < THREADS_NUM_COL; ++idx) bias[i] += temp[idx];
      bias[i] /= (Dtype) N;
    }
    __syncthreads();
    for (int j = j_start; j < j_end; ++j) in[i * N + j] -= bias[i];
  }
  temp[id] = 0;
  for (int j = j_start; j < j_end; ++j) temp[id] += gpu_abs(in[i * N + j]);
  __syncthreads();
  if (id == 0) {
    delta[i] = 0;
    for (int idx = 0; idx < THREADS_NUM_COL; ++idx) delta[i] += temp[idx];
    delta[i] *= (Dtype) 0.7 / (Dtype) N;
  }
  __syncthreads();
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
    Dtype sum = 0;
    for (int idx = 0; idx < THREADS_NUM_COL; ++idx) {
      scale[i] += temp[idx];
      sum += temp2[idx];
    }
    scale[i] /= sum;
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
    for (int j = j_start; j < j_end; ++j) {
      out[i * N + j] += bias[i];
      in[i * N + j] += bias[i];
    }
  }
}

template <typename Dtype>
__global__ void ternary_approx_kernel_1(
    const int M, const int N, bool use_bias, Dtype *in, Dtype *out,
    Dtype *scale, Dtype *bias, Dtype *delta) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = threadIdx.y;
  const int i_start = id * M / THREADS_NUM_COL;
  const int i_end = (id + 1) * M / THREADS_NUM_COL;
  volatile __shared__ Dtype temp[THREADS_NUM_COL];
  volatile __shared__ Dtype temp2[THREADS_NUM_COL];
  temp[id] = 0;
  temp2[id] = 0;
  if (j >= N) return;
  if (use_bias) {
    temp[id] = 0;
    for (int i = i_start; i < i_end; ++i) temp[id] += in[i * N + j];
    __syncthreads();
    if (id == 0) {
      bias[j] = 0;
      for (int idx = 0; idx < THREADS_NUM_COL; ++idx) bias[j] += temp[idx];
      bias[j] /= (Dtype) M;
    }
    __syncthreads();
    for (int i = i_start; i < i_end; ++i) in[i * N + j] -= bias[j];
  }

  temp[id] = 0;
  for (int i = i_start; i < i_end; ++i) temp[id] += gpu_abs(in[i * N + j]);
  __syncthreads();
  if (id == 0) {
    delta[j] = 0;
    for (int idx = 0; idx < THREADS_NUM_COL; ++idx) delta[j] += temp[idx];
    delta[j] *= (Dtype) 0.7 / (Dtype) M;
  }
  __syncthreads();
  temp[id] = 0;
  temp2[id] = 0;
  for (int i = i_start; i < i_end; ++i) {
    Dtype val = gpu_abs(in[i * N + j]);
    if (val > delta[j]) {
      temp[id] += val;
      ++temp2[id];
    }
  }
  __syncthreads();
  if (id == 0) {
    scale[j] = 0;
    Dtype sum = 0;
    for (int idx = 0; idx < THREADS_NUM_COL; ++idx) {
      scale[j] += temp[idx];
      sum += temp2[idx];
    }
    scale[j] /= sum;
  }
  __syncthreads();
  for (int i = i_start; i < i_end; ++i) {
    if (in[i * N + j] > delta[j])
      out[i * N + j] = scale[j];
    else if (in[i * N + j] < -delta[j])
      out[i * N + j] = -scale[j];
    else
      out[i * N + j] = 0;
  }
  if (use_bias) {
    for (int i = i_start; i < i_end; ++i) {
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
    dim3 dim_blocks((M - 1) / THREADS_NUM_ROW + 1, 1, 1);
    ternary_approx_kernel_0<Dtype><<<dim_blocks, dim_threads>>>(
        M, N, use_bias, in, out, scale, bias, delta);
  } else {
    dim3 dim_blocks((N - 1) / THREADS_NUM_ROW + 1, 1, 1);
    ternary_approx_kernel_1<Dtype><<<dim_blocks, dim_threads>>>(
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
#endif  // CPU_ONLY
}

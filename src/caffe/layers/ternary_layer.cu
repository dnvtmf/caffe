#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <thrust/functional.h>
#include "caffe/layers/ternary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/cuda_reduce.hpp"

namespace caffe {

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }

template <typename Dtype>
void __global__ beta_div_add_kernel(
    const int n, const Dtype *sum, const Dtype add_value, Dtype *beta) {
  CUDA_KERNEL_LOOP(index, n) {
    if (sum[index] > 0) beta[index] = (beta[index] + add_value) / sum[index];
  }
}

template <typename Dtype>
void __global__ forward_kernel(
    const int n, const Dtype delta, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > delta ? 1 : (in[index] < -delta ? -1 : 0);
  }
}

template <typename Dtype>
void __global__ scale_kernel(
    const int channels, const int dim, const Dtype *in, const Dtype *out,
    Dtype *beta, Dtype *sum) {
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
void __global__ backward_kernel(
    const int n, const int channels, const int dim, const Dtype *out,
    const Dtype *beta, Dtype *diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if (out[index] != 0) {
      const int beta_index = (index / dim / channels) * dim + (index % dim);
      diff[index] *= beta[beta_index];
    }
  }
}

template <typename Dtype>
__global__ void backward_kernel(
    const int channels, const int dim, const Dtype *T, const Dtype *scale,
    const Dtype *sum, Dtype *diff) {
  const int idx = blockIdx.x;
  const int id  = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  int offset = idx / dim * channels * dim + idx % dim;
  T += offset;
  diff += offset;
  Dtype val = 0;
  for (int c = id; c < channels; c += blockDim.x) {
    val += T[c * dim] * diff[c * dim];
  }
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k];
    temp[id] = val;
  }
  if (id < 16) temp[id] += temp[id + 16];
  if (id < 8) temp[id] += temp[id + 8];
  if (id < 4) temp[id] += temp[id + 4];
  if (id < 2) temp[id] += temp[id + 2];
  if (id < 1) temp[id] += temp[id + 1];
  __syncthreads();
  for (int c = id; c < channels; c += blockDim.x) {
    diff[c * dim] =
        (temp[0] * T[c * dim] + diff[c * dim] * scale[idx]) / sum[idx];
  }
}

template <typename Dtype>
void ternary_count(const int n, const Dtype *out) {
  int cnt[3] = {};
  for (int i = 0; i < n; ++i) {
    if (out[i] == -1)
      ++cnt[0];
    else if (out[i] == 0)
      ++cnt[1];
    else if (out[i] == 1)
      ++cnt[2];
    else
      CHECK(false) << "Error value: " << out[i];
  }
  LOG(INFO) << "-1: " << cnt[0] << ", 0: " << cnt[1] << ", 1: " << cnt[2];
}
const double clip_value = 1;
template <typename Dtype>
void TernaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int count = bottom[0]->count();
  caffe_gpu_clip<Dtype>(
      count, -clip_value, clip_value, bottom[0]->mutable_gpu_data());
  forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, threshold_t_, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  // ternary_count<Dtype>(top[0]->count(), top[0]->cpu_data());
  CUDA_POST_KERNEL_CHECK;
  if (scale_term_) {
    scale_kernel<Dtype><<<num_ * dim_, CAFFE_CUDA_NUM_THREADS>>>(
        channels_, dim_, bottom[0]->gpu_data(), top[0]->gpu_data(),
        top[1]->mutable_gpu_data(), top[2]->mutable_gpu_data());
  }
}

template <typename Dtype>
void TernaryLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  if (!propagate_down[0]) return;
  const int count = bottom[0]->count();
  caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  if (scale_term_) {
    beta_div_add_kernel<Dtype>
        <<<CAFFE_GET_BLOCKS(top[1]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            top[1]->count(), top[2]->gpu_data(), 1, top[1]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels_, dim_, top[0]->gpu_data(), top[1]->gpu_data(),
        bottom[0]->mutable_gpu_diff());
    //    backward_kernel<Dtype><<<num_ * dim_, CAFFE_CUDA_NUM_THREADS>>>(
    //        channels_, dim_, top[0]->gpu_data(),
    //        top[1]->gpu_data(), top[2]->gpu_data(),
    //        bottom[0]->mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
  }
  caffe_gpu_clip_grad(
      count, (Dtype) clip_value, bottom[0]->gpu_data(),
      bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryLayer);
}  // namespace caffe

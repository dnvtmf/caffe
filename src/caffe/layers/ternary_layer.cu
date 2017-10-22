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
    if (sum[index] > 0) beta[index] /= sum[index];
    beta[index] += add_value;
  }
}

/**
\delta_c = \frac{t}{num * dim} \sum_{n=1}^{num}{\sum_{i=1}^{dim}{|in[n][c][i]|}}
*/
template <typename Dtype>
void __global__ delta_kernel(const int num, const int channels, const int dim,
    const Dtype threshold_t, const Dtype *in, Dtype *delta) {
  const int c  = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val = 0;
  in += c * dim;
  for (int n = 0; n < num; ++n) {
    for (int j = id; j < dim; j += blockDim.x) {
      val += gpu_abs(in[j]);
    }
    in += channels * dim;
  }
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k];
    temp[id] = val;
  }
  // __syncthreads();
  if (id == 0) {
    for (int k = 1; k < WARP_SIZE; ++k) val += temp[k];
    delta[c]   = val * threshold_t;
  }
}

template <typename Dtype>
void __global__ forward_kernel(const int n, const int channels, const int dim,
    const Dtype *delta, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int c = index / dim % channels;
    out[index]  = in[index] > delta[c] ? 1 : (in[index] < -delta[c] ? -1 : 0);
  }
}

template <typename Dtype>
void __global__ backward_kernel(const int n, const int channels,
    const int group_channels, const int dim, const Dtype *delta,
    const Dtype *beta, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    int y      = index % dim;
    int index2 = index / dim;
    int c      = index2 % channels;
    index2     = (index2 / group_channels) * dim + y;
    if (gpu_abs(in[index]) > delta[c]) out[index] *= beta[index2];
  }
}

template <typename Dtype>
void TernaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int count = bottom[0]->count();
  if (!use_global_stats_) {
    Dtype threshold_t = threshold_t_ / Dtype(count / channels_);
    delta_kernel<Dtype><<<channels_, CAFFE_CUDA_NUM_THREADS>>>(num_, channels_,
        dim_, threshold_t, bottom[0]->gpu_data(), delta_.mutable_gpu_data());
    caffe_gpu_axpby<Dtype>(channels_, 1. - moving_average_fraction_,
        delta_.gpu_data(), moving_average_fraction_,
        this->blobs_[0]->mutable_gpu_data());
  }

  const Dtype *delta =
      use_global_stats_ ? this->blobs_[0]->gpu_data() : delta_.gpu_data();
  forward_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, channels_,
          dim_, delta, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  if (scale_term_) {
    caffe_gpu_input_scale<Dtype>(num_ * group_, channels_ / group_, dim_,
        bottom[0]->gpu_data(), top[0]->gpu_data(), top[1]->mutable_gpu_data(),
        top[2]->mutable_gpu_data());
  }
}

template <typename Dtype>
void TernaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
    if (scale_term_) {
      beta_div_add_kernel<Dtype>
          <<<CAFFE_GET_BLOCKS(top[1]->count()), CAFFE_CUDA_NUM_THREADS>>>(
              top[1]->count(), top[2]->gpu_data(),
              Dtype(1.) / Dtype(channels_ / group_),
              top[1]->mutable_gpu_data());
      backward_kernel<Dtype>
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
              channels_, channels_ / group_, dim_, delta_.gpu_data(),
              top[1]->gpu_data(), bottom[0]->gpu_data(),
              bottom[0]->mutable_gpu_diff());
    }
    caffe_gpu_clip_grad(
        count, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryLayer);
}  // namespace caffe

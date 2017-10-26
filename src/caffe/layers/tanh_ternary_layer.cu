#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <thrust/functional.h>
#include "caffe/layers/tanh_ternary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/cuda_reduce.hpp"

namespace caffe {
#define HARD_TANH

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }
inline __device__ float gpu_floor(float x) { return floorf(x); }
inline __device__ double gpu_floor(double x) { return floor(x); }

template <typename Dtype>
void __global__ beta_div_add_kernel(
    const int n, const Dtype *sum, const Dtype add_value, Dtype *beta) {
  CUDA_KERNEL_LOOP(index, n) {
    if (sum[index] > 0) beta[index] = (beta[index] + add_value) / sum[index];
  }
}

template <typename Dtype>
void __global__ forward_kernel(const int n, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
#ifdef HARD_TANH
    out[index] = gpu_floor(in[index] + out[index]);
#else
    Dtype tanhx = tanh(in[index]);
    out[index]  = tanhx >= 0 ? out[index] < tanhx : -(out[index] < -tanhx);
#endif
  }
}

template <typename Dtype>
void __global__ tanh_backward_kernel(
    const int n, const Dtype *in, Dtype *out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype tanhx = tanh(in[index]);
    out_diff[index] *= (1 - tanhx * tanhx);
  }
}

template <typename Dtype>
void __global__ backward_kernel(const int n, const int group_channels,
    const int dim, const Dtype *out, const Dtype *beta, Dtype *diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if (out[index] != 0) {
      const int beta_index =
          (index / dim / group_channels) * dim + (index % dim);
      diff[index] *= beta[beta_index];
    }
  }
}

template <typename Dtype>
void TanHTernaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int count = bottom[0]->count();
  caffe_gpu_rng_uniform<Dtype>(count, 0, 1, top[0]->mutable_gpu_data());
#ifdef HARD_TANH
  caffe_gpu_clip<Dtype>(count, -1, 1, bottom[0]->mutable_gpu_data());
#endif
  forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  if (scale_term_) {
    caffe_gpu_input_scale<Dtype>(num_ * group_, channels_ / group_, dim_,
        bottom[0]->gpu_data(), top[0]->gpu_data(), top[1]->mutable_gpu_data(),
        top[2]->mutable_gpu_data());
  }
}

template <typename Dtype>
void TanHTernaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
    if (scale_term_) {
      beta_div_add_kernel<Dtype>
          <<<CAFFE_GET_BLOCKS(top[1]->count()), CAFFE_CUDA_NUM_THREADS>>>(
              top[1]->count(), top[2]->gpu_data(), 1,
              top[1]->mutable_gpu_data());
      backward_kernel<Dtype>
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
              channels_ / group_, dim_, top[0]->gpu_data(), top[1]->gpu_data(),
              bottom[0]->mutable_gpu_diff());
    }
#ifdef HARD_TANH
    caffe_gpu_clip_grad(count, (Dtype) 1., bottom[0]->gpu_data(),
        bottom[0]->mutable_gpu_diff());
#else
    tanh_backward_kernel<Dtype>
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TanHTernaryLayer);
}  // namespace caffe

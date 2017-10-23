#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <thrust/functional.h>
#include "caffe/layers/tanh_ternary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/cuda_reduce.hpp"

namespace caffe {

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }
inline __device__ float gpu_floor(float x) { return floorf(x); }
inline __device__ double gpu_floor(double x) { return floor(x); }

template <typename Dtype>
void __global__ forward_kernel(const int n, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype tanhx = tanh(in[index]);
    out[index]  = tanhx >= 0 ? out[index] < tanhx : -(out[index] < -tanhx);
    // gpu_floor(tanh(in[index]) + out[index]);
  }
}

template <typename Dtype>
void __global__ backward_kernel(const int n, const Dtype *in, Dtype *out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype tanhx = tanh(in[index]);
    out_diff[index] *= (1 - tanhx * tanhx);
  }
}

template <typename Dtype>
void __global__ backward_kernel(const int n, const int group_channels,
    const int dim, const Dtype *beta, Dtype *diff) {
  CUDA_KERNEL_LOOP(index, n) {
    diff[index] *= beta[index / dim % group_channels];
  }
}
template <typename Dtype>
void TanHTernaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int count = bottom[0]->count();
  caffe_gpu_rng_uniform<Dtype>(count, 0, 1, top[0]->mutable_gpu_data());
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
      caffe_gpu_add_scalar<Dtype>(
          top[1]->count(), Dtype(1.), top[1]->mutable_gpu_diff());
      caffe_gpu_div<Dtype>(top[1]->count(), top[1]->gpu_data(),
          top[2]->gpu_data(), top[1]->mutable_gpu_diff());
      backward_kernel<Dtype>
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
              channels_ / group_, dim_, top[1]->gpu_diff(),
              bottom[0]->mutable_gpu_diff());
      caffe_gpu_clip_grad(
          count, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
    } else {
      backward_kernel<Dtype>
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TanHTernaryLayer);
}  // namespace caffe

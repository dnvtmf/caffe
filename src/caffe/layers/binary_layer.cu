#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <thrust/functional.h>
#include "caffe/layers/binary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/cuda_reduce.hpp"

namespace caffe {

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }

template <typename Dtype>
void __global__ backward_kernel(const int n, const int group_channels,
    const int dim, const Dtype *beta, Dtype *diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const int beta_index = index / dim / group_channels * dim + index % dim;
    diff[index] *= beta[beta_index];
  }
}

template <typename Dtype>
void BinaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int count = bottom[0]->count();
  caffe_gpu_sign<Dtype>(
      count, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  if (scale_term_) {
    caffe_gpu_input_scale<Dtype>(num_ * group_, channels_ / group_, dim_,
        bottom[0]->gpu_data(), top[0]->gpu_data(), top[1]->mutable_gpu_data(),
        top[2]->mutable_gpu_data());
  }
}

template <typename Dtype>
void BinaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
    if (scale_term_) {
      caffe_gpu_add_scalar<Dtype>(
          top[1]->count(), 1, top[1]->mutable_gpu_data());
      caffe_gpu_div<Dtype>(top[1]->count(), top[1]->gpu_data(),
          top[2]->gpu_data(), top[1]->mutable_gpu_diff());
      backward_kernel<Dtype>
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
              channels_ / group_, dim_, top[1]->gpu_diff(),
              bottom[0]->mutable_gpu_diff());
    }
    caffe_gpu_clip_grad(
        count, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryLayer);
}  // namespace caffe

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

#if TERNARY_METHOD == 2
template <typename Dtype>
void __global__ forward_kernel(const int n, const int channels, const int dim,
    const Dtype threshold_t, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype delta         = 0;
    const int offset    = (index / dim) * channels * dim + (index % dim);
    const Dtype *in_ptr = in + offset;
    Dtype *out_ptr      = out + offset;
    for (int c = 0; c < channels; ++c) {
      delta += gpu_abs(in_ptr[c * dim]);
    }
    delta = delta * threshold_t / channels;
    for (int c = 0; c < channels; ++c) {
      out_ptr[c * dim] =
          in_ptr[c * dim] > delta ? 1 : (in_ptr[c * dim] < -delta ? -1 : 0);
    }
  }
}
#else
template <typename Dtype>
void __global__ forward_kernel(
    const int n, const Dtype delta, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > delta ? 1 : (in[index] < -delta ? -1 : 0);
  }
}
#endif

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

template <typename Dtype>
void TernaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  caffe_gpu_clip<Dtype>(
      bottom[0]->count(), -1., 1., bottom[0]->mutable_gpu_data());
#if TERNARY_METHOD == 2
  const int count = num_ * group_ * dim_;
  forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels_ / group_, dim_, threshold_t_, bottom[0]->gpu_data(),
      top[0]->mutable_gpu_data());
#else
  const int count = bottom[0]->count();
#if TERNARY_METHOD == 0
  Dtype delta     = threshold_t_;
#else
  Dtype delta = 0;
  caffe_gpu_asum(count, bottom[0]->gpu_data(), &delta);
  delta = delta * threshold_t_ / bottom[0]->count();
#endif
  forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, delta, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
#endif
  // ternary_count<Dtype>(top[0]->count(), top[0]->cpu_data());
  CUDA_POST_KERNEL_CHECK;
  if (scale_term_) {
    caffe_gpu_input_scale<Dtype>(num_ * group_, channels_ / group_, dim_,
        bottom[0]->gpu_data(), top[0]->gpu_data(), top[1]->mutable_gpu_data(),
        top[2]->mutable_gpu_data());
  }
}

template <typename Dtype>
void TernaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  if (!propagate_down[0]) return;
  const int count = bottom[0]->count();
  caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  if (scale_term_) {
    beta_div_add_kernel<Dtype>
        <<<CAFFE_GET_BLOCKS(top[1]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            top[1]->count(), top[2]->gpu_data(), 1, top[1]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels_ / group_, dim_, top[0]->gpu_data(), top[1]->gpu_data(),
        bottom[0]->mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
  }
  caffe_gpu_clip_grad(
      count, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryLayer);
}  // namespace caffe

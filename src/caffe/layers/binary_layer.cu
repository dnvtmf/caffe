#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <thrust/functional.h>
#include "caffe/layers/binary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/cuda_reduce.hpp"

namespace caffe {

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }

template <typename Dtype>
void __global__ forward_kernel(const int channels, const int dim,
    const Dtype *in, Dtype *out, Dtype *beta) {
  const int idx = blockIdx.x;
  const int id  = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val  = 0;
  int offset = idx / dim * channels * dim + idx % dim;
  in += offset;
  out += offset;
  for (int c = id; c < channels; c += blockDim.x) {
    offset = c * dim;
    if (in[offset] >= 0) {
      out[offset] = 1;
      val += in[offset];
    } else {
      out[offset] = -1;
      val -= in[offset];
    }
  }
  if (id >= WARP_SIZE) {
    temp[id - WARP_SIZE] = val;
  }
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
    beta[idx]  = val;
  }
}

template <typename Dtype>
void __global__ backward_kernel(const int n, const int group_channels,
    const int dim, const Dtype *in, const Dtype *beta, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] *= (gpu_abs(in[index]) < 1) * beta[index / dim % group_channels];
  }
}

template <typename Dtype>
void BinaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data          = top[0]->mutable_gpu_data();
  Dtype *beta_data         = top[1]->mutable_gpu_data();
  caffe_gpu_set<Dtype>(
      top[2]->count(), channels_ / group_, top[2]->mutable_gpu_data());
  forward_kernel<Dtype><<<num_ * group_ * dim_, CAFFE_CUDA_NUM_THREADS>>>(
      channels_ / group_, dim_, bottom_data, top_data, beta_data);
}

template <typename Dtype>
void BinaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const int count       = bottom[0]->count();
    const Dtype *top_diff = top[0]->gpu_diff();
    caffe_gpu_scal<Dtype>(
        top[1]->count(), 1. / (channels_ / group_), top[1]->mutable_gpu_data());
    caffe_gpu_add_scalar<Dtype>(
        top[1]->count(), 1. / (channels_ / group_), top[1]->mutable_gpu_data());
    caffe_copy(count, top_diff, bottom[0]->mutable_gpu_diff());
    backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels_ / group_, dim_, bottom[0]->gpu_data(),
        top[1]->gpu_data(), bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryLayer);
}  // namespace caffe

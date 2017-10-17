#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <thrust/functional.h>
#include "caffe/layers/ternary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/cuda_reduce.hpp"

namespace caffe {

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }

template <typename Dtype>
void __global__ delta_kernel(
    const int n, const int dim, const int channels, const Dtype* in,
    Dtype* delta) {
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
}

template <typename Dtype>
struct abs_data : public thrust::unary_function<Dtype, int> {
  const Dtype* data;
  abs_data(const Dtype* _data) : data(_data){};
  __host__ __device__ Dtype operator()(int i) { return gpu_abs(data[i]); }
};

template <typename Dtype>
struct greater_fetch : public thrust::unary_function<Dtype, int> {
  const Dtype *data, *diff, value;
  greater_fetch(const Dtype* _data, const Dtype* _diff, const Dtype _value)
      : data(_data), diff(_diff), value(_value){};
  __host__ __device__ Dtype operator()(int i) {
    return diff[i] * (data[i] > value);
  }
};

template <typename Dtype>
struct less_fetch : public thrust::unary_function<Dtype, int> {
  const Dtype *data, *diff, value;
  less_fetch(const Dtype* _data, const Dtype* _diff, const Dtype _value)
      : data(_data), diff(_diff), value(_value){};
  __host__ __device__ Dtype operator()(int i) {
    return diff[i] * (data[i] < value);
  }
};

template <typename Dtype>
void __global__ ternary_forward_kernel(
    const int n, const Dtype* in, const Dtype wp, const Dtype wn,
    const Dtype threshold, Dtype* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = wp * (in[idx] > threshold) - wn * (in[idx] < -threshold);
  }
}

template <typename Dtype>
void __global__ ternary_backward_input_kernel(
    const int N, const Dtype wp, const Dtype wn, const Dtype threshold,
    const Dtype* bottom_data, const Dtype* top_diff, Dtype* bottom_diff) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    bottom_diff[idx] = top_diff[idx];
    bottom_diff[idx] *=
        (bottom_data[idx] > threshold ? wp
                                      : bottom_data[idx] < -threshold ? wn : 1);
  }
}

template <typename Dtype>
void TernaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  abs_data<Dtype> bottom_data(bottom[0]->mutable_gpu_data());
  thrust::maximum<Dtype> fcn1;
  threshold_ = functionReduce<Dtype>(count, &temp_, 0, bottom_data, fcn1);
  // Dtype mx       = 0;
  // const Dtype* A = bottom[0]->cpu_data();
  // for (int i = 0; i < count; ++i) mx += std::abs(A[i]);
  // LOG(INFO) << "threshold: " << threshold_ << ' ' << mx;
  // CHECK(std::abs(threshold_ - mx) < 1e-6);
  threshold_ *= 0.05;
  // LOG(INFO) << "threshold: " << threshold_;
  // LOG(INFO) << "wp: " << *Wp_;
  // LOG(INFO) << "Wn: " << *Wn_;
  ternary_forward_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[0]->gpu_data(), *Wp_, *Wn_, threshold_,
          top[0]->mutable_gpu_data());
}
template <typename Dtype>
void TernaryLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int count          = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff    = top[0]->gpu_diff();
  if (propagate_down[0]) {
    ternary_backward_input_kernel<Dtype>
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, *Wp_, *Wn_, threshold_, bottom_data, top_diff,
            bottom[0]->mutable_gpu_diff());
  }

  greater_fetch<Dtype> gf_fcn(bottom_data, top_diff, threshold_);
  less_fetch<Dtype> lf_fcn(bottom_data, top_diff, -threshold_);
  thrust::plus<Dtype> fcn1;
  const int num = bottom[0]->shape(0);
  this->blobs_[0]->mutable_cpu_diff()[0] =
      functionReduce<Dtype>(count, &temp_, 0, gf_fcn, fcn1) / num;
  this->blobs_[0]->mutable_cpu_diff()[1] =
      functionReduce<Dtype>(count, &temp_, 0, lf_fcn, fcn1) / num;
}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryLayer);
}  // namespace caffe
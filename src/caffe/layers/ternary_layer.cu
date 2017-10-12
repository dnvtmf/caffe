#include "caffe/layers/ternary_layer.hpp"

namespace caffe {
template <typename Dtype>
void __global__
    ternary_forward_kernel_1(const int n, const Dtype* x, Dtype* y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    if (y[idx] >= 0)
      y[idx] = y[idx] <= x[idx];
    else
      y[idx] = -(y[idx] <= -x[idx]);
  }
}
template <typename Dtype>
void __global__ ternary_backward_kernel_1(
    const int n, const Dtype* x, const Dtype* y, Dtype* z) {
  const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) z[idx] = y[idx] * (x[idx] < 1);
}
template <typename Dtype>
void TernaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* top_data          = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  int count                = bottom[0]->count();
  caffe_gpu_rng_uniform<Dtype>(count, 0., 1., top_data);
  ternary_forward_kernel_1<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, top_data);
}
template <typename Dtype>
void TernaryLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff    = bottom[0]->mutable_gpu_diff();
    const int count       = bottom[0]->count();
    if (top_diff != bottom_diff)
      caffe_copy<Dtype>(count, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryLayer);
}  // namespace caffe
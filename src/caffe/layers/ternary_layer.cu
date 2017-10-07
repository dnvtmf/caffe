#include "caffe/layers/ternary_layer.hpp"

namespace caffe {
template <typename Dtype>
void __global__
    ternary_forward_kernel_1(const int n, const Dtype* x, Dtype* y) {
  const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = (x[idx] > 0.3) - (x[idx] < -0.3);
}
template <typename Dtype>
void __global__ ternary_backward_kernel_1(
    const int n, const Dtype* x, const Dtype* y, Dtype* z) {
  const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) z[idx] = y[idx] * ((x[idx] > 0.3) - (x[idx] < -0.3));
}
template <typename Dtype>
void TernaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ternary_forward_kernel_1<Dtype>
      <<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
          bottom[0]->count(), bottom[0]->gpu_data(),
          top[0]->mutable_gpu_data());
}
template <typename Dtype>
void TernaryLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    ternary_backward_kernel_1<Dtype>
        <<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            bottom[0]->count(), bottom[0]->gpu_data(), top[0]->gpu_diff(),
            bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryLayer);
}  // namespace caffe
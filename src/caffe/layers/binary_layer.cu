#include "caffe/layers/binary_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }

template <typename Dtype>
void __global__
    binary_backward_kernel(int n, const Dtype* x, const Dtype* y, Dtype* z) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) z[idx] = y[idx] * (gpu_abs(x[idx]) < 1);
}

template <typename Dtype>
void BinaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  caffe_gpu_sign<Dtype>(
      bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}
template <typename Dtype>
void BinaryLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    binary_backward_kernel<Dtype>
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom[0]->gpu_data(), top[0]->gpu_diff(),
            bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryLayer);
}  // namespace caffe
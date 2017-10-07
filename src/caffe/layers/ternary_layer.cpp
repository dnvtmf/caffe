#include <algorithm>
#include <vector>

#include "caffe/layers/ternary_layer.hpp"

namespace caffe {

template <typename Dtype>
void TernaryLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data          = top[0]->mutable_cpu_data();
  const int count          = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }
}

template <typename Dtype>
void TernaryLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff    = bottom[0]->mutable_cpu_diff();
    const int count       = bottom[0]->count();
    Dtype tanhx;
    for (int i = 0; i < count; ++i) {
      tanhx          = top_data[i];
      bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TernaryLayer);
#endif

INSTANTIATE_CLASS(TernaryLayer);
REGISTER_LAYER_CLASS(Ternary);

}  // namespace caffe

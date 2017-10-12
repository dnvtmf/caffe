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
  caffe_rng_uniform<Dtype>(count, 0, 1., top_data);
  for (int i = 0; i < count; ++i) {
    if (bottom_data[i] >= 0)
      top_data[i] = top_data[i] <= bottom_data[i];
    else
      top_data[i] = -(top_data[i] <= -bottom_data[i]);
  }
}

template <typename Dtype>
void TernaryLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff    = bottom[0]->mutable_cpu_diff();
    const int count       = bottom[0]->count();
    if (top_diff != bottom_diff)
      caffe_copy<Dtype>(count, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(TernaryLayer);
#endif

INSTANTIATE_CLASS(TernaryLayer);
REGISTER_LAYER_CLASS(Ternary);

}  // namespace caffe

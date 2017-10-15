#include <algorithm>
#include <vector>

#include "caffe/layers/ternary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"

namespace caffe {
template <typename Dtype>
void TernaryLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->blobs_.size() > 0) {
    CHECK_EQ(this->blobs_.size(), 1);
    CHECK_EQ(this->blobs_[0]->count(), 2);
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>({2}));
    caffe_set(2, Dtype(1), this->blobs_[0]->mutable_cpu_data());
  }
  Wp_   = this->blobs_[0]->mutable_cpu_data();
  Wn_   = Wp_ + 1;
  temp_ = NULL;
}
template <typename Dtype>
void TernaryLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void TernaryLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void TernaryLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
  }
}

#ifdef CPU_ONLY
STUB_GPU(TernaryLayer);
#endif

INSTANTIATE_CLASS(TernaryLayer);
REGISTER_LAYER_CLASS(Ternary);

}  // namespace caffe

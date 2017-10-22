#include <algorithm>
#include <vector>

#include "caffe/layers/tanh_ternary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"

namespace caffe {
template <typename Dtype>
void TanHTernaryLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  TernaryParameter param = this->layer_param_.ternary_param();

  group_ = param.group();
  CHECK_EQ(bottom[0]->num_axes(), 4);
  channels_ = bottom[0]->shape(1);
  dim_      = bottom[0]->count(2);
  CHECK(channels_ % group_ == 0);
  CHECK(top.size() == 1 || top.size() == 3);
  scale_term_ = top.size() == 3;
}
template <typename Dtype>
void TanHTernaryLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  if (scale_term_) {
    CHECK(top.size() == 3);
    num_ = bottom[0]->shape(0);
    top[1]->Reshape({num_, group_, dim_});
    top[2]->Reshape({num_, group_, dim_});
  } else {
    CHECK(top.size() == 1);
  }
}

template <typename Dtype>
void TanHTernaryLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void TanHTernaryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
  }
}

#ifdef CPU_ONLY
STUB_GPU(TanHTernaryLayer);
#endif

INSTANTIATE_CLASS(TanHTernaryLayer);
REGISTER_LAYER_CLASS(TanHTernary);

}  // namespace caffe

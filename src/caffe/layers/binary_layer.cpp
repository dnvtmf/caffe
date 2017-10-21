#include <algorithm>
#include <vector>

#include "caffe/layers/binary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"

namespace caffe {
template <typename Dtype>
void BinaryLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  TernaryParameter param = this->layer_param_.ternary_param();
  group_                 = param.group();
  CHECK_EQ(bottom[0]->num_axes(), 4);
  channels_ = bottom[0]->shape(1);
  dim_      = bottom[0]->count(2);
  CHECK(channels_ % group_ == 0);
}
template <typename Dtype>
void BinaryLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(1), channels_);
  num_ = bottom[0]->shape(0);
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->Reshape({num_, group_, dim_});
  top[2]->Reshape({num_, group_, dim_});
}

template <typename Dtype>
void BinaryLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void BinaryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
  }
}

#ifdef CPU_ONLY
STUB_GPU(BinaryLayer);
#endif

INSTANTIATE_CLASS(BinaryLayer);
REGISTER_LAYER_CLASS(Binary);

}  // namespace caffe

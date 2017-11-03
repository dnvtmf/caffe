#include <algorithm>
#include <vector>

#include "caffe/layers/ternary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"

namespace caffe {
template <typename Dtype>
void TernaryLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  TernaryParameter param   = this->layer_param_.ternary_param();
  moving_average_fraction_ = param.moving_average_fraction();
  threshold_t_             = param.threshold_t();
  group_                   = param.group();
  use_global_stats_        = this->phase_ == TEST;
  CHECK_EQ(bottom[0]->num_axes(), 4);
  channels_ = bottom[0]->shape(1);
  CHECK(group_ > 0 && channels_ % group_ == 0);
  channels_ /= group_;
  CHECK(top.size() == 1 || top.size() == 3);
  scale_term_ = top.size() == 3;
}
template <typename Dtype>
void TernaryLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  CHECK_EQ(bottom[0]->num_axes(), 4);
  num_ = bottom[0]->shape(0);
  dim_ = bottom[0]->count(2);
  if (scale_term_) {
    CHECK(top.size() == 3);
    top[1]->Reshape({num_, group_, dim_});
    top[2]->Reshape({num_, group_, dim_});
  } else {
    CHECK(top.size() == 1);
  }
  num_ *= group_;
}

template <typename Dtype>
void TernaryLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void TernaryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
  if (propagate_down[0]) {
  }
}

#ifdef CPU_ONLY
STUB_GPU(TernaryLayer);
#endif

INSTANTIATE_CLASS(TernaryLayer);
REGISTER_LAYER_CLASS(Ternary);

}  // namespace caffe

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
  CHECK_EQ(bottom[0]->num_axis(), 4);
  channels_ = bottom[0]->shape(1);
  CHECK(channels_ % group_ == 0);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(this->blobs_.size(), 1);
    CHECK_EQ(this->blobs_[0]->count(), channels_);
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>({channels_}));
    caffe_set(channels_, Dtype(0), this->blobs_[0]->mutable_cpu_data());
  }

  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    } else {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure batch normalization statistics as layer "
          << "parameters.";
    }
  }
}
template <typename Dtype>
void TernaryLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(1), channels_);
  num_ = bottom[0]->shape(0);
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->Reshape({num_, group_});
  top[2]->Reshape({num_, group_});
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

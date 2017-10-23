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
  dim_      = bottom[0]->count(2);
  CHECK(channels_ % group_ == 0);
  CHECK(top.size() == 1 || top.size() == 3);
  scale_term_ = top.size() == 3;

  if (this->blobs_.size() > 0) {
    CHECK_EQ(this->blobs_.size(), 2);
    CHECK_EQ(this->blobs_[0]->count(), channels_);
    CHECK_EQ(this->blobs_[1]->count(), 1);
  } else {
    this->blobs_.resize(2);
    this->blobs_[0].reset(new Blob<Dtype>({channels_}));
    this->blobs_[1].reset(new Blob<Dtype>({1}));
    for (int i = 0; i < (int) this->blobs_.size(); ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
          this->blobs_[i]->mutable_cpu_data());
    }
  }

  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    } else {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure ternary statistics as layer parameters.";
    }
  }
  delta_.Reshape({channels_});
}
template <typename Dtype>
void TernaryLayer<Dtype>::Reshape(
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

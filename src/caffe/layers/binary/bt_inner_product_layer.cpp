#include "caffe/layers/binary/bt_inner_product_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>

namespace caffe {

template<typename Dtype>
void BTInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom,
    const vector<Blob<Dtype>*> &top) {
  auto &params = this->layer_param_.bt_inner_product_param();
  N_ = params.num_output();
  bias_term_ = params.bias_term();
  const int axis = bottom[0]->CanonicalAxisIndex(params.axis());
  K_ = bottom[0]->count(axis);
  bK_ = (K_ + BINARY_SIZE - 1) / BINARY_SIZE;
  if(this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if(bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape = {K_, N_};
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>
                                            (params.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if(bias_term_) {
      vector<int> bias_shape = {N_};
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>
                                            (params.bias_filler()));
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void BTInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
    const vector<Blob<Dtype>*> &top) {
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.bt_inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  if(bias_term_) {
    vector<int> bias_shape = {M_};
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template<typename Dtype>
void BTInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  auto bottom_data = bottom[0]->mutable_cpu_data();
  auto top_data = top[0]->mutable_cpu_data();
  auto weight = this->blobs_[0]->cpu_data();
  caffe_cpu_ternary<Dtype>(1, K_, N_, weight, w_pos_, w_neg_, w_delta_, w_scale_);
  caffe_cpu_ternary<Dtype>(0, M_, K_, bottom_data, in_, in_2_, in_delta_, in_scale_);
//  caffe_cpu_binary<Dtype>(0, M_, K_, bottom_data, in_, in_scale_);
//  for (auto &x : w_scale_) x = 1;
//  for (auto &x : in_scale_) x = 1;
  caffe_cpu_binary_gemm_and<Dtype>(false, false, M_, N_, K_,
      Dtype(1), &in_[0], &w_pos_[0], &in_scale_[0], &w_scale_[0],
      Dtype(0), top_data);
  caffe_cpu_binary_gemm_and<Dtype>(false, false, M_, N_, K_,
      Dtype(-1), &in_[0], &w_neg_[0], &in_scale_[0], &w_scale_[0],
      Dtype(1), top_data);

  if(bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
        (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
        (Dtype)1., top_data);
  }
}

template<typename Dtype>
void BTInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype>*> &bottom) {
  auto top_diff = top[0]->cpu_diff();
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
    caffe_cpu_ternary<Dtype>(1, M_, N_, top_diff, g_, g_2_,
                             g_delta_, g_scale_);
    caffe_cpu_ternary<Dtype>(1, M_, K_, bottom[0]->cpu_data(), in_, in_2_,
                             in_delta_, in_scale_);
    caffe_cpu_binary_gemm_and<Dtype>(true, false, K_, N_, M_,
        Dtype(1), &in_[0], &g_[0], &in_scale_[0], &g_scale_[0],
        Dtype(0), this->blobs_[0]->mutable_cpu_diff());
    caffe_cpu_binary_gemm_and<Dtype>(true, false, K_, N_, M_,
        Dtype(1), &in_[0], &g_2_[0], &in_scale_[0], &g_scale_[0],
        Dtype(0), this->blobs_[0]->mutable_cpu_diff());
//    caffe_cpu_binary<Dtype>(1, M_, N_, top_diff, g_, g_scale_);
//    caffe_cpu_binary<Dtype>(1, M_, K_, bottom[0]->cpu_data(), in_, in_scale_);
//    caffe_cpu_binary_gemm_xor<Dtype>(true, false, K_, N_, M_,
//        &in_[0], &g_[0], &in_scale_[0], &g_scale_[0],
//        this->blobs_[0]->mutable_cpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                          bias_multiplier_.cpu_data(), (Dtype)1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    caffe_cpu_ternary<Dtype>(0, M_, N_, top_diff, g_, g_2_, g_delta_, g_scale_);
    caffe_cpu_ternary<Dtype>(0, K_, N_, this->blobs_[0]->cpu_data(),
                             w_pos_, w_neg_, w_delta_, w_scale_);
    caffe_cpu_binary_gemm_and<Dtype>(false, true, M_, K_, N_,
        Dtype(1), &g_[0], &w_pos_[0], &g_scale_[0], &w_scale_[0],
        Dtype(0), bottom[0]->mutable_cpu_diff());
    caffe_cpu_binary_gemm_and<Dtype>(false, true, M_, K_, N_,
        Dtype(-1), &g_2_[0], &w_pos_[0], &g_scale_[0], &w_scale_[0],
        Dtype(1), bottom[0]->mutable_cpu_diff());
    caffe_cpu_binary_gemm_and<Dtype>(false, true, M_, K_, N_,
        Dtype(-1), &g_[0], &w_neg_[0], &g_scale_[0], &w_scale_[0],
        Dtype(1), bottom[0]->mutable_cpu_diff());
    caffe_cpu_binary_gemm_and<Dtype>(false, true, M_, K_, N_,
        Dtype(1), &g_2_[0], &w_neg_[0], &g_scale_[0], &w_scale_[0],
        Dtype(1), bottom[0]->mutable_cpu_diff());
    auto p = bottom[0]->cpu_data();
    auto q = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < M_ * K_; ++i, ++p) {
      if(*p < -in_delta_) {
        *q = 0;
      }
    }
//    caffe_cpu_binary<Dtype>(0, M_, N_, top_diff, g_, g_scale_);
//    caffe_cpu_ternary<Dtype>(0, K_, N_, this->blobs_[0]->cpu_data(),
//        w_pos_, w_neg_, w_delta_, w_scale_);
//    caffe_cpu_binary_gemm_and<Dtype>(false, true, M_, K_, N_,
//        Dtype(1), &g_[0], &w_pos_[0], &g_scale_[0], &w_scale_[0],
//        Dtype(0), bottom[0]->mutable_cpu_diff());
//    caffe_cpu_binary_gemm_and<Dtype>(false, true, M_, K_, N_,
//        Dtype(-1), &g_[0], &w_neg_[0], &g_scale_[0], &w_scale_[0],
//        Dtype(1), bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(BTInnerProductLayer);
REGISTER_LAYER_CLASS(BTInnerProduct);

} //namespcae caffe

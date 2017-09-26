#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/xnor_net_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void XnorNetInnerProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  w_scale_  = weight_s_.mutable_gpu_data();
  w_bias_   = weight_s_.mutable_gpu_diff();
  in_scale_ = in_s_.mutable_gpu_data();
  in_bias_  = in_s_.mutable_gpu_diff();

  Dtype *weight      = weight_.mutable_gpu_data();
  Dtype *in          = in_.mutable_gpu_data();
  Dtype *bottom_data = bottom[0]->mutable_gpu_data();
  Dtype *top_data    = top[0]->mutable_gpu_data();
  Dtype *weight_data = this->blobs_[0]->mutable_gpu_data();

  // mean cetner params
  caffe_gpu_gemv<Dtype>(
      CblasTrans, K_, N_, Dtype(-1. / K_), weight_data,
      sum_multiplier_.gpu_data(), Dtype(0), w_bias_);
  caffe_gpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans, K_, N_, 1, Dtype(1),
      sum_multiplier_.gpu_data(), w_bias_, Dtype(1), weight_data);
  // clamp params
  caffe_gpu_clip<Dtype>(K_ * N_, -1, 1, weight_data);
  // binary the weight
  caffe_gpu_binary_approx<Dtype>(
      1, K_, N_, false, weight_data, weight, w_scale_, w_bias_);
  // binary the input
  caffe_gpu_binary_approx<Dtype>(
      0, M_, K_, false, bottom_data, in, in_scale_, in_bias_);
  // matrix multiply
  caffe_gpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype) 1., in, weight,
      (Dtype) 0., top_data);

  // bias
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(
        CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
        bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(), (Dtype) 1.,
        top_data);
  }
}

template <typename Dtype>
void XnorNetInnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  const Dtype *top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
    Dtype *weight_diff  = this->blobs_[0]->mutable_gpu_diff();
    const Dtype *weight = this->blobs_[0]->gpu_data();
    // Gradient with respect to weight
    const Dtype *in = in_.mutable_gpu_data();
    caffe_gpu_gemm<Dtype>(
        CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype) 1., in, top_diff,
        (Dtype) 1., weight_diff);
    caffe_gpu_binary_gradient<Dtype>(
        1, K_, N_, false, weight, w_scale_, w_bias_, weight_diff);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(
        CblasTrans, M_, N_, (Dtype) 1., top_diff, bias_multiplier_.gpu_data(),
        (Dtype) 1., this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    const Dtype *weight      = weight_.mutable_gpu_data();
    Dtype *in_diff           = bottom[0]->mutable_gpu_diff();
    const Dtype *bottom_data = bottom[0]->gpu_data();
    // dI' = g * W'^
    caffe_gpu_gemm<Dtype>(
        CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1., top_diff, weight,
        (Dtype) 0., in_diff);

    caffe_gpu_binary_gradient<Dtype>(
        0, M_, K_, false, bottom_data, in_scale_, in_bias_, in_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(XnorNetInnerProductLayer);
}  // namespace caffe

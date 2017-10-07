#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/xnor_net_conv_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  w_scale_ = weight_s_.mutable_gpu_data();
  w_bias_  = weight_s_.mutable_gpu_diff();

  Dtype *weight_data = this->blobs_[0]->mutable_gpu_data();
  // meen center
  const int c_in = this->conv_in_channels_;
  mean_center<Dtype>(M_, c_in, K_ / c_in, weight_data);
  caffe_gpu_clip<Dtype>(M_ * K_, -1, 1, weight_data);

  caffe_gpu_binary_approx<Dtype>(
      0, M_, K_, false, weight_data, weight_.mutable_gpu_data(), w_scale_,
      w_bias_);
  weight_data = weight_.mutable_gpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->gpu_data();
    Dtype *top_data          = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(
          bottom_data + n * this->bottom_dim_, weight_data,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype *bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  const Dtype *weight = this->blobs_[0]->gpu_data();
  Dtype *weight_diff  = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype *top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype *bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype *bottom_data = bottom[i]->gpu_data();
      Dtype *bottom_diff       = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(
              bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(
              top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
  caffe_gpu_binary_gradient<Dtype>(
      0, M_, K_, false, this->blobs_[0]->gpu_data(), w_scale_, w_bias_,
      weight_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(XnorNetConvolutionLayer);
}  // namespace caffe

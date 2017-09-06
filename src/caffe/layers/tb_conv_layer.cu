#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/binary_math_functions.hpp"

namespace caffe {
template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_gpu_gemm(
  const Dtype *input, Dtype *output, bool skip_im2col) {
  /*
    const Dtype *col_buff = input;
    const Dtype *weights = (is_in_bin_ || is_w_bin_) ? weight_.gpu_data() :
                           this->blobs_[0]->gpu_data();
    if (!this->is_1x1_) {
      if (!skip_im2col) {
        this->conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());
      }
      col_buff = this->col_buffer_.gpu_data();
    }
    if (is_in_bin_) {
      caffe_gpu_binary_approx<Dtype>(
        1, K_, N_, col_buff,
        in_.mutable_gpu_data(), in_s_.mutable_gpu_data());
      col_buff = in_.gpu_data();
    }
    else if (is_w_bin_) {
      caffe_gpu_ternary_approx<Dtype>(
        1, K_, N_, col_buff,
        in_.mutable_gpu_data(), in_s_.mutable_gpu_data(), in_s_.mutable_gpu_diff());
      col_buff = in_.gpu_data();
    }
    caffe_gpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans, M_, N_, K_,
      (Dtype)1., weights, col_buff, (Dtype)0., output);
    /*
    caffe_gpu_ternary_norm<Dtype>(
      1, K_, N_, col_buff,
      binary_in_.data(), mask_in_.data(), delta_in_.data(), scale_in_.data(),
      bias_in_.data(), sum_in_.data(), sum2_in_.data(), use_bias_);
    caffe_gpu_bt_gemm<Dtype>(
      false, false, M_, N_, K_,
      binary_w_.data(), scale_w_.data(),
      binary_in_.data(), mask_in_.data(), scale_in_.data(), sum2_in_.data(),
      output,
      use_bias_, bias_w_.data(), sum_w_.data(), bias_in_.data(), sum_in_.data());
    */
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_gpu_gemm(
  const Dtype *input, const Dtype* top_diff,
  Dtype *input_diff, Dtype *weight_diff) {
  /*
  const Dtype *weight = (is_in_bin_ || is_w_bin_) ? weight_.gpu_data() :
                      this->blobs_[0]->gpu_data();
  const Dtype* col_buff = input;
  Dtype* col_buff_diff = this->col_buffer_.mutable_gpu_diff();
  if (this->is_1x1_) {
  col_buff_diff = input_diff;
  }
  else {
  this->conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());
  col_buff = this->col_buffer_.gpu_data();
  }
  const Dtype *in = (is_in_bin_ || is_w_bin_) ? in_.gpu_data() : col_buff;
  if (is_in_bin_) {
  caffe_gpu_binary_approx<Dtype>(
    1, K_, N_, col_buff, in_.mutable_gpu_data(), in_s_.mutable_gpu_data());
  }
  else if (is_w_bin_) {
  caffe_gpu_ternary_approx<Dtype>(
    1, K_, N_, col_buff, in_.mutable_gpu_data(),
    in_s_.mutable_gpu_data(), in_s_.mutable_gpu_diff());
  }
  caffe_gpu_gemm<Dtype>(
  CblasTrans, CblasNoTrans, K_, N_, M_,
  (Dtype)1., weight, top_diff, (Dtype)0., col_buff_diff);
  caffe_gpu_gemm<Dtype>(
  CblasNoTrans, CblasTrans, M_, K_, N_,
  (Dtype)1., top_diff, in, (Dtype)1., weight_diff);
  if (is_in_bin_) {
  caffe_gpu_binary_gradient<Dtype>(
    1, K_, N_, col_buff, in_s_.gpu_data(), col_buff_diff);
  }
  else if (is_w_bin_) {
  caffe_gpu_ternary_gradient(
    1, K_, N_, col_buff, in_s_.gpu_data(), in_s_.gpu_diff(), col_buff_diff);
  }
  if (!this->is_1x1_) {
  this->conv_col2im_gpu(col_buff_diff, input_diff);
  }
  */
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
//  const Dtype *weight = this->blobs_[0]->gpu_data();
//  Dtype *p = this->blobs_[0]->mutable_gpu_data();
//  for (int i = 0; i < K_ * M_; ++i) {
//    *p = std::max(std::min(*p, max_), min_);
//    p++;
//  }
  if (is_w_bin_) {
    caffe_gpu_binary_approx<Dtype>(
      0, M_, K_, this->blobs_[0]->gpu_data(),
      weight_.mutable_gpu_data(), weight_s_.mutable_gpu_data());
  }
  else {
    caffe_gpu_ternary_approx<Dtype>(
      0, M_, K_, this->blobs_[0]->gpu_data(), weight_.mutable_gpu_data(),
      weight_s_.mutable_gpu_data(), weight_s_.mutable_gpu_diff());
  }
//  caffe_gpu_binary_norm<Dtype>(
//    0, M_, K_, weight, binary_w_.data(), scale_w_.data(),
//    bias_w_.data(), sum_w_.data(), use_bias_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->gpu_data();
    Dtype *top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(
        bottom_data + n * this->bottom_dim_, top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype *bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom) {
  Dtype *weight_diff  = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype *top_diff = top[i]->gpu_diff();
    const Dtype *bottom_data = bottom[i]->gpu_data();
    Dtype *bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype *bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        backward_gpu_gemm(bottom_data + n * this->bottom_dim_,
                          top_diff + n * this->top_dim_,
                          bottom_diff + n * this->bottom_dim_, weight_diff);
      }
    }
  }
  if (is_w_bin_) {
    caffe_gpu_binary_gradient<Dtype>(0, M_, K_, this->blobs_[0]->gpu_data(),
                                     weight_s_.gpu_data(), weight_diff);
  }
  else if (is_in_bin_) {
    caffe_gpu_ternary_gradient<Dtype>(
      0, M_, K_, this->blobs_[0]->gpu_data(),
      weight_s_.gpu_data(), weight_s_.gpu_diff(), weight_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TBConvolutionLayer);
}  // namespace caffe

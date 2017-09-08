#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_conv_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_gpu_gemm(
    Dtype *input, Dtype *output, bool skip_im2col) {
  Dtype *col_buff = input;
  Dtype *weights = (is_in_bin_ || is_w_bin_)
                       ? weight_.mutable_gpu_data()
                       : this->blobs_[0]->mutable_gpu_data();
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      this->conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());
    }
    col_buff = this->col_buffer_.mutable_gpu_data();
  }

  if (clip_ & 2) {
    caffe_gpu_clip<Dtype>(K_ * N_, (Dtype) -1., (Dtype) 1., col_buff);
  }
  if (is_in_bin_) {
    caffe_gpu_binary_approx<Dtype>(
        1, K_, N_, use_bias_, col_buff, in_.mutable_gpu_data(),
        in_s_.mutable_gpu_data());
    col_buff = in_.mutable_gpu_data();
  } else if (is_w_bin_) {
    caffe_gpu_ternary_approx<Dtype>(
        1, K_, N_, use_bias_, col_buff, in_.mutable_gpu_data(),
        in_s_.mutable_gpu_data(), in_s_.mutable_gpu_diff(),
        sum_.mutable_gpu_diff());
    col_buff = in_.mutable_gpu_data();
  }
  caffe_gpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype) 1., weights, col_buff,
      (Dtype) 0., output);
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_gpu_gemm(
    Dtype *input, const Dtype *top_diff, Dtype *input_diff,
    Dtype *weight_diff) {
  Dtype *weight = (is_in_bin_ || is_w_bin_)
                      ? weight_.mutable_gpu_data()
                      : this->blobs_[0]->mutable_gpu_data();
  Dtype *col_buff = input;
  Dtype *col_buff_diff = this->col_buffer_.mutable_gpu_diff();
  if (this->is_1x1_) {
    col_buff_diff = input_diff;
  } else {
    this->conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());
    col_buff = this->col_buffer_.mutable_gpu_data();
  }
  Dtype *in = (is_in_bin_ || is_w_bin_) ? in_.mutable_gpu_data() : col_buff;
  if (is_in_bin_) {
    caffe_gpu_binary_approx<Dtype>(
        1, K_, N_, use_bias_, col_buff, in_.mutable_gpu_data(),
        in_s_.mutable_gpu_data());
  } else if (is_w_bin_) {
    caffe_gpu_ternary_approx<Dtype>(
        1, K_, N_, use_bias_, col_buff, in_.mutable_gpu_data(),
        in_s_.mutable_gpu_data(), in_s_.mutable_gpu_diff(),
        sum_.mutable_gpu_diff());
  }
  caffe_gpu_gemm<Dtype>(
      CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype) 1., weight, top_diff,
      (Dtype) 0., col_buff_diff);
  caffe_gpu_gemm<Dtype>(
      CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1., top_diff, in,
      (Dtype) 1., weight_diff);
  if (is_in_bin_) {
    caffe_gpu_binary_gradient<Dtype>(
        1, K_, N_, use_bias_, col_buff, in_s_.gpu_data(), col_buff_diff);
  } else if (is_w_bin_) {
    caffe_gpu_ternary_gradient(
        1, K_, N_, use_bias_, col_buff, in_s_.gpu_data(), in_s_.gpu_diff(),
        col_buff_diff);
  }
  if (!this->is_1x1_) {
    this->conv_col2im_gpu(col_buff_diff, input_diff);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  if (clip_ & 1) {
    const Dtype value = sqrt(6. / (M_ + K_));
    caffe_gpu_clip<Dtype>(
        M_ * K_, -value, value, this->blobs_[0]->mutable_gpu_data());
  }
  if (is_w_bin_) {
    caffe_gpu_binary_approx<Dtype>(
        0, M_, K_, use_bias_, this->blobs_[0]->mutable_gpu_data(),
        weight_.mutable_gpu_data(), weight_s_.mutable_gpu_data());
  } else {
    caffe_gpu_ternary_approx<Dtype>(
        0, M_, K_, use_bias_, this->blobs_[0]->mutable_gpu_data(),
        weight_.mutable_gpu_data(), weight_s_.mutable_gpu_data(),
        weight_s_.mutable_gpu_diff(), sum_.mutable_cpu_diff());
  }
  for (int i = 0; i < bottom.size(); ++i) {
    Dtype *bottom_data = bottom[i]->mutable_gpu_data();
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
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  Dtype *weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype *top_diff = top[i]->gpu_diff();
    Dtype *bottom_data = bottom[i]->mutable_gpu_data();
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
        backward_gpu_gemm(
            bottom_data + n * this->bottom_dim_, top_diff + n * this->top_dim_,
            bottom_diff + n * this->bottom_dim_, weight_diff);
      }
    }
  }
  if (is_w_bin_) {
    caffe_gpu_binary_gradient<Dtype>(
        0, M_, K_, use_bias_, this->blobs_[0]->gpu_data(), weight_s_.gpu_data(),
        weight_diff);
  } else if (is_in_bin_) {
    caffe_gpu_ternary_gradient<Dtype>(
        0, M_, K_, use_bias_, this->blobs_[0]->gpu_data(), weight_s_.gpu_data(),
        weight_s_.gpu_diff(), weight_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TBConvolutionLayer);
}  // namespace caffe

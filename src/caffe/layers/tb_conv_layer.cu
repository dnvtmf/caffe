#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_conv_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_gpu_gemm(Dtype *input, Dtype *output) {
  Dtype *col_buff = input;
  Dtype *weight   = (is_in_bin_ || is_w_bin_)
                      ? weight_.mutable_gpu_data()
                      : this->blobs_[0]->mutable_gpu_data();
  if (!this->is_1x1_) {
    col_buff = this->col_buffer_.mutable_gpu_data();
    this->conv_im2col_gpu(input, col_buff);
  }
  // clip
  if (clip_ & 2) {
    caffe_gpu_clip<Dtype>(
        K_ * N_ * this->group_, (Dtype) -1., (Dtype) 1., col_buff);
  }
  // mean
  if (use_bias_) {
    for (int g = 0; g < this->group_; ++g) {
      caffe_gpu_gemv<Dtype>(
          CblasTrans, K_, N_, Dtype(1. / K_), col_buff + this->col_offset_ * g,
          sum_multiplier_.gpu_data(), 0, in_bias_ + N_ * g);
    }
  }
  // ternary

  for (int g = 0; g < this->group_; ++g) {
    const int offset = this->col_offset_ * g;
    caffe_gpu_ternary_approx<Dtype>(
        1, K_, N_, use_bias_, col_buff + offset,
        in_.mutable_gpu_data() + offset, in_scale_ + N_ * g, in_bias_ + N_ * g,
        in_delta_ + N_ * g);
  }
  col_buff = in_.mutable_gpu_data();

  // forward
  for (int g = 0; g < this->group_; ++g) {
    caffe_gpu_gemm<Dtype>(
        CblasNoTrans, CblasNoTrans, M_ / this->group_, N_, K_, (Dtype) 1.,
        weight + this->weight_offset_ * g, col_buff + this->col_offset_ * g,
        (Dtype) 0., output + this->output_offset_ * g);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_gpu_gemm(
    Dtype *input, const Dtype *top_diff, Dtype *input_diff,
    Dtype *weight_diff) {
  Dtype *weight = (is_in_bin_ || is_w_bin_)
                      ? weight_.mutable_gpu_data()
                      : this->blobs_[0]->mutable_gpu_data();
  Dtype *col_buff      = input;
  Dtype *col_buff_diff = input_diff;
  if (!this->is_1x1_) {
    this->conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());
    col_buff      = this->col_buffer_.mutable_gpu_data();
    col_buff_diff = this->col_buffer_.mutable_gpu_diff();
  }
  Dtype *in = (is_in_bin_ || is_w_bin_) ? in_.mutable_gpu_data() : col_buff;
  // mean
  if (use_bias_) {
    for (int g = 0; g < this->group_; ++g) {
      caffe_gpu_gemv<Dtype>(
          CblasTrans, K_, N_, Dtype(1. / K_), col_buff + this->col_offset_ * g,
          sum_multiplier_.gpu_data(), 0, in_bias_ + N_ * g);
    }
  }
  // binary or ternary
  if (is_in_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      const int offset = this->col_offset_ * g;
      caffe_gpu_binary_approx<Dtype>(
          1, K_, N_, use_bias_, col_buff + offset, in + offset,
          in_scale_ + N_ * g, in_bias_ + N_ * g);
    }
  } else if (is_w_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      const int offset = this->col_offset_ * g;
      caffe_gpu_ternary_approx<Dtype>(
          1, K_, N_, use_bias_, col_buff + offset, in + offset,
          in_scale_ + N_ * g, in_bias_ + N_ * g, in_delta_ + N_ * g);
    }
  }
  // grad
  for (int g = 0; g < this->group_; ++g) {
    caffe_gpu_gemm<Dtype>(
        CblasTrans, CblasNoTrans, K_, N_, M_ / this->group_, (Dtype) 1.,
        weight + this->weight_offset_ * g, top_diff + this->output_offset_ * g,
        (Dtype) 0., col_buff_diff + this->col_offset_ * g);
    caffe_gpu_gemm<Dtype>(
        CblasNoTrans, CblasTrans, M_ / this->group_, K_, N_, (Dtype) 1.,
        top_diff + this->output_offset_ * g, in + this->col_offset_ * g,
        (Dtype) 1., weight_diff + this->weight_offset_ * g);
  }
  Dtype *delta = in_.mutable_gpu_diff();
  if (have_reg_) {
    // D = I' - I
    caffe_gpu_sub<Dtype>(
        K_ * N_ * this->group_, in_.gpu_data(), col_buff, delta);
    // dI' += reg_ * D
    caffe_gpu_axpy<Dtype>(K_ * N_ * this->group_, reg_, delta, col_buff_diff);
  }
  if (is_in_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      const int offset = this->col_offset_ * g;
      caffe_gpu_binary_gradient<Dtype>(
          1, K_, N_, use_bias_, col_buff + offset, in_scale_ + N_ * g,
          in_bias_ + N_ * g, col_buff_diff + offset);
    }
  } else if (is_w_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      const int offset = this->col_offset_ * g;
      caffe_gpu_ternary_gradient<Dtype>(
          1, K_, N_, use_bias_, col_buff + offset, in_scale_ + N_ * g,
          in_bias_ + N_ * g, in_delta_ + N_ * g, col_buff_diff + offset);
    }
  }
  if (have_reg_) {
    // dI += -reg_ * D
    caffe_gpu_axpy<Dtype>(K_ * N_ * this->group_, -reg_, delta, col_buff_diff);
  }
  if (!this->is_1x1_) {
    this->conv_col2im_gpu(col_buff_diff, input_diff);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  w_scale_  = weight_s_.mutable_gpu_data();
  w_bias_   = weight_s_.mutable_gpu_diff();
  w_delta_  = delta_.mutable_gpu_data();
  in_scale_ = in_s_.mutable_gpu_data();
  in_bias_  = in_s_.mutable_gpu_diff();
  in_delta_ = delta_.mutable_gpu_diff();

  Dtype *weight_data = this->blobs_[0]->mutable_gpu_data();
  const int c_in     = this->conv_in_channels_;
  mean_center<Dtype>(M_, c_in, K_ / c_in, weight_data);

  if (clip_ & 1) {
    const Dtype value = sqrt(3. / K_);
    caffe_gpu_clip<Dtype>(M_ * K_, -value, value, weight_data);
  } else {
    caffe_gpu_clip<Dtype>(M_ * K_, -1, 1, weight_data);
  }
  // mean
  if (use_bias_) {
    caffe_gpu_gemv<Dtype>(
        CblasNoTrans, M_, K_, Dtype(1. / K_), weight_data,
        sum_multiplier_.gpu_data(), 0, w_bias_);
  }
  if (is_w_bin_) {
    caffe_gpu_binary_approx<Dtype>(
        0, M_, K_, use_bias_, weight_data, weight_.mutable_gpu_data(), w_scale_,
        w_bias_);
  } else if (is_in_bin_) {
    caffe_gpu_ternary_approx<Dtype>(
        0, M_, K_, use_bias_, weight_data, weight_.mutable_gpu_data(), w_scale_,
        w_bias_, w_delta_);
  }

  for (int i = 0; i < bottom.size(); ++i) {
    Dtype *bottom_data = bottom[i]->mutable_gpu_data();
    Dtype *top_data    = top[i]->mutable_gpu_data();
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
    Dtype *bottom_data    = bottom[i]->mutable_gpu_data();
    Dtype *bottom_diff    = bottom[i]->mutable_gpu_diff();

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
  Dtype *delta = weight_.mutable_gpu_diff();
  if (have_reg_) {
    // D = W' - W
    caffe_gpu_sub<Dtype>(
        M_ * K_, weight_.gpu_data(), this->blobs_[0]->gpu_data(), delta);
    // dW' += reg_ * D
    caffe_gpu_axpy<Dtype>(M_ * K_, reg_, delta, weight_diff);
  }
  if (is_w_bin_) {
    caffe_gpu_binary_gradient<Dtype>(
        0, M_, K_, use_bias_, this->blobs_[0]->gpu_data(), w_scale_, w_bias_,
        weight_diff);
  } else if (is_in_bin_) {
    caffe_gpu_ternary_gradient<Dtype>(
        0, M_, K_, use_bias_, this->blobs_[0]->gpu_data(), w_scale_, w_bias_,
        w_delta_, weight_diff);
  }
  if (have_reg_) {
    // dW += -reg_ * D
    caffe_gpu_axpy<Dtype>(M_ * K_, -reg_, delta, weight_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TBConvolutionLayer);
}  // namespace caffe

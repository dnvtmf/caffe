#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/xnor_net_conv_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::forward_gpu_gemm(
    Dtype *input, Dtype *output) {
  Dtype *col_buff = input;
  Dtype *weights  = weight_.mutable_gpu_data();
  if (!this->is_1x1_) {
    this->conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());
    col_buff = this->col_buffer_.mutable_gpu_data();
  }

  for (int g = 0; g < this->group_; ++g) {
    const int offset = this->col_offset_ * g;
    /*
    caffe_gpu_gemv<Dtype>(
         CblasTrans, K_, N_, -1. / K_, col_buff + offset,
         sum_multiplier_.gpu_data(), 0., in_bias_ + N_ * g);
     caffe_gpu_gemm<Dtype>(
         CblasNoTrans, CblasNoTrans, K_, N_, 1, 1., sum_multiplier_.gpu_data(),
         in_bias_ + N_ * g, 1., col_buff + offset);
    */
    caffe_gpu_binary_approx<Dtype>(
        1, K_, N_, false, col_buff + offset, in_.mutable_gpu_data() + offset,
        in_scale_ + N_ * g, in_bias_ + N_ * g);
  }
  col_buff = in_.mutable_gpu_data();

  for (int g = 0; g < this->group_; ++g) {
    caffe_gpu_gemm<Dtype>(
        CblasNoTrans, CblasNoTrans, M_ / this->group_, N_, K_, (Dtype) 1.,
        weights + this->weight_offset_ * g, col_buff + this->col_offset_ * g,
        (Dtype) 0., output + this->output_offset_ * g);
  }
}

template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::backward_gpu_gemm(
    Dtype *input, const Dtype *top_diff, Dtype *input_diff,
    Dtype *weight_diff) {
  Dtype *weight        = weight_.mutable_gpu_data();
  Dtype *in            = in_.mutable_gpu_data();
  Dtype *col_buff      = input;
  Dtype *col_buff_diff = input_diff;
  if (!this->is_1x1_) {
    col_buff_diff = this->col_buffer_.mutable_gpu_diff();
    this->conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());
    col_buff = this->col_buffer_.mutable_gpu_data();
  }

  for (int g = 0; g < this->group_; ++g) {
    const int offset = this->col_offset_ * g;
    caffe_gpu_binary_approx<Dtype>(
        1, K_, N_, false, col_buff + offset, in + offset, in_scale_ + N_ * g,
        in_bias_ + N_ * g);
  }

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

  for (int g = 0; g < this->group_; ++g) {
    const int offset = this->col_offset_ * g;
    caffe_gpu_binary_gradient<Dtype>(
        1, K_, N_, false, col_buff + offset, in_scale_ + N_ * g,
        in_bias_ + N_ * g, col_buff_diff + offset);
  }

  if (!this->is_1x1_) {
    this->conv_col2im_gpu(col_buff_diff, input_diff);
  }
}

template <typename Dtype>
__global__ void mean_center_kernel(
    const int c_out, const int c_in, const int wh, Dtype *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int x = i / wh;
  const int y = i % wh;
  if (x >= c_out) return;
  Dtype mean  = 0;
  Dtype *p_in = in + x * c_in * wh + y;
  for (int i = 0; i < c_in; ++i, p_in += wh) mean += *p_in;
  mean /= c_in;
  p_in = in + x * c_in * wh + y;
  for (int i = 0; i < c_in; ++i, p_in += wh) *p_in -= mean;
}

template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  w_scale_  = weight_s_.mutable_gpu_data();
  w_bias_   = weight_s_.mutable_gpu_diff();
  in_scale_ = in_s_.mutable_gpu_data();
  in_bias_  = in_s_.mutable_gpu_diff();

  Dtype *weight_data = this->blobs_[0]->mutable_gpu_data();
  // meen center
  const int c_out = this->conv_out_channels_;
  const int c_in  = this->conv_in_channels_;
  const int wh    = K_ / c_in;
  mean_center_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(c_out * wh), CAFFE_CUDA_NUM_THREADS>>>(
          c_out, c_in, wh, weight_data);
  caffe_gpu_clip<Dtype>(M_ * K_, -1, 1, weight_data);

  caffe_gpu_binary_approx<Dtype>(
      0, M_, K_, false, this->blobs_[0]->mutable_gpu_data(),
      weight_.mutable_gpu_data(), w_scale_, w_bias_);

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
void XnorNetConvolutionLayer<Dtype>::Backward_gpu(
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
  caffe_gpu_binary_gradient<Dtype>(
      0, M_, K_, false, this->blobs_[0]->gpu_data(), w_scale_, w_bias_,
      weight_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(XnorNetConvolutionLayer);
}  // namespace caffe

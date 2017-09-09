#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_conv_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void TBConvolutionLayer<Dtype>::LayerSetUp_more() {
  // Initialize the vectors
  M_ = this->conv_out_channels_ / this->group_;
  K_ = this->kernel_dim_;
  CHECK_EQ(this->group_, 1);
  weight_.Reshape({M_, K_});
  weight_s_.Reshape({M_});
  full_train_ = this->layer_param_.tb_param().full_train();
  use_bias_ = this->layer_param_.tb_param().use_bias();
  is_w_bin_ = this->layer_param_.tb_param().w_binary();
  is_in_bin_ = this->layer_param_.tb_param().in_binary();
  clip_ = this->layer_param_.tb_param().clip();
  reg_ = this->layer_param_.tb_param().reg();
  have_reg_ = (is_w_bin_ || is_in_bin_) && abs(reg_) < 1e-10;
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Reshape_more() {
  N_ = this->conv_out_spatial_dim_;
  in_.Reshape({K_, N_});
  in_s_.Reshape({N_});
  sum_.Reshape({max(M_, N_)});
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::compute_output_shape() {
  const int *kernel_shape_data = this->kernel_shape_.cpu_data();
  const int *stride_data = this->stride_.cpu_data();
  const int *pad_data = this->pad_.cpu_data();
  const int *dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim =
        (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_cpu_gemm(
    const Dtype *input, Dtype *output, bool skip_im2col) {
  const Dtype *col_buff = input;
  const Dtype *weights = (is_in_bin_ || is_w_bin_)
                             ? weight_.cpu_data()
                             : this->blobs_[0]->cpu_data();
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      this->conv_im2col_cpu(input, this->col_buffer_.mutable_cpu_data());
    }
    col_buff = this->col_buffer_.cpu_data();
  }
  if (is_in_bin_) {
    caffe_cpu_binary_approx<Dtype>(
        1, K_, N_, col_buff, in_.mutable_cpu_data(), in_s_.mutable_cpu_data());
    col_buff = in_.cpu_data();
  } else if (is_w_bin_) {
    caffe_cpu_ternary_approx<Dtype>(
        1, K_, N_, col_buff, in_.mutable_cpu_data(), in_s_.mutable_cpu_data(),
        in_s_.mutable_cpu_diff(), sum_.mutable_cpu_diff());
    col_buff = in_.cpu_data();
  }
  caffe_cpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype) 1., weights, col_buff,
      (Dtype) 0., output);
  /*
  caffe_cpu_ternary_norm<Dtype>(
    1, K_, N_, col_buff,
    binary_in_.data(), mask_in_.data(), delta_in_.data(), scale_in_.data(),
    bias_in_.data(), sum_in_.data(), sum2_in_.data(), use_bias_);
  caffe_cpu_bt_gemm<Dtype>(
    false, false, M_, N_, K_,
    binary_w_.data(), scale_w_.data(),
    binary_in_.data(), mask_in_.data(), scale_in_.data(), sum2_in_.data(),
    output,
    use_bias_, bias_w_.data(), sum_w_.data(), bias_in_.data(), sum_in_.data());
  */
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_cpu_gemm(
    const Dtype *input, const Dtype *top_diff, Dtype *input_diff,
    Dtype *weight_diff) {
  const Dtype *weights = (is_in_bin_ || is_w_bin_)
                             ? weight_.cpu_data()
                             : this->blobs_[0]->cpu_data();
  const Dtype *col_buff = input;
  Dtype *col_buff_diff = this->col_buffer_.mutable_cpu_diff();
  if (this->is_1x1_) {
    col_buff_diff = input_diff;
  } else {
    this->conv_im2col_cpu(input, this->col_buffer_.mutable_cpu_data());
    col_buff = this->col_buffer_.cpu_data();
  }
  const Dtype *in = (is_in_bin_ || is_w_bin_) ? in_.cpu_data() : col_buff;
  if (is_in_bin_) {
    caffe_cpu_binary_approx<Dtype>(
        1, K_, N_, col_buff, in_.mutable_cpu_data(), in_s_.mutable_cpu_data());
  } else if (is_w_bin_) {
    caffe_cpu_ternary_approx<Dtype>(
        1, K_, N_, col_buff, in_.mutable_cpu_data(), in_s_.mutable_cpu_data(),
        in_s_.mutable_cpu_diff(), sum_.mutable_cpu_diff());
  }
  caffe_cpu_gemm<Dtype>(
      CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype) 1., weights, top_diff,
      (Dtype) 0., col_buff_diff);
  caffe_cpu_gemm<Dtype>(
      CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1., top_diff, in,
      (Dtype) 1., weight_diff);
  if (is_in_bin_) {
    caffe_cpu_binary_gradient<Dtype>(
        1, K_, N_, col_buff, in_s_.cpu_data(), col_buff_diff);
  } else if (is_w_bin_) {
    caffe_cpu_ternary_gradient(
        1, K_, N_, col_buff, in_s_.cpu_data(), in_s_.cpu_diff(), col_buff_diff);
  }
  if (!this->is_1x1_) {
    this->conv_col2im_cpu(col_buff_diff, input_diff);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  //  const Dtype *weight = this->blobs_[0]->cpu_data();
  //  Dtype *p = this->blobs_[0]->mutable_cpu_data();
  //  for (int i = 0; i < K_ * M_; ++i) {
  //    *p = std::max(std::min(*p, max_), min_);
  //    p++;
  //  }
  if (is_w_bin_) {
    caffe_cpu_binary_approx<Dtype>(
        0, M_, K_, this->blobs_[0]->cpu_data(), weight_.mutable_cpu_data(),
        weight_s_.mutable_cpu_data());
  } else {
    caffe_cpu_ternary_approx<Dtype>(
        0, M_, K_, this->blobs_[0]->cpu_data(), weight_.mutable_cpu_data(),
        weight_s_.mutable_cpu_data(), weight_s_.mutable_cpu_diff(),
        sum_.mutable_cpu_data());
  }
  //  caffe_cpu_binary_norm<Dtype>(
  //    0, M_, K_, weight, binary_w_.data(), scale_w_.data(),
  //    bias_w_.data(), sum_w_.data(), use_bias_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->cpu_data();
    Dtype *top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(
          bottom_data + n * this->bottom_dim_, top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype *bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype *top_diff = top[i]->cpu_diff();
    const Dtype *bottom_data = bottom[i]->cpu_data();
    Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype *bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        backward_cpu_gemm(
            bottom_data + n * this->bottom_dim_, top_diff + n * this->top_dim_,
            bottom_diff + n * this->bottom_dim_, weight_diff);
      }
    }
  }
  if (is_w_bin_) {
    caffe_cpu_binary_gradient<Dtype>(
        0, M_, K_, this->blobs_[0]->cpu_data(), weight_s_.cpu_data(),
        weight_diff);
  } else if (is_in_bin_) {
    caffe_cpu_ternary_gradient<Dtype>(
        0, M_, K_, this->blobs_[0]->cpu_data(), weight_s_.cpu_data(),
        weight_s_.cpu_diff(), weight_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(TBConvolutionLayer);
#endif

INSTANTIATE_CLASS(TBConvolutionLayer);
REGISTER_LAYER_CLASS(TBConvolution);
}  // namespace caffe

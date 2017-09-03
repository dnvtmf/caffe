#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/binary_math_functions.hpp"

namespace caffe {
template<typename Dtype>
void TBConvolutionLayer<Dtype>::LayerSetUp_more() {
  // Initialize the vectors
  M_ = this->conv_out_channels_ / this->group_;
  K_ = this->kernel_dim_;
  BM_ = (N_ - 1) / BINARY_SIZE + 1;
  BK_ = (K_ - 1) / BINARY_SIZE + 1;
  binary_w_.resize(max(BM_ * K_, M_ * BK_));
  scale_w_ .resize(max(K_, M_));
  bias_w_  .resize(max(K_, M_));
  sum_w_   .resize(max(K_, M_));
  max_ = sqrt(12. / M_);
  min_ = -max_;
  full_train_ = this->layer_param_.tb_param().full_train();
  use_bias_   = this->layer_param_.tb_param().use_bias();
  w_method_   = this->layer_param_.tb_param().w_binary();
  in_method_  = this->layer_param_.tb_param().in_binary();
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Reshape_more() {
  N_ = this->conv_out_spatial_dim_;
  BN_ = (N_ - 1) / BINARY_SIZE + 1;
  binary_in_.resize(max(N_ * BK_, K_ * BN_));
  mask_in_  .resize(max(N_ * BK_, K_ * BN_));
  scale_in_ .resize(max(N_, K_));
  bias_in_  .resize(max(N_, K_));
  sum_in_   .resize(max(N_, K_));
  sum2_in_  .resize(max(N_, K_));
  delta_in_ .resize(max(N_, K_));
  binary_g_ .resize(max(M_ * BN_, N_ * BM_));
  mask_g_   .resize(max(M_ * BN_, N_ * BM_));
  scale_g_  .resize(max(M_, N_));
  bias_g_   .resize(max(M_, N_));
  sum_g_    .resize(max(M_, N_));
  sum2_g_   .resize(max(M_, N_));
  delta_g_  .resize(max(M_, N_));
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_cpu_binary_gemm(
  const Dtype *input, Dtype *output, bool skip_im2col) {
  const Dtype *col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      this->conv_im2col_cpu(input, this->col_buffer_.mutable_cpu_data());
    }
    col_buff = this->col_buffer_.cpu_data();
  }
//    for (int g = 0; g < group_; ++g) {
//      caffe_cpu_gemm<Dtype>(
//        CblasNoTrans, CblasNoTrans, conv_out_channels_ /
//        group_, conv_out_spatial_dim_, kernel_dim_,
//        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
//        (Dtype)0., output + output_offset_ * g);
//    }
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
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::tb_backward_cpu_gemm(const Dtype *output,
    const Dtype *weights, Dtype *input) {
  Dtype *col_buff = this->col_buffer_.mutable_cpu_data();
  if (this->is_1x1_) {
    col_buff = input;
  }
  if (!skip_weight_binary_) {
    skip_weight_binary_ = true;
//    for (int g = 0; g < group_; ++g) {
    caffe_cpu_binary_norm<Dtype>(
      1, M_, K_, weights, binary_w_.data(), scale_w_.data(), bias_w_.data(),
      sum_w_.data(), use_bias_);
//    }
  }
  // dI = W' * dO
  caffe_cpu_ternary_norm<Dtype>(
    1, M_, N_, output,
    binary_g_.data(), mask_g_.data(), delta_g_.data(), scale_g_.data(),
    bias_g_.data(), sum_g_.data(), sum2_g_.data(), use_bias_);
  caffe_cpu_bt_gemm<Dtype>(
    true, false, K_, N_, M_,
    binary_w_.data(), scale_w_.data(),
    binary_g_.data(), mask_g_.data(), scale_g_.data(), sum2_g_.data(), col_buff,
    true, bias_w_.data(), sum_w_.data(), bias_g_.data(), sum_g_.data());
  if (!this->is_1x1_) {
    this->conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::tb_weight_cpu_gemm(const Dtype *input,
    const Dtype *output, Dtype *weights) {
  const Dtype *col_buff = input;
  if (!this->is_1x1_) {
    this->conv_im2col_cpu(input, this->col_buffer_.mutable_cpu_data());
    col_buff = this->col_buffer_.cpu_data();
  }
  Dtype *aux_weights = aux_[0]->mutable_cpu_diff();
  // dW = O * I'
//  for (int g = 0; g < group_; ++g) {
  caffe_cpu_binary_norm<Dtype>(
    0, M_, N_, output, binary_g_.data(), scale_g_.data(), bias_g_.data(),
    sum_g_.data(), use_bias_);
  caffe_cpu_ternary_norm<Dtype>(
    0, K_, N_, col_buff,
    binary_in_.data(), mask_in_.data(), delta_in_.data(), scale_in_.data(),
    bias_in_.data(), sum_in_.data(), sum2_in_.data(), use_bias_);
  caffe_cpu_bt_gemm<Dtype>(
    false, true, M_, K_, N_,
    binary_g_.data(), scale_g_.data(),
    binary_in_.data(), mask_in_.data(), scale_in_.data(), sum2_in_.data(),
    aux_weights,
    true, bias_g_.data(), sum_g_.data(), bias_in_.data(), sum_in_.data());
  caffe_cpu_axpby<Dtype>(this->conv_out_channels_ * K_,
                         Dtype(1.), aux_weights, Dtype(1.), weights);
//  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  const Dtype *weight = this->blobs_[0]->cpu_data();
//  Dtype *p = this->blobs_[0]->mutable_cpu_data();
//  for (int i = 0; i < K_ * M_; ++i) {
//    *p = std::max(std::min(*p, max_), min_);
//    p++;
//  }
  caffe_cpu_binary_norm<Dtype>(
    0, M_, K_, weight, binary_w_.data(), scale_w_.data(),
    bias_w_.data(), sum_w_.data(), use_bias_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->cpu_data();
    Dtype *top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_binary_gemm(bottom_data + n * this->bottom_dim_,
                                    top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype *bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom) {
  const Dtype *weight = this->blobs_[0]->cpu_data();
  Dtype *weight_diff  = this->blobs_[0]->mutable_cpu_diff();
  skip_weight_binary_ = false;
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
    if (full_train_) {
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                  top_diff + n * this->top_dim_, weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                    bottom_diff + n * this->bottom_dim_);
          }
        }
      }
    } else {
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        for (int n = 0; n < this->num_; ++n) {
          this->tb_weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                   top_diff + n * this->top_dim_, weight_diff);
        }
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[i]) {
        for (int n = 0; n < this->num_; ++n) {
          this->tb_backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                     bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

/*
#ifdef CPU_ONLY
STUB_GPU(TBConvolutionLayer);
#endif
*/
INSTANTIATE_CLASS(TBConvolutionLayer);
REGISTER_LAYER_CLASS(TBConvolution);
}  // namespace caffe

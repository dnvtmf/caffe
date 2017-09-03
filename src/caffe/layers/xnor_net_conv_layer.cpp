#include <vector>

#include "caffe/layers/xnor_net_conv_layer.hpp"

namespace caffe {
template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::LayerSetUp_more() {
  CHECK_EQ(this->group_, 1) << "in XnorNet Conv Layer, gourp must be 1.";
  M_ = this->conv_out_channels_;
  K_ = this->kernel_dim_;
  BM_ = (N_ - 1) / BINARY_SIZE + 1;
  BK_ = (K_ - 1) / BINARY_SIZE + 1;
  w_code_ .resize(M_ * BK_);
  w2_     .resize(M_ * K_);
  w_scale_.resize(M_);
}

template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::Reshape_more() {
  N_ = this->conv_out_spatial_dim_;
  BN_ = (N_ - 1) / BINARY_SIZE + 1;
  in_code_ .resize(N_ * BK_);
  in2_     .resize(N_ * K_);
  in_scale_.resize(N_);
}

template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::compute_output_shape() {
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
void XnorNetConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      this->conv_im2col_cpu(input, this->col_buffer_.mutable_cpu_data());
    }
    col_buff = this->col_buffer_.cpu_data();
  }
  caffe_cpu_binary<Dtype>(1, K_, N_, col_buff, in_code_.data(), in_scale_.data());
  caffe_cpu_binary_gemm<Dtype>(
    false, false, M_, N_, K_,
    w_code_.data(), w_scale_.data(), in_code_.data(), in_scale_.data(),
    output, false, nullptr, nullptr, nullptr, nullptr);
}
template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>:: backward_cpu_gemm(
  const Dtype *input, const Dtype *weight, const Dtype* top_diff,
  Dtype *input_diff, Dtype *weight_diff) {
  const Dtype* col_buff = input;
  Dtype* col_buff_diff = this->col_buffer_.mutable_cpu_diff();
  if (this->is_1x1_) {
    col_buff_diff = input_diff;
  } else {
    this->conv_im2col_cpu(input, this->col_buffer_.mutable_cpu_data());
    col_buff = this->col_buffer_.cpu_data();
  }
  caffe_cpu_binary<Dtype>(1, K_, N_, col_buff, in_code_.data(), in_scale_.data());
  caffe_cpu_binary_restore<Dtype>(
    1, K_, N_, in_code_.data(), in_scale_.data(), nullptr, false, in2_.data());
  caffe_cpu_gemm<Dtype>(
    CblasTrans, CblasNoTrans, K_, N_, M_,
    (Dtype)1., w2_.data(), top_diff, (Dtype)0., col_buff_diff);
  caffe_cpu_gemm<Dtype>(
    CblasNoTrans, CblasTrans, M_, K_, N_,
    (Dtype)1., top_diff, in2_.data(), (Dtype)1., weight_diff);
  caffe_cpu_binary_gradient<Dtype>(
    1, K_, N_, col_buff, in_scale_.data(), col_buff_diff);
  if (!this->is_1x1_) {
    this->conv_col2im_cpu(col_buff_diff, input_diff);
  }
}

template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_binary<Dtype>(0, M_, K_, weight, w_code_.data(), w_scale_.data());
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                             top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}
template <typename Dtype>
void XnorNetConvolutionLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>&top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_binary_restore<Dtype>(
    0, M_, K_, w_code_.data(), w_scale_.data(), nullptr, false, w2_.data());
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n)
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        backward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                          top_diff + n * this->top_dim_,
                          bottom_diff + n * this->bottom_dim_, weight_diff);
      }
    }
  }
  if (this->param_propagate_down_[0]) {
    caffe_cpu_binary_gradient<Dtype>(
      0, M_, K_, weight, w_scale_.data(),  weight_diff);
  }
}
#ifdef CPU_ONLY
STUB_GPU(XnorNetConvolutionLayer);
#endif
INSTANTIATE_CLASS(XnorNetConvolutionLayer);
REGISTER_LAYER_CLASS(XnorNetConvolution);
}  // namespace caffe

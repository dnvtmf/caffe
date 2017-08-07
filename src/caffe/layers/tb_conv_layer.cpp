#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/binary_math_function.hpp"

namespace caffe {

template <typename Dtype>
void TBConvolutionLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int *kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  }
  else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    for (int i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] =
        conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
    }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int *stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  }
  else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
                       conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int *pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  }
  else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
                    conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int *dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
      kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) {
      break;
    }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  CHECK_EQ(group_, 1) << "TBConvolutonLayer Cannot support group now!";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  }
  else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape = {conv_out_channels_, conv_in_channels_ / group_};
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
                 << weight_shaped_blob.shape_string() << "; instead, shape was "
                 << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
                 << bias_shaped_blob.shape_string() << "; instead, shape was "
                 << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  }
  else {
    if (bias_term_) {
      this->blobs_.resize(2);
    }
    else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>(
                                              this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
    // Initialize thw aux_
    aux_.resize(2);
    aux_[0].reset(new Blob<Dtype>(weight_shape));
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Initialize the vectors
  M_ = conv_out_channels_ / group_;
  K_ = kernel_dim_;
  BM_ = (N_ - 1) / BINARY_SIZE + 1;
  BK_ = (K_ - 1) / BINARY_SIZE + 1;
  binary_w_.resize(max(BM_ * K_, M_ * BK_));
  scale_w_ .resize(max(K_, M_));
  bias_w_  .resize(max(K_, M_));
  sum_w_   .resize(max(K_, M_));
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  full_train_ = this->layer_param_.tb_param().full_train();
  tb_use_bias_ = this->layer_param_.tb_param().use_bias();
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
                                        const vector<Blob<Dtype>*> &top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
                        bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  }
  else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  N_ = conv_out_spatial_dim_;
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
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int *conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    }
    else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    }
    else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
              bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_cpu_binary_gemm(
  const Dtype *input, Dtype *output, bool skip_im2col) {
  const Dtype *col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
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
    bias_in_.data(), sum_in_.data(), sum2_in_.data(), tb_use_bias_);
  caffe_cpu_bt_gemm<Dtype>(
    false, false, M_, N_, K_,
    binary_w_.data(), scale_w_.data(),
    binary_in_.data(), mask_in_.data(), scale_in_.data(), sum2_in_.data(),
    output,
    true, bias_w_.data(), sum_w_.data(), bias_in_.data(), sum_in_.data());
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_cpu_bias(Dtype *output,
    const Dtype *bias) {
  caffe_cpu_gemm<Dtype>(
    CblasNoTrans, CblasNoTrans, num_output_,
    out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
    (Dtype)1., output);
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::tb_backward_cpu_gemm(const Dtype *output,
    const Dtype *weights, Dtype *input) {
  Dtype *col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  if (!skip_weight_binary_) {
    skip_weight_binary_ = true;
//    for (int g = 0; g < group_; ++g) {
    caffe_cpu_binary_norm<Dtype>(
      1, M_, K_, weights, binary_w_.data(), scale_w_.data(), bias_w_.data(),
      sum_w_.data(), tb_use_bias_);
//    }
  }
  // dI = W' * dO
  caffe_cpu_ternary_norm<Dtype>(
    1, M_, N_, output,
    binary_g_.data(), mask_g_.data(), delta_g_.data(), scale_g_.data(),
    bias_g_.data(), sum_g_.data(), sum2_g_.data(), tb_use_bias_);
  caffe_cpu_bt_gemm<Dtype>(
    true, false, K_, N_, M_,
    binary_w_.data(), scale_w_.data(),
    binary_g_.data(), mask_g_.data(), scale_g_.data(), sum2_g_.data(), col_buff,
    true, bias_w_.data(), sum_w_.data(), bias_g_.data(), sum_g_.data());
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::tb_weight_cpu_gemm(const Dtype *input,
    const Dtype *output, Dtype *weights) {
  const Dtype *col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  Dtype *aux_weights = aux_[0]->mutable_cpu_diff();
  // dW = O * I'
//  for (int g = 0; g < group_; ++g) {
  caffe_cpu_binary_norm<Dtype>(
    0, M_, N_, output, binary_g_.data(), scale_g_.data(), bias_g_.data(),
    sum_g_.data(), tb_use_bias_);
  caffe_cpu_ternary_norm<Dtype>(
    0, K_, N_, col_buff,
    binary_in_.data(), mask_in_.data(), delta_in_.data(), scale_in_.data(),
    bias_in_.data(), sum_in_.data(), sum2_in_.data(), tb_use_bias_);
  caffe_cpu_bt_gemm<Dtype>(
    false, true, M_, K_, N_,
    binary_g_.data(), scale_g_.data(),
    binary_in_.data(), mask_in_.data(), scale_in_.data(), sum2_in_.data(),
    aux_weights,
    true, bias_g_.data(), sum_g_.data(), bias_in_.data(), sum_in_.data());
  caffe_cpu_axpby<Dtype>(conv_out_channels_ * kernel_dim_,
                         Dtype(1.), aux_weights, Dtype(1.), weights);
//  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_cpu_bias(Dtype *bias,
    const Dtype *input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                        input, bias_multiplier_.cpu_data(), 1., bias);
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
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
                           / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  const Dtype *weight = this->blobs_[0]->cpu_data();
  caffe_cpu_binary_norm<Dtype>(
    0, M_, K_, weight, binary_w_.data(), scale_w_.data(),
    bias_w_.data(), sum_w_.data(), tb_use_bias_);
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
    }
    else {
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


template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype *output,
    const Dtype *weights, Dtype *input) {
  Dtype *col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(
      CblasTrans, CblasNoTrans, kernel_dim_,
      conv_out_spatial_dim_, conv_out_channels_ / group_,
      (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
      (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype *input,
    const Dtype *output, Dtype *weights) {
  const Dtype *col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(
      CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
      kernel_dim_, conv_out_spatial_dim_,
      (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
      (Dtype)1., weights + weight_offset_ * g);
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

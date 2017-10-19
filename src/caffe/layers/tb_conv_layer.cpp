#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_conv_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TBConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes           = bottom[0]->num_axes();
  num_spatial_axes_            = num_axes - first_spatial_axis;
  CHECK_EQ(bottom[0]->num_axes(), 4) << "Only support 4D tensor";
  CHECK_EQ(num_spatial_axes_, 2);
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
  } else {
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
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0)
                           ? kDefaultStride
                           : conv_param.stride((num_stride_dims == 1) ? 0 : i);
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
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; " << num_spatial_axes_
        << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0)
                        ? kDefaultPad
                        : conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int *dilation_data          = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] =
        (num_dilation_dims == 0)
            ? kDefaultDilation
            : conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
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
  in_channels_ = bottom[0]->shape(channel_axis_);
  num_output_  = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(in_channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = num_output_;
  weight_shape[1] = in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  vector<int> scale_shape(1, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(2 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
                 << weight_shaped_blob.shape_string() << "; instead, shape was "
                 << this->blobs_[0]->shape_string();
    }
    if (scale_shape != this->blobs_[1]->shape()) {
      LOG(FATAL) << "Incorrect weight scale shape";
    }
    if (bias_term_ && bias_shape != this->blobs_[2]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
                 << bias_shaped_blob.shape_string() << "; instead, shape was "
                 << this->blobs_[2]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2 + bias_term_);
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // Initialize and fill the weight scales
    this->blobs_[1].reset(new Blob<Dtype>(scale_shape));
    caffe_set<Dtype>(num_output_, 1, this->blobs_[1]->mutable_cpu_data());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
    }
  }
  kernel_dim_    = this->blobs_[0]->count(1);
  out_channels_  = num_output_ / group_;
  weight_offset_ = out_channels_ * kernel_dim_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  sum_multiplier_.Reshape({kernel_dim_});
  caffe_set<Dtype>(
      sum_multiplier_.count(), 1, sum_multiplier_.mutable_cpu_data());
  weight_.ReshapeLike(*this->blobs_[0]);
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), in_channels_)
      << "Input size incompatible with convolution kernel.";
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(
      bottom[0]->shape().begin(), bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  top[0]->Reshape(top_shape);

  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  in_spatial_dim_  = bottom[0]->count(first_spatial_axis);
  col_offset_      = kernel_dim_ * out_spatial_dim_;
  output_offset_   = out_channels_ * out_spatial_dim_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int *conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    col_buffer_shape_.push_back(output_shape_[i]);
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_         = bottom[0]->count(channel_axis_);
  top_dim_            = top[0]->count(channel_axis_);
  num_kernels_im2col_ = in_channels_ * out_spatial_dim_;
  num_kernels_col2im_ = bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
  beta_.Reshape({num_ * group_, out_spatial_dim_});
  sum_.Reshape({num_ * group_, out_spatial_dim_});
  vector<int> b1_shape = {num_, group_, in_spatial_dim_};
  CHECK_EQ(bottom.size(), 3) << "the number of bottom is incorrect";
  CHECK(bottom[1]->shape() == b1_shape);
  CHECK(bottom[2]->shape() == b1_shape);
  beta_dim_ = group_ * out_spatial_dim_;
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::compute_output_shape() {
  const int *kernel_shape_data = this->kernel_shape_.cpu_data();
  const int *stride_data       = this->stride_.cpu_data();
  const int *pad_data          = this->pad_.cpu_data();
  const int *dilation_data     = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim     = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim =
        (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_cpu_gemm(Dtype *input, Dtype *output) {
  /*
  Dtype *col_buff = input;
  Dtype *weights  = (is_in_bin_ || is_w_bin_)
                       ? weight_.mutable_cpu_data()
                       : this->blobs_[0]->mutable_cpu_data();
  if (!this->is_1x1_) {
    col_buff = this->col_buffer_.mutable_cpu_data();
    this->conv_im2col_cpu(input, col_buff);
  }
  Dtype *in = (is_in_bin_ || is_w_bin_) ? in_.mutable_cpu_data() : col_buff;
  if (is_in_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      int offset = this->col_offset_ * g;
      caffe_cpu_binary_approx<Dtype>(
          1, K_, N_, use_bias_, col_buff + offset, in + offset,
          in_scale_ + N_ * g, in_bias_ + N_ * g);
    }
  } else if (is_w_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      int offset = this->col_offset_ * g;
      caffe_cpu_ternary_approx<Dtype>(
          1, K_, N_, use_bias_, col_buff + offset, in + offset,
          in_scale_ + N_ * g, in_bias_ + N_ * g, in_delta_ + N_ * g);
    }
  }
  for (int g = 0; g < this->group_; ++g) {
    caffe_cpu_gemm<Dtype>(
        CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype) 1.,
        weights + this->weight_offset_ * g, col_buff + this->col_offset_ * g,
        (Dtype) 0., output + this->output_offset_ * g);
  }
  */
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_cpu_gemm(Dtype *input,
    const Dtype *top_diff, Dtype *input_diff, Dtype *weight_diff) {
  /*
  Dtype *weight = (is_in_bin_ || is_w_bin_)
                      ? weight_.mutable_cpu_data()
                      : this->blobs_[0]->mutable_cpu_data();
  Dtype *col_buff      = input;
  Dtype *col_buff_diff = this->col_buffer_.mutable_cpu_diff();

  if (this->is_1x1_) {
    col_buff_diff = input_diff;
  } else {
    col_buff = this->col_buffer_.mutable_cpu_data();
    this->conv_im2col_cpu(input, col_buff);
  }
  Dtype *in = (is_in_bin_ || is_w_bin_) ? in_.mutable_cpu_data() : col_buff;

  if (is_in_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      int offset = this->col_offset_ * g;
      caffe_cpu_binary_approx<Dtype>(
          1, K_, N_, use_bias_, col_buff + offset, in + offset,
          in_scale_ + N_ * g, in_bias_ + N_ * g);
    }
  } else if (is_w_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      int offset = this->col_offset_ * g;
      caffe_cpu_ternary_approx<Dtype>(
          1, K_, N_, use_bias_, col_buff + offset, in + offset,
          in_scale_ + N_ * g, in_bias_ + N_ * g, in_delta_ + N_ * g);
    }
  }
  for (int g = 0; g < this->group_; ++g) {
    caffe_cpu_gemm<Dtype>(
        CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype) 1.,
        weight + this->weight_offset_ * g, top_diff + this->output_offset_ * g,
        (Dtype) 0., col_buff_diff + this->col_offset_ * g);
    caffe_cpu_gemm<Dtype>(
        CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1.,
        top_diff + this->output_offset_ * g, in + this->col_offset_ * g,
        (Dtype) 1., weight_diff + this->weight_offset_ * g);
  }

  if (is_in_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      int offset = this->col_offset_ * g;
      caffe_cpu_binary_gradient<Dtype>(
          1, K_, N_, use_bias_, col_buff + offset, in_scale_ + N_ * g,
          in_bias_ + N_ * g, col_buff_diff + offset);
    }
  } else if (is_w_bin_) {
    for (int g = 0; g < this->group_; ++g) {
      int offset = this->col_offset_ * g;
      caffe_cpu_ternary_gradient(
          1, K_, N_, use_bias_, col_buff + offset, in_scale_ + N_ * g,
          in_bias_ + N_ * g, in_delta_ + N_ * g, col_buff_diff + offset);
    }
  }
  if (!this->is_1x1_) {
    this->conv_col2im_cpu(col_buff_diff, input_diff);
  }
  */
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::weight_cpu_gemm(
    const Dtype *input, const Dtype *output, Dtype *weights) {
  /*
const Dtype *col_buff = input;
if (!is_1x1_) {
conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
col_buff = col_buffer_.cpu_data();
}
for (int g = 0; g < group_; ++g) {
caffe_cpu_gemm<Dtype>(
    CblasNoTrans, CblasTrans, out_channels_ / group_, kernel_dim_,
    out_spatial_dim_, (Dtype) 1., output + output_offset_ * g,
    col_buff + col_offset_ * g, (Dtype) 1., weights + weight_offset_ * g);
}
*/
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_cpu_bias(
    Dtype *bias, const Dtype *input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, out_channels_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  /*
  w_scale_  = weight_s_.mutable_cpu_data();
  w_bias_   = weight_s_.mutable_cpu_diff();
  w_delta_  = delta_.mutable_cpu_data();
  in_scale_ = in_s_.mutable_cpu_data();
  in_bias_  = in_s_.mutable_cpu_diff();
  in_delta_ = delta_.mutable_cpu_diff();

  Dtype *weight = this->blobs_[0]->mutable_cpu_data();
  if (clip_ & 1) {
    Dtype val = sqrt(6.0 / (M_ + K_));
    caffe_cpu_clip<Dtype>(M_ * K_, -val, val, weight);
  } else {
    caffe_cpu_clip<Dtype>(M_ * K_, -1, 1, weight);
  }

  if (is_w_bin_) {
    caffe_cpu_binary_approx<Dtype>(
        0, M_, K_, use_bias_, weight, weight_.mutable_cpu_data(), w_scale_,
        w_bias_);
  } else {
    caffe_cpu_ternary_approx<Dtype>(
        0, M_, K_, use_bias_, weight, weight_.mutable_cpu_data(), w_scale_,
        w_bias_, w_delta_);
  }
  //  caffe_cpu_binary_norm<Dtype>(
  //    0, M_, K_, weight, binary_w_.data(), scale_w_.data(),
  //    bias_w_.data(), sum_w_.data(), use_bias_);
  for (int i = 0; i < bottom.size(); ++i) {
    Dtype *bottom_data = bottom[i]->mutable_cpu_data();
    Dtype *top_data    = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(
          bottom_data + n * this->bottom_dim_, top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype *bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
  */
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  /*
  Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype *top_diff = top[i]->cpu_diff();
    Dtype *bottom_data    = bottom[i]->mutable_cpu_data();
    Dtype *bottom_diff    = bottom[i]->mutable_cpu_diff();
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
        0, M_, K_, use_bias_, this->blobs_[0]->cpu_data(), w_scale_, w_bias_,
        weight_diff);
  } else if (is_in_bin_) {
    caffe_cpu_ternary_gradient<Dtype>(
        0, M_, K_, use_bias_, this->blobs_[0]->cpu_data(), w_scale_, w_bias_,
        w_delta_, weight_diff);
  }
  */
}

#ifdef CPU_ONLY
STUB_GPU(TBConvolutionLayer);
#endif

INSTANTIATE_CLASS(TBConvolutionLayer);
REGISTER_LAYER_CLASS(TBConvolution);
}  // namespace caffe

#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/depthwise_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes           = bottom[0]->num_axes();
  num_spatial_axes_            = num_axes - first_spatial_axis;
  CHECK_EQ(num_spatial_axes_, 2) << "must have 2D spatial shape";
  // Setup filter kernel dimensions (kernel_shape_).
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    kernel_h_ = conv_param.kernel_size(0);
    kernel_w_ = conv_param.kernel_size((num_kernel_dims == 1) ? 0 : 1);
  }

  CHECK_GT(kernel_h_, 0) << "Filter dimensions must be nonzero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions must be nonzero.";

  // Setup stride dimensions (stride_).
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(
        num_stride_dims == 0 || num_stride_dims == 1 ||
        num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;

    stride_h_ = (num_stride_dims == 0) ? kDefaultStride : conv_param.stride(0);
    stride_w_ = (num_stride_dims == 0)
                    ? kDefaultStride
                    : conv_param.stride((num_stride_dims == 1) ? 0 : 1);
  }
  CHECK_GT(stride_h_, 0) << "Stride dimensions must be nonzero.";
  CHECK_GT(stride_w_, 0) << "Stride dimensions must be nonzero.";
  // Setup pad dimensions (pad_).
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(
        num_pad_dims == 0 || num_pad_dims == 1 ||
        num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; " << num_spatial_axes_
        << " spatial dims).";
    const int kDefaultPad = 0;
    pad_h_ = (num_pad_dims == 0) ? kDefaultPad : conv_param.pad(0);
    pad_w_ = (num_pad_dims == 0) ? kDefaultPad
                                 : conv_param.pad((num_pad_dims == 1) ? 0 : 1);
  }
  CHECK(conv_param.dilation_size() == 0) << "Dilation is not supported";
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape = {channels_, kernel_h_, kernel_w_};
  weight_spatial_dim_      = kernel_h_ * kernel_w_;

  bias_term_ = conv_param.bias_term();
  vector<int> bias_shape(bias_term_, channels_);
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
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype>> weight_filler(
        GetFiller<Dtype>(conv_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype>> bias_filler(
          GetFiller<Dtype>(conv_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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
  in_h_  = bottom[0]->shape(channel_axis_ + 1);
  in_w_  = bottom[0]->shape(channel_axis_ + 1);
  out_h_ = (in_h_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  out_w_ = (in_w_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  vector<int> top_shape(
      bottom[0]->shape().begin(), bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(channels_);
  top_shape.push_back(out_h_);
  top_shape.push_back(out_w_);
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  out_spatial_dim_ = out_h_ * out_w_;
  in_spatial_dim_  = in_h_ * in_w_;
  bottom_dim_      = bottom[0]->count(channel_axis_);
  top_dim_         = top[0]->count(channel_axis_);

  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(
        bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::forward_cpu_conv2D(
    const Dtype* input, const Dtype* weight, Dtype* output) {
  caffe_set(out_spatial_dim_, Dtype(0), output);
  for (int i = 0; i < kernel_h_; ++i) {
    for (int j = 0; j < kernel_w_; ++j) {
      for (int h = 0; h < out_h_; ++h) {
        for (int w = 0; w < out_w_; ++w) {
          int x = h * stride_h_ - pad_h_ + i;
          int y = w * stride_w_ - pad_w_ + j;
          if (is_a_ge_zero_and_a_lt_b(x, in_h_) &&
              is_a_ge_zero_and_a_lt_b(y, in_w_)) {
            output[h * out_w_ + w] +=
                input[x * in_w_ + y] * weight[i * kernel_w_ + j];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::backward_cpu_conv2D(
    const Dtype* diff, const Dtype* weight, Dtype* in_diff) {
  caffe_set(in_spatial_dim_, Dtype(0), in_diff);
  // in_diff = diff * rot180(weight)
  for (int h = 0; h < out_h_; ++h) {
    for (int w = 0; w < out_w_; ++w) {
      for (int i = 0; i < kernel_h_; ++i) {
        for (int j = 0; j < kernel_w_; ++j) {
          int x  = h * stride_h_ - pad_h_ + i;
          int y  = w * stride_w_ - pad_w_ + j;
          int ii = kernel_h_ - 1 - i;
          int jj = kernel_w_ - 1 - j;
          if (is_a_ge_zero_and_a_lt_b(x, in_h_) &&
              is_a_ge_zero_and_a_lt_b(y, in_w_)) {
            in_diff[x * in_w_ + y] +=
                diff[h * out_w_ + w] * weight[ii * kernel_w_ + jj];
          }
        }
      }
    }
  }
}
template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::weight_cpu_conv2D(
    const Dtype* diff, const Dtype* input, Dtype* w_diff) {
  caffe_set(weight_spatial_dim_, Dtype(0), w_diff);
  // w_diff = rot180(input) * diff
  for (int i = 0; i < kernel_h_; ++i) {
    for (int j = 0; j < kernel_w_; ++j) {
      for (int h = 0; h < out_h_; ++h) {
        for (int w = 0; w < out_w_; ++w) {
          int x = pad_h_ + in_h_ - 1 - (h * stride_h_ + i);
          int y = pad_w_ + in_w_ - 1 - (w * stride_w_ + j);
          if (is_a_ge_zero_and_a_lt_b(x, in_h_) &&
              is_a_ge_zero_and_a_lt_b(y, in_w_)) {
            w_diff[i * kernel_w_ + j] +=
                input[x * in_w_ + y] * diff[h * out_w_ + w];
          }
        }
      }
    }
  }
}
template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data          = top[i]->mutable_cpu_data();
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        forward_cpu_conv2D(
            bottom_data + n * bottom_dim_ + c * in_spatial_dim_,
            weight + c * weight_spatial_dim_,
            top_data + n * top_dim_ + c * out_spatial_dim_);
      }
    }
  }
  if (this->bias_term_) {
    const Dtype* bias = this->blobs_[1]->cpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
      Dtype* top_data = top[i]->mutable_cpu_data();
      for (int n = 0; n < this->num_; ++n) {
        caffe_cpu_gemm<Dtype>(
            CblasNoTrans, CblasNoTrans, channels_, out_spatial_dim_, 1,
            (Dtype) 1., bias, bias_multiplier_.cpu_data(), (Dtype) 1.,
            top_data + n * this->top_dim_);
      }
    }
  }
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    for (int i = 0; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->cpu_diff();
      Dtype* bias_diff      = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        caffe_cpu_gemv<Dtype>(
            CblasNoTrans, channels_, out_spatial_dim_, 1.,
            top_diff + n * this->top_dim_, bias_multiplier_.cpu_data(), 1.,
            bias_diff);
      }
    }
  }
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff  = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff    = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff       = bottom[i]->mutable_cpu_diff();
    // gradient w.r.t.weight.Note that we will accumulate diffs.
    if (this->param_propagate_down_[0]) {
      for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {
          weight_cpu_conv2D(
              bottom_data + n * bottom_dim_ + c * in_spatial_dim_,
              top_diff + n * top_dim_ + c * out_spatial_dim_,
              weight_diff + c * weight_spatial_dim_);
        }
      }
    }
    // gradient w.r.t. bottom data, if necessary.
    if (propagate_down[i]) {
      for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {
          backward_cpu_conv2D(
              top_diff + n * top_dim_ + c * out_spatial_dim_,
              weight + c * weight_spatial_dim_,
              bottom_diff + n * bottom_dim_ + c * in_spatial_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DepthwiseConvolutionLayer);
#endif

INSTANTIATE_CLASS(DepthwiseConvolutionLayer);
REGISTER_LAYER_CLASS(DepthwiseConvolution);
}  // namespace caffe

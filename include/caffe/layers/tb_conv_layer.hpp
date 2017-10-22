#ifndef CAFFE_BINARY_CONV_LAYER_HPP_
#define CAFFE_BINARY_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
class TBConvolutionLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - dilation (\b optional, default 1). The filter
   *  dilation, given by dilation_size for equal dimensions for different
   *  dilation. By default the convolution has dilation 1.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel
   * dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit TBConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int MinNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "TBConvolution"; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(Dtype* input, Dtype* output);
  void backward_cpu_gemm(Dtype* input, const Dtype* top_diff, Dtype* input_diff,
      Dtype* weight_diff);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void conv2D_gpu(const Dtype* in, const int num, Dtype* out);
  void forward_gpu_gemm(const Dtype* input, const Dtype* beta,
      const Dtype* weight, Dtype* output);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(
      const Dtype* output, const Dtype* weights, Dtype* input);
  void weight_gpu_gemm(const Dtype* input, const Dtype* output,
      const Dtype* beta, Dtype* weight_diff);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) { return (*bottom_shape_)[channel_axis_ + i]; }
  // Compute height_out_ and width_out_ from other parameters.
  void compute_output_shape();

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;
  int beta_dim_;

  int channel_axis_;
  int num_;
  int group_;
  int out_spatial_dim_;
  int in_spatial_dim_;
  int kernel_dim_;
  int in_channels_;
  int out_channels_;
  int num_output_;
  int weight_offset_;
  int col_offset_;
  int output_offset_;
  bool bias_term_;
  int num_kernels_im2col_;
  int num_kernels_col2im_;
  bool is_1x1_;
  bool beta_term_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> weight_;
  Blob<Dtype> beta_, sum_;

  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (num_spatial_axes_ == 2) {
      im2col_cpu(data, in_channels_, conv_input_shape_.cpu_data()[1],
          conv_input_shape_.cpu_data()[2], kernel_shape_.cpu_data()[0],
          kernel_shape_.cpu_data()[1], pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], dilation_.cpu_data()[0],
          dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(), pad_.cpu_data(),
          stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, in_channels_, conv_input_shape_.cpu_data()[1],
          conv_input_shape_.cpu_data()[2], kernel_shape_.cpu_data()[0],
          kernel_shape_.cpu_data()[1], pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], dilation_.cpu_data()[0],
          dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(), pad_.cpu_data(),
          stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (num_spatial_axes_ == 2) {
      im2col_gpu(data, in_channels_, conv_input_shape_.cpu_data()[1],
          conv_input_shape_.cpu_data()[2], kernel_shape_.cpu_data()[0],
          kernel_shape_.cpu_data()[1], pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], dilation_.cpu_data()[0],
          dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, in_channels_, conv_input_shape_.cpu_data()[1],
          conv_input_shape_.cpu_data()[2], kernel_shape_.cpu_data()[0],
          kernel_shape_.cpu_data()[1], pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], dilation_.cpu_data()[0],
          dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif
};

}  // namespace caffe

#endif  // CAFFE_BINARY_CONV_LAYER_HPP_

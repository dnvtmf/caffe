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
class TBConvolutionLayer : public BaseConvolutionLayer<Dtype> {
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
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "TBConvolution"; }

 protected:
  void LayerSetUp_more();
  void Reshape_more();
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(
      const Dtype* input, Dtype* output, bool skip_im2col = false);
  void backward_cpu_gemm(
      const Dtype* input, const Dtype* top_diff, Dtype* input_diff,
      Dtype* weight_diff);
#ifndef CPU_ONLY
  void forward_gpu_gemm(Dtype* input, Dtype* output, bool skip_im2col = false);
  void backward_gpu_gemm(
      Dtype* input, const Dtype* top_diff, Dtype* input_diff,
      Dtype* weight_diff);
#endif  // CPU_ONLY
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

 private:
  int M_, N_, K_;
  Blob<Dtype> in_, weight_;
  Blob<Dtype> in_s_, weight_s_, delta_;
  Dtype *w_scale_, *w_bias_, *w_delta_;
  Dtype *in_scale_, *in_bias_, *in_delta_;
  bool full_train_;
  bool use_bias_;
  bool is_w_bin_, is_in_bin_;
  int clip_;
  bool have_reg_;
  Dtype reg_;
};

}  // namespace caffe

#endif  // CAFFE_BINARY_CONV_LAYER_HPP_

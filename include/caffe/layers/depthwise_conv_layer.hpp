#ifndef CAFFE_DEPTHWISE_CONVOLUTION_LAYER_HPP_
#define CAFFE_DEPTHWISE_CONVOLUTION_LAYER_HPP_
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
class DepthwiseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit DepthwiseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }
  virtual inline const char* type() const { return "DepthwiseConvolution"; }

 protected:
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

 private:
  void forward_cpu_conv2D(
      const Dtype* input, const Dtype* weight, Dtype* output);
  void backward_cpu_conv2D(
      const Dtype* diff, const Dtype* weight, Dtype* in_diff);
  void weight_cpu_conv2D(const Dtype* diff, const Dtype* input, Dtype* w_diff);
#ifndef CPU_ONLY
  void forward_gpu_conv2D(
      const Dtype* input, const Dtype* weight, Dtype* output);
  void backward_gpu_conv2D(
      const Dtype* diff, const Dtype* weight, Dtype* in_diff);
  void weight_gpu_conv2D(const Dtype* diff, const Dtype* input, Dtype* w_diff);
#endif  // CPU_ONLY

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int out_spatial_dim_;
  int weight_spatial_dim_;
  int in_spatial_dim_;

  bool bias_term_;

  Blob<Dtype> bias_multiplier_;

  int in_h_, in_w_, kernel_h_, kernel_w_, out_h_, out_w_;
  int pad_h_, pad_w_, stride_h_, stride_w_;
};
}
#endif  // CAFFE_DEPTHWISE_CONVOLUTION_LAYER_HPP_
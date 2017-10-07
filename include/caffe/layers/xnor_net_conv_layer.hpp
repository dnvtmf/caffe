#ifndef CAFFE_CONV_LAYER_HPP_
#define CAFFE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"

namespace caffe {

template <typename Dtype>
class XnorNetConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit XnorNetConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "XnorNetConvolution"; }

 protected:
  virtual void LayerSetUp_more();
  virtual void Reshape_more();
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

 private:
  Blob<Dtype> weight_, weight_s_;
  Dtype *w_scale_, *w_bias_;
  int M_, N_, K_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_

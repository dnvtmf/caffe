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

  virtual inline const char* type() const {return "XnorNetConvolution";}

 protected:
  virtual void LayerSetUp_more();
  virtual void Reshape_more();
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output,
                        bool skip_im2col = false);
  void backward_cpu_gemm(
    const Dtype *input, const Dtype *weight, const Dtype* top_diff,
    Dtype *input_diff, Dtype *weight_diff);
  void backward_cpu_gemm(
    const Dtype* input, const Dtype* weights, Dtype* output);
  void weight_cpu_gemm(
    const Dtype* input, const Dtype* output, Dtype* weights);

  virtual void Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CHECK(false) << "no XnorNetConvolutionLayer::Forward_gpu()";
  }
  virtual void Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    CHECK(false) << "no XnorNetConvolutionLayer::Backward_gpu";
  }
  virtual inline bool reverse_dimensions() {return false;}
  virtual void compute_output_shape();
 private:
  int M_, K_, N_;
  int BM_, BK_, BN_;
  vector<Btype> w_code_, in_code_;
  vector<Dtype> w2_, in2_;
  vector<Dtype> w_scale_, in_scale_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_

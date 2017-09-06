#ifndef CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/binary_math_functions.hpp"

namespace caffe {

/**
 * @brief It is same with inner product layer.
 */
template <typename Dtype>
class XnorNetInnerProductLayer : public Layer<Dtype> {
 public:
  explicit XnorNetInnerProductLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "XnorInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

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


  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

 private:
  int binary_K_;
  vector<Btype> binary_weight_;
  vector<Dtype> binary_weight_scale_;
  vector<Btype> binary_input_;
  vector<Dtype> binary_input_scale_;
  vector<Dtype> input_temp_;
  vector<Dtype> weight_temp_;
};

}  // namespace caffe

#endif  // CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_

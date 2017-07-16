#ifndef CAFFE_BINARY_BT_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_BINARY_BT_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/binary_math_function.hpp"

namespace caffe {

template<typename Dtype>
class BTInnerProductLayer : public Layer<Dtype> {
public:
  explicit BTInnerProductLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                          const vector<Blob<Dtype>*> &top);
  virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
                       const vector<Blob<Dtype>*> &top);
  virtual inline const char* type() const {return "BTInnerProduct";}
  virtual inline int ExactNumBottomBlobs() const {return 1;}
  virtual inline int ExactNumTopBlobs() const {return 1;}
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                           const vector<Blob<Dtype>*> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype>*> &bottom);

  int M_, K_, N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
private:
  int bK_;
  vector<binary_t> w_pos_, w_neg_;
  vector<Dtype> w_scale_;
  Dtype w_delta_;
  vector<binary_t> in_, in_2_;
  vector<Dtype> in_scale_;
  vector<binary_t> g_, g_2_;
  vector<Dtype> g_scale_;
  Dtype in_delta_;
  Dtype g_delta_;
};

} // namespace caffe

#endif // CAFFE_BINARY_BT_INNER_PRODUCT_LAYER_HPP_

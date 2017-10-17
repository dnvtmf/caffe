#ifndef CAFFE_TERNARY_LAYER_HPP_
#define CAFFE_TERNARY

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class TernaryLayer : public Layer<Dtype> {
 public:
  explicit TernaryLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline const char* type() const { return "TernaryLayer"; }

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

 private:
  int channels_, group_, num_;
  float moving_average_fraction_;
  float threshold_t_;
};
}  // namespace caffe

#endif  // CAFFE_TERNARY_LAYER_HPP_

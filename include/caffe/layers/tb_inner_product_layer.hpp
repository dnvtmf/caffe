#ifndef CAFFE_TB_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_TB_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/binary_math_function.hpp"

namespace caffe {

/**
 * @brief It is same with inner product layer.
 */
template <typename Dtype>
class TBInnerProductLayer : public Layer<Dtype> {
public:
  explicit TBInnerProductLayer(const LayerParameter &param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                          const vector<Blob<Dtype>*> &top);
  virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
                       const vector<Blob<Dtype>*> &top);

  virtual inline const char *type() const { return "TBInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                           const vector<Blob<Dtype>*> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
                            const vector<bool> &propagate_down, const vector<Blob<Dtype>*> &bottom);
  /*
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  */

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;

private:
  int BM_;
  int BK_;
  int BN_;
  vector<Btype> binary_w_, binary_in_, binary_g_;
  vector<Btype>            mask_in_,   mask_g_;
  vector<Dtype> scale_w_,  scale_in_,  scale_g_;
  vector<Dtype> bias_w_,   bias_in_,   bias_g_;
  vector<Dtype>            delta_in_,  delta_g_;
  vector<Dtype> sum_w_,    sum_in_,    sum_g_;
  vector<Dtype>            sum2_in_,   sum2_g_;
//  vector<shared_ptr<Blob<Dtype> > > aux_;
  bool full_train_;
  Dtype min_, max_;
};

}  // namespace caffe

#endif  // CAFFE_TB_INNER_PRODUCT_LAYER_HPP_

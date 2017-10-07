#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/depthwise_conv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class DepthwiseConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DepthwiseConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4))
      , blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4))
      , blob_top_(new Blob<Dtype>())
      , blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_2_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_2_.push_back(blob_top_2_);
  }

  virtual ~DepthwiseConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_2_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_vec_2_;

 public:
  friend class DepthwiseConvolutionLayer<Dtype>;
  friend class ConvolutionLayer<Dtype>;
  friend class Layer<Dtype>;
};

TYPED_TEST_CASE(DepthwiseConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(DepthwiseConvolutionLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  Caffe::set_random_seed(1234);
  shared_ptr<Layer<Dtype>> layer(new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Caffe::set_random_seed(1234);
  shared_ptr<Layer<Dtype>> dw_layer(new ConvolutionLayer<Dtype>(layer_param));
  dw_layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_2_);
  dw_layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_2_);
  // Check against reference convolution.
  const Dtype* top_data_1 = this->blob_top_->cpu_data();
  const Dtype* top_data_2 = this->blob_top_2_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data_1[i], top_data_2[i], 1e-4);
  }
  // dw_layer->Backward(this->blob_top_vec_, {true}, this->blob_bottom_vec_);
}

TYPED_TEST(DepthwiseConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  // convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  DepthwiseConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(
      &layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}  // namespace caffe

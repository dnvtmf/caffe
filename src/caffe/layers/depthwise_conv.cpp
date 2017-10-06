#include "caffe/layers/depthwise_conv.hpp"

namespace caffe {

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {}
#ifdef CPU_ONLY
STUB_GPU(DepthwiseConvolutionLayer);
#endif

INSTANTIATE_CLASS(DepthwiseConvolutionLayer);
REGISTER_LAYER_CLASS(DepthwiseConvolution);
}  // namespace caffe

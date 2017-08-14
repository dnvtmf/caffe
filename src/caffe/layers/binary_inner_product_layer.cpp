#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/binary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  auto &params = this->layer_param_.inner_product_param();
  const int num_output = params.num_output();
  bias_term_ = params.bias_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(params.axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  }
  else {
    if (bias_term_) {
      this->blobs_.resize(2);
    }
    else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = N_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // Initialize the vectors
    BN_ = (N_ - 1) / BINARY_SIZE + 1;
    BK_ = (K_ - 1) / BINARY_SIZE + 1;
    binary_w_.resize(max(BN_ * K_, N_ * BK_));
    scale_w_ .resize(max(K_, N_));
    bias_w_  .resize(max(K_, N_));
    sum_w_   .resize(max(K_, N_));
    // fill the weights
    shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
        params.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>(params.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  full_train_  = true; // this->layer_param_.tb_param().full_train();
  use_bias_ = this->layer_param_.tb_param().use_bias();
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
                     this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  BM_ = (M_ - 1) / BINARY_SIZE + 1;
  binary_in_.resize(max(M_ * BK_, K_ * BM_));
  scale_in_ .resize(max(M_, K_));
  bias_in_  .resize(max(M_, K_));
  sum_in_   .resize(max(M_, K_));
  binary_g_ .resize(max(M_ * BN_, N_ * BM_));
  scale_g_  .resize(max(M_, N_));
  bias_g_   .resize(max(M_, N_));
  sum_g_    .resize(max(M_, N_));
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  const Dtype *weight = this->blobs_[0]->cpu_data();
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  caffe_cpu_binary<Dtype>(
    0, M_, K_, bottom[0]->mutable_cpu_data(), binary_in_.data());
  caffe_cpu_binary_norm<Dtype>(
    1, K_, N_, this->blobs_[0]->mutable_cpu_data(), binary_w_.data());
  caffe_cpu_binary_gemm<Dtype>(
    false, false, M_, N_, K_,
    binary_in_.data(), scale_in_.data(), binary_w_.data(), scale_w_.data(),
    top_data,
    use_bias_, bias_in_.data(), sum_in_.data(), bias_w_.data(), sum_w_.data());
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                          bias_multiplier_.cpu_data(),
                          this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom) {
  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *weight = this->blobs_[0]->cpu_data();
  if (this->param_propagate_down_[0]) {
    auto weight_diff = this->blobs_[0]->mutable_cpu_diff();
    // dW = In' x dO
    if (full_train_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                            K_, N_, M_,
                            (Dtype)1., bottom_data, top_diff,
                            (Dtype)0., weight_diff);
    }
    else {
      caffe_cpu_binary_norm<Dtype>(
        1, M_, K_, bottom_data, binary_in_.data(), scale_in_.data(),
        bias_in_.data(), sum_in_.data(), use_bias_);
      caffe_cpu_binary_norm<Dtype>(
        1, M_, N_, top_diff, binary_g_.data(), scale_g_.data(),
        bias_g_.data(), sum_g_.data(), use_bias_);
      caffe_cpu_binary_gemm<Dtype>(
        true, false, K_, N_, M_,
        binary_in_.data(), scale_in_.data(), binary_g_.data(), scale_g_.data(),
        weight_diff, use_bias_,
        bias_in_.data(), sum_in_.data(), bias_g_.data(), sum_g_.data());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                          bias_multiplier_.cpu_data(), (Dtype)1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    // dIn = dO x W'
    if (full_train_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                            Dtype(1.), top_diff, weight,
                            Dtype(0.), bottom[0]->mutable_cpu_diff());
    }
    else {
      caffe_cpu_binary_norm<Dtype>(
        0, M_, N_, top_diff, binary_g_.data(), scale_g_.data(),
        bias_g_.data(), sum_g_.data(), use_bias_);
      caffe_cpu_binary_norm<Dtype>(
        0, K_, N_, weight, binary_w_.data(), scale_w_.data(),
        bias_w_.data(), sum_w_.data(), use_bias_);
      caffe_cpu_binary_gemm<Dtype>(
        false, true, M_, K_, N_,
        binary_g_.data(), scale_g_.data(), binary_w_.data(), scale_w_.data(),
        bottom[0]->mutable_cpu_diff(), use_bias_,
        bias_g_.data(), sum_g_.data(), bias_w_.data(), sum_w_.data());
    }
  }
}

/*
#ifdef CPU_ONLY
STUB_GPU(BinaryInnerProductLayer);
#endif
*/

INSTANTIATE_CLASS(BinaryInnerProductLayer);
REGISTER_LAYER_CLASS(BinaryInnerProduct);

}  // namespace caffe

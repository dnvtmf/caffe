#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TBInnerProductLayer<Dtype>::LayerSetUp(
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
  }
  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  is_w_bin_  = this->layer_param_.tb_param().w_binary();
  is_in_bin_ = this->layer_param_.tb_param().in_binary();
  weight_.Reshape({K_, N_});
  weight_s_.Reshape({N_});
//  reg_ = this->layer_param_.tb_param().reg();
}

template <typename Dtype>
void TBInnerProductLayer<Dtype>::Reshape(
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
  in_.Reshape({M_, K_});
  in_s_.Reshape({M_});
  sum_.Reshape({max(M_, N_)});
}

template <typename Dtype>
void TBInnerProductLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  const Dtype *weight      = weight_.cpu_data();
  const Dtype *bottom_data = in_.cpu_data();
  Dtype *top_data          = top[0]->mutable_cpu_data();
  Dtype value = sqrt(6. / (K_ + N_));
  caffe_cpu_clip<Dtype>(K_ * N_, -value, value,
                        this->blobs_[0]->mutable_cpu_data());
  // binary or ternary the weight
  if (is_w_bin_) {
    caffe_cpu_binary_approx<Dtype>(
      1, K_, N_, this->blobs_[0]->cpu_data(),
      weight_.mutable_cpu_data(), weight_s_.mutable_cpu_data());
  }
  else if (is_in_bin_) {
    caffe_cpu_ternary_approx<Dtype>(
      1, K_, N_, this->blobs_[0]->cpu_data(), weight_.mutable_cpu_data(),
      weight_s_.mutable_cpu_data(), weight_s_.mutable_cpu_diff());
  }
  else
    weight = this->blobs_[0]->cpu_data();
  // ternary or binary the input
  if (is_in_bin_) {
    caffe_cpu_binary_approx<Dtype>(
      0, M_, K_, bottom[0]->cpu_data(),
      in_.mutable_cpu_data(), in_s_.mutable_cpu_data());
  }
  else if (is_w_bin_) {
    caffe_cpu_ternary_approx<Dtype>(
      0, M_, K_, bottom[0]->cpu_data(), in_.mutable_cpu_data(),
      in_s_.mutable_cpu_data(), in_s_.mutable_cpu_diff());
  }
  else
    bottom_data = bottom[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(
    CblasNoTrans, CblasNoTrans, M_, N_, K_,
    (Dtype)1., bottom_data, weight, (Dtype)0., top_data);
  // bias
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans, M_, N_, 1,
      (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
      (Dtype)1., top_data);
  }
}

template <typename Dtype>
void TBInnerProductLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
    const Dtype *bottom_data =
      (is_w_bin_ || is_in_bin_) ? in_.cpu_data() : bottom[0]->cpu_data();
    caffe_cpu_gemm<Dtype>(
      CblasTrans, CblasNoTrans, K_, N_, M_,
      (Dtype)1., bottom_data, top_diff,
      (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    if (is_w_bin_) {
      caffe_cpu_binary_gradient<Dtype>(
        1, K_, N_, this->blobs_[0]->cpu_data(), weight_s_.cpu_data(),
        this->blobs_[0]->mutable_cpu_diff());
    }
    else if (is_in_bin_) {
      caffe_cpu_ternary_gradient<Dtype>(
        1, K_, N_, this->blobs_[0]->cpu_data(), weight_s_.cpu_data(),
        weight_s_.cpu_diff(), this->blobs_[0]->mutable_cpu_diff());
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
    const Dtype* weight = (is_w_bin_ || is_in_bin_) ? weight_.cpu_data() :
                          this->blobs_[0]->cpu_data();
    caffe_cpu_gemm<Dtype>(
      CblasNoTrans, CblasTrans, M_, K_, N_,
      (Dtype)1., top_diff, weight,
      (Dtype)0., bottom[0]->mutable_cpu_diff());
    if (is_in_bin_) {
      caffe_cpu_binary_gradient<Dtype>(
        0, M_, K_, bottom[0]->cpu_data(), in_s_.cpu_data(),
        bottom[0]->mutable_cpu_diff());
    }
    else if (is_w_bin_) {
      caffe_cpu_ternary_gradient<Dtype>(
        0, M_, K_, bottom[0]->cpu_data(), in_s_.cpu_data(),
        in_s_.cpu_diff(), bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TBInnerProductLayer);
#endif

INSTANTIATE_CLASS(TBInnerProductLayer);
REGISTER_LAYER_CLASS(TBInnerProduct);

}  // namespace caffe

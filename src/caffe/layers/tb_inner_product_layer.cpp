#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TBInnerProductLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  auto &params         = this->layer_param_.inner_product_param();
  const int num_output = params.num_output();
  bias_term_           = params.bias_term();
  N_                   = num_output;
  const int axis       = bottom[0]->CanonicalAxisIndex(params.axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = N_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype>> weight_filler(
        GetFiller<Dtype>(params.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype>> bias_filler(
          GetFiller<Dtype>(params.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  full_train_ = true;
  use_bias_   = false;
  is_w_bin_   = true; 
  is_in_bin_  = false;
  clip_       = 0;
  reg_        = 0;
  have_reg_   = (is_w_bin_ || is_in_bin_) && abs(reg_) < 1e-10;
  weight_.Reshape({K_, N_});
  weight_s_.Reshape({N_});
  sum_multiplier_.Reshape({K_});
  caffe_set(K_, Dtype(1.), sum_multiplier_.mutable_cpu_data());
  LOG(INFO) << "\033[30;47m fc weight: " << (is_w_bin_ ? "binary" : "ternary")
            << "; input: " << (is_in_bin_ ? "binary" : "ternary")
            << "; bias: " << (use_bias_ ? "YES" : "NO") << "; clip: " << clip_
            << "; reg: " << reg_ << "\033[0m";
}

template <typename Dtype>
void TBInnerProductLayer<Dtype>::Reshape(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
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
  delta_.Reshape({max(M_, N_)});
}

template <typename Dtype>
void TBInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  w_scale_  = weight_s_.mutable_cpu_data();
  w_bias_   = weight_s_.mutable_cpu_diff();
  w_delta_  = delta_.mutable_cpu_data();
  in_scale_ = in_s_.mutable_cpu_data();
  in_bias_  = in_s_.mutable_cpu_diff();
  in_delta_ = delta_.mutable_cpu_diff();

  Dtype *weight   = weight_.mutable_cpu_data();
  Dtype *input    = in_.mutable_cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  if (clip_ & 1) {
    Dtype value = sqrt(6. / (K_ + N_));
    caffe_cpu_clip<Dtype>(K_ * N_, -value, value, weight);
  }
  if (clip_ & 2) {
    caffe_cpu_clip<Dtype>(M_ * K_, -1, 1., input);
  }
  // binary or ternary the weight
  if (is_w_bin_) {
    caffe_cpu_binary_approx<Dtype>(
        1, K_, N_, use_bias_, this->blobs_[0]->mutable_cpu_data(), weight,
        w_scale_, w_bias_);
  } else if (is_in_bin_) {
    caffe_cpu_ternary_approx<Dtype>(
        1, K_, N_, use_bias_, this->blobs_[0]->mutable_cpu_data(), weight,
        w_scale_, w_bias_, w_delta_);
  } else
    weight = this->blobs_[0]->mutable_cpu_data();

  // ternary or binary the input
  if (is_in_bin_) {
    caffe_cpu_binary_approx<Dtype>(
        0, M_, K_, use_bias_, bottom[0]->mutable_cpu_data(), input, in_scale_,
        in_bias_);
  } else if (is_w_bin_) {
    caffe_cpu_ternary_approx<Dtype>(
        0, M_, K_, use_bias_, bottom[0]->mutable_cpu_data(), input, in_scale_,
        in_bias_, in_delta_);
  } else
    input = bottom[0]->mutable_cpu_data();

  caffe_cpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype) 1., input, weight,
      (Dtype) 0., top_data);
  // bias
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(
        CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
        bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(), (Dtype) 1.,
        top_data);
  }
}

template <typename Dtype>
void TBInnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  const Dtype *top_diff = top[0]->cpu_diff();
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
    const Dtype *bottom_data =
        (is_w_bin_ || is_in_bin_) ? in_.cpu_data() : bottom[0]->cpu_data();
    Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_cpu_gemm<Dtype>(
        CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype) 1., bottom_data, top_diff,
        (Dtype) 1., weight_diff);
    if (is_w_bin_) {
      caffe_cpu_binary_gradient<Dtype>(
          1, K_, N_, use_bias_, this->blobs_[0]->cpu_data(), w_scale_, w_bias_,
          weight_diff);
    } else if (is_in_bin_) {
      caffe_cpu_ternary_gradient<Dtype>(
          1, K_, N_, use_bias_, this->blobs_[0]->cpu_data(), w_scale_, w_bias_,
          w_delta_, weight_diff);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(
        CblasTrans, M_, N_, (Dtype) 1., top_diff, bias_multiplier_.cpu_data(),
        (Dtype) 1., this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    const Dtype *weight = (is_w_bin_ || is_in_bin_)
                              ? weight_.cpu_data()
                              : this->blobs_[0]->cpu_data();
    Dtype *in_diff = bottom[0]->mutable_cpu_diff();
    caffe_cpu_gemm<Dtype>(
        CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1., top_diff, weight,
        (Dtype) 0., in_diff);
    if (is_in_bin_) {
      caffe_cpu_binary_gradient<Dtype>(
          0, M_, K_, use_bias_, bottom[0]->cpu_data(), in_scale_, in_bias_,
          in_diff);
    } else if (is_w_bin_) {
      caffe_cpu_ternary_gradient<Dtype>(
          0, M_, K_, use_bias_, bottom[0]->cpu_data(), in_scale_, in_bias_,
          in_delta_, in_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TBInnerProductLayer);
#endif

INSTANTIATE_CLASS(TBInnerProductLayer);
REGISTER_LAYER_CLASS(TBInnerProduct);

}  // namespace caffe

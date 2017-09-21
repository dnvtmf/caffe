#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void TBInnerProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  w_scale_  = weight_s_.mutable_gpu_data();
  w_bias_   = weight_s_.mutable_gpu_diff();
  w_delta_  = delta_.mutable_gpu_data();
  in_scale_ = in_s_.mutable_gpu_data();
  in_bias_  = in_s_.mutable_gpu_diff();
  in_delta_ = delta_.mutable_gpu_diff();

  Dtype *weight      = weight_.mutable_gpu_data();
  Dtype *bottom_data = in_.mutable_gpu_data();
  Dtype *top_data    = top[0]->mutable_gpu_data();

  if (clip_ & 1) {
    Dtype value = sqrt(6. / (K_ + N_));
    caffe_gpu_clip<Dtype>(
        K_ * N_, -value, value, this->blobs_[0]->mutable_gpu_data());
  }
  if (clip_ & 2) {
    caffe_gpu_clip<Dtype>(M_ * K_, -1., 1., bottom[0]->mutable_gpu_data());
  }

  // binary or ternary the weight
  if (is_w_bin_) {
    caffe_gpu_binary_approx<Dtype>(
        1, K_, N_, use_bias_, this->blobs_[0]->mutable_gpu_data(), weight,
        w_scale_, w_bias_);
  } else if (is_in_bin_) {
    caffe_gpu_ternary_approx<Dtype>(
        1, K_, N_, use_bias_, this->blobs_[0]->mutable_gpu_data(), weight,
        w_scale_, w_bias_, w_delta_);
  } else
    weight = this->blobs_[0]->mutable_gpu_data();

  // ternary or binary the input
  if (is_in_bin_) {
    caffe_gpu_binary_approx<Dtype>(
        0, M_, K_, use_bias_, bottom[0]->mutable_gpu_data(), bottom_data,
        in_scale_, in_bias_);
  } else if (is_w_bin_) {
    caffe_gpu_ternary_approx<Dtype>(
        0, M_, K_, use_bias_, bottom[0]->mutable_gpu_data(), bottom_data,
        in_scale_, in_bias_, in_delta_);
  } else
    bottom_data = bottom[0]->mutable_gpu_data();

  caffe_gpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype) 1., bottom_data, weight,
      (Dtype) 0., top_data);

  // bias
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(
        CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
        bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(), (Dtype) 1.,
        top_data);
  }
}

template <typename Dtype>
void TBInnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  const Dtype *top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
    Dtype *      weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype *weight      = this->blobs_[0]->gpu_data();
    Dtype *      delta       = weight_.mutable_gpu_diff();
    // Gradient with respect to weight
    const Dtype *bottom_data =
        (is_w_bin_ || is_in_bin_) ? in_.gpu_data() : bottom[0]->gpu_data();
    caffe_gpu_gemm<Dtype>(
        CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype) 1., bottom_data, top_diff,
        (Dtype) 1., weight_diff);
    if (have_reg_) {
      // D = (W' - W)
      caffe_gpu_sub<Dtype>(K_ * N_, weight_.gpu_data(), weight, delta);
      // W'_diff += reg_ * D
      caffe_gpu_axpy<Dtype>(K_ * N_, reg_, delta, weight_diff);
    }
    if (is_w_bin_) {
      caffe_gpu_binary_gradient<Dtype>(
          1, K_, N_, use_bias_, weight, w_scale_, w_bias_, weight_diff);
    } else if (is_in_bin_) {
      caffe_gpu_ternary_gradient<Dtype>(
          1, K_, N_, use_bias_, weight, w_scale_, w_bias_, w_delta_,
          weight_diff);
    }
    if (have_reg_) {
      // dW += -reg_ * D
      caffe_gpu_axpy<Dtype>(K_ * N_, -reg_, delta, weight_diff);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(
        CblasTrans, M_, N_, (Dtype) 1., top_diff, bias_multiplier_.gpu_data(),
        (Dtype) 1., this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    const Dtype *weight = (is_w_bin_ || is_in_bin_)
                              ? weight_.gpu_data()
                              : this->blobs_[0]->gpu_data();
    Dtype *      in_diff = bottom[0]->mutable_gpu_diff();
    const Dtype *in      = bottom[0]->gpu_data();
    Dtype *      delta   = in_.mutable_gpu_diff();
    // dI' = g * W'^
    caffe_gpu_gemm<Dtype>(
        CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1., top_diff, weight,
        (Dtype) 0., in_diff);
    if (have_reg_) {
      caffe_gpu_sub<Dtype>(M_ * K_, in_.gpu_data(), in, delta);  // D = I' - I
      caffe_gpu_axpy<Dtype>(M_ * K_, reg_, delta, in_diff);  // dI' += reg_ * D
    }
    if (is_in_bin_) {
      caffe_gpu_binary_gradient<Dtype>(
          0, M_, K_, use_bias_, in, in_scale_, in_bias_, in_diff);
    } else if (is_w_bin_) {
      caffe_gpu_ternary_gradient<Dtype>(
          0, M_, K_, use_bias_, in, in_scale_, in_bias_, in_delta_, in_diff);
    }
    if (have_reg_) {
      // dI += -reg_ * rD
      caffe_gpu_axpy<Dtype>(M_ * K_, -reg_, delta, in_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TBInnerProductLayer);
}  // namespace caffe

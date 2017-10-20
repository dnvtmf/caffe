#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/tb_conv_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void out(const char* info, const Blob<Dtype>& x) {
  printf("%s: (", info);
  for (auto y : x.shape()) printf("%d ", y);
  printf(")\n");
  for (int i = 0; i < min(100, x.count()); ++i) printf("%g ", x.cpu_data()[i]);
  if (x.count() > 100) printf("......");
  puts("");
}
template <typename Dtype>
void outt(const char* info, const Blob<Dtype>& x) {
  printf("%s: (", info);
  for (auto y : x.shape()) printf("%d ", y);
  printf(")\n");
  for (int i = 0; i < min(100, x.count()); ++i) printf("%g ", x.cpu_diff()[i]);
  if (x.count() > 100) printf("......");
  puts("");
}

template <typename Dtype>
void __global__ scale_kernel_1(
    const int n, const int width, const Dtype* alpha, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) { out[index] = out[index] * alpha[index / width]; }
}
template <typename Dtype>
void __global__ scale_kernel_2(
    const int n, const int width, const Dtype* beta, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) { out[index] = out[index] * beta[index % width]; }
}

template <typename Dtype>
void __global__ beta_div_kernel(const int n, const Dtype* sum, Dtype* beta) {
  CUDA_KERNEL_LOOP(index, n) {
    if (sum[index] > 0) beta[index] /= sum[index];
  }
}

template <typename Dtype>
__global__ void conv2D_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int height_col,
    const int width_col, Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index   = index / width_col;
    const int h_col     = h_index % height_col;
    const int w_col     = index % width_col;
    const int c_im      = h_index / height_col;
    const int h_offset  = h_col * stride_h - pad_h;
    const int w_offset  = w_col * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (c_im * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    *data_col_ptr = 0;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr +=
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im_ptr[i * dilation_h * width + j * dilation_w]
                : 0;
      }
    }
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::conv2D_gpu(
    const Dtype* in, const int num, Dtype* out) {
  const int height     = conv_input_shape_.cpu_data()[1];
  const int width      = conv_input_shape_.cpu_data()[2];
  const int kernel_h   = kernel_shape_.cpu_data()[0];
  const int kernel_w   = kernel_shape_.cpu_data()[1];
  const int pad_h      = pad_.cpu_data()[0];
  const int pad_w      = pad_.cpu_data()[1];
  const int stride_h   = stride_.cpu_data()[0];
  const int stride_w   = stride_.cpu_data()[1];
  const int dilation_h = dilation_.cpu_data()[0];
  const int dilation_w = dilation_.cpu_data()[1];

  int height_col =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = num * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  conv2D_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(num_kernels,
          in, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
          stride_w, dilation_h, dilation_w, height_col, width_col, out);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_gpu_gemm(
    const Dtype* input, const Dtype* beta, const Dtype* weight, Dtype* output) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    Dtype* temp = col_buffer_.mutable_gpu_data();
    for (int g = 0; g < group_; ++g) {
      scale_kernel_2<Dtype>
          <<<CAFFE_GET_BLOCKS(col_offset_), CAFFE_CUDA_NUM_THREADS>>>(
              col_offset_, out_spatial_dim_, beta + out_spatial_dim_ * g,
              temp + col_offset_ * g);
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, out_channels_,
        out_spatial_dim_, kernel_dim_, (Dtype) 1., weight + weight_offset_ * g,
        col_buff + col_offset_ * g, (Dtype) 0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::forward_gpu_bias(
    Dtype* output, const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype) 1., bias, bias_multiplier_.gpu_data(),
      (Dtype) 1., output);
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_gpu_gemm(
    const Dtype* output, const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        out_spatial_dim_, out_channels_, (Dtype) 1.,
        weights + weight_offset_ * g, output + output_offset_ * g, (Dtype) 0.,
        col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, const Dtype* beta, Dtype* weight_diff) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    Dtype* temp = col_buffer_.mutable_gpu_data();
    for (int g = 0; g < group_; ++g) {
      scale_kernel_2<Dtype>
          <<<CAFFE_GET_BLOCKS(col_offset_), CAFFE_CUDA_NUM_THREADS>>>(
              col_offset_, out_spatial_dim_, beta + out_spatial_dim_ * g,
              temp + col_offset_ * g);
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, out_channels_, kernel_dim_,
        out_spatial_dim_, (Dtype) 1., output + output_offset_ * g,
        col_buff + col_offset_ * g, (Dtype) 1.,
        weight_diff + weight_offset_ * g);
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::backward_gpu_bias(
    Dtype* bias, const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1., input,
      bias_multiplier_.gpu_data(), 1., bias);
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* alpha   = this->blobs_[1]->gpu_data();
  const Dtype* b1_data = bottom[1]->gpu_data();
  const Dtype* b2_data = bottom[2]->gpu_data();
  int count            = this->blobs_[0]->count();

  caffe_gpu_clip<Dtype>(count, -1, 1, this->blobs_[0]->mutable_gpu_data());
  caffe_gpu_sign<Dtype>(
      count, this->blobs_[0]->gpu_data(), weight_.mutable_gpu_data());
  scale_kernel_1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, kernel_dim_, alpha, weight_.mutable_gpu_data());

  if (!is_1x1_) {
    conv2D_gpu(b1_data, num_ * group_, beta_.mutable_gpu_data());
    conv2D_gpu(b2_data, num_ * group_, sum_.mutable_gpu_data());
  } else {
    caffe_copy<Dtype>(beta_.count(), b1_data, beta_.mutable_gpu_data());
    caffe_copy<Dtype>(sum_.count(), b2_data, sum_.mutable_gpu_data());
  }
  beta_div_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(beta_.count()), CAFFE_CUDA_NUM_THREADS>>>(
          beta_.count(), sum_.gpu_data(), beta_.mutable_gpu_data());
  const Dtype* beta = beta_.gpu_data();

  if (is_1x1_) {
    Dtype* bottom_data = bottom[0]->mutable_gpu_data();
    for (int i = 0; i < num_ * group_; ++i) {
      scale_kernel_2<Dtype>
          <<<CAFFE_GET_BLOCKS(col_offset_), CAFFE_CUDA_NUM_THREADS>>>(
              col_offset_, out_spatial_dim_, beta + out_spatial_dim_ * i,
              bottom_data + col_offset_ * i);
    }
  }

  const Dtype* weight      = weight_.gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data          = top[0]->mutable_gpu_data();
  for (int n = 0; n < num_; ++n) {
    forward_gpu_gemm(bottom_data + n * bottom_dim_, beta + n * beta_dim_,
        weight, top_data + n * top_dim_);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[2]->gpu_data();
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          out_spatial_dim_, 1, (Dtype) 1., bias, bias_multiplier_.gpu_data(),
          (Dtype) 1., top_data + n * top_dim_);
    }
  }
}

template <typename Dtype>
void TBConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight      = weight_.gpu_data();
  const Dtype* alpha       = this->blobs_[1]->gpu_data();
  const Dtype* top_diff    = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* beta        = beta_.gpu_data();

  // Bias gradient, if necessary.
  if (bias_term_ && this->param_propagate_down_[2]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    for (int n = 0; n < num_; ++n) {
      backward_gpu_bias(bias_diff, top_diff + n * top_dim_);
    }
  }
  // gradient w.r.t. weight. Note that we will accumulate diffs.
  if (this->param_propagate_down_[0]) {
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    for (int n = 0; n < num_; ++n) {
      weight_gpu_gemm(bottom_data + n * bottom_dim_, top_diff + n * top_dim_,
          beta + n * beta_dim_, weight_diff);
    }
  }
  // gradient w.r.t. bottom data, if necessary.
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    for (int n = 0; n < num_; ++n) {
      backward_gpu_gemm(
          top_diff + n * top_dim_, weight, bottom_diff + n * bottom_dim_);
    }
  }
  // gradient w.r.t weight scales and weight.
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* alpha_diff  = this->blobs_[1]->mutable_gpu_diff();
  const int count    = this->blobs_[0]->count();
  Dtype* temp_diff   = weight_.mutable_gpu_diff();
  caffe_gpu_sign<Dtype>(
      count, this->blobs_[0]->gpu_data(), weight_.mutable_gpu_data());
  caffe_gpu_mul<Dtype>(count, weight_diff, weight_.gpu_data(), temp_diff);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, kernel_dim_, (Dtype) 1.,
      weight_.gpu_diff(), sum_multiplier_.gpu_data(), (Dtype) 0., alpha_diff);
  scale_kernel_1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, kernel_dim_, alpha, weight_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(TBConvolutionLayer);
}  // namespace caffe

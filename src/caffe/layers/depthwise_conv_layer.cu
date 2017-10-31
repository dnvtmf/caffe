#include "caffe/layers/depthwise_conv_layer.hpp"
namespace caffe {
#define WARP_SIZE 32
template <typename Dtype>
__global__ void forward_kernel(
    const int n, const int height_in, const int width_in, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int height_out, const int width_out,
    const Dtype* in, const Dtype* weight, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index  = index / width_out;
    const int h_out    = h_index % height_out;
    const int w_out    = index % width_out;
    const int c_im     = h_index / height_out;
    const int h_offset = h_out * stride_h - pad_h;
    const int w_offset = w_out * stride_w - pad_w;
    const Dtype* in_ptr =
        in + (c_im * height_in + h_offset) * width_in + w_offset;
    const Dtype* weight_ptr = weight + c_im * kernel_h * kernel_w;

    Dtype temp = 0;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j, ++weight_ptr) {
        int h_im = h_offset + i;
        int w_im = w_offset + j;
        temp += (h_im >= 0 && w_im >= 0 && h_im < height_in && w_im < width_in)
                    ? in_ptr[i * width_in + j] * *weight_ptr
                    : 0;
      }
    }
    out[index] = temp;
  }
}

template <typename Dtype>
__global__ void backward_kernel(
    const int n, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int height_col, const int width_col,
    const Dtype* weight, const Dtype* out_diff, Dtype* in_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index       = index / width_col;
    const int h_col         = h_index % height_col;
    const int w_col         = index % width_col;
    const int c_im          = h_index / height_col;
    const int h_offset      = h_col * stride_h - pad_h;
    const int w_offset      = w_col * stride_w - pad_w;
    const Dtype* weight_ptr = weight + c_im * kernel_h * kernel_w;
    const Dtype temp_diff   = out_diff[index];

    Dtype* diff = in_diff + (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j, ++weight_ptr) {
        int h_im = h_offset + i;
        int w_im = w_offset + j;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          diff[i * width + j] += *weight_ptr * temp_diff;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void weight_backward_kernel(
    const int n, const int channels, const int height_in, const int width_in,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int height_out,
    const int width_out, const Dtype* in, const Dtype* out_diff,
    Dtype* weight_diff) {
  const int weight_index   = blockIdx.x;
  const int id             = threadIdx.x;
  const int weight_h_index = weight_index / kernel_w;
  const int i              = weight_h_index % kernel_h;
  const int j              = weight_index % kernel_w;
  const int c              = weight_h_index / kernel_h;

  Dtype val = 0;
  for (int index = id; index <= n / channels; index += blockDim.x) {
    const int h_index = index / width_out;
    const int h_out   = h_index % height_out;
    const int w_out   = index % width_out;
    const int offset  = h_index / height_out * channels + c;
    const int h_in    = h_out * stride_h - pad_h + i;
    const int w_in    = w_out * stride_w - pad_w + j;
    if (0 <= h_in && h_in < height_in && 0 <= w_in && w_in < width_in) {
      val += out_diff[(offset * height_out + h_out) * width_out + w_out] *
             in[(offset * height_in + h_in) * width_in + w_in];
    }
  }
  // reduce
  volatile __shared__ Dtype smem[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  if (id >= WARP_SIZE) smem[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
    for (int k = id; k < CAFFE_CUDA_NUM_THREADS - WARP_SIZE; k += WARP_SIZE)
      val += smem[k];
    smem[id] = val;
  }
  if (id < 16) smem[id] += smem[id + 16];
  if (id < 8) smem[id] += smem[id + 8];
  if (id < 4) smem[id] += smem[id + 4];
  if (id < 2) smem[id] += smem[id + 2];
  if (id < 1) smem[id] += smem[id + 1];
  if (id == 0) weight_diff[weight_index] = smem[0];
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();

  forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, in_h_, in_w_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_,
      stride_w_, out_h_, out_w_, bottom[0]->gpu_data(),
      this->blobs_[0]->gpu_data(), top_data);
  if (bias_term_) {
    const Dtype* bias_data = this->blobs_[1]->gpu_data();
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemm<Dtype>(
          CblasNoTrans, CblasNoTrans, channels_, out_spatial_dim_, 1,
          (Dtype) 1., bias_data, bias_multiplier_.gpu_data(), (Dtype) 1.,
          top_data + n * top_dim_);
    }
  }
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int count          = top[0]->count();
  const Dtype* top_diff    = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();

  if (bias_term_) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemv<Dtype>(
          CblasNoTrans, channels_, out_spatial_dim_, 1.,
          top_diff + n * top_dim_, bias_multiplier_.gpu_data(), 1., bias_diff);
    }
  }
  // gradient w.r.t. bottom data, if necessary.
  if (propagate_down[0]) {
    backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, in_h_, in_w_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_,
        stride_w_, out_h_, out_w_, weight_data, top_diff,
        bottom[0]->mutable_gpu_diff());
  }

  // gradient w.r.t. weight. Note that we will accumulate diffs.
  if (this->param_propagate_down_[0]) {
    weight_backward_kernel<Dtype>
        <<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
           CAFFE_CUDA_NUM_THREADS>>>(
            count, channels_, in_h_, in_w_, kernel_h_, kernel_w_, pad_h_,
            pad_w_, stride_h_, stride_w_, out_h_, out_w_, bottom_data, top_diff,
            this->blobs_[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DepthwiseConvolutionLayer);
}  // namespace caffe

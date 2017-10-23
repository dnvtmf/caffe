#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <thrust/functional.h>
#include "caffe/layers/ternary_layer.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/cuda_reduce.hpp"

namespace caffe {

inline __device__ float gpu_abs(float x) { return fabsf(x); }
inline __device__ double gpu_abs(double x) { return fabs(x); }

template <typename Dtype>
void __global__ beta_div_add_kernel(
    const int n, const Dtype *sum, const Dtype add_value, Dtype *beta) {
  CUDA_KERNEL_LOOP(index, n) {
    if (sum[index] > 0) beta[index] = (beta[index] + add_value) / sum[index];
  }
}

/**
\delta_c = \frac{t}{num * dim} \sum_{n=1}^{num}{\sum_{i=1}^{dim}{|in[n][c][i]|}}
*/
template <typename Dtype>
void __global__ delta_kernel(const int num, const int channels, const int dim,
    const Dtype threshold_t, const Dtype *in, Dtype *delta) {
  const int c  = blockIdx.x;
  const int id = threadIdx.x;
  volatile __shared__ Dtype temp[CAFFE_CUDA_NUM_THREADS - WARP_SIZE];
  Dtype val = 0;
  in += c * dim;
  for (int n = 0; n < num; ++n) {
    for (int j = id; j < dim; j += blockDim.x) {
      val += gpu_abs(in[j]);
    }
    in += channels * dim;
  }
  if (id >= WARP_SIZE) temp[id - WARP_SIZE] = val;
  __syncthreads();
  if (id < WARP_SIZE) {
#pragma unroll
    for (int k = id; k < (CAFFE_CUDA_NUM_THREADS - WARP_SIZE); k += WARP_SIZE)
      val += temp[k];
    temp[id] = val;
  }
  // reduce shared memory.
  if (id < 16) temp[id] = temp[id] + temp[id + 16];
  if (id < 8) temp[id]  = temp[id] + temp[id + 8];
  if (id < 4) temp[id]  = temp[id] + temp[id + 4];
  if (id < 2) temp[id]  = temp[id] + temp[id + 2];
  if (id < 1) temp[id]  = temp[id] + temp[id + 1];
  // save result
  if (id == 0) delta[blockIdx.x] = temp[id] * threshold_t;
}

template <typename Dtype>
bool eps_eq(Dtype a, Dtype b, Dtype eps = 1e-5) {
  return std::abs(a - b) <= eps * gpu_abs(a);
}
template <typename Dtype>
void delta_cpu_test(const int num, const int channels, const int dim,
    const Dtype t, const Dtype *in, const Dtype *delta) {
  for (int c = 0; c < channels; ++c) {
    Dtype val = 0;
    for (int n = 0; n < num; ++n) {
      for (int i = 0; i < dim; ++i) {
        val += std::abs(in[(n * channels + c) * dim + i]);
      }
    }
    val *= t;
    CHECK(eps_eq(val, delta[c])) << "Delta compute Error: " << val
                                 << "(test) vs. " << delta[c] << "(gpu)";
    CHECK_LE(val, 1.) << "Delta too big: " << val;
  }
}

template <typename Dtype>
void __global__ forward_kernel(const int n, const int channels, const int dim,
    const Dtype *delta, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int c = index / dim % channels;
    out[index]  = in[index] > delta[c] ? 1 : (in[index] < -delta[c] ? -1 : 0);
  }
}
template <typename Dtype>
void top0_cpu_test(const int num, const int channels, const int dim,
    const Dtype *delta, const Dtype *in, const Dtype *top0) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < dim; ++i) {
        Dtype in_value = in[(n * channels + c) * dim + i];
        Dtype val = in_value > delta[c] ? 1 : in_value < -delta[c] ? -1 : 0;
        CHECK(eps_eq(val, top0[(n * channels + c) * dim + i]))
            << "Top 0 Error: " << val << "(test) vs "
            << top0[(n * channels + c) * dim + i] << "(gpu)";
      }
    }
  }
}

template <typename Dtype>
void scale_cpu_test(const int num, const int group, const int channels,
    const int dim, const Dtype *in, const Dtype *out, const Dtype *beta,
    const Dtype *sum) {
  for (int n = 0; n < num; ++n) {
    for (int g = 0; g < group; ++g) {
      for (int i = 0; i < dim; ++i) {
        Dtype t_beta = 0, t_sum = 0;
        for (int c = 0; c < channels; ++c) {
          const int idx = ((n * group + g) * channels + c) * dim + i;
          t_beta += in[idx] * out[idx];
          t_sum += std::abs(out[idx]);
        }
        const int idx2 = (n * group + g) * dim + i;
        CHECK(eps_eq(t_beta, beta[idx2]))
            << "beta Error: " << t_beta << "(test) vs " << beta[idx2]
            << "(gpu)";
        CHECK(eps_eq(t_sum, sum[idx2]))
            << "sum Error: " << t_sum << "(test) vs " << sum[idx2] << "(gpu)";
        CHECK_NE(t_sum, 0) << "sum is zero";
      }
    }
  }
}

template <typename Dtype>
void __global__ backward_kernel(const int n, const int group_channels,
    const int dim, const Dtype *beta, const Dtype *out, Dtype *diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if (out[index] != 0) diff[index] *= beta[index / dim % group_channels];
  }
}

template <typename Dtype>
void TernaryLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int count = bottom[0]->count();
  // caffe_gpu_clip<Dtype>(count, -1., 1., bottom[0]->mutable_gpu_data());
  if (use_global_stats_) {
    // use the stored delta estimates.
    Dtype scale_factor                  = this->blobs_[1]->cpu_data()[0];
    if (scale_factor != 0) scale_factor = 1 / scale_factor;
    caffe_gpu_scale(delta_.count(), scale_factor, this->blobs_[0]->gpu_data(),
        delta_.mutable_gpu_data());
  } else {
    Dtype threshold_t = threshold_t_ / Dtype(count / channels_);
    delta_kernel<Dtype><<<channels_, CAFFE_CUDA_NUM_THREADS>>>(num_, channels_,
        dim_, threshold_t, bottom[0]->gpu_data(), delta_.mutable_gpu_data());
    /*
      CHECK_EQ(count, num_ * channels_ * dim_) << "Error shape";
      delta_cpu_test<Dtype>(num_, channels_, dim_, threshold_t,
          bottom[0]->cpu_data(), delta_.cpu_data());
    */
    // compute and save moving average
    this->blobs_[1]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[1]->mutable_cpu_data()[0] += 1;
    caffe_gpu_axpby<Dtype>(delta_.count(), 1, delta_.gpu_data(),
        moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
  }

  forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels_, dim_, delta_.gpu_data(), bottom[0]->gpu_data(),
      top[0]->mutable_gpu_data());
  /*
  top0_cpu_test<Dtype>(num_, channels_, dim_, delta_.cpu_data(),
      bottom[0]->cpu_data(), top[0]->cpu_data());
  */
  if (scale_term_) {
    caffe_gpu_input_scale<Dtype>(num_ * group_, channels_ / group_, dim_,
        bottom[0]->gpu_data(), top[0]->gpu_data(), top[1]->mutable_gpu_data(),
        top[2]->mutable_gpu_data());
    /*
    scale_cpu_test(num_, group_, channels_ / group_, dim_,
        bottom[0]->cpu_data(), top[0]->cpu_data(), top[1]->cpu_data(),
        top[2]->cpu_data());
    */
  }
}

template <typename Dtype>
void TernaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
    if (scale_term_) {
      beta_div_add_kernel<Dtype>
          <<<CAFFE_GET_BLOCKS(top[1]->count()), CAFFE_CUDA_NUM_THREADS>>>(
              top[1]->count(), top[2]->gpu_data(), 1,
              top[1]->mutable_gpu_diff());
      backward_kernel<Dtype>
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
              channels_ / group_, dim_, top[1]->gpu_diff(), top[0]->gpu_data(),
              bottom[0]->mutable_gpu_diff());
    }
    // caffe_gpu_clip_grad(
    // count, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryLayer);
}  // namespace caffe

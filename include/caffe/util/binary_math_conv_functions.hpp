#ifndef BINARY_MATH_CONV_FUNCTION_HPP_INCLUDED
#define BINARY_MATH_CONV_FUNCTION_HPP_INCLUDED

#include "caffe/util/binary_math_functions.hpp"
namespace caffe {
template <typename Dtype>
void caffe_cpu_binary_conv2D(
  const int in_c, const int in_h, const int in_w,
  const Btype *in, const Dtype *in_scale,
  const int filter_num, const int filter_h, const int filter_w,
  const Btype *filter, const Dtype *filter_scale,
  const int stride_h, const int stride_w,
  const int pad_h, const int pad_w,
  Dtype *out);

template <typename Dtype>
void caffe_cpu_binary(
  const int num, const int channel, const int height, const int width,
  const Dtype *in, Btype *out, Dtype *scale);
}
#endif // BINARY_MATH_CONV_FUNCTION_HPP_INCLUDED

#include <algorithm>
#include "caffe/util/math_functions.hpp"
#include "caffe/util/binary_math_conv_functions.hpp"

using std::min;
namespace caffe {
template <typename Dtype>
void caffe_cpu_binary_conv2D(
  const int in_c, const int in_h, const int in_w,
  const Btype *in, const Dtype *in_scale,
  const int filter_num, const int filter_h, const int filter_w,
  const Btype *filter, const Dtype *filter_scale,
  const int stride_h, const int stride_w,
  const int pad_h, const int pad_w,
  Dtype *out) {
  const int BC = (in_c - 1) / BINARY_SIZE + 1;
  const int out_c = filter_num;
  const int out_h = (in_h - filter_h + 2 * pad_h) / stride_h + 1;
  const int out_w = (in_w - filter_w + 2 * pad_w) / stride_w + 1;
  const int off_in_wc  = in_w * BC;
  const int off_in_swc = stride_h * in_w * BC;
  const int off_in_sw = stride_h * in_w;
  const int off_in_sc  = stride_w * BC;

  caffe_set<Dtype>(out_c * out_h * out_w, 0, out);
  const Btype *filter_p = filter;
  const Dtype *fs       = filter_scale;
  for (int fid = 0; fid < filter_num; ++fid) {
    for (int i = 0; i < filter_h; ++i) {
      for (int j = 0; j < filter_w; ++j) {
        Dtype *out_st = out + fid * out_h * out_w;
        const Btype *w      = filter_p; filter_p += BC;
        const Dtype *ws     = fs; fs++;
        const Btype *x      = in + (i - pad_h) * off_in_wc;
        const Dtype *xs     = in_scale + (i - pad_h) * in_w;
        const Btype *x_end  =
          in + min(in_h, in_h + pad_h - filter_h + i + 1) * off_in_wc;

        for (; x < in   ; x += off_in_swc, out_st += out_w, xs += off_in_sw);
        for (; x < x_end; x += off_in_swc, out_st += out_w, xs += off_in_sw) {
                Dtype *out_p   = out_st;
          const Btype *y       = x + (j - pad_w) * BC;
          const Dtype *ys      = xs + (j - pad_w);
          const Btype *y_end   =
            x + min(in_w, in_w + pad_w - filter_w + j + 1) * BC;

          for (; y < x    ; y += off_in_sc, ++out_p, ys += stride_w);
          for (; y < y_end; y += off_in_sc, ++out_p, ys += stride_w) {
            Btype temp = 0;
            const Btype *z = y;
            for (; w < filter_p; ++w) {
              temp += bitcount(*z++ ^ *w);
            }
            *out_p += *ys * *ws * (in_c - (int)(temp << 1));
            w -= BC;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void caffe_cpu_binary(
  const int num, const int channel, const int height, const int width,
  const Dtype *in, Btype *out, Dtype *scale) {
  const int binary_channel = (channel - 1) / BINARY_SIZE + 1;
  caffe_set<Btype>(num * binary_channel * height * width, 0, out);
  caffe_set<Dtype>(num * height * width, 0, scale);
  for (int n = 0; n < num; ++n) {
    Btype *pp = out   + n * height * width * binary_channel;
    Dtype *ss = scale + n * height * width;
    for (int c = 0; c < channel;) {
      for (int k = 0; k < BINARY_SIZE && c < channel; ++c, ++k) {
        Btype *p = pp;
        Dtype *s = ss;
        for (int j = 0; j < height * width; ++j) {
          if (*in >= 0) {
            *p |= Btype(1) << k;
            *s += *in;
          }
          else {
            *s -= *in;
          }
          ++in;
          ++s;
          p += binary_channel;
        }
      }
      ++pp;
    }
  }

  caffe_scal<Dtype>(num * height * width, Dtype(1. / channel), scale);
}

#define INIT_FUNC(Dtype) \
  \
template void caffe_cpu_binary_conv2D<Dtype>( \
  const int in_c, const int in_h, const int in_w, \
  const Btype *in, const Dtype *in_scale,  \
  const int filter_num, const int filter_h, const int filter_w, \
  const Btype *filter, const Dtype *filter_scale, \
  const int stride_h, const int stride_w, \
  const int pad_h, const int pad_w, \
  Dtype *out);  \
  \
template void caffe_cpu_binary<Dtype>(  \
  const int num, const int channel, const int height, const int width, \
  const Dtype *in, Btype *out, Dtype *scale);

INIT_FUNC(float);
INIT_FUNC(double);
}

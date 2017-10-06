#include <bits/stdc++.h>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
namespace caffe {
template <typename Dtype>
class SpeedupTest : public ::testing::Test {
 private:
  Blob<Dtype> R[2], res[3];
  Blob<Btype> B[4];
  Blob<Dtype> scale[3], bias[3], sum[3], sum2, delta;
  Dtype Error(int N, const Dtype *a, const Dtype *b) {
    Dtype sum = 0;
    for (int i = 0; i < N; ++i) {
      sum += abs(b[i] - a[i]);
    }
    sum /= N;
    return sum;
  }
  void gen_data(int m, int k, int n) {
    R[0].Reshape({m, k});
    R[1].Reshape({k, n});
    int bk = (k - 1) / BINARY_SIZE + 1;
    caffe_rng_uniform<Dtype>(m * k, -1., 1., R[0].mutable_cpu_data());
    caffe_rng_uniform<Dtype>(k * n, -1., 1., R[1].mutable_cpu_data());
    B[0].Reshape({m, bk});
    scale[0].Reshape({m});
    bias[0].Reshape({m});
    sum[0].Reshape({m});
    caffe_cpu_binary_norm<Dtype>(
        0, m, k, R[0].mutable_cpu_data(), B[0].mutable_cpu_data(),
        scale[0].mutable_cpu_data(), bias[0].mutable_cpu_data(),
        sum[0].mutable_cpu_data(), false);
    B[1].Reshape({bk, n});
    B[2].Reshape({bk, n});
    B[3].Reshape({bk, n});
    scale[1].Reshape({n});
    bias[1].Reshape({n});
    sum[1].Reshape({n});
    scale[2].Reshape({n});
    bias[2].Reshape({n});
    sum[2].Reshape({n});
    sum2.Reshape({n});
    delta.Reshape({n});
    caffe_cpu_binary_norm<Dtype>(
        1, k, n, R[1].mutable_cpu_data(), B[1].mutable_cpu_data(),
        scale[1].mutable_cpu_data(), bias[1].mutable_cpu_data(),
        sum[1].mutable_cpu_data(), false);
    caffe_cpu_ternary_norm<Dtype>(
        1, k, n, R[1].mutable_cpu_data(), B[2].mutable_cpu_data(),
        B[3].mutable_cpu_data(), delta.mutable_cpu_data(),
        scale[2].mutable_cpu_data(), bias[2].mutable_cpu_data(),
        sum[2].mutable_cpu_data(), sum2.mutable_cpu_data(), false);

    // result
    res[0].Reshape({m, n});
    res[1].Reshape({m, n});
    res[2].Reshape({m, n});
  }

 protected:
  void test_cpu(int m, int k, int n, int test_iter = 10) {
    Timer timer;
    gen_data(m, k, n);
    // full precision
    timer.Start();
    for (int it = 0; it < test_iter; ++it) {
      caffe_cpu_gemm<Dtype>(
          CblasNoTrans, CblasNoTrans, m, n, k, (Dtype) 1., R[0].cpu_data(),
          R[1].cpu_data(), (Dtype) 0., res[0].mutable_cpu_data());
    }
    timer.Stop();
    float full_average_time = timer.MilliSeconds() / float(test_iter);
    // binary_binary_gemm
    timer.Start();
    for (int it = 0; it < test_iter; ++it) {
      caffe_cpu_binary_gemm<Dtype>(
          false, false, m, n, k, B[0].cpu_data(), scale[0].cpu_data(),
          B[1].cpu_data(), scale[1].cpu_data(), res[1].mutable_cpu_data(),
          false, bias[0].cpu_data(), sum[0].cpu_data(), bias[1].cpu_data(),
          sum[1].cpu_data());
    }
    timer.Stop();
    float bb_average_time = timer.MilliSeconds() / float(test_iter);
    // binary_ternary_gemm
    timer.Start();
    for (int it = 0; it < test_iter; ++it) {
      caffe_cpu_bt_gemm<Dtype>(
          false, false, m, n, k, B[0].cpu_data(), scale[0].cpu_data(),
          B[2].cpu_data(), B[3].cpu_data(), scale[2].cpu_data(),
          sum2.cpu_data(), res[2].mutable_cpu_data(), false, bias[0].cpu_data(),
          sum[0].cpu_data(), bias[2].cpu_data(), sum[2].cpu_data());
    }
    timer.Stop();
    float bt_average_time = timer.MilliSeconds() / float(test_iter);
    Dtype error_full_bb   = Error(m * n, res[0].cpu_data(), res[1].cpu_data());
    Dtype error_full_bt   = Error(m * n, res[0].cpu_data(), res[2].cpu_data());
    cout << "full gemm use time: " << full_average_time << "ms\n";
    cout << "binary gemm use time: " << bb_average_time << "ms\n";
    cout << "ternary-binary gemm use time: " << bt_average_time << "ms\n";
    cout << "speed up binary gemm: \033[31m"
         << full_average_time / bb_average_time << "\033[0m\n";
    cout << "speed up ternary-binary gemm: \033[31m"
         << full_average_time / bt_average_time << "\033[0m\n";
    cout << "average error binary: " << error_full_bb << "(binary),  "
         << error_full_bt << "(binary-ternary)\n";
  }
};
TYPED_TEST_CASE(SpeedupTest, TestDtypes);
TYPED_TEST(SpeedupTest, K1000) {
  this->test_cpu(1000, 64, 1000);
  this->test_cpu(1000, 128, 1000);
  this->test_cpu(1000, 256, 1000);
  this->test_cpu(1000, 512, 1000);
  this->test_cpu(1000, 1024, 1000);
  this->test_cpu(1000, 2048, 1000);
  this->test_cpu(1000, 4096, 1000);
}
}  // namespace caffe
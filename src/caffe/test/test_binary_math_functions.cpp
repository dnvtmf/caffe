#include <bits/stdc++.h>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
namespace caffe {
#ifndef CPU_ONLY
template <typename Dtype>
class BinaryMathFunctionsTest : public ::testing::Test {
 private:
  Blob<Dtype> data, result;
  vector<Btype> code, mask;
  Blob<Dtype> scale, bias, delta, sum, sum2;
  Dtype eps;

 protected:
  BinaryMathFunctionsTest() {
    if (sizeof(Dtype) == sizeof(double))
      eps = 1e-6;
    else
      eps = 1e-5;
  }
  Dtype Error(int N, const Dtype *a, const Dtype *b) {
    Dtype sum = 0;
    for (int i = 0; i < N; ++i) {
      sum += abs(b[i] - a[i]);
    }
    sum /= N;
    if (sum >= eps) {
      printf("\033[031m \n");
      for (int i = 0; i < N; ++i) {
        if (abs(a[i] - b[i]) >= eps) printf("(%d: %g) ", i, abs(a[i] - b[i]));
      }
      printf("\033[0m\n");
    }
    return sum;
  }

  void binary(int M, int N, int axis, bool use_bias) {
    data.Reshape({M, N});
    result.Reshape({M, N});
    code.resize(M * N);
    scale.Reshape({max(M, N)});
    bias.Reshape({max(M, N)});
    sum.Reshape({max(M, N)});
    caffe_rng_uniform<Dtype>(M * N, -1., 1., data.mutable_cpu_data());
    caffe_copy(M * N, data.cpu_data(), data.mutable_cpu_diff());
    caffe_cpu_binary_norm<Dtype>(
        axis, M, N, data.cpu_data(), code.data(), scale.mutable_cpu_data(),
        bias.mutable_cpu_data(), sum.mutable_cpu_data(), use_bias);
    caffe_cpu_binary_restore<Dtype>(
        axis, M, N, code.data(), scale.cpu_data(), bias.cpu_data(), use_bias,
        result.mutable_cpu_data());
    caffe_gpu_binary_approx<Dtype>(
        axis, M, N, use_bias, data.mutable_gpu_data(),
        result.mutable_gpu_diff(), scale.mutable_gpu_diff(),
        bias.mutable_gpu_diff());
    Dtype diff;
    diff = Error(M * N, data.cpu_data(), data.cpu_diff());
    EXPECT_LT(diff, eps) << "data check error " << axis << ' ' << use_bias;
    diff = Error(M * N, result.cpu_data(), result.cpu_diff());
    EXPECT_LT(diff, eps) << "approx check error " << axis << ' ' << use_bias;
    diff = Error(axis ? N : M, scale.cpu_data(), scale.cpu_diff());
    EXPECT_LT(diff, eps) << "scale check error " << axis << ' ' << use_bias;
    if (use_bias) {
      diff = Error(axis ? N : M, bias.cpu_data(), bias.cpu_diff());
      EXPECT_LT(diff, eps) << "bias check error " << axis << ' ' << use_bias;
    }

    // check binary gradient
    caffe_cpu_binary_gradient<Dtype>(
        axis, M, N, use_bias, data.cpu_data(), scale.cpu_data(),
        bias.cpu_data(), result.mutable_cpu_data());
    caffe_gpu_binary_gradient<Dtype>(
        axis, M, N, use_bias, data.gpu_data(), scale.gpu_diff(),
        bias.gpu_diff(), result.mutable_gpu_diff());
    diff = Error(M * N, result.cpu_data(), result.cpu_diff());
    EXPECT_LT(diff, eps) << "gradient check error " << axis << ' ' << use_bias;
  }
  void ternary(int M, int N, int axis, bool use_bias) {
    data.Reshape({M, N});
    result.Reshape({M, N});
    code.resize(M * N);
    mask.resize(M * N);
    scale.Reshape({max(M, N)});
    bias.Reshape({max(M, N)});
    delta.Reshape({max(M, N)});
    sum.Reshape({max(M, N)});
    sum2.Reshape({max(M, N)});
    caffe_rng_uniform<Dtype>(M * N, -1., 1., data.mutable_cpu_data());
    caffe_copy(M * N, data.cpu_data(), data.mutable_cpu_diff());
    caffe_cpu_ternary_norm<Dtype>(
        axis, M, N, data.cpu_data(), code.data(), mask.data(),
        delta.mutable_cpu_data(), scale.mutable_cpu_data(),
        bias.mutable_cpu_data(), sum.mutable_cpu_data(),
        sum2.mutable_cpu_data(), use_bias);
    caffe_cpu_ternary_restore<Dtype>(
        axis, M, N, code.data(), mask.data(), scale.cpu_data(), bias.cpu_data(),
        use_bias, result.mutable_cpu_data());
    caffe_gpu_ternary_approx<Dtype>(
        axis, M, N, use_bias, data.mutable_gpu_data(),
        result.mutable_gpu_diff(), scale.mutable_gpu_diff(),
        bias.mutable_gpu_diff(), delta.mutable_gpu_diff());
    Dtype diff;
    diff = Error(M * N, data.cpu_data(), data.cpu_diff());
    EXPECT_LT(diff, eps) << "data check error " << axis << ' ' << use_bias;
    diff = Error(M * N, result.cpu_data(), result.cpu_diff());
    EXPECT_LT(diff, eps) << "approx check error " << axis << ' ' << use_bias;
    diff = Error(axis ? N : M, scale.cpu_data(), scale.cpu_diff());
    EXPECT_LT(diff, eps) << "scale check error " << axis << ' ' << use_bias;
    diff = Error(axis ? N : M, delta.cpu_data(), delta.cpu_diff());
    EXPECT_LT(diff, eps) << "delta check error " << axis << ' ' << use_bias;
    if (use_bias) {
      diff = Error(axis ? N : M, bias.cpu_data(), bias.cpu_diff());
      EXPECT_LT(diff, eps) << "bias check error " << axis << ' ' << use_bias;
    }

    // check ternary gradient
    caffe_cpu_ternary_gradient<Dtype>(
        axis, M, N, use_bias, data.cpu_data(), scale.cpu_data(),
        bias.cpu_data(), delta.cpu_data(), result.mutable_cpu_data());
    caffe_gpu_ternary_gradient<Dtype>(
        axis, M, N, use_bias, data.gpu_data(), scale.gpu_diff(),
        bias.gpu_diff(), delta.gpu_diff(), result.mutable_gpu_diff());
    diff = Error(M * N, result.cpu_data(), result.cpu_diff());
    EXPECT_LT(diff, eps) << "gradient check error " << axis << ' ' << use_bias;
  }
};
const int M = 1000, N = 1111;
TYPED_TEST_CASE(BinaryMathFunctionsTest, TestDtypes);

TYPED_TEST(BinaryMathFunctionsTest, Binary) {
  this->binary(M, N, 0, false);
  this->binary(M, N, 1, false);
}

TYPED_TEST(BinaryMathFunctionsTest, BinaryBias) {
  this->binary(M, N, 0, true);
  this->binary(M, N, 1, true);
}

TYPED_TEST(BinaryMathFunctionsTest, Ternary) {
  this->ternary(M, N, 0, false);
  this->ternary(M, N, 1, false);
}

TYPED_TEST(BinaryMathFunctionsTest, TernaryBias) {
  this->ternary(M, N, 0, true);
  this->ternary(M, N, 1, true);
}
#endif  // CPU_ONLY
}  // namespace caffe

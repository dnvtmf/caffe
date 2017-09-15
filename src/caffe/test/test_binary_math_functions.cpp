#include <bits/stdc++.h>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/binary_math_functions.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
namespace caffe {
const double eps = 1e-6;
template <typename Dtype>
class BinaryMathFunctionsTest : public ::testing::Test {
 private:
  Blob<Dtype> data, result;
  mt19937 mt;
  uniform_real_distribution<Dtype> dis;
  vector<Btype> code, mask;
  Blob<Dtype> scale, bias, delta, sum, sum2;

 protected:
  BinaryMathFunctionsTest() : mt(time(0)), dis(-1, 1) {}
  Dtype Error(int N, const Dtype *a, const Dtype *b) {
    Dtype sum = 0;
    printf("\n");
    for (int i = 0; i < N; ++i) {
      printf("%.6g ", abs(b[i] - a[i]));
      sum += abs(b[i] - a[i]);  // / max(Dtype(eps), a[i]);
    }
    printf("\n");
    sum /= N;
    printf("\033[031m error sum: %f\033[0m\n", sum);
    return sum;
  }

  void binary(int M, int N, int axis, bool use_bias) {
    data.Reshape({M, N});
    result.Reshape({M, N});
    code.resize(M * N);
    scale.Reshape({max(M, N)});
    bias.Reshape({max(M, N)});
    sum.Reshape({max(M, N)});
    auto p = data.mutable_cpu_data();
    for (int i = 0; i < M * N; ++i) {
      *p++ = dis(mt);
    }
    caffe_copy(M * N, data.cpu_data(), data.mutable_cpu_diff());
    // p = data.mutable_cpu_data();
    // for (int i = 0; i < M * N; ++i) cout << *p++ << ' ';
    // cout << endl;
    caffe_cpu_binary_norm<Dtype>(
        axis, M, N, data.cpu_data(), code.data(), scale.mutable_cpu_data(),
        bias.mutable_cpu_data(), sum.mutable_cpu_data(), use_bias);
    caffe_cpu_binary_restore<Dtype>(
        axis, M, N, code.data(), scale.cpu_data(), bias.cpu_data(), use_bias,
        result.mutable_cpu_data());
    // p = result.mutable_cpu_data();
    caffe_gpu_binary_approx<Dtype>(
        axis, M, N, use_bias, data.mutable_gpu_data(),
        result.mutable_gpu_diff(), scale.mutable_gpu_diff(),
        bias.mutable_gpu_diff());
    // p = result.mutable_cpu_diff();
    Dtype diff;
    diff = Error(M * N, data.cpu_data(), data.cpu_diff());
    EXPECT_LT(diff, eps);
    diff = Error(M * N, result.cpu_data(), result.cpu_diff());
    EXPECT_LT(diff, eps);
    diff = Error(axis ? N : M, scale.cpu_data(), scale.cpu_diff());
    EXPECT_LT(diff, eps);
    if (use_bias) {
      diff = Error(axis ? N : M, bias.cpu_data(), bias.cpu_diff());
      EXPECT_LT(diff, eps);
    }
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
    auto p = data.mutable_cpu_data();
    for (int i = 0; i < M * N; ++i) {
      *p++ = dis(mt);
    }
    caffe_copy(M * N, data.cpu_data(), data.mutable_cpu_diff());
    // p = data.mutable_cpu_data();
    // for (int i = 0; i < M * N; ++i) cout << *p++ << ' ';
    // cout << endl;
    caffe_cpu_ternary_norm<Dtype>(
        axis, M, N, data.cpu_data(), code.data(), mask.data(),
        delta.mutable_cpu_data(), scale.mutable_cpu_data(),
        bias.mutable_cpu_data(), sum.mutable_cpu_data(),
        sum2.mutable_cpu_data(), use_bias);
    caffe_cpu_ternary_restore<Dtype>(
        axis, M, N, code.data(), mask.data(), scale.cpu_data(), bias.cpu_data(),
        use_bias, result.mutable_cpu_data());
    // p = result.mutable_cpu_data();
    // for (int i = 0; i < M * N; ++i) cout << *p++ << ' ';
    // cout << endl;
    caffe_gpu_ternary_approx<Dtype>(
        axis, M, N, use_bias, data.mutable_gpu_data(),
        result.mutable_gpu_diff(), scale.mutable_gpu_diff(),
        bias.mutable_gpu_diff(), delta.mutable_gpu_diff());
    // p = result.mutable_cpu_diff();
    // for (int i = 0; i < M * N; ++i) cout << *p++ << ' ';
    // cout << endl;
    Dtype diff;
    diff = Error(M * N, data.cpu_data(), data.cpu_diff());
    EXPECT_LT(diff, eps);
    diff = Error(M * N, result.cpu_data(), result.cpu_diff());
    EXPECT_LT(diff, eps);
    diff = Error(axis ? N : M, scale.cpu_data(), scale.cpu_diff());
    EXPECT_LT(diff, eps);
    diff = Error(axis ? N : M, delta.cpu_data(), delta.cpu_diff());
    EXPECT_LT(diff, eps);
    if (use_bias) {
      diff = Error(axis ? N : M, bias.cpu_data(), bias.cpu_diff());
      EXPECT_LT(diff, eps);
    }
  }
};
const int M = 10, N = 11;
TYPED_TEST_CASE(BinaryMathFunctionsTest, TestDtypes);

TYPED_TEST(BinaryMathFunctionsTest, Binary) {
  this->binary(M, N, 0, false);
  // this->binary(M, N, 1, false);
}
/*
TYPED_TEST(BinaryMathFunctionsTest, BinaryBias) {
  TypeParam re = this->binary(M, N, 0, true);
  EXPECT_NEAR(re, 0, (TypeParam) 1e-6);

  re = this->binary(M, N, 1, true);
  EXPECT_NEAR(re, 1, (TypeParam) 1e-6);
}
TYPED_TEST(BinaryMathFunctionsTest, Ternary) {
  TypeParam re = this->ternary(M, N, 0, false);
  EXPECT_NEAR(re, 0, (TypeParam) 1e-6);

  re = this->ternary(M, N, 1, false);
  EXPECT_NEAR(re, 1, (TypeParam) 1e-6);
}
TYPED_TEST(BinaryMathFunctionsTest, TernaryBias) {
  TypeParam re = this->ternary(M, N, 0, true);
  EXPECT_NEAR(re, 0, (TypeParam) 1e-6);

  re = this->ternary(M, N, 1, true);
  EXPECT_NEAR(re, 1, (TypeParam) 1e-6);
}
*/
}

#include "caffe/util/binary_math_function.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template<typename Dtype>
void caffe_cpu_bgemm(const int M, const int N, const int K, const binary_t* A,
                     const binary_t* B, Dtype *C) {
  caffe_set(M * N, 0, C);
  binary_t *pA = A, *pB;
  Dtype *pC;
  for(int i = 0; i < M; ++i) {
    pB = B;
    for(int k = 0; k < K; ++k) {
      pC = C + i * N;
      binary_t vA = *(pA++);
      for(int j = 0; j < N; ++j) {
        *(pC++) += bitcount(vA & *(pB++));
      }
    }
  }
}

}

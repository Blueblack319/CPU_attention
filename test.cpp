#include <cuda_fp16.h>
#include <immintrin.h>

#include <cstdint>
#include <iostream>
#include <thread>

int main() {
  size_t Dh = 128, num_head = 32, ITER = 51, batch_size = 32, K = 40;
  // Allocate memory
  half *values;
  half *logits;
  half *result;

  values = static_cast<half *>(
      aligned_alloc(32, num_head * batch_size * K * Dh * sizeof(half)));
  logits = static_cast<half *>(
      aligned_alloc(32, num_head * batch_size * K * sizeof(half)));
  result = static_cast<half *>(
      aligned_alloc(32, num_head * batch_size * Dh * sizeof(half)));

  for (int i = 0; i < 32; ++i) {
    values[i] = static_cast<half>(i + 1);
    logits[i] = static_cast<half>(i + 1);
    result[i] = 0;
  }
  std::cout << "Before Multiply and Add" << std::endl;
  for (int i = 0; i < 32; ++i) {
    printf("values[%d]: %f, logits[%d]: %f, result[%d]: %f\n", i,
           __half2float(values[i]), i, __half2float(logits[i]), i,
           __half2float(result[i]));
  }

  float logit = __half2float(logits[1]);
  __m256 logit_vec = _mm256_set1_ps(logit);

  __m256 c00 = _mm256_setzero_ps();

  __m256 v00 = _mm256_cvtph_ps(_mm_load_si128((__m128i *)(values)));

  c00 = _mm256_fmadd_ps(logit_vec, v00, c00);

  _mm_store_si128(
      (__m128i *)(result),
      _mm256_cvtps_ph(c00, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

  std::cout << "\nAfter Multiply and Add" << "\n\n";
  for (int i = 0; i < 32; ++i) {
    printf("values[%d]: %f, logits[%d]: %f, result[%d]: %f\n", i,
           __half2float(values[i]), i, __half2float(logits[i]), i,
           __half2float(result[i]));
  }

  free(values);
  free(logits);
  free(result);
}
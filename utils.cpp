#include "utils.h"

inline float hsum(__m128 x) {
  x = _mm_add_ps(x, _mm_movehl_ps(x, x));
  x = _mm_add_ss(x, _mm_movehdup_ps(x));

  // __m128 t;
  // t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
  // x = _mm_add_ps(x, t);
  // t = _mm_movehl_ps(t, x);
  // x = _mm_add_ss(x, t);

  return _mm_cvtss_f32(x);
  // __m128 t = _mm_add_ps(x, _mm_movehl_ps(x, x));  // add high and low
  // t = _mm_add_ps(
  //     t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 0, 3, 2)));  // add across lanes
  // return _mm_cvtss_f32(t);                                // get the result
}

inline float hsum(__m256 x) {
  return hsum(
      _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}

void flush_cache() {
  size_t cache_flusher_size = 512 * 1024 * 1024;  // 512 MB
  char *cache_flusher = (char *)malloc(cache_flusher_size);

  for (size_t i = 0; i < cache_flusher_size; i += 4096) {
    cache_flusher[i] = 0;
  }

  free(cache_flusher);
}

float calculate_mse(const float *C, const float *golden_output,
                    const size_t m) {
  float mse = 0.0;
  for (size_t i = 0; i < m; ++i) {
    mse += std::pow(C[i] - golden_output[i], 2);
  }
  return mse / m;
}
float calculate_mae(const float *C, const float *golden_output,
                    const size_t m) {
  float mae = 0.0;
  for (size_t i = 0; i < m; ++i) {
    mae = std::max(mae, std::fabs(C[i] - golden_output[i]));
  }
  return mae;
}

float calculate_mse_half(const half *C, const float *golden_output,
                         const size_t m) {
  float mse = 0.0;
  for (size_t i = 0; i < m; ++i) {
    mse += std::pow(__half2float(C[i]) - golden_output[i], 2);
  }
  return mse / m;
}
float calculate_mae_half(const half *C, const float *golden_output,
                         const size_t m) {
  float mae = 0.0;
  for (size_t i = 0; i < m; ++i) {
    mae = std::max(mae, std::fabs(__half2float(C[i]) - golden_output[i]));
  }
  return mae;
}

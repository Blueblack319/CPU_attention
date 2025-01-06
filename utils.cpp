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

float calculate_mse_half_gqa(const half *C, const float *golden_output,
                             const int topk_num, const int q_head_num,
                             const int kv_head_num, const int batch_size,
                             const int Dh, const int result_batch_offset,
                             const int result_head_offset) {
  float mse = 0.0;
  const int q_per_kv = q_head_num / kv_head_num;
  const int m = batch_size * kv_head_num * Dh;

  for (size_t i = 0; i < batch_size; ++i) {                    // B
    for (size_t kv_idx = 0; kv_idx < kv_head_num; ++kv_idx) {  // H
      int q_idx = kv_idx * q_per_kv;
      // if (q_idx >= 24) {
      //   printf("B: %d, q_head: %d\n", i, q_idx);
      //   printf("C: %f, G: %f ",
      //          __half2float(
      //              C[i * result_batch_offset + q_idx * result_head_offset]),
      //          golden_output[i * result_batch_offset +
      //                        q_idx * result_head_offset]);
      // }
      for (size_t l = 0; l < Dh; ++l) {  // Dh
        if (golden_output[i * result_batch_offset + q_idx * result_head_offset +
                          l] != 0.f) {
          mse += std::pow(
              __half2float(
                  C[i * result_batch_offset + q_idx * result_head_offset + l]) -
                  golden_output[i * result_batch_offset +
                                q_idx * result_head_offset + l],
              2);
        } else {
          printf("B: %d, H: %d, ERROR!!!!\n\n\n", i, kv_idx);
        }
      }
    }
  }

  return mse / m;
}
float calculate_mae_half_gqa(const half *C, const float *golden_output,
                             const size_t m) {
  float mae = 0.0;
  for (size_t i = 0; i < m; ++i) {
    mae = std::max(mae, std::fabs(__half2float(C[i]) - golden_output[i]));
  }
  return mae;
}

void generateUniqueRandomValues(int *arr, int size, int min, int max) {
  int count = 0;
  while (count < size) {
    int num = rand() % (max - min + 1) + min;
    bool isUnique = true;

    // Check if the number is already in the array
    for (int i = 0; i < count; ++i) {
      if (arr[i] == num) {
        isUnique = false;
        break;
      }
    }

    // Add the number to the array if it is unique
    if (isUnique) {
      arr[count] = num;
      ++count;
    }
  }
}

bool fileExists(const char *filename) {
  std::ifstream file(filename);
  return file.good();
}

void saveBinary(const char *filename, void *data, size_t size) {
  std::ofstream out(filename, std::ios::binary);
  out.write(reinterpret_cast<char *>(data), size);
  out.close();
}

void loadBinary(const char *filename, void *data, size_t size) {
  std::ifstream in(filename, std::ios::binary);
  in.read(reinterpret_cast<char *>(data), size);
  in.close();
}
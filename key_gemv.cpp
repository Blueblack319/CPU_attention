#include "key_gemv.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

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
  //     t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 0, 3, 2)));  // add across
  // lanes
  // return _mm_cvtss_f32(t);                                // get the
  // result
}

inline float hsum(__m256 x) {
  return hsum(
      _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}

void key_gemv__threaded_2(float* keys, const float* queries, float* logits,
                          int const num_head, int const batch_size, int const K,
                          int const Dh, int const keys_head_offset,
                          int const keys_batch_offset, int const q_haed_offset,
                          int const q_batch_offset,
                          int const logits_head_offset,
                          int const logits_batch_offset) {
  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      for (int k = 0; k < K; k += 8) {
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        __m256 c04 = _mm256_setzero_ps();
        __m256 c05 = _mm256_setzero_ps();
        __m256 c06 = _mm256_setzero_ps();
        __m256 c07 = _mm256_setzero_ps();
        for (int l = 0; l < Dh; l += 8) {
          // Prefetching the next query and keys for the next iteration
          //   if (l + 8 < Dh)
          //     _mm_prefetch((const char*)(queries + i * q_haed_offset +
          //                                j * q_batch_offset + l + 8),
          //                  _MM_HINT_T0);
          //   if (k + 8 < K)
          //     _mm_prefetch(
          //         (const char*)(keys + i * keys_head_offset +
          //                       j * keys_batch_offset + (k + 8) * Dh + l),
          //         _MM_HINT_T0);

          __m256 q0 = _mm256_loadu_ps(queries + i * q_haed_offset +
                                      j * q_batch_offset + l);

          __m256 k0 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + k * Dh + l);
          __m256 k1 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 1) * Dh + l);
          __m256 k2 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 2) * Dh + l);
          __m256 k3 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 3) * Dh + l);
          __m256 k4 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 4) * Dh + l);
          __m256 k5 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 5) * Dh + l);
          __m256 k6 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 6) * Dh + l);
          __m256 k7 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 7) * Dh + l);
          c00 = _mm256_fmadd_ps(q0, k0, c00);
          c01 = _mm256_fmadd_ps(q0, k1, c01);
          c02 = _mm256_fmadd_ps(q0, k2, c02);
          c03 = _mm256_fmadd_ps(q0, k3, c03);
          c04 = _mm256_fmadd_ps(q0, k4, c04);
          c05 = _mm256_fmadd_ps(q0, k5, c05);
          c06 = _mm256_fmadd_ps(q0, k6, c06);
          c07 = _mm256_fmadd_ps(q0, k7, c07);
        }
        logits[i * logits_head_offset + j * logits_batch_offset + k] =
            hsum(c00);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 1] =
            hsum(c01);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 2] =
            hsum(c02);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 3] =
            hsum(c03);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 4] =
            hsum(c04);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 5] =
            hsum(c05);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 6] =
            hsum(c06);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 7] =
            hsum(c07);
      }
    }
  }
}

void key_gemv_threaded_1(
    float** keys_arr, float** queries_arr, float** logits_arr,
    int const num_head, int const batch_size, int const K, int const Dh,
    int const keys_head_offset, int const keys_batch_offset,
    int const queries_haed_offset, int const queries_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool>* ready_flag,
    std::atomic<bool>* finished_flag, std::atomic<bool>* stop_flag,
    std::atomic<int>* iter_num) {
  float* keys;
  float* queries;
  float* logits;
  while (!stop_flag->load(std::memory_order_acquire)) {
    while (!(ready_flag->load(std::memory_order_acquire) &&
             !finished_flag->load(std::memory_order_acquire) &&
             !stop_flag->load(std::memory_order_acquire))) {
      // ready_flag: true
      // finished_flag: false
      // stop_flag: false
      // Busy-wait (spinlock) until the main thread signals ready
      if (stop_flag->load(std::memory_order_acquire)) return;
      keys = keys_arr[iter_num->load(std::memory_order_acquire)];
      queries = queries_arr[iter_num->load(std::memory_order_acquire)];
      logits = logits_arr[iter_num->load(std::memory_order_acquire)];
    }
    if (stop_flag->load(std::memory_order_acquire)) return;

    // Multiply and Add
    for (int idx = start_idx; idx < end_idx; ++idx) {
      int i = idx / batch_size;
      int j = idx % batch_size;
      // if (end_idx == 1024) {
      //   printf("i: %d, j: %d\n", i, j);
      // }
      // i = 31, j = 31;

      for (int k = 0; k < K; k += 16) {
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        __m256 c04 = _mm256_setzero_ps();
        __m256 c05 = _mm256_setzero_ps();
        __m256 c06 = _mm256_setzero_ps();
        __m256 c07 = _mm256_setzero_ps();
        __m256 c08 = _mm256_setzero_ps();
        __m256 c09 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c11 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps();
        __m256 c13 = _mm256_setzero_ps();
        __m256 c14 = _mm256_setzero_ps();
        __m256 c15 = _mm256_setzero_ps();

        for (int l = 0; l < Dh; l += 8) {
          // Prefetching the next query and keys for the next iteration
          // if (l + 8 < Dh)
          //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
          //                              j * q_batch_offset + l + 8),
          //                _MM_HINT_T0);
          // if (k + 8 < K)
          //   _mm_prefetch(
          //       (const char*)(keys + i * keys_head_offset +
          //                     j * keys_batch_offset + (k + 8) * Dh + l),
          //       _MM_HINT_T0);

          __m256 q0 = _mm256_loadu_ps(queries + i * queries_haed_offset +
                                      j * queries_batch_offset + l);

          __m256 k0 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + k * Dh + l);
          __m256 k1 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 1) * Dh + l);
          __m256 k2 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 2) * Dh + l);
          __m256 k3 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 3) * Dh + l);
          __m256 k4 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 4) * Dh + l);
          __m256 k5 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 5) * Dh + l);
          __m256 k6 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 6) * Dh + l);
          __m256 k7 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 7) * Dh + l);
          __m256 k8 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 8) * Dh + l);
          __m256 k9 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 9) * Dh + l);
          __m256 k10 =
              _mm256_loadu_ps(keys + i * keys_head_offset +
                              j * keys_batch_offset + (k + 10) * Dh + l);
          __m256 k11 =
              _mm256_loadu_ps(keys + i * keys_head_offset +
                              j * keys_batch_offset + (k + 11) * Dh + l);
          __m256 k12 =
              _mm256_loadu_ps(keys + i * keys_head_offset +
                              j * keys_batch_offset + (k + 12) * Dh + l);
          __m256 k13 =
              _mm256_loadu_ps(keys + i * keys_head_offset +
                              j * keys_batch_offset + (k + 13) * Dh + l);
          __m256 k14 =
              _mm256_loadu_ps(keys + i * keys_head_offset +
                              j * keys_batch_offset + (k + 14) * Dh + l);
          __m256 k15 =
              _mm256_loadu_ps(keys + i * keys_head_offset +
                              j * keys_batch_offset + (k + 15) * Dh + l);
          c00 = _mm256_fmadd_ps(q0, k0, c00);
          c01 = _mm256_fmadd_ps(q0, k1, c01);
          c02 = _mm256_fmadd_ps(q0, k2, c02);
          c03 = _mm256_fmadd_ps(q0, k3, c03);
          c04 = _mm256_fmadd_ps(q0, k4, c04);
          c05 = _mm256_fmadd_ps(q0, k5, c05);
          c06 = _mm256_fmadd_ps(q0, k6, c06);
          c07 = _mm256_fmadd_ps(q0, k7, c07);
          c08 = _mm256_fmadd_ps(q0, k8, c08);
          c09 = _mm256_fmadd_ps(q0, k9, c09);
          c10 = _mm256_fmadd_ps(q0, k10, c10);
          c11 = _mm256_fmadd_ps(q0, k11, c11);
          c12 = _mm256_fmadd_ps(q0, k12, c12);
          c13 = _mm256_fmadd_ps(q0, k13, c13);
          c14 = _mm256_fmadd_ps(q0, k14, c14);
          c15 = _mm256_fmadd_ps(q0, k15, c15);
        }
        logits[i * logits_head_offset + j * logits_batch_offset + k] =
            hsum(c00);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 1] =
            hsum(c01);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 2] =
            hsum(c02);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 3] =
            hsum(c03);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 4] =
            hsum(c04);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 5] =
            hsum(c05);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 6] =
            hsum(c06);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 7] =
            hsum(c07);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 8] =
            hsum(c08);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 9] =
            hsum(c09);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 10] =
            hsum(c10);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 11] =
            hsum(c11);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 12] =
            hsum(c12);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 13] =
            hsum(c13);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 14] =
            hsum(c14);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 15] =
            hsum(c15);
      }
    }
    // Mark this thread as finished
    finished_flag->store(true, std::memory_order_release);
    while (ready_flag->load(std::memory_order_acquire)) {
      // Wait until ready_flag is reset
    }
  }
}

void key_gemv_trusted(float* keys, const float* queries, float* logits,
                      int const num_head, int const batch_size, int const K,
                      int const Dh, int const keys_head_offset,
                      int const keys_batch_offset, int const q_haed_offset,
                      int const q_batch_offset, int const logits_head_offset,
                      int const logits_batch_offset) {
  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      // Pointers to the current head and batch query vector
      const float* query_vec = queries + i * q_haed_offset + j * q_batch_offset;

      // For each key, perform GEMV (dot product) between the query and the key
      for (int k = 0; k < K; ++k) {
        // Pointer to the current key vector in the matrix
        float* key_row =
            keys + i * keys_head_offset + j * keys_batch_offset + k * Dh;
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans, 1, Dh, 1.0f, key_row, Dh, query_vec, 1,
            0.0f, logits + i * logits_head_offset + j * logits_batch_offset + k,
            1);
      }
    }
  }
}
#include "value_gemv.h"
/*
I: (B, H_kv, K)
V: (B, H_kv, S, Dh)
L: (B, H_q, K)
O: (B, H_q, Dh)
*/
void value_gemv_trusted(
    half *values, const half *logits, float *result, const int *topk_indices,
    int const topk_num, int const q_head_num, int const kv_head_num,
    int const batch_size, int const S_len, int const Dh,
    int const logits_head_offset, int const logits_batch_offset,
    int const values_head_offset, int const values_batch_offset,
    int const result_head_offset, int const result_batch_offset) {
  int const indices_batch_offset = kv_head_num * topk_num;
  int const indices_head_offset = topk_num;
  int const q_per_kv = q_head_num / kv_head_num;

  // Multiply
  for (int i = 0; i < batch_size; ++i) {
    for (int kv_idx = 0; kv_idx < kv_head_num; ++kv_idx) {
      int q_idx = kv_idx * q_per_kv;
      // printf("\nTrusted B: %d, H_q: %d\n", i, q_idx);
      for (int k_idx = 0; k_idx < topk_num; ++k_idx) {
        int k = topk_indices[i * indices_batch_offset +
                             kv_idx * indices_head_offset + k_idx];
        // if (q_idx > 0) {
        // printf("k: %d,", k);
        // printf("logits: %f,",
        //        __half2float(logits[i * logits_batch_offset +
        //                            q_idx * logits_head_offset + k]));
        // printf("value: %f, ",
        //        __half2float(values[kv_idx * values_head_offset +
        //                            i * values_batch_offset + k * Dh]));
        // }
        // Scale the i-th row of the matrix by the corresponding logit
        float logit = __half2float(logits[i * logits_batch_offset +
                                          q_idx * logits_head_offset + k_idx]);
        for (int l = 0; l < Dh; ++l) {
          values[i * values_batch_offset + kv_idx * values_head_offset +
                 k * Dh + l] =
              values[i * values_batch_offset + kv_idx * values_head_offset +
                     k * Dh + l] *
              logits[i * logits_batch_offset + q_idx * logits_head_offset +
                     k_idx];
        }
        // if (i == 255 && q_idx == 28) {
        //   printf("k: %d, value: %f, logit: %f\n", k,
        //          __half2float(values[i * values_batch_offset +
        //                              kv_idx * values_head_offset + k * Dh]),
        //          logit);
        // }
        // cblas_sscal(
        //     Dh,
        //     logits[i * logits_batch_offset + q_idx * logits_head_offset +
        // k],
        //     values + i * values_batch_offset + kv_idx * values_head_offset
        // +
        //         k * Dh,
        //     1);
      }

      // printf("===================================\n");
    }
  }

  // Add
  // for (int i = 0; i < batch_size; ++i) {
  //   for (int kv_idx = 0; kv_idx < kv_head_num; ++kv_idx) {
  //     int q_idx = kv_idx * q_per_kv;
  //     for (int l = 0; l < Dh; l += 64) {
  //       __m256 c00 = _mm256_setzero_ps();
  //       __m256 c01 = _mm256_setzero_ps();
  //       __m256 c02 = _mm256_setzero_ps();
  //       __m256 c03 = _mm256_setzero_ps();
  //       __m256 c04 = _mm256_setzero_ps();
  //       __m256 c05 = _mm256_setzero_ps();
  //       __m256 c06 = _mm256_setzero_ps();
  //       __m256 c07 = _mm256_setzero_ps();
  //       for (int k_idx = 0; k_idx < topk_num; ++k_idx) {
  //         int k = topk_indices[i * indices_batch_offset +
  //                              kv_idx * indices_head_offset + k_idx];
  //         __m256 v0 = _mm256_load_ps(values + kv_idx * values_head_offset +
  //                                    i * values_batch_offset + k * Dh + l);
  //         __m256 v1 = _mm256_load_ps(values + kv_idx * values_head_offset +
  //                                    i * values_batch_offset + k * Dh + l +
  //                                    8);
  //         __m256 v2 = _mm256_load_ps(values + kv_idx * values_head_offset +
  //                                    i * values_batch_offset + k * Dh + l +
  //                                    16);
  //         __m256 v3 = _mm256_load_ps(values + kv_idx * values_head_offset +
  //                                    i * values_batch_offset + k * Dh + l +
  //                                    24);
  //         __m256 v4 = _mm256_load_ps(values + kv_idx * values_head_offset +
  //                                    i * values_batch_offset + k * Dh + l +
  //                                    32);
  //         __m256 v5 = _mm256_load_ps(values + kv_idx * values_head_offset +
  //                                    i * values_batch_offset + k * Dh + l +
  //                                    40);
  //         __m256 v6 = _mm256_load_ps(values + kv_idx * values_head_offset +
  //                                    i * values_batch_offset + k * Dh + l +
  //                                    48);
  //         __m256 v7 = _mm256_load_ps(values + kv_idx * values_head_offset +
  //                                    i * values_batch_offset + k * Dh + l +
  //                                    56);
  //         c00 = _mm256_add_ps(c00, v0);
  //         c01 = _mm256_add_ps(c01, v1);
  //         c02 = _mm256_add_ps(c02, v2);
  //         c03 = _mm256_add_ps(c03, v3);
  //         c04 = _mm256_add_ps(c04, v4);
  //         c05 = _mm256_add_ps(c05, v5);
  //         c06 = _mm256_add_ps(c06, v6);
  //         c07 = _mm256_add_ps(c07, v7);
  //       }
  //       // Store the accumulated result back into the result array
  //       _mm256_store_ps(
  //           result + q_idx * result_head_offset + i * result_batch_offset +
  //           l, c00);
  //       _mm256_store_ps(result + q_idx * result_head_offset +
  //                           i * result_batch_offset + l + 8,
  //                       c01);
  //       _mm256_store_ps(result + q_idx * result_head_offset +
  //                           i * result_batch_offset + l + 16,
  //                       c02);
  //       _mm256_store_ps(result + q_idx * result_head_offset +
  //                           i * result_batch_offset + l + 24,
  //                       c03);
  //       _mm256_store_ps(result + q_idx * result_head_offset +
  //                           i * result_batch_offset + l + 32,
  //                       c04);
  //       _mm256_store_ps(result + q_idx * result_head_offset +
  //                           i * result_batch_offset + l + 40,
  //                       c05);
  //       _mm256_store_ps(result + q_idx * result_head_offset +
  //                           i * result_batch_offset + l + 48,
  //                       c06);
  //       _mm256_store_ps(result + q_idx * result_head_offset +
  //                           i * result_batch_offset + l + 56,
  //                       c07);
  //     }
  //   }
  // }
  for (int i = 0; i < batch_size; ++i) {  // B
    // for (int q_idx = 0; q_idx < q_head_num; ++q_idx) {  // H
    //   int kv_idx = q_idx / q_per_kv;
    for (int kv_idx = 0; kv_idx < kv_head_num; ++kv_idx) {  // H
      int q_idx = kv_idx * q_per_kv;
      //   int i = idx / batch_size; // B
      //   int q_idx = idx % batch_size; // H

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
      // printf("\nB: %d, H_q: %d\n", i, q_idx);

      for (int k_idx = 0; k_idx < topk_num; ++k_idx) {
        int k = topk_indices[i * indices_batch_offset +
                             kv_idx * indices_head_offset + k_idx];
        float logit = __half2float(logits[i * logits_batch_offset +
                                          q_idx * logits_head_offset + k_idx]);

        // float logit = __half2float(logits[i * logits_batch_offset +
        //                                   q_idx * logits_head_offset +
        //                                   k_idx]);

        // if (q_idx > 0) {
        // printf("k: %d,", k);
        // printf("logits: %f,",
        //        __half2float(logits[i * logits_batch_offset +
        //                            q_idx * logits_head_offset + k]));
        // printf("logits: %f, ", logit);
        // printf("value: %f, ",
        //        __half2float(values[kv_idx * values_head_offset +
        //                            i * values_batch_offset + k * Dh]));
        // }
        __m256 logit_vec = _mm256_set1_ps(logit);

        //   if (k + 1 < K) {
        //     _mm_prefetch((const char *)(values + i * values_head_offset +
        //                                 q_idx * values_batch_offset + (k +
        //                                 1) * Dh),
        //                  _MM_HINT_T0);
        //   }
        __m256 v00 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh)));
        __m256 v01 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 8)));
        __m256 v02 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 16)));
        __m256 v03 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 24)));
        __m256 v04 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 32)));
        __m256 v05 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 40)));
        __m256 v06 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 48)));
        __m256 v07 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 56)));
        __m256 v08 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 64)));
        __m256 v09 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 72)));
        __m256 v10 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 80)));
        __m256 v11 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 88)));
        __m256 v12 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                       i * values_batch_offset + k * Dh + 96)));
        __m256 v13 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values + kv_idx * values_head_offset +
                        i * values_batch_offset + k * Dh + 104)));
        __m256 v14 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values + kv_idx * values_head_offset +
                        i * values_batch_offset + k * Dh + 112)));
        __m256 v15 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values + kv_idx * values_head_offset +
                        i * values_batch_offset + k * Dh + 120)));
        // c00 = _mm256_fmadd_ps(logit_vec, v00, c00);
        // c01 = _mm256_fmadd_ps(logit_vec, v01, c01);
        // c02 = _mm256_fmadd_ps(logit_vec, v02, c02);
        // c03 = _mm256_fmadd_ps(logit_vec, v03, c03);
        // c04 = _mm256_fmadd_ps(logit_vec, v04, c04);
        // c05 = _mm256_fmadd_ps(logit_vec, v05, c05);
        // c06 = _mm256_fmadd_ps(logit_vec, v06, c06);
        // c07 = _mm256_fmadd_ps(logit_vec, v07, c07);
        // c08 = _mm256_fmadd_ps(logit_vec, v08, c08);
        // c09 = _mm256_fmadd_ps(logit_vec, v09, c09);
        // c10 = _mm256_fmadd_ps(logit_vec, v10, c10);
        // c11 = _mm256_fmadd_ps(logit_vec, v11, c11);
        // c12 = _mm256_fmadd_ps(logit_vec, v12, c12);
        // c13 = _mm256_fmadd_ps(logit_vec, v13, c13);
        // c14 = _mm256_fmadd_ps(logit_vec, v14, c14);
        // c15 = _mm256_fmadd_ps(logit_vec, v15, c15);

        c00 = _mm256_add_ps(v00, c00);
        c01 = _mm256_add_ps(v01, c01);
        c02 = _mm256_add_ps(v02, c02);
        c03 = _mm256_add_ps(v03, c03);
        c04 = _mm256_add_ps(v04, c04);
        c05 = _mm256_add_ps(v05, c05);
        c06 = _mm256_add_ps(v06, c06);
        c07 = _mm256_add_ps(v07, c07);
        c08 = _mm256_add_ps(v08, c08);
        c09 = _mm256_add_ps(v09, c09);
        c10 = _mm256_add_ps(v10, c10);
        c11 = _mm256_add_ps(v11, c11);
        c12 = _mm256_add_ps(v12, c12);
        c13 = _mm256_add_ps(v13, c13);
        c14 = _mm256_add_ps(v14, c14);
        c15 = _mm256_add_ps(v15, c15);
      }
      // Store the accumulated result back into the result array
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset, c00);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 8,
          c01);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 16,
          c02);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 24,
          c03);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 32,
          c04);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 40,
          c05);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 48,
          c06);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 56,
          c07);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 64,
          c08);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 72,
          c09);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 80,
          c10);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 88,
          c11);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 96,
          c12);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 104,
          c13);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 112,
          c14);
      _mm256_store_ps(
          result + q_idx * result_head_offset + i * result_batch_offset + 120,
          c15);
    }
  }
}

void value_gemv_trusted_threaded(
    float *values, const float *logits, float *result, int const head_num,
    int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset, int thread_id,
    int num_threads, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag) {
  while (!ready_flag->load(std::memory_order_acquire)) {
    // Busy-wait (spinlock) until the main thread signals ready
  }

  // Each thread works on its slice
  int total_work = head_num * batch_size;
  int work_per_thread = (total_work + num_threads - 1) / num_threads;
  int start_idx = thread_id * work_per_thread;
  int end_idx = std::min(start_idx + work_per_thread, total_work);

  // Multiply
  for (int idx = start_idx; idx < end_idx; ++idx) {
    int i = idx / batch_size;
    int j = idx % batch_size;

    for (int k = 0; k < K; ++k) {
      // Scale the i-th row of the matrix by the corresponding logit
      cblas_sscal(
          Dh, logits[i * logits_head_offset + j * logits_batch_offset + k],
          values + i * values_head_offset + j * values_batch_offset + k * Dh,
          1);
    }
  }

  // Add
  for (int idx = start_idx; idx < end_idx; ++idx) {
    int i = idx / batch_size;
    int j = idx % batch_size;
    for (int l = 0; l < Dh; l += 64) {
      __m256 c00 = _mm256_setzero_ps();
      __m256 c01 = _mm256_setzero_ps();
      __m256 c02 = _mm256_setzero_ps();
      __m256 c03 = _mm256_setzero_ps();
      __m256 c04 = _mm256_setzero_ps();
      __m256 c05 = _mm256_setzero_ps();
      __m256 c06 = _mm256_setzero_ps();
      __m256 c07 = _mm256_setzero_ps();
      for (int k = 0; k < K; ++k) {
        __m256 v0 = _mm256_load_ps(values + i * values_head_offset +
                                   j * values_batch_offset + k * Dh + l);
        __m256 v1 = _mm256_load_ps(values + i * values_head_offset +
                                   j * values_batch_offset + k * Dh + l + 8);
        __m256 v2 = _mm256_load_ps(values + i * values_head_offset +
                                   j * values_batch_offset + k * Dh + l + 16);
        __m256 v3 = _mm256_load_ps(values + i * values_head_offset +
                                   j * values_batch_offset + k * Dh + l + 24);
        __m256 v4 = _mm256_load_ps(values + i * values_head_offset +
                                   j * values_batch_offset + k * Dh + l + 32);
        __m256 v5 = _mm256_load_ps(values + i * values_head_offset +
                                   j * values_batch_offset + k * Dh + l + 40);
        __m256 v6 = _mm256_load_ps(values + i * values_head_offset +
                                   j * values_batch_offset + k * Dh + l + 48);
        __m256 v7 = _mm256_load_ps(values + i * values_head_offset +
                                   j * values_batch_offset + k * Dh + l + 56);
        c00 = _mm256_add_ps(c00, v0);
        c01 = _mm256_add_ps(c01, v1);
        c02 = _mm256_add_ps(c02, v2);
        c03 = _mm256_add_ps(c03, v3);
        c04 = _mm256_add_ps(c04, v4);
        c05 = _mm256_add_ps(c05, v5);
        c06 = _mm256_add_ps(c06, v6);
        c07 = _mm256_add_ps(c07, v7);
      }
      // Store the accumulated result back into the result array
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + l, c00);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + l + 8,
          c01);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + l + 16,
          c02);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + l + 24,
          c03);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + l + 32,
          c04);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + l + 40,
          c05);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + l + 48,
          c06);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + l + 56,
          c07);
    }
  }
}

void value_gemv_threaded(
    float **values_arr, float **logits_arr, float **result_arr,
    int const head_num, int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, std::atomic<bool> *stop_flag,
    std::atomic<int> *iter_num, double *end_time) {
  struct timespec _end, _start;
  float *values;
  float *logits;
  float *result;
  while (!stop_flag->load(std::memory_order_acquire)) {
    while (!(ready_flag->load(std::memory_order_acquire) &&
             !finished_flag->load(std::memory_order_acquire) &&
             !stop_flag->load(std::memory_order_acquire))) {
      // ready_flag: true
      // finished_flag: false
      // stop_flag: false
      // Busy-wait (spinlock) until the main thread signals ready
      if (stop_flag->load(std::memory_order_acquire)) return;
      values = values_arr[iter_num->load(std::memory_order_acquire)];
      logits = logits_arr[iter_num->load(std::memory_order_acquire)];
      result = result_arr[iter_num->load(std::memory_order_acquire)];
      // printf("Value ptr: %p\n", static_cast<void*>(values));
    }
    clock_gettime(CLOCK_REALTIME, &_start);
    if (stop_flag->load(std::memory_order_acquire)) return;

    // Multiply and Add
    for (int idx = start_idx; idx < end_idx; ++idx) {
      int i = idx / batch_size;
      int j = idx % batch_size;

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

      for (int k = 0; k < K; ++k) {
        float logit =
            logits[i * logits_head_offset + j * logits_batch_offset + k];
        __m256 logit_vec = _mm256_set1_ps(logit);

        if (k + 1 < K) {
          _mm_prefetch((const char *)(values + i * values_head_offset +
                                      j * values_batch_offset + (k + 1) * Dh),
                       _MM_HINT_T0);
        }
        __m256 v00 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh);
        __m256 v01 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 8);
        __m256 v02 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 16);
        __m256 v03 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 24);
        __m256 v04 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 32);
        __m256 v05 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 40);
        __m256 v06 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 48);
        __m256 v07 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 56);
        __m256 v08 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 64);
        __m256 v09 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 72);
        __m256 v10 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 80);
        __m256 v11 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 88);
        __m256 v12 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 96);
        __m256 v13 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 104);
        __m256 v14 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 112);
        __m256 v15 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 120);
        c00 = _mm256_fmadd_ps(logit_vec, v00, c00);
        c01 = _mm256_fmadd_ps(logit_vec, v01, c01);
        c02 = _mm256_fmadd_ps(logit_vec, v02, c02);
        c03 = _mm256_fmadd_ps(logit_vec, v03, c03);
        c04 = _mm256_fmadd_ps(logit_vec, v04, c04);
        c05 = _mm256_fmadd_ps(logit_vec, v05, c05);
        c06 = _mm256_fmadd_ps(logit_vec, v06, c06);
        c07 = _mm256_fmadd_ps(logit_vec, v07, c07);
        c08 = _mm256_fmadd_ps(logit_vec, v08, c08);
        c09 = _mm256_fmadd_ps(logit_vec, v09, c09);
        c10 = _mm256_fmadd_ps(logit_vec, v10, c10);
        c11 = _mm256_fmadd_ps(logit_vec, v11, c11);
        c12 = _mm256_fmadd_ps(logit_vec, v12, c12);
        c13 = _mm256_fmadd_ps(logit_vec, v13, c13);
        c14 = _mm256_fmadd_ps(logit_vec, v14, c14);
        c15 = _mm256_fmadd_ps(logit_vec, v15, c15);
      }
      // Store the accumulated result back into the result array
      _mm256_store_ps(result + i * result_head_offset + j * result_batch_offset,
                      c00);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 8, c01);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 16, c02);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 24, c03);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 32, c04);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 40, c05);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 48, c06);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 56, c07);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 64, c08);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 72, c09);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 80, c10);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 88, c11);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 96, c12);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 104, c13);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 112, c14);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 120, c15);
    }
    // Mark this thread as finished
    finished_flag->store(true, std::memory_order_release);
    clock_gettime(CLOCK_REALTIME, &_end);
    *end_time = ((_end.tv_sec - _start.tv_sec) +
                 (_end.tv_nsec - _start.tv_nsec) / 1e9) *
                1e6;
    while (ready_flag->load(std::memory_order_acquire)) {
      // Wait until ready_flag is reset
    }
  }
}

void value_gemv_threaded_half(
    half **values_arr, half **logits_arr, half **result_arr,
    int **topk_indices_arr, int const topk_num, int const q_head_num,
    int const kv_head_num, int const batch_size, int const S_len, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, std::atomic<bool> *stop_flag,
    std::atomic<int> *iter_num, double *end_time) {
  struct timespec _end, _start;
  half *values;
  half *logits;
  half *result;
  int *topk_indices;
  int const indices_batch_offset = kv_head_num * topk_num;
  int const indices_head_offset = topk_num;
  int const q_per_kv = q_head_num / kv_head_num;

  while (!stop_flag->load(std::memory_order_acquire)) {
    while (!(ready_flag->load(std::memory_order_acquire) &&
             !finished_flag->load(std::memory_order_acquire) &&
             !stop_flag->load(std::memory_order_acquire))) {
      // ready_flag: true
      // finished_flag: false
      // stop_flag: false
      // Busy-wait (spinlock) until the main thread signals ready
      if (stop_flag->load(std::memory_order_acquire)) return;
      values = values_arr[iter_num->load(std::memory_order_acquire)];
      logits = logits_arr[iter_num->load(std::memory_order_acquire)];
      result = result_arr[iter_num->load(std::memory_order_acquire)];
      topk_indices =
          topk_indices_arr[iter_num->load(std::memory_order_acquire)];

      for (int i = start_idx; i < end_idx; ++i) {           // B
        for (int q_idx = 0; q_idx < q_head_num; ++q_idx) {  // H
          int kv_idx = q_idx / q_per_kv;

          for (int k_idx = 0; k_idx < topk_num; ++k_idx) {
            // int k = topk_indices[i * indices_batch_offset +
            //                      kv_idx * indices_head_offset + k_idx];
            int k = k_idx;

            volatile __m256 v00 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh)));
            volatile __m256 v01 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 8)));
            volatile __m256 v02 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 16)));
            volatile __m256 v03 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 24)));
            volatile __m256 v04 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 32)));
            volatile __m256 v05 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 40)));
            volatile __m256 v06 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 48)));
            volatile __m256 v07 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 56)));
            volatile __m256 v08 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 64)));
            volatile __m256 v09 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 72)));
            volatile __m256 v10 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 80)));
            volatile __m256 v11 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 88)));
            volatile __m256 v12 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 96)));
            volatile __m256 v13 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 104)));
            volatile __m256 v14 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 112)));
            volatile __m256 v15 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(values + kv_idx * values_head_offset +
                            i * values_batch_offset + k * Dh + 120)));

            if (ready_flag->load(std::memory_order_acquire))
              goto startExecution;
          }
        }
      }
    }

  startExecution:
    clock_gettime(CLOCK_REALTIME, &_start);
    if (stop_flag->load(std::memory_order_acquire)) return;

    // Multiply and Add
    for (int i = start_idx; i < end_idx; ++i) {           // B
      for (int q_idx = 0; q_idx < q_head_num; ++q_idx) {  // H
        int kv_idx = q_idx / q_per_kv;
        // for (int kv_idx = 0; kv_idx < kv_head_num; ++kv_idx) {  // H
        //   int q_idx = kv_idx * q_per_kv;
        //   int i = idx / batch_size; // B
        //   int q_idx = idx % batch_size; // H

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
        // printf("\nB: %d, H_q: %d\n", i, q_idx);

        for (int k_idx = 0; k_idx < topk_num; ++k_idx) {
          int k = topk_indices[i * indices_batch_offset +
                               kv_idx * indices_head_offset + k_idx];
          float logit =
              __half2float(logits[i * logits_batch_offset +
                                  q_idx * logits_head_offset + k_idx]);

          // float logit = __half2float(
          //     logits[i * logits_batch_offset + q_idx * logits_head_offset +
          //     k]);

          __m256 logit_vec = _mm256_set1_ps(logit);

          __m256 v00 = _mm256_cvtph_ps(
              _mm_load_si128((__m128i *)(values + kv_idx * values_head_offset +
                                         i * values_batch_offset + k * Dh)));
          __m256 v01 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 8)));
          __m256 v02 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 16)));
          __m256 v03 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 24)));
          __m256 v04 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 32)));
          __m256 v05 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 40)));
          __m256 v06 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 48)));
          __m256 v07 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 56)));
          __m256 v08 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 64)));
          __m256 v09 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 72)));
          __m256 v10 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 80)));
          __m256 v11 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 88)));
          __m256 v12 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 96)));
          __m256 v13 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 104)));
          __m256 v14 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 112)));
          __m256 v15 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(values + kv_idx * values_head_offset +
                          i * values_batch_offset + k * Dh + 120)));
          c00 = _mm256_fmadd_ps(logit_vec, v00, c00);
          c01 = _mm256_fmadd_ps(logit_vec, v01, c01);
          c02 = _mm256_fmadd_ps(logit_vec, v02, c02);
          c03 = _mm256_fmadd_ps(logit_vec, v03, c03);
          c04 = _mm256_fmadd_ps(logit_vec, v04, c04);
          c05 = _mm256_fmadd_ps(logit_vec, v05, c05);
          c06 = _mm256_fmadd_ps(logit_vec, v06, c06);
          c07 = _mm256_fmadd_ps(logit_vec, v07, c07);
          c08 = _mm256_fmadd_ps(logit_vec, v08, c08);
          c09 = _mm256_fmadd_ps(logit_vec, v09, c09);
          c10 = _mm256_fmadd_ps(logit_vec, v10, c10);
          c11 = _mm256_fmadd_ps(logit_vec, v11, c11);
          c12 = _mm256_fmadd_ps(logit_vec, v12, c12);
          c13 = _mm256_fmadd_ps(logit_vec, v13, c13);
          c14 = _mm256_fmadd_ps(logit_vec, v14, c14);
          c15 = _mm256_fmadd_ps(logit_vec, v15, c15);

          // c00 = _mm256_add_ps(v00, c00);
          // c01 = _mm256_add_ps(v01, c01);
          // c02 = _mm256_add_ps(v02, c02);
          // c03 = _mm256_add_ps(v03, c03);
          // c04 = _mm256_add_ps(v04, c04);
          // c05 = _mm256_add_ps(v05, c05);
          // c06 = _mm256_add_ps(v06, c06);
          // c07 = _mm256_add_ps(v07, c07);
          // c08 = _mm256_add_ps(v08, c08);
          // c09 = _mm256_add_ps(v09, c09);
          // c10 = _mm256_add_ps(v10, c10);
          // c11 = _mm256_add_ps(v11, c11);
          // c12 = _mm256_add_ps(v12, c12);
          // c13 = _mm256_add_ps(v13, c13);
          // c14 = _mm256_add_ps(v14, c14);
          // c15 = _mm256_add_ps(v15, c15);
        }
        // Store the accumulated result back into the result array
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset),
                        _mm256_cvtps_ph(c00, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 8),
                        _mm256_cvtps_ph(c01, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 16),
                        _mm256_cvtps_ph(c02, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 24),
                        _mm256_cvtps_ph(c03, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 32),
                        _mm256_cvtps_ph(c04, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 40),
                        _mm256_cvtps_ph(c05, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 48),
                        _mm256_cvtps_ph(c06, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 56),
                        _mm256_cvtps_ph(c07, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 64),
                        _mm256_cvtps_ph(c08, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 72),
                        _mm256_cvtps_ph(c09, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 80),
                        _mm256_cvtps_ph(c10, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 88),
                        _mm256_cvtps_ph(c11, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 96),
                        _mm256_cvtps_ph(c12, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 104),
                        _mm256_cvtps_ph(c13, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 112),
                        _mm256_cvtps_ph(c14, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
        _mm_store_si128((__m128i *)(result + q_idx * result_head_offset +
                                    i * result_batch_offset + 120),
                        _mm256_cvtps_ph(c15, _MM_FROUND_TO_NEAREST_INT |
                                                 _MM_FROUND_NO_EXC));
      }
    }
    // Mark this thread as finished
    finished_flag->store(true, std::memory_order_release);
    clock_gettime(CLOCK_REALTIME, &_end);
    *end_time = ((_end.tv_sec - _start.tv_sec) +
                 (_end.tv_nsec - _start.tv_nsec) / 1e9) *
                1e6;
    while (ready_flag->load(std::memory_order_acquire)) {
      // Wait until ready_flag is reset
    }
  }
}

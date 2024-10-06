#include "attention_output.h"

#include "shared.h"  // Include the shared header for external variable access

void attn_output_trusted(float* values, const float* logits, float* result,
                         int const num_head, int const batch_size, int const K,
                         int const Dh, int const logits_haed_offset,
                         int const logits_batch_offset,
                         int const values_head_offset,
                         int const values_batch_offset,
                         int const result_head_offset,
                         int const result_batch_offset) {
  // Multiply
  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      for (int k = 0; k < K; ++k) {
        // Scale the i-th row of the matrix by the corresponding logit
        cblas_sscal(
            Dh, logits[i * logits_haed_offset + j * logits_batch_offset + k],
            values + i * values_head_offset + j * values_batch_offset + k * Dh,
            1);
      }
    }
  }

  // Add
  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < batch_size; ++j) {
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
}

void attn_output_trusted_threaded(
    float* values, const float* logits, float* result, int const num_head,
    int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset, int thread_id,
    int num_threads) {
  while (!ready_flag.load(std::memory_order_acquire)) {
    // Busy-wait (spinlock) until the main thread signals ready
  }

  // Each thread works on its slice
  int total_work = num_head * batch_size;
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
          Dh, logits[i * logits_haed_offset + j * logits_batch_offset + k],
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

void attn_output_3_threaded(
    float* values, const float* logits, float* result, int const num_head,
    int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset, int thread_id,
    int num_threads) {
  while (!ready_flag.load(std::memory_order_acquire)) {
    // Busy-wait (spinlock) until the main thread signals ready
  }
  // Multiply and Add
  // Parallelize over num_head and batch_size
  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < batch_size; ++j) {
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
          float logit =
              logits[i * logits_haed_offset + j * logits_batch_offset + k];
          __m256 logit_vec = _mm256_set1_ps(logit);
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
          c00 = _mm256_fmadd_ps(logit_vec, v0, c00);
          c01 = _mm256_fmadd_ps(logit_vec, v1, c01);
          c02 = _mm256_fmadd_ps(logit_vec, v2, c02);
          c03 = _mm256_fmadd_ps(logit_vec, v3, c03);
          c04 = _mm256_fmadd_ps(logit_vec, v4, c04);
          c05 = _mm256_fmadd_ps(logit_vec, v5, c05);
          c06 = _mm256_fmadd_ps(logit_vec, v6, c06);
          c07 = _mm256_fmadd_ps(logit_vec, v7, c07);
          // Prefetch next values
          if (k + 1 < K) {
            _mm_prefetch(
                (const char*)(values + i * values_head_offset +
                              j * values_batch_offset + (k + 1) * Dh + l),
                _MM_HINT_T0);
          }
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
}

void attn_output_test(float* values, const float* logits, float* result,
                      int const num_head, int const batch_size, int const K,
                      int const Dh, int const values_head_offset,
                      int const values_batch_offset,
                      int const logits_haed_offset,
                      int const logits_batch_offset,
                      int const result_head_offset,
                      int const result_batch_offset) {
  // Multiply and Add
  // Parallelize over num_head and batch_size
  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      // for (int l = 0; l < Dh; l += 128) {
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
            logits[i * logits_haed_offset + j * logits_batch_offset + k];
        __m256 logit_vec = _mm256_set1_ps(logit);
        if (k + 1 < K) {
          _mm_prefetch((const char*)(values + i * values_head_offset +
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
  }
}

void attn_output_threaded(float* values, const float* logits, float* result,
                          int const num_head, int const batch_size, int const K,
                          int const Dh, int const values_head_offset,
                          int const values_batch_offset,
                          int const logits_haed_offset,
                          int const logits_batch_offset,
                          int const result_head_offset,
                          int const result_batch_offset, int const thread_id,
                          int const num_threads, int const start_idx,
                          int const end_idx) {
  while (!ready_flag.load(std::memory_order_acquire)) {
    // Busy-wait (spinlock) until the main thread signals ready
  }

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
          logits[i * logits_haed_offset + j * logits_batch_offset + k];
      __m256 logit_vec = _mm256_set1_ps(logit);

      if (k + 1 < K) {
        _mm_prefetch((const char*)(values + i * values_head_offset +
                                   j * values_batch_offset + (k + 1) * Dh),
                     _MM_HINT_T0);
      }
      __m256 one = _mm256_set1_ps(1.0f);
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
  finished_flags[thread_id].store(true, std::memory_order_release);
}

void attn_output_mul_add_threaded(
    float* values, const float* logits, float* result, int const num_head,
    int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset, int thread_id,
    int num_threads) {
  while (!ready_flag.load(std::memory_order_acquire)) {
    // Busy-wait (spinlock) until the main thread signals ready
  }

  // Each thread works on its slice
  int total_work = num_head * batch_size;
  int work_per_thread = (total_work + num_threads - 1) / num_threads;
  int start_idx = thread_id * work_per_thread;
  int end_idx = std::min(start_idx + work_per_thread, total_work);

  // Multiply
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
        float logit =
            logits[i * logits_haed_offset + j * logits_batch_offset + k];
        __m256 logit_vec = _mm256_set1_ps(logit);
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
        v0 = _mm256_mul_ps(logit_vec, v0);
        v1 = _mm256_mul_ps(logit_vec, v1);
        v2 = _mm256_mul_ps(logit_vec, v2);
        v3 = _mm256_mul_ps(logit_vec, v3);
        v4 = _mm256_mul_ps(logit_vec, v4);
        v5 = _mm256_mul_ps(logit_vec, v5);
        v6 = _mm256_mul_ps(logit_vec, v6);
        v7 = _mm256_mul_ps(logit_vec, v7);
        // Prefetch next values
        if (k + 1 < K) {
          _mm_prefetch(
              (const char*)(values + i * values_head_offset +
                            j * values_batch_offset + (k + 1) * Dh + l),
              _MM_HINT_T0);
        }
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
          float logit =
              logits[i * logits_haed_offset + j * logits_batch_offset + k];
          __m256 logit_vec = _mm256_set1_ps(logit);
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
          c00 = _mm256_add_ps(v0, c00);
          c01 = _mm256_add_ps(v1, c01);
          c02 = _mm256_add_ps(v2, c02);
          c03 = _mm256_add_ps(v3, c03);
          c04 = _mm256_add_ps(v4, c04);
          c05 = _mm256_add_ps(v5, c05);
          c06 = _mm256_add_ps(v6, c06);
          c07 = _mm256_add_ps(v7, c07);
          // Prefetch next values
          if (k + 1 < K) {
            _mm_prefetch(
                (const char*)(values + i * values_head_offset +
                              j * values_batch_offset + (k + 1) * Dh + l),
                _MM_HINT_T0);
          }
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
}

void attn_output_mul_add(float* values, const float* logits, float* result,
                         int const num_head, int const batch_size, int const K,
                         int const Dh, int const values_head_offset,
                         int const values_batch_offset,
                         int const logits_haed_offset,
                         int const logits_batch_offset,
                         int const result_head_offset,
                         int const result_batch_offset) {
  // Multiply
  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < batch_size; ++j) {
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
          float logit =
              logits[i * logits_haed_offset + j * logits_batch_offset + k];
          __m256 logit_vec = _mm256_set1_ps(logit);
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
          v0 = _mm256_mul_ps(logit_vec, v0);
          v1 = _mm256_mul_ps(logit_vec, v1);
          v2 = _mm256_mul_ps(logit_vec, v2);
          v3 = _mm256_mul_ps(logit_vec, v3);
          v4 = _mm256_mul_ps(logit_vec, v4);
          v5 = _mm256_mul_ps(logit_vec, v5);
          v6 = _mm256_mul_ps(logit_vec, v6);
          v7 = _mm256_mul_ps(logit_vec, v7);
          // Prefetch next values
          if (k + 1 < K) {
            _mm_prefetch(
                (const char*)(values + i * values_head_offset +
                              j * values_batch_offset + (k + 1) * Dh + l),
                _MM_HINT_T0);
          }
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

  // Add
  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < batch_size; ++j) {
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
          float logit =
              logits[i * logits_haed_offset + j * logits_batch_offset + k];
          __m256 logit_vec = _mm256_set1_ps(logit);
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
          c00 = _mm256_add_ps(v0, c00);
          c01 = _mm256_add_ps(v1, c01);
          c02 = _mm256_add_ps(v2, c02);
          c03 = _mm256_add_ps(v3, c03);
          c04 = _mm256_add_ps(v4, c04);
          c05 = _mm256_add_ps(v5, c05);
          c06 = _mm256_add_ps(v6, c06);
          c07 = _mm256_add_ps(v7, c07);
          // Prefetch next values
          if (k + 1 < K) {
            _mm_prefetch(
                (const char*)(values + i * values_head_offset +
                              j * values_batch_offset + (k + 1) * Dh + l),
                _MM_HINT_T0);
          }
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
}

void attn_output_1(float* values, float* values_t, const float* logits,
                   float* result, int K, int Dh) {
  transpose_matrix_avx2_2(values, values_t, K, Dh);
  for (int i = 0; i < Dh; i += 8) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c02 = _mm256_setzero_ps();
    __m256 c03 = _mm256_setzero_ps();
    __m256 c04 = _mm256_setzero_ps();
    __m256 c05 = _mm256_setzero_ps();
    __m256 c06 = _mm256_setzero_ps();
    __m256 c07 = _mm256_setzero_ps();
    for (int j = 0; j < K; j += 8) {
      __m256 v0 = _mm256_loadu_ps(values_t + i * K + j);
      __m256 v1 = _mm256_loadu_ps(values_t + (i + 1) * K + j);
      __m256 v2 = _mm256_loadu_ps(values_t + (i + 2) * K + j);
      __m256 v3 = _mm256_loadu_ps(values_t + (i + 3) * K + j);
      __m256 v4 = _mm256_loadu_ps(values_t + (i + 4) * K + j);
      __m256 v5 = _mm256_loadu_ps(values_t + (i + 5) * K + j);
      __m256 v6 = _mm256_loadu_ps(values_t + (i + 6) * K + j);
      __m256 v7 = _mm256_loadu_ps(values_t + (i + 7) * K + j);
      __m256 l0 = _mm256_loadu_ps(logits + j);
      c00 = _mm256_fmadd_ps(l0, v0, c00);
      c01 = _mm256_fmadd_ps(l0, v1, c01);
      c02 = _mm256_fmadd_ps(l0, v2, c02);
      c03 = _mm256_fmadd_ps(l0, v3, c03);
      c04 = _mm256_fmadd_ps(l0, v4, c04);
      c05 = _mm256_fmadd_ps(l0, v5, c05);
      c06 = _mm256_fmadd_ps(l0, v6, c06);
      c07 = _mm256_fmadd_ps(l0, v7, c07);
    }
    result[i] = hsum(c00);
    result[i + 1] = hsum(c01);
    result[i + 2] = hsum(c02);
    result[i + 3] = hsum(c03);
    result[i + 4] = hsum(c04);
    result[i + 5] = hsum(c05);
    result[i + 6] = hsum(c06);
    result[i + 7] = hsum(c07);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Transpose a K x Dh matrix into a Dh x K matrix using AVX2
// https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
void transpose_matrix_avx2_1(const float* src, float* dst, int K, int Dh) {
  int block_size = 8;  // Transpose 8x8 blocks
  __m256 r0, r1, r2, r3, r4, r5, r6, r7;
  __m256 t0, t1, t2, t3, t4, t5, t6, t7;

  // Transpose in blocks of 8x8 using AVX2
  for (int i = 0; i < K; i += block_size) {
    for (int j = 0; j < Dh; j += block_size) {
      // Load 8 rows of 8 elements each from src
      r0 = _mm256_load_ps(&src[(i + 0) * Dh + j]);
      r1 = _mm256_load_ps(&src[(i + 1) * Dh + j]);
      r2 = _mm256_load_ps(&src[(i + 2) * Dh + j]);
      r3 = _mm256_load_ps(&src[(i + 3) * Dh + j]);
      r4 = _mm256_load_ps(&src[(i + 4) * Dh + j]);
      r5 = _mm256_load_ps(&src[(i + 5) * Dh + j]);
      r6 = _mm256_load_ps(&src[(i + 6) * Dh + j]);
      r7 = _mm256_load_ps(&src[(i + 7) * Dh + j]);

      // Transpose 8x8 block
      t0 = _mm256_unpacklo_ps(r0, r1);
      t1 = _mm256_unpackhi_ps(r0, r1);
      t2 = _mm256_unpacklo_ps(r2, r3);
      t3 = _mm256_unpackhi_ps(r2, r3);
      t4 = _mm256_unpacklo_ps(r4, r5);
      t5 = _mm256_unpackhi_ps(r4, r5);
      t6 = _mm256_unpacklo_ps(r6, r7);
      t7 = _mm256_unpackhi_ps(r6, r7);

      // Shuffle and store
      r0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
      r1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
      r2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
      r3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
      r4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
      r5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
      r6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
      r7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

      t0 = _mm256_permute2f128_ps(r0, r4, 0x20);
      t1 = _mm256_permute2f128_ps(r1, r5, 0x20);
      t2 = _mm256_permute2f128_ps(r2, r6, 0x20);
      t3 = _mm256_permute2f128_ps(r3, r7, 0x20);
      t4 = _mm256_permute2f128_ps(r0, r4, 0x31);
      t5 = _mm256_permute2f128_ps(r1, r5, 0x31);
      t6 = _mm256_permute2f128_ps(r2, r6, 0x31);
      t7 = _mm256_permute2f128_ps(r3, r7, 0x31);

      // Store the transposed rows in dst
      _mm256_store_ps(&dst[(j + 0) * K + i], t0);
      _mm256_store_ps(&dst[(j + 1) * K + i], t1);
      _mm256_store_ps(&dst[(j + 2) * K + i], t2);
      _mm256_store_ps(&dst[(j + 3) * K + i], t3);
      _mm256_store_ps(&dst[(j + 4) * K + i], t4);
      _mm256_store_ps(&dst[(j + 5) * K + i], t5);
      _mm256_store_ps(&dst[(j + 6) * K + i], t6);
      _mm256_store_ps(&dst[(j + 7) * K + i], t7);
    }
  }
}

void transpose_matrix_avx2_2(const float* src, float* dst, int K, int Dh) {
  int block_size = 8;  // Transpose 8x8 blocks
  __m256 r0, r1, r2, r3, r4, r5, r6, r7;
  __m256 t0, t1, t2, t3, t4, t5, t6, t7;

  // Transpose in blocks of 8x8 using AVX2
  for (int i = 0; i < K; i += block_size) {
    for (int j = 0; j < Dh; j += block_size) {
      // Load 8 rows of 8 elements each from src
      r0 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_load_ps(&src[(i + 0) * Dh + j])),
          _mm_load_ps(&src[(i + 4) * Dh + j]), 1);
      r1 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_load_ps(&src[(i + 1) * Dh + j])),
          _mm_load_ps(&src[(i + 5) * Dh + j]), 1);
      r2 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_load_ps(&src[(i + 2) * Dh + j])),
          _mm_load_ps(&src[(i + 6) * Dh + j]), 1);
      r3 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_load_ps(&src[(i + 3) * Dh + j])),
          _mm_load_ps(&src[(i + 7) * Dh + j]), 1);
      r4 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_load_ps(&src[(i + 0) * Dh + j + 4])),
          _mm_load_ps(&src[(i + 4) * Dh + j + 4]), 1);
      r5 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_load_ps(&src[(i + 1) * Dh + j + 4])),
          _mm_load_ps(&src[(i + 5) * Dh + j + 4]), 1);
      r6 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_load_ps(&src[(i + 2) * Dh + j + 4])),
          _mm_load_ps(&src[(i + 6) * Dh + j + 4]), 1);
      r7 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_load_ps(&src[(i + 3) * Dh + j + 4])),
          _mm_load_ps(&src[(i + 7) * Dh + j + 4]), 1);

      // Transpose 8x8 block
      t0 = _mm256_unpacklo_ps(r0, r1);
      t1 = _mm256_unpackhi_ps(r0, r1);
      t2 = _mm256_unpacklo_ps(r2, r3);
      t3 = _mm256_unpackhi_ps(r2, r3);
      t4 = _mm256_unpacklo_ps(r4, r5);
      t5 = _mm256_unpackhi_ps(r4, r5);
      t6 = _mm256_unpacklo_ps(r6, r7);
      t7 = _mm256_unpackhi_ps(r6, r7);

      __m256 v;

      // r0 = _mm256_shuffle_ps(t0,t2, 0x44);
      // r1 = _mm256_shuffle_ps(t0,t2, 0xEE);
      v = _mm256_shuffle_ps(t0, t2, 0x4E);
      r0 = _mm256_blend_ps(t0, v, 0xCC);
      r1 = _mm256_blend_ps(t2, v, 0x33);

      // r2 = _mm256_shuffle_ps(t1,t3, 0x44);
      // r3 = _mm256_shuffle_ps(t1,t3, 0xEE);
      v = _mm256_shuffle_ps(t1, t3, 0x4E);
      r2 = _mm256_blend_ps(t1, v, 0xCC);
      r3 = _mm256_blend_ps(t3, v, 0x33);

      // r4 = _mm256_shuffle_ps(t4,t6, 0x44);
      // r5 = _mm256_shuffle_ps(t4,t6, 0xEE);
      v = _mm256_shuffle_ps(t4, t6, 0x4E);
      r4 = _mm256_blend_ps(t4, v, 0xCC);
      r5 = _mm256_blend_ps(t6, v, 0x33);

      // r6 = _mm256_shuffle_ps(t5,t7, 0x44);
      // r7 = _mm256_shuffle_ps(t5,t7, 0xEE);
      v = _mm256_shuffle_ps(t5, t7, 0x4E);
      r6 = _mm256_blend_ps(t5, v, 0xCC);
      r7 = _mm256_blend_ps(t7, v, 0x33);

      // Store the transposed rows in dst
      _mm256_store_ps(&dst[(j + 0) * K + i], r0);
      _mm256_store_ps(&dst[(j + 1) * K + i], r1);
      _mm256_store_ps(&dst[(j + 2) * K + i], r2);
      _mm256_store_ps(&dst[(j + 3) * K + i], r3);
      _mm256_store_ps(&dst[(j + 4) * K + i], r4);
      _mm256_store_ps(&dst[(j + 5) * K + i], r5);
      _mm256_store_ps(&dst[(j + 6) * K + i], r6);
      _mm256_store_ps(&dst[(j + 7) * K + i], r7);
    }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <cuda_fp16.h>
#include <immintrin.h>
#include <math.h>
#include <sched.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
#define THREAD_NUM 48

extern "C" {
////////////////////////////////////////////////////////////////////
// For Key GEMV
float *keys, *values, *queries, *logits;
half *keys_half, *values_half, *queries_half, *logits_half, *results_half;

inline float hsum_128(__m128 x) {
  x = _mm_add_ps(x, _mm_movehl_ps(x, x));
  x = _mm_add_ss(x, _mm_movehdup_ps(x));
  return _mm_cvtss_f32(x);
}

inline float hsum(__m256 x) {
  return hsum_128(
      _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}

inline void gemv_15(const int k, const int Dh, const int i, const int j,
                    const int queries_head_offset,
                    const int queries_batch_offset, const int keys_head_offset,
                    const int keys_batch_offset, const int logits_head_offset,
                    const int logits_batch_offset) {
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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    __m256 k10 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 10) * Dh + l);
    __m256 k11 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 11) * Dh + l);
    __m256 k12 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 12) * Dh + l);
    __m256 k13 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 13) * Dh + l);
    __m256 k14 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 14) * Dh + l);
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
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 6] = hsum(c06);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 7] = hsum(c07);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 8] = hsum(c08);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 9] = hsum(c09);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 10] = hsum(c10);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 11] = hsum(c11);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 12] = hsum(c12);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 13] = hsum(c13);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 14] = hsum(c14);
}

inline void gemv_14(const int k, const int Dh, const int i, const int j,
                    const int queries_head_offset,
                    const int queries_batch_offset, const int keys_head_offset,
                    const int keys_batch_offset, const int logits_head_offset,
                    const int logits_batch_offset) {
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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    __m256 k10 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 10) * Dh + l);
    __m256 k11 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 11) * Dh + l);
    __m256 k12 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 12) * Dh + l);
    __m256 k13 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 13) * Dh + l);
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
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 6] = hsum(c06);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 7] = hsum(c07);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 8] = hsum(c08);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 9] = hsum(c09);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 10] = hsum(c10);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 11] = hsum(c11);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 12] = hsum(c12);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 13] = hsum(c13);
}

inline void gemv_13(const int k, const int Dh, const int i, const int j,
                    const int queries_head_offset,
                    const int queries_batch_offset, const int keys_head_offset,
                    const int keys_batch_offset, const int logits_head_offset,
                    const int logits_batch_offset) {
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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    __m256 k10 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 10) * Dh + l);
    __m256 k11 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 11) * Dh + l);
    __m256 k12 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 12) * Dh + l);
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
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 6] = hsum(c06);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 7] = hsum(c07);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 8] = hsum(c08);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 9] = hsum(c09);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 10] = hsum(c10);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 11] = hsum(c11);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 12] = hsum(c12);
}

inline void gemv_12(const int k, const int Dh, const int i, const int j,
                    const int queries_head_offset,
                    const int queries_batch_offset, const int keys_head_offset,
                    const int keys_batch_offset, const int logits_head_offset,
                    const int logits_batch_offset) {
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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    __m256 k10 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 10) * Dh + l);
    __m256 k11 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 11) * Dh + l);
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
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 6] = hsum(c06);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 7] = hsum(c07);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 8] = hsum(c08);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 9] = hsum(c09);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 10] = hsum(c10);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 11] = hsum(c11);
}

inline void gemv_11(const int k, const int Dh, const int i, const int j,
                    const int queries_head_offset,
                    const int queries_batch_offset, const int keys_head_offset,
                    const int keys_batch_offset, const int logits_head_offset,
                    const int logits_batch_offset) {
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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    __m256 k10 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                 j * keys_batch_offset + (k + 10) * Dh + l);
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
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 6] = hsum(c06);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 7] = hsum(c07);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 8] = hsum(c08);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 9] = hsum(c09);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 10] = hsum(c10);
}

inline void gemv_10(const int k, const int Dh, const int i, const int j,
                    const int queries_head_offset,
                    const int queries_batch_offset, const int keys_head_offset,
                    const int keys_batch_offset, const int logits_head_offset,
                    const int logits_batch_offset) {
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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 6] = hsum(c06);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 7] = hsum(c07);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 8] = hsum(c08);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 9] = hsum(c09);
}

inline void gemv_9(const int k, const int Dh, const int i, const int j,
                   const int queries_head_offset,
                   const int queries_batch_offset, const int keys_head_offset,
                   const int keys_batch_offset, const int logits_head_offset,
                   const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();
  __m256 c04 = _mm256_setzero_ps();
  __m256 c05 = _mm256_setzero_ps();
  __m256 c06 = _mm256_setzero_ps();
  __m256 c07 = _mm256_setzero_ps();
  __m256 c08 = _mm256_setzero_ps();

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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
    c05 = _mm256_fmadd_ps(q0, k5, c05);
    c06 = _mm256_fmadd_ps(q0, k6, c06);
    c07 = _mm256_fmadd_ps(q0, k7, c07);
    c08 = _mm256_fmadd_ps(q0, k8, c08);
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 6] = hsum(c06);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 7] = hsum(c07);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 8] = hsum(c08);
}

inline void gemv_8(const int k, const int Dh, const int i, const int j,
                   const int queries_head_offset,
                   const int queries_batch_offset, const int keys_head_offset,
                   const int keys_batch_offset, const int logits_head_offset,
                   const int logits_batch_offset) {
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
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
    c05 = _mm256_fmadd_ps(q0, k5, c05);
    c06 = _mm256_fmadd_ps(q0, k6, c06);
    c07 = _mm256_fmadd_ps(q0, k7, c07);
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 6] = hsum(c06);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 7] = hsum(c07);
}

inline void gemv_7(const int k, const int Dh, const int i, const int j,
                   const int queries_head_offset,
                   const int queries_batch_offset, const int keys_head_offset,
                   const int keys_batch_offset, const int logits_head_offset,
                   const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();
  __m256 c04 = _mm256_setzero_ps();
  __m256 c05 = _mm256_setzero_ps();
  __m256 c06 = _mm256_setzero_ps();

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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
    c05 = _mm256_fmadd_ps(q0, k5, c05);
    c06 = _mm256_fmadd_ps(q0, k6, c06);
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 6] = hsum(c06);
}

inline void gemv_6(const int k, const int Dh, const int i, const int j,
                   const int queries_head_offset,
                   const int queries_batch_offset, const int keys_head_offset,
                   const int keys_batch_offset, const int logits_head_offset,
                   const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();
  __m256 c04 = _mm256_setzero_ps();
  __m256 c05 = _mm256_setzero_ps();

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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
    c05 = _mm256_fmadd_ps(q0, k5, c05);
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 5] = hsum(c05);
}

inline void gemv_5(const int k, const int Dh, const int i, const int j,
                   const int queries_head_offset,
                   const int queries_batch_offset, const int keys_head_offset,
                   const int keys_batch_offset, const int logits_head_offset,
                   const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();
  __m256 c04 = _mm256_setzero_ps();

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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 4] = hsum(c04);
}

inline void gemv_4(const int k, const int Dh, const int i, const int j,
                   const int queries_head_offset,
                   const int queries_batch_offset, const int keys_head_offset,
                   const int keys_batch_offset, const int logits_head_offset,
                   const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();

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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
                                j * queries_batch_offset + l);

    __m256 k0 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + k * Dh + l);
    __m256 k1 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + (k + 1) * Dh + l);
    __m256 k2 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + (k + 2) * Dh + l);
    __m256 k3 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + (k + 3) * Dh + l);
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 3] = hsum(c03);
}

inline void gemv_3(const int k, const int Dh, const int i, const int j,
                   const int queries_head_offset,
                   const int queries_batch_offset, const int keys_head_offset,
                   const int keys_batch_offset, const int logits_head_offset,
                   const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();

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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
                                j * queries_batch_offset + l);

    __m256 k0 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + k * Dh + l);
    __m256 k1 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + (k + 1) * Dh + l);
    __m256 k2 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + (k + 2) * Dh + l);
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 2] = hsum(c02);
}

inline void gemv_2(const int k, const int Dh, const int i, const int j,
                   const int queries_head_offset,
                   const int queries_batch_offset, const int keys_head_offset,
                   const int keys_batch_offset, const int logits_head_offset,
                   const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();

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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
                                j * queries_batch_offset + l);

    __m256 k0 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + k * Dh + l);
    __m256 k1 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + (k + 1) * Dh + l);
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
  logits[i * logits_head_offset + j * logits_batch_offset + k + 1] = hsum(c01);
}

inline void gemv_1(const int k, const int Dh, const int i, const int j,
                   const int queries_head_offset,
                   const int queries_batch_offset, const int keys_head_offset,
                   const int keys_batch_offset, const int logits_head_offset,
                   const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();

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

    __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
                                j * queries_batch_offset + l);

    __m256 k0 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                j * keys_batch_offset + k * Dh + l);
    c00 = _mm256_fmadd_ps(q0, k0, c00);
  }
  logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
}

inline void gemv_15_half(
    const int k, const int Dh, const int i_q, const int i_kv, const int j,
    const int queries_head_offset, const int queries_batch_offset,
    const int keys_head_offset, const int keys_batch_offset,
    const int logits_head_offset, const int logits_batch_offset) {
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

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    __m256 k6 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 6) * Dh + l)));
    __m256 k7 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 7) * Dh + l)));
    __m256 k8 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 8) * Dh + l)));
    __m256 k9 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 9) * Dh + l)));
    __m256 k10 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 10) * Dh + l)));
    __m256 k11 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 11) * Dh + l)));
    __m256 k12 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 12) * Dh + l)));
    __m256 k13 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 13) * Dh + l)));
    __m256 k14 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 14) * Dh + l)));
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
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 6] =
      __float2half(hsum(c06));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 7] =
      __float2half(hsum(c07));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 8] =
      __float2half(hsum(c08));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 9] =
      __float2half(hsum(c09));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 10] =
      __float2half(hsum(c10));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 11] =
      __float2half(hsum(c11));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 12] =
      __float2half(hsum(c12));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 13] =
      __float2half(hsum(c13));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 14] =
      __float2half(hsum(c14));
}

inline void gemv_14_half(
    const int k, const int Dh, const int i_q, const int i_kv, const int j,
    const int queries_head_offset, const int queries_batch_offset,
    const int keys_head_offset, const int keys_batch_offset,
    const int logits_head_offset, const int logits_batch_offset) {
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
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    __m256 k6 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 6) * Dh + l)));
    __m256 k7 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 7) * Dh + l)));
    __m256 k8 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 8) * Dh + l)));
    __m256 k9 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 9) * Dh + l)));
    __m256 k10 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 10) * Dh + l)));
    __m256 k11 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 11) * Dh + l)));
    __m256 k12 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 12) * Dh + l)));
    __m256 k13 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 13) * Dh + l)));
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
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 6] =
      __float2half(hsum(c06));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 7] =
      __float2half(hsum(c07));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 8] =
      __float2half(hsum(c08));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 9] =
      __float2half(hsum(c09));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 10] =
      __float2half(hsum(c10));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 11] =
      __float2half(hsum(c11));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 12] =
      __float2half(hsum(c12));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 13] =
      __float2half(hsum(c13));
}

inline void gemv_13_half(
    const int k, const int Dh, const int i_q, const int i_kv, const int j,
    const int queries_head_offset, const int queries_batch_offset,
    const int keys_head_offset, const int keys_batch_offset,
    const int logits_head_offset, const int logits_batch_offset) {
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
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    __m256 k6 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 6) * Dh + l)));
    __m256 k7 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 7) * Dh + l)));
    __m256 k8 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 8) * Dh + l)));
    __m256 k9 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 9) * Dh + l)));
    __m256 k10 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 10) * Dh + l)));
    __m256 k11 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 11) * Dh + l)));
    __m256 k12 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 12) * Dh + l)));
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
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 6] =
      __float2half(hsum(c06));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 7] =
      __float2half(hsum(c07));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 8] =
      __float2half(hsum(c08));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 9] =
      __float2half(hsum(c09));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 10] =
      __float2half(hsum(c10));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 11] =
      __float2half(hsum(c11));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 12] =
      __float2half(hsum(c12));
}

inline void gemv_12_half(
    const int k, const int Dh, const int i_q, const int i_kv, const int j,
    const int queries_head_offset, const int queries_batch_offset,
    const int keys_head_offset, const int keys_batch_offset,
    const int logits_head_offset, const int logits_batch_offset) {
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
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    __m256 k6 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 6) * Dh + l)));
    __m256 k7 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 7) * Dh + l)));
    __m256 k8 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 8) * Dh + l)));
    __m256 k9 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 9) * Dh + l)));
    __m256 k10 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 10) * Dh + l)));
    __m256 k11 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 11) * Dh + l)));
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
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 6] =
      __float2half(hsum(c06));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 7] =
      __float2half(hsum(c07));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 8] =
      __float2half(hsum(c08));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 9] =
      __float2half(hsum(c09));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 10] =
      __float2half(hsum(c10));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 11] =
      __float2half(hsum(c11));
}

inline void gemv_11_half(
    const int k, const int Dh, const int i_q, const int i_kv, const int j,
    const int queries_head_offset, const int queries_batch_offset,
    const int keys_head_offset, const int keys_batch_offset,
    const int logits_head_offset, const int logits_batch_offset) {
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
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    __m256 k6 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 6) * Dh + l)));
    __m256 k7 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 7) * Dh + l)));
    __m256 k8 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 8) * Dh + l)));
    __m256 k9 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 9) * Dh + l)));
    __m256 k10 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 10) * Dh + l)));
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
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 6] =
      __float2half(hsum(c06));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 7] =
      __float2half(hsum(c07));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 8] =
      __float2half(hsum(c08));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 9] =
      __float2half(hsum(c09));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 10] =
      __float2half(hsum(c10));
}

inline void gemv_10_half(
    const int k, const int Dh, const int i_q, const int i_kv, const int j,
    const int queries_head_offset, const int queries_batch_offset,
    const int keys_head_offset, const int keys_batch_offset,
    const int logits_head_offset, const int logits_batch_offset) {
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
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    __m256 k6 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 6) * Dh + l)));
    __m256 k7 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 7) * Dh + l)));
    __m256 k8 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 8) * Dh + l)));
    __m256 k9 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 9) * Dh + l)));
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
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 6] =
      __float2half(hsum(c06));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 7] =
      __float2half(hsum(c07));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 8] =
      __float2half(hsum(c08));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 9] =
      __float2half(hsum(c09));
}

inline void gemv_9_half(const int k, const int Dh, const int i_q,
                        const int i_kv, const int j,
                        const int queries_head_offset,
                        const int queries_batch_offset,
                        const int keys_head_offset, const int keys_batch_offset,
                        const int logits_head_offset,
                        const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();
  __m256 c04 = _mm256_setzero_ps();
  __m256 c05 = _mm256_setzero_ps();
  __m256 c06 = _mm256_setzero_ps();
  __m256 c07 = _mm256_setzero_ps();
  __m256 c08 = _mm256_setzero_ps();
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    __m256 k6 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 6) * Dh + l)));
    __m256 k7 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 7) * Dh + l)));
    __m256 k8 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 8) * Dh + l)));
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
    c05 = _mm256_fmadd_ps(q0, k5, c05);
    c06 = _mm256_fmadd_ps(q0, k6, c06);
    c07 = _mm256_fmadd_ps(q0, k7, c07);
    c08 = _mm256_fmadd_ps(q0, k8, c08);
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 6] =
      __float2half(hsum(c06));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 7] =
      __float2half(hsum(c07));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 8] =
      __float2half(hsum(c08));
}

inline void gemv_8_half(const int k, const int Dh, const int i_q,
                        const int i_kv, const int j,
                        const int queries_head_offset,
                        const int queries_batch_offset,
                        const int keys_head_offset, const int keys_batch_offset,
                        const int logits_head_offset,
                        const int logits_batch_offset) {
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
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    __m256 k6 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 6) * Dh + l)));
    __m256 k7 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 7) * Dh + l)));
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
    c05 = _mm256_fmadd_ps(q0, k5, c05);
    c06 = _mm256_fmadd_ps(q0, k6, c06);
    c07 = _mm256_fmadd_ps(q0, k7, c07);
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 6] =
      __float2half(hsum(c06));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 7] =
      __float2half(hsum(c07));
}

inline void gemv_7_half(const int k, const int Dh, const int i_q,
                        const int i_kv, const int j,
                        const int queries_head_offset,
                        const int queries_batch_offset,
                        const int keys_head_offset, const int keys_batch_offset,
                        const int logits_head_offset,
                        const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();
  __m256 c04 = _mm256_setzero_ps();
  __m256 c05 = _mm256_setzero_ps();
  __m256 c06 = _mm256_setzero_ps();
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    __m256 k6 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 6) * Dh + l)));
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
    c05 = _mm256_fmadd_ps(q0, k5, c05);
    c06 = _mm256_fmadd_ps(q0, k6, c06);
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 6] =
      __float2half(hsum(c06));
}

inline void gemv_6_half(const int k, const int Dh, const int i_q,
                        const int i_kv, const int j,
                        const int queries_head_offset,
                        const int queries_batch_offset,
                        const int keys_head_offset, const int keys_batch_offset,
                        const int logits_head_offset,
                        const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();
  __m256 c04 = _mm256_setzero_ps();
  __m256 c05 = _mm256_setzero_ps();
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    __m256 k5 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 5) * Dh + l)));
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
    c05 = _mm256_fmadd_ps(q0, k5, c05);
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 5] =
      __float2half(hsum(c05));
}

inline void gemv_5_half(const int k, const int Dh, const int i_q,
                        const int i_kv, const int j,
                        const int queries_head_offset,
                        const int queries_batch_offset,
                        const int keys_head_offset, const int keys_batch_offset,
                        const int logits_head_offset,
                        const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();
  __m256 c04 = _mm256_setzero_ps();
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    __m256 k4 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 4) * Dh + l)));
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
    c04 = _mm256_fmadd_ps(q0, k4, c04);
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 4] =
      __float2half(hsum(c04));
}

inline void gemv_4_half(const int k, const int Dh, const int i_q,
                        const int i_kv, const int j,
                        const int queries_head_offset,
                        const int queries_batch_offset,
                        const int keys_head_offset, const int keys_batch_offset,
                        const int logits_head_offset,
                        const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  __m256 c03 = _mm256_setzero_ps();
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    __m256 k3 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 3) * Dh + l)));
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
    c03 = _mm256_fmadd_ps(q0, k3, c03);
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 3] =
      __float2half(hsum(c03));
}

inline void gemv_3_half(const int k, const int Dh, const int i_q,
                        const int i_kv, const int j,
                        const int queries_head_offset,
                        const int queries_batch_offset,
                        const int keys_head_offset, const int keys_batch_offset,
                        const int logits_head_offset,
                        const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  __m256 c02 = _mm256_setzero_ps();
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    __m256 k2 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 2) * Dh + l)));
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
    c02 = _mm256_fmadd_ps(q0, k2, c02);
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 2] =
      __float2half(hsum(c02));
}

inline void gemv_2_half(const int k, const int Dh, const int i_q,
                        const int i_kv, const int j,
                        const int queries_head_offset,
                        const int queries_batch_offset,
                        const int keys_head_offset, const int keys_batch_offset,
                        const int logits_head_offset,
                        const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  __m256 c01 = _mm256_setzero_ps();
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    __m256 k1 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + (k + 1) * Dh + l)));
    c00 = _mm256_fmadd_ps(q0, k0, c00);
    c01 = _mm256_fmadd_ps(q0, k1, c01);
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k + 1] =
      __float2half(hsum(c01));
}

inline void gemv_1_half(const int k, const int Dh, const int i_q,
                        const int i_kv, const int j,
                        const int queries_head_offset,
                        const int queries_batch_offset,
                        const int keys_head_offset, const int keys_batch_offset,
                        const int logits_head_offset,
                        const int logits_batch_offset) {
  __m256 c00 = _mm256_setzero_ps();
  for (int l = 0; l < Dh; l += 8) {
    // Prefetching the next query and keys for the next iteration
    // if (l + 8 < Dh)
    //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //                              j * q_batch_offset + l + 8),
    //                _MM_HINT_T0);
    // if (k + 8 < K)
    //   _mm_prefetch(
    //       (const char*)(keys + i_kv * keys_head_offset +
    //                     j * keys_batch_offset + (k + 8) * Dh + l),
    //       _MM_HINT_T0);

    __m256 q0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(queries_half + i_q * queries_head_offset +
                                   j * queries_batch_offset + l)));

    __m256 k0 = _mm256_cvtph_ps(
        _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                   j * keys_batch_offset + k * Dh + l)));
    c00 = _mm256_fmadd_ps(q0, k0, c00);
  }
  // Store the accumulated result back into the result array
  logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
      __float2half(hsum(c00));
}

////////////////////////////////////////////////////////////////////

// Define the finished flag for each thread
std::atomic<bool> finished_flags[THREAD_NUM];
std::atomic<bool> done_flag(false);
std::atomic<bool> ready_flag(false);

// Store the time or duration for each thread
typedef std::pair<int, long> pair_tr;
pair_tr thread_results[THREAD_NUM];
// double thread_results[THREAD_NUM];
static struct timespec _start, _end, _end_1;

// [x] Value GEMV with FP32
void value_gemv_threaded(float *values, float *logits, float *result,
                         int const head_num, int const batch_size, int const K,
                         int const Dh, int const values_head_offset,
                         int const values_batch_offset,
                         int const logits_head_offset,
                         int const logits_batch_offset,
                         int const result_head_offset,
                         int const result_batch_offset, int const thread_id,
                         int const thread_num, int const start_idx,
                         int const end_idx, std::atomic<bool> *ready_flag,
                         std::atomic<bool> *finished_flag, pair_tr *duration) {
  struct timespec start, end;

  while (!(ready_flag->load(std::memory_order_acquire))) {
    // while (!(*ready_flag)) {
  }
  clock_gettime(CLOCK_REALTIME, &start);
  // clock_gettime(CLOCK_MONOTONIC, &start);

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
  clock_gettime(CLOCK_REALTIME, &end);
  // clock_gettime(CLOCK_MONOTONIC, &end);
  duration->first = thread_id;
  // duration->second = start.tv_sec * 1e9 + start.tv_nsec;
  // duration->second = (end.tv_nsec) / 1e3;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
}

// [x] Key GEMV with FP32
void key_gemv_threaded(
    float *keys_, float *queries_, float *logits_, int const head_num,
    int const batch_size, int const K, int const Dh, int const keys_head_offset,
    int const keys_batch_offset, int const queries_head_offset,
    int const queries_batch_offset, int const logits_head_offset,
    int const logits_batch_offset, int const thread_id, int const num_threads,
    int const start_idx, int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, pair_tr *duration) {
  struct timespec start, end;
  keys = keys_;
  queries = queries_;
  logits = logits_;
  const int last_case = K % 16;
  // DEBUGGING
  // printf("lastcase %d", last_case);

  while (!(ready_flag->load(std::memory_order_acquire))) {
    // while (!(*ready_flag)) {
  }
  clock_gettime(CLOCK_REALTIME, &start);
  // clock_gettime(CLOCK_MONOTONIC, &start);

  // Multiply and Add
  for (int idx = start_idx; idx < end_idx; ++idx) {
    const int i = idx / batch_size;
    int j = idx % batch_size;

    for (int k = 0; k < K; k += 16) {
      if (k + 16 > K) {
        switch (last_case) {
          case 1:
            gemv_1(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                   keys_head_offset, keys_batch_offset, logits_head_offset,
                   logits_batch_offset);
            break;
          case 2:
            gemv_2(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                   keys_head_offset, keys_batch_offset, logits_head_offset,
                   logits_batch_offset);
            break;
          case 3:
            gemv_3(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                   keys_head_offset, keys_batch_offset, logits_head_offset,
                   logits_batch_offset);
            break;
          case 4:
            gemv_4(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                   keys_head_offset, keys_batch_offset, logits_head_offset,
                   logits_batch_offset);
            break;
          case 5:
            gemv_5(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                   keys_head_offset, keys_batch_offset, logits_head_offset,
                   logits_batch_offset);
            break;
          case 6:
            gemv_6(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                   keys_head_offset, keys_batch_offset, logits_head_offset,
                   logits_batch_offset);

            break;
          case 7:
            gemv_7(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                   keys_head_offset, keys_batch_offset, logits_head_offset,
                   logits_batch_offset);

            break;
          case 8:
            gemv_8(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                   keys_head_offset, keys_batch_offset, logits_head_offset,
                   logits_batch_offset);

            break;
          case 9:
            gemv_9(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                   keys_head_offset, keys_batch_offset, logits_head_offset,
                   logits_batch_offset);

            break;
          case 10:
            gemv_10(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                    keys_head_offset, keys_batch_offset, logits_head_offset,
                    logits_batch_offset);

            break;
          case 11:
            gemv_11(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                    keys_head_offset, keys_batch_offset, logits_head_offset,
                    logits_batch_offset);

            break;
          case 12:
            gemv_12(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                    keys_head_offset, keys_batch_offset, logits_head_offset,
                    logits_batch_offset);

            break;
          case 13:
            gemv_13(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                    keys_head_offset, keys_batch_offset, logits_head_offset,
                    logits_batch_offset);

            break;
          case 14:
            gemv_14(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                    keys_head_offset, keys_batch_offset, logits_head_offset,
                    logits_batch_offset);

            break;
          case 15:
            gemv_15(k, Dh, i, j, queries_head_offset, queries_batch_offset,
                    keys_head_offset, keys_batch_offset, logits_head_offset,
                    logits_batch_offset);

            break;
          default:
            break;
        }
      } else {
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

          __m256 q0 = _mm256_loadu_ps(queries + i * queries_head_offset +
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
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);
  clock_gettime(CLOCK_REALTIME, &end);
  // clock_gettime(CLOCK_MONOTONIC, &end);
  duration->first = thread_id;
  // duration->second = start.tv_sec * 1e9 + start.tv_nsec;
  // duration->second = (end.tv_nsec) / 1e3;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
}

// [x] Value GEMV with FP16
void value_gemv_threaded_half(
    half *values_, half *logits_, half *results_, int const q_head_num,
    int const kv_head_num, int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, pair_tr *duration) {
  struct timespec start, end;
  values_half = values_;
  logits_half = logits_;
  results_half = results_;
  const int last_case = K % 16;
  // [ ] Reduce memory footprint
  const int q_per_kv = q_head_num / kv_head_num;

  while (!(ready_flag->load(std::memory_order_acquire))) {
  }
  clock_gettime(CLOCK_REALTIME, &start);

  // Multiply and Add
  for (int idx = start_idx; idx < end_idx; ++idx) {
    const int i_q = idx / batch_size;
    const int i_kv = i_q / q_per_kv;
    const int j = idx % batch_size;

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
      float logit = __half2float(
          logits_half[i_q * logits_haed_offset + j * logits_batch_offset + k]);
      __m256 logit_vec = _mm256_set1_ps(logit);

      if (k + 1 < K) {
        _mm_prefetch((const char *)(values_half + i_kv * values_head_offset +
                                    j * values_batch_offset + (k + 1) * Dh),
                     _MM_HINT_T0);
      }
      __m256 v00 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh)));
      __m256 v01 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 8)));
      __m256 v02 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 16)));
      __m256 v03 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 24)));
      __m256 v04 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 32)));
      __m256 v05 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 40)));
      __m256 v06 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 48)));
      __m256 v07 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 56)));
      __m256 v08 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 64)));
      __m256 v09 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 72)));
      __m256 v10 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 80)));
      __m256 v11 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 88)));
      __m256 v12 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 96)));
      __m256 v13 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 104)));
      __m256 v14 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 112)));
      __m256 v15 = _mm256_cvtph_ps(
          _mm_load_si128((__m128i *)(values_half + i_kv * values_head_offset +
                                     j * values_batch_offset + k * Dh + 120)));
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
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset),
        _mm256_cvtps_ph(c00, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 8),
        _mm256_cvtps_ph(c01, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 16),
        _mm256_cvtps_ph(c02, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 24),
        _mm256_cvtps_ph(c03, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 32),
        _mm256_cvtps_ph(c04, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 40),
        _mm256_cvtps_ph(c05, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 48),
        _mm256_cvtps_ph(c06, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 56),
        _mm256_cvtps_ph(c07, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 64),
        _mm256_cvtps_ph(c08, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 72),
        _mm256_cvtps_ph(c09, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 80),
        _mm256_cvtps_ph(c10, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 88),
        _mm256_cvtps_ph(c11, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 96),
        _mm256_cvtps_ph(c12, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 104),
        _mm256_cvtps_ph(c13, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 112),
        _mm256_cvtps_ph(c14, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_store_si128(
        (__m128i *)(results_half + i_q * result_head_offset +
                    j * result_batch_offset + 120),
        _mm256_cvtps_ph(c15, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);
  clock_gettime(CLOCK_REALTIME, &end);
  duration->first = thread_id;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
}

void value_gemv_threaded_half_1(
    half *values_, half *logits_, half *results_, int const q_head_num,
    int const kv_head_num, int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, pair_tr *duration) {
  struct timespec start, end;
  values_half = values_;
  logits_half = logits_;
  results_half = results_;
  const int last_case = K % 16;
  // [ ] Reduce memory footprint
  const int q_per_kv = q_head_num / kv_head_num;

  while (!(ready_flag->load(std::memory_order_acquire))) {
    // // Multiply and Add
    // for (int idx = start_idx; idx < end_idx; ++idx) {  // batch index
    //   for (int q_head_idx = 0; q_head_idx < q_head_num; ++q_head_idx) {
    //     //   const int i_q = idx / batch_size;
    //     //   const int i_kv = i_q / q_per_kv;
    //     //   int j = idx % batch_size;
    //     int const kv_head_idx = q_head_idx / q_per_kv;

    //     __m256 c00 = _mm256_setzero_ps();
    //     __m256 c01 = _mm256_setzero_ps();
    //     __m256 c02 = _mm256_setzero_ps();
    //     __m256 c03 = _mm256_setzero_ps();
    //     __m256 c04 = _mm256_setzero_ps();
    //     __m256 c05 = _mm256_setzero_ps();
    //     __m256 c06 = _mm256_setzero_ps();
    //     __m256 c07 = _mm256_setzero_ps();
    //     __m256 c08 = _mm256_setzero_ps();
    //     __m256 c09 = _mm256_setzero_ps();
    //     __m256 c10 = _mm256_setzero_ps();
    //     __m256 c11 = _mm256_setzero_ps();
    //     __m256 c12 = _mm256_setzero_ps();
    //     __m256 c13 = _mm256_setzero_ps();
    //     __m256 c14 = _mm256_setzero_ps();
    //     __m256 c15 = _mm256_setzero_ps();

    //     for (int k = 0; k < K; ++k) {
    //       float logit =
    //           __half2float(logits_half[q_head_idx * logits_haed_offset +
    //                                    idx * logits_batch_offset + k]);
    //       __m256 logit_vec = _mm256_set1_ps(logit);

    //       if (k + 1 < K) {
    //         _mm_prefetch(
    //             (const char *)(values_half + kv_head_idx * values_head_offset
    //             +
    //                            idx * values_batch_offset + (k + 1) * Dh),
    //             _MM_HINT_T0);
    //       }
    //       __m256 v00 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh)));
    //       __m256 v01 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 8)));
    //       __m256 v02 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 16)));
    //       __m256 v03 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 24)));
    //       __m256 v04 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 32)));
    //       __m256 v05 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 40)));
    //       __m256 v06 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 48)));
    //       __m256 v07 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 56)));
    //       __m256 v08 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 64)));
    //       __m256 v09 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 72)));
    //       __m256 v10 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 80)));
    //       __m256 v11 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 88)));
    //       __m256 v12 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 96)));
    //       __m256 v13 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 104)));
    //       __m256 v14 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 112)));
    //       __m256 v15 = _mm256_cvtph_ps(_mm_load_si128(
    //           (__m128i *)(values_half + kv_head_idx * values_head_offset +
    //                       idx * values_batch_offset + k * Dh + 120)));
    //       c00 = _mm256_fmadd_ps(logit_vec, v00, c00);
    //       c01 = _mm256_fmadd_ps(logit_vec, v01, c01);
    //       c02 = _mm256_fmadd_ps(logit_vec, v02, c02);
    //       c03 = _mm256_fmadd_ps(logit_vec, v03, c03);
    //       c04 = _mm256_fmadd_ps(logit_vec, v04, c04);
    //       c05 = _mm256_fmadd_ps(logit_vec, v05, c05);
    //       c06 = _mm256_fmadd_ps(logit_vec, v06, c06);
    //       c07 = _mm256_fmadd_ps(logit_vec, v07, c07);
    //       c08 = _mm256_fmadd_ps(logit_vec, v08, c08);
    //       c09 = _mm256_fmadd_ps(logit_vec, v09, c09);
    //       c10 = _mm256_fmadd_ps(logit_vec, v10, c10);
    //       c11 = _mm256_fmadd_ps(logit_vec, v11, c11);
    //       c12 = _mm256_fmadd_ps(logit_vec, v12, c12);
    //       c13 = _mm256_fmadd_ps(logit_vec, v13, c13);
    //       c14 = _mm256_fmadd_ps(logit_vec, v14, c14);
    //       c15 = _mm256_fmadd_ps(logit_vec, v15, c15);
    //     }
    //     // Store the accumulated result back into the result array
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset),
    //         _mm256_cvtps_ph(c00,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 8),
    //         _mm256_cvtps_ph(c01,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 16),
    //         _mm256_cvtps_ph(c02,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 24),
    //         _mm256_cvtps_ph(c03,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 32),
    //         _mm256_cvtps_ph(c04,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 40),
    //         _mm256_cvtps_ph(c05,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 48),
    //         _mm256_cvtps_ph(c06,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 56),
    //         _mm256_cvtps_ph(c07,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 64),
    //         _mm256_cvtps_ph(c08,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 72),
    //         _mm256_cvtps_ph(c09,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 80),
    //         _mm256_cvtps_ph(c10,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 88),
    //         _mm256_cvtps_ph(c11,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 96),
    //         _mm256_cvtps_ph(c12,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 104),
    //         _mm256_cvtps_ph(c13,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 112),
    //         _mm256_cvtps_ph(c14,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //     _mm_store_si128(
    //         (__m128i *)(results_half + q_head_idx * result_head_offset +
    //                     idx * result_batch_offset + 120),
    //         _mm256_cvtps_ph(c15,
    //                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    //   }
    // }
  }
  clock_gettime(CLOCK_REALTIME, &start);

  // Multiply and Add
  for (int idx = start_idx; idx < end_idx; ++idx) {  // batch index
    for (int q_head_idx = 0; q_head_idx < q_head_num; ++q_head_idx) {
      //   const int i_q = idx / batch_size;
      //   const int i_kv = i_q / q_per_kv;
      //   int j = idx % batch_size;
      int const kv_head_idx = q_head_idx / q_per_kv;

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
        float logit = __half2float(logits_half[q_head_idx * logits_haed_offset +
                                               idx * logits_batch_offset + k]);
        __m256 logit_vec = _mm256_set1_ps(logit);

        if (k + 1 < K) {
          _mm_prefetch(
              (const char *)(values_half + kv_head_idx * values_head_offset +
                             idx * values_batch_offset + (k + 1) * Dh),
              _MM_HINT_T0);
        }
        __m256 v00 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh)));
        __m256 v01 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 8)));
        __m256 v02 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 16)));
        __m256 v03 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 24)));
        __m256 v04 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 32)));
        __m256 v05 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 40)));
        __m256 v06 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 48)));
        __m256 v07 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 56)));
        __m256 v08 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 64)));
        __m256 v09 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 72)));
        __m256 v10 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 80)));
        __m256 v11 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 88)));
        __m256 v12 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 96)));
        __m256 v13 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 104)));
        __m256 v14 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 112)));
        __m256 v15 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 120)));
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
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset),
          _mm256_cvtps_ph(c00, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 8),
          _mm256_cvtps_ph(c01, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 16),
          _mm256_cvtps_ph(c02, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 24),
          _mm256_cvtps_ph(c03, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 32),
          _mm256_cvtps_ph(c04, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 40),
          _mm256_cvtps_ph(c05, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 48),
          _mm256_cvtps_ph(c06, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 56),
          _mm256_cvtps_ph(c07, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 64),
          _mm256_cvtps_ph(c08, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 72),
          _mm256_cvtps_ph(c09, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 80),
          _mm256_cvtps_ph(c10, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 88),
          _mm256_cvtps_ph(c11, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 96),
          _mm256_cvtps_ph(c12, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 104),
          _mm256_cvtps_ph(c13, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 112),
          _mm256_cvtps_ph(c14, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      _mm_store_si128(
          (__m128i *)(results_half + q_head_idx * result_head_offset +
                      idx * result_batch_offset + 120),
          _mm256_cvtps_ph(c15, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);
  clock_gettime(CLOCK_REALTIME, &end);
  duration->first = thread_id;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
  //   duration->second = (start.tv_sec + start.tv_nsec / 1e9) * 1e6;
}

void value_bandwidth_test(
    half *values_, half *logits_, half *results_, int const q_head_num,
    int const kv_head_num, int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, pair_tr *duration) {
  struct timespec start, end;
  values_half = values_;
  logits_half = logits_;
  results_half = results_;
  const int last_case = K % 16;
  // [ ] Reduce memory footprint
  const int q_per_kv = q_head_num / kv_head_num;

  while (!(ready_flag->load(std::memory_order_acquire))) {
  }
  clock_gettime(CLOCK_REALTIME, &start);

  // Multiply and Add
  for (int idx = start_idx; idx < end_idx; ++idx) {  // batch index
    for (int q_head_idx = 0; q_head_idx < q_head_num;
         ++q_head_idx) {  // head_idx
      int const kv_head_idx = q_head_idx / q_per_kv;

      //   volatile __m256 c00 = _mm256_setzero_ps();
      //   volatile __m256 c01 = _mm256_setzero_ps();
      //   volatile __m256 c02 = _mm256_setzero_ps();
      //   volatile __m256 c03 = _mm256_setzero_ps();
      //   volatile __m256 c04 = _mm256_setzero_ps();
      //   volatile __m256 c05 = _mm256_setzero_ps();
      //   volatile __m256 c06 = _mm256_setzero_ps();
      //   volatile __m256 c07 = _mm256_setzero_ps();
      //   volatile __m256 c08 = _mm256_setzero_ps();
      //   volatile __m256 c09 = _mm256_setzero_ps();
      //   volatile __m256 c10 = _mm256_setzero_ps();
      //   volatile __m256 c11 = _mm256_setzero_ps();
      //   volatile __m256 c12 = _mm256_setzero_ps();
      //   volatile __m256 c13 = _mm256_setzero_ps();
      //   volatile __m256 c14 = _mm256_setzero_ps();
      //   volatile __m256 c15 = _mm256_setzero_ps();

      for (int k = 0; k < K; ++k) {
        float logit = __half2float(logits_half[q_head_idx * logits_haed_offset +
                                               idx * logits_batch_offset + k]);
        volatile __m256 logit_vec = _mm256_set1_ps(logit);

        if (k + 1 < K) {
          _mm_prefetch(
              (const char *)(values_half + kv_head_idx * values_head_offset +
                             idx * values_batch_offset + (k + 1) * Dh),
              _MM_HINT_T0);
        }
        volatile __m256 v00 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh)));
        volatile __m256 v01 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 8)));
        volatile __m256 v02 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 16)));
        volatile __m256 v03 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 24)));
        volatile __m256 v04 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 32)));
        volatile __m256 v05 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 40)));
        volatile __m256 v06 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 48)));
        volatile __m256 v07 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 56)));
        volatile __m256 v08 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 64)));
        volatile __m256 v09 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 72)));
        volatile __m256 v10 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 80)));
        volatile __m256 v11 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 88)));
        volatile __m256 v12 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 96)));
        volatile __m256 v13 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 104)));
        volatile __m256 v14 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 112)));
        volatile __m256 v15 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(values_half + kv_head_idx * values_head_offset +
                        idx * values_batch_offset + k * Dh + 120)));
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
      }
      // Store the accumulated result back into the result array
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset),
      //       _mm256_cvtps_ph(c00, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 8),
      //       _mm256_cvtps_ph(c01, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 16),
      //       _mm256_cvtps_ph(c02, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 24),
      //       _mm256_cvtps_ph(c03, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 32),
      //       _mm256_cvtps_ph(c04, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 40),
      //       _mm256_cvtps_ph(c05, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 48),
      //       _mm256_cvtps_ph(c06, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 56),
      //       _mm256_cvtps_ph(c07, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 64),
      //       _mm256_cvtps_ph(c08, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 72),
      //       _mm256_cvtps_ph(c09, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 80),
      //       _mm256_cvtps_ph(c10, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 88),
      //       _mm256_cvtps_ph(c11, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 96),
      //       _mm256_cvtps_ph(c12, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 104),
      //       _mm256_cvtps_ph(c13, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 112),
      //       _mm256_cvtps_ph(c14, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
      //   _mm_store_si128(
      //       (__m128i *)(results_half + q_head_idx * result_head_offset +
      //                   idx * result_batch_offset + 120),
      //       _mm256_cvtps_ph(c15, _MM_FROUND_TO_NEAREST_INT |
      //       _MM_FROUND_NO_EXC));
    }
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);
  clock_gettime(CLOCK_REALTIME, &end);
  duration->first = thread_id;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
  //   duration->second = (start.tv_sec + start.tv_nsec / 1e9) * 1e6;
}

// [x] Key GEMV with FP16
void key_gemv_threaded_half(
    half *keys_, half *queries_, half *logits_, int const q_head_num,
    int const kv_head_num, int const batch_size, int const K, int const Dh,
    int const keys_head_offset, int const keys_batch_offset,
    int const queries_head_offset, int const queries_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, pair_tr *duration) {
  struct timespec start, end;
  keys_half = keys_;
  queries_half = queries_;
  logits_half = logits_;
  const int last_case = K % 16;
  // [ ] Reduce memory footprint
  const int q_per_kv = q_head_num / kv_head_num;
  // DEBUG
  // printf("lastcase %d\n", last_case);

  while (!(ready_flag->load(std::memory_order_acquire))) {
    // while (!(*ready_flag)) {
  }
  clock_gettime(CLOCK_REALTIME, &start);
  // clock_gettime(CLOCK_MONOTONIC, &start);

  // Multiply and Add
  for (int idx = start_idx; idx < end_idx; ++idx) {
    const int i_q = idx / batch_size;
    const int i_kv = i_q / q_per_kv;
    int j = idx % batch_size;

    for (int k = 0; k < K; k += 16) {
      if (k + 16 > K) {
        switch (last_case) {
          case 1:
            gemv_1_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                        queries_batch_offset, keys_head_offset,
                        keys_batch_offset, logits_head_offset,
                        logits_batch_offset);
            break;
          case 2:
            gemv_2_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                        queries_batch_offset, keys_head_offset,
                        keys_batch_offset, logits_head_offset,
                        logits_batch_offset);
            break;
          case 3:
            gemv_3_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                        queries_batch_offset, keys_head_offset,
                        keys_batch_offset, logits_head_offset,
                        logits_batch_offset);
            break;
          case 4:
            gemv_4_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                        queries_batch_offset, keys_head_offset,
                        keys_batch_offset, logits_head_offset,
                        logits_batch_offset);
            break;
          case 5:
            gemv_5_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                        queries_batch_offset, keys_head_offset,
                        keys_batch_offset, logits_head_offset,
                        logits_batch_offset);
            break;
          case 6:
            gemv_6_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                        queries_batch_offset, keys_head_offset,
                        keys_batch_offset, logits_head_offset,
                        logits_batch_offset);

            break;
          case 7:
            gemv_7_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                        queries_batch_offset, keys_head_offset,
                        keys_batch_offset, logits_head_offset,
                        logits_batch_offset);

            break;
          case 8:
            gemv_8_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                        queries_batch_offset, keys_head_offset,
                        keys_batch_offset, logits_head_offset,
                        logits_batch_offset);

            break;
          case 9:
            gemv_9_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                        queries_batch_offset, keys_head_offset,
                        keys_batch_offset, logits_head_offset,
                        logits_batch_offset);

            break;
          case 10:
            gemv_10_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                         queries_batch_offset, keys_head_offset,
                         keys_batch_offset, logits_head_offset,
                         logits_batch_offset);

            break;
          case 11:
            gemv_11_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                         queries_batch_offset, keys_head_offset,
                         keys_batch_offset, logits_head_offset,
                         logits_batch_offset);

            break;
          case 12:
            gemv_12_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                         queries_batch_offset, keys_head_offset,
                         keys_batch_offset, logits_head_offset,
                         logits_batch_offset);

            break;
          case 13:
            gemv_13_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                         queries_batch_offset, keys_head_offset,
                         keys_batch_offset, logits_head_offset,
                         logits_batch_offset);

            break;
          case 14:
            gemv_14_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                         queries_batch_offset, keys_head_offset,
                         keys_batch_offset, logits_head_offset,
                         logits_batch_offset);

            break;
          case 15:
            gemv_15_half(k, Dh, i_q, i_kv, j, queries_head_offset,
                         queries_batch_offset, keys_head_offset,
                         keys_batch_offset, logits_head_offset,
                         logits_batch_offset);

            break;
          default:
            break;
        }
      } else {
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

          __m256 q0 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(queries_half + i_q * queries_head_offset +
                          j * queries_batch_offset + l)));

          __m256 k0 = _mm256_cvtph_ps(
              _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                         j * keys_batch_offset + k * Dh + l)));
          __m256 k1 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 1) * Dh + l)));
          __m256 k2 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 2) * Dh + l)));
          __m256 k3 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 3) * Dh + l)));
          __m256 k4 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 4) * Dh + l)));
          __m256 k5 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 5) * Dh + l)));
          __m256 k6 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 6) * Dh + l)));
          __m256 k7 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 7) * Dh + l)));
          __m256 k8 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 8) * Dh + l)));
          __m256 k9 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 9) * Dh + l)));
          __m256 k10 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 10) * Dh + l)));
          __m256 k11 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 11) * Dh + l)));
          __m256 k12 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 12) * Dh + l)));
          __m256 k13 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 13) * Dh + l)));
          __m256 k14 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 14) * Dh + l)));
          __m256 k15 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + i_kv * keys_head_offset +
                          j * keys_batch_offset + (k + 15) * Dh + l)));
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
        // Store the accumulated result back into the result array
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k] =
            __float2half(hsum(c00));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    1] = __float2half(hsum(c01));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    2] = __float2half(hsum(c02));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    3] = __float2half(hsum(c03));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    4] = __float2half(hsum(c04));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    5] = __float2half(hsum(c05));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    6] = __float2half(hsum(c06));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    7] = __float2half(hsum(c07));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    8] = __float2half(hsum(c08));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    9] = __float2half(hsum(c09));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    10] = __float2half(hsum(c10));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    11] = __float2half(hsum(c11));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    12] = __float2half(hsum(c12));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    13] = __float2half(hsum(c13));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    14] = __float2half(hsum(c14));
        logits_half[i_q * logits_head_offset + j * logits_batch_offset + k +
                    15] = __float2half(hsum(c15));
      }
    }
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);
  clock_gettime(CLOCK_REALTIME, &end);
  duration->first = thread_id;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
}

void key_bandwidth_test(half *keys_, half *queries_, half *logits_,
                        int const q_head_num, int const kv_head_num,
                        int const batch_size, int const K, int const Dh,
                        int const keys_head_offset, int const keys_batch_offset,
                        int const queries_head_offset,
                        int const queries_batch_offset,
                        int const logits_head_offset,
                        int const logits_batch_offset, int const thread_id,
                        int const num_threads, int const start_idx,
                        int const end_idx, std::atomic<bool> *ready_flag,
                        std::atomic<bool> *finished_flag, pair_tr *duration) {
  struct timespec start, end;
  keys_half = keys_;
  queries_half = queries_;
  logits_half = logits_;
  const int last_case = K % 16;
  // [ ] Reduce memory footprint
  const int q_per_kv = q_head_num / kv_head_num;

  while (!(ready_flag->load(std::memory_order_acquire))) {
    // while (!(*ready_flag)) {
  }
  clock_gettime(CLOCK_REALTIME, &start);
  // clock_gettime(CLOCK_MONOTONIC, &start);

  // Multiply and Add
  for (int idx = start_idx; idx < end_idx; ++idx) {
    const int i_q = idx / batch_size;
    const int i_kv = i_q / q_per_kv;
    int j = idx % batch_size;

    for (int k = 0; k < K; k += 16) {
      for (int l = 0; l < Dh; l += 8) {
        volatile __m256 q0 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(queries_half + i_q * queries_head_offset +
                        j * queries_batch_offset + l)));

        volatile __m256 k0 = _mm256_cvtph_ps(
            _mm_load_si128((__m128i *)(keys_half + i_kv * keys_head_offset +
                                       j * keys_batch_offset + k * Dh + l)));
        volatile __m256 k1 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 1) * Dh + l)));
        volatile __m256 k2 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 2) * Dh + l)));
        volatile __m256 k3 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 3) * Dh + l)));
        volatile __m256 k4 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 4) * Dh + l)));
        volatile __m256 k5 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 5) * Dh + l)));
        volatile __m256 k6 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 6) * Dh + l)));
        volatile __m256 k7 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 7) * Dh + l)));
        volatile __m256 k8 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 8) * Dh + l)));
        volatile __m256 k9 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 9) * Dh + l)));
        volatile __m256 k10 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 10) * Dh + l)));
        volatile __m256 k11 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 11) * Dh + l)));
        volatile __m256 k12 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 12) * Dh + l)));
        volatile __m256 k13 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 13) * Dh + l)));
        volatile __m256 k14 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 14) * Dh + l)));
        volatile __m256 k15 = _mm256_cvtph_ps(_mm_load_si128(
            (__m128i *)(keys_half + i_kv * keys_head_offset +
                        j * keys_batch_offset + (k + 15) * Dh + l)));
      }
    }
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);
  clock_gettime(CLOCK_REALTIME, &end);
  duration->first = thread_id;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
}

void key_gemv_threaded_half_1(
    half *keys_, half *queries_, half *logits_, int const q_head_num,
    int const kv_head_num, int const batch_size, int const K, int const Dh,
    int const keys_head_offset, int const keys_batch_offset,
    int const queries_head_offset, int const queries_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, pair_tr *duration) {
  struct timespec start, end;
  keys_half = keys_;
  queries_half = queries_;
  logits_half = logits_;
  const int last_case = K % 16;
  // [ ] Reduce memory footprint
  const int q_per_kv = q_head_num / kv_head_num;
  //   printf("q_per_kv: %d\n", q_per_kv);
  // DEBUG
  //   printf("lastcase %d\n", last_case);

  while (!(ready_flag->load(std::memory_order_acquire))) {
    // Multiply and Add
    // for (int idx = start_idx; idx < end_idx; ++idx) {  // batch_idx
    //   for (int k = 0; k < K; k += 16) {
    //     bool is_remainder = k + 16 > K;
    //     for (int q_head_idx = 0; q_head_idx < q_head_num; ++q_head_idx) {
    //       //   const int i_q = idx / batch_size;
    //       //   const int i_kv = i_q / q_per_kv;
    //       //   int j = idx % batch_size;
    //       int const kv_head_idx = q_head_idx / q_per_kv;

    //       if (is_remainder) {
    //         switch (last_case) {
    //           case 1:
    //             gemv_1_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                         queries_head_offset, queries_batch_offset,
    //                         keys_head_offset, keys_batch_offset,
    //                         logits_head_offset, logits_batch_offset);
    //             break;
    //           case 2:
    //             gemv_2_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                         queries_head_offset, queries_batch_offset,
    //                         keys_head_offset, keys_batch_offset,
    //                         logits_head_offset, logits_batch_offset);
    //             break;
    //           case 3:
    //             gemv_3_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                         queries_head_offset, queries_batch_offset,
    //                         keys_head_offset, keys_batch_offset,
    //                         logits_head_offset, logits_batch_offset);
    //             break;
    //           case 4:
    //             gemv_4_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                         queries_head_offset, queries_batch_offset,
    //                         keys_head_offset, keys_batch_offset,
    //                         logits_head_offset, logits_batch_offset);
    //             break;
    //           case 5:
    //             gemv_5_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                         queries_head_offset, queries_batch_offset,
    //                         keys_head_offset, keys_batch_offset,
    //                         logits_head_offset, logits_batch_offset);
    //             break;
    //           case 6:
    //             gemv_6_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                         queries_head_offset, queries_batch_offset,
    //                         keys_head_offset, keys_batch_offset,
    //                         logits_head_offset, logits_batch_offset);

    //             break;
    //           case 7:
    //             gemv_7_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                         queries_head_offset, queries_batch_offset,
    //                         keys_head_offset, keys_batch_offset,
    //                         logits_head_offset, logits_batch_offset);

    //             break;
    //           case 8:
    //             gemv_8_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                         queries_head_offset, queries_batch_offset,
    //                         keys_head_offset, keys_batch_offset,
    //                         logits_head_offset, logits_batch_offset);

    //             break;
    //           case 9:
    //             gemv_9_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                         queries_head_offset, queries_batch_offset,
    //                         keys_head_offset, keys_batch_offset,
    //                         logits_head_offset, logits_batch_offset);

    //             break;
    //           case 10:
    //             gemv_10_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                          queries_head_offset, queries_batch_offset,
    //                          keys_head_offset, keys_batch_offset,
    //                          logits_head_offset, logits_batch_offset);

    //             break;
    //           case 11:
    //             gemv_11_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                          queries_head_offset, queries_batch_offset,
    //                          keys_head_offset, keys_batch_offset,
    //                          logits_head_offset, logits_batch_offset);

    //             break;
    //           case 12:
    //             gemv_12_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                          queries_head_offset, queries_batch_offset,
    //                          keys_head_offset, keys_batch_offset,
    //                          logits_head_offset, logits_batch_offset);

    //             break;
    //           case 13:
    //             gemv_13_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                          queries_head_offset, queries_batch_offset,
    //                          keys_head_offset, keys_batch_offset,
    //                          logits_head_offset, logits_batch_offset);

    //             break;
    //           case 14:
    //             gemv_14_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                          queries_head_offset, queries_batch_offset,
    //                          keys_head_offset, keys_batch_offset,
    //                          logits_head_offset, logits_batch_offset);

    //             break;
    //           case 15:
    //             gemv_15_half(k, Dh, q_head_idx, kv_head_idx, idx,
    //                          queries_head_offset, queries_batch_offset,
    //                          keys_head_offset, keys_batch_offset,
    //                          logits_head_offset, logits_batch_offset);

    //             break;
    //           default:
    //             break;
    //         }
    //       } else {
    //         __m256 c00 = _mm256_setzero_ps();
    //         __m256 c01 = _mm256_setzero_ps();
    //         __m256 c02 = _mm256_setzero_ps();
    //         __m256 c03 = _mm256_setzero_ps();
    //         __m256 c04 = _mm256_setzero_ps();
    //         __m256 c05 = _mm256_setzero_ps();
    //         __m256 c06 = _mm256_setzero_ps();
    //         __m256 c07 = _mm256_setzero_ps();
    //         __m256 c08 = _mm256_setzero_ps();
    //         __m256 c09 = _mm256_setzero_ps();
    //         __m256 c10 = _mm256_setzero_ps();
    //         __m256 c11 = _mm256_setzero_ps();
    //         __m256 c12 = _mm256_setzero_ps();
    //         __m256 c13 = _mm256_setzero_ps();
    //         __m256 c14 = _mm256_setzero_ps();
    //         __m256 c15 = _mm256_setzero_ps();

    //         for (int l = 0; l < Dh; l += 8) {
    //           // Prefetching the next query and keys for the next iteration
    //           // if (l + 8 < Dh)
    //           //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
    //           //                              j * q_batch_offset + l + 8),
    //           //                _MM_HINT_T0);
    //           // if (k + 8 < K)
    //           //   _mm_prefetch(
    //           //       (const char*)(keys + i * keys_head_offset +
    //           //                     j * keys_batch_offset + (k + 8) * Dh +
    //           //   l),
    //           //       _MM_HINT_T0);

    //           __m256 q0 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(queries_half + q_head_idx * queries_head_offset
    //               +
    //                           idx * queries_batch_offset + l)));

    //           __m256 k0 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + k * Dh + l)));
    //           __m256 k1 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 1) * Dh + l)));
    //           __m256 k2 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 2) * Dh + l)));
    //           __m256 k3 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 3) * Dh + l)));
    //           __m256 k4 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 4) * Dh + l)));
    //           __m256 k5 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 5) * Dh + l)));
    //           __m256 k6 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 6) * Dh + l)));
    //           __m256 k7 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 7) * Dh + l)));
    //           __m256 k8 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 8) * Dh + l)));
    //           __m256 k9 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 9) * Dh + l)));
    //           __m256 k10 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 10) * Dh + l)));
    //           __m256 k11 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 11) * Dh + l)));
    //           __m256 k12 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 12) * Dh + l)));
    //           __m256 k13 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 13) * Dh + l)));
    //           __m256 k14 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 14) * Dh + l)));
    //           __m256 k15 = _mm256_cvtph_ps(_mm_load_si128(
    //               (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
    //                           idx * keys_batch_offset + (k + 15) * Dh + l)));
    //           c00 = _mm256_fmadd_ps(q0, k0, c00);
    //           c01 = _mm256_fmadd_ps(q0, k1, c01);
    //           c02 = _mm256_fmadd_ps(q0, k2, c02);
    //           c03 = _mm256_fmadd_ps(q0, k3, c03);
    //           c04 = _mm256_fmadd_ps(q0, k4, c04);
    //           c05 = _mm256_fmadd_ps(q0, k5, c05);
    //           c06 = _mm256_fmadd_ps(q0, k6, c06);
    //           c07 = _mm256_fmadd_ps(q0, k7, c07);
    //           c08 = _mm256_fmadd_ps(q0, k8, c08);
    //           c09 = _mm256_fmadd_ps(q0, k9, c09);
    //           c10 = _mm256_fmadd_ps(q0, k10, c10);
    //           c11 = _mm256_fmadd_ps(q0, k11, c11);
    //           c12 = _mm256_fmadd_ps(q0, k12, c12);
    //           c13 = _mm256_fmadd_ps(q0, k13, c13);
    //           c14 = _mm256_fmadd_ps(q0, k14, c14);
    //           c15 = _mm256_fmadd_ps(q0, k15, c15);
    //         }
    //         // Store the accumulated result back into the result array
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k] =
    //         //     __float2half(hsum(c00));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 1] =
    //         //     __float2half(hsum(c01));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 2] =
    //         //     __float2half(hsum(c02));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 3] =
    //         //     __float2half(hsum(c03));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 4] =
    //         //     __float2half(hsum(c04));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 5] =
    //         //     __float2half(hsum(c05));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 6] =
    //         //     __float2half(hsum(c06));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 7] =
    //         //     __float2half(hsum(c07));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 8] =
    //         //     __float2half(hsum(c08));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 9] =
    //         //     __float2half(hsum(c09));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 10] =
    //         //     __float2half(hsum(c10));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 11] =
    //         //     __float2half(hsum(c11));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 12] =
    //         //     __float2half(hsum(c12));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 13] =
    //         //     __float2half(hsum(c13));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 14] =
    //         //     __float2half(hsum(c14));
    //         // logits_half[q_head_idx * logits_head_offset +
    //         //             idx * logits_batch_offset + k + 15] =
    //         //     __float2half(hsum(c15));
    //       }
    //     }
    //   }
    // }
  }
  clock_gettime(CLOCK_REALTIME, &start);
  // clock_gettime(CLOCK_MONOTONIC, &start);

  // Multiply and Add
  for (int idx = start_idx; idx < end_idx; ++idx) {  // batch_idx
    for (int k = 0; k < K; k += 16) {
      for (int q_head_idx = 0; q_head_idx < q_head_num;
           ++q_head_idx) {  // head_idx
        bool is_remainder = k + 16 > K;
        //   const int i_q = idx / batch_size;
        //   const int i_kv = i_q / q_per_kv;
        //   int j = idx % batch_size;
        int const kv_head_idx = q_head_idx / q_per_kv;

        if (is_remainder) {
          switch (last_case) {
            case 1:
              gemv_1_half(k, Dh, q_head_idx, kv_head_idx, idx,
                          queries_head_offset, queries_batch_offset,
                          keys_head_offset, keys_batch_offset,
                          logits_head_offset, logits_batch_offset);
              break;
            case 2:
              gemv_2_half(k, Dh, q_head_idx, kv_head_idx, idx,
                          queries_head_offset, queries_batch_offset,
                          keys_head_offset, keys_batch_offset,
                          logits_head_offset, logits_batch_offset);
              break;
            case 3:
              gemv_3_half(k, Dh, q_head_idx, kv_head_idx, idx,
                          queries_head_offset, queries_batch_offset,
                          keys_head_offset, keys_batch_offset,
                          logits_head_offset, logits_batch_offset);
              break;
            case 4:
              gemv_4_half(k, Dh, q_head_idx, kv_head_idx, idx,
                          queries_head_offset, queries_batch_offset,
                          keys_head_offset, keys_batch_offset,
                          logits_head_offset, logits_batch_offset);
              break;
            case 5:
              gemv_5_half(k, Dh, q_head_idx, kv_head_idx, idx,
                          queries_head_offset, queries_batch_offset,
                          keys_head_offset, keys_batch_offset,
                          logits_head_offset, logits_batch_offset);
              break;
            case 6:
              gemv_6_half(k, Dh, q_head_idx, kv_head_idx, idx,
                          queries_head_offset, queries_batch_offset,
                          keys_head_offset, keys_batch_offset,
                          logits_head_offset, logits_batch_offset);

              break;
            case 7:
              gemv_7_half(k, Dh, q_head_idx, kv_head_idx, idx,
                          queries_head_offset, queries_batch_offset,
                          keys_head_offset, keys_batch_offset,
                          logits_head_offset, logits_batch_offset);

              break;
            case 8:
              gemv_8_half(k, Dh, q_head_idx, kv_head_idx, idx,
                          queries_head_offset, queries_batch_offset,
                          keys_head_offset, keys_batch_offset,
                          logits_head_offset, logits_batch_offset);

              break;
            case 9:
              gemv_9_half(k, Dh, q_head_idx, kv_head_idx, idx,
                          queries_head_offset, queries_batch_offset,
                          keys_head_offset, keys_batch_offset,
                          logits_head_offset, logits_batch_offset);

              break;
            case 10:
              gemv_10_half(k, Dh, q_head_idx, kv_head_idx, idx,
                           queries_head_offset, queries_batch_offset,
                           keys_head_offset, keys_batch_offset,
                           logits_head_offset, logits_batch_offset);

              break;
            case 11:
              gemv_11_half(k, Dh, q_head_idx, kv_head_idx, idx,
                           queries_head_offset, queries_batch_offset,
                           keys_head_offset, keys_batch_offset,
                           logits_head_offset, logits_batch_offset);

              break;
            case 12:
              gemv_12_half(k, Dh, q_head_idx, kv_head_idx, idx,
                           queries_head_offset, queries_batch_offset,
                           keys_head_offset, keys_batch_offset,
                           logits_head_offset, logits_batch_offset);

              break;
            case 13:
              gemv_13_half(k, Dh, q_head_idx, kv_head_idx, idx,
                           queries_head_offset, queries_batch_offset,
                           keys_head_offset, keys_batch_offset,
                           logits_head_offset, logits_batch_offset);

              break;
            case 14:
              gemv_14_half(k, Dh, q_head_idx, kv_head_idx, idx,
                           queries_head_offset, queries_batch_offset,
                           keys_head_offset, keys_batch_offset,
                           logits_head_offset, logits_batch_offset);

              break;
            case 15:
              gemv_15_half(k, Dh, q_head_idx, kv_head_idx, idx,
                           queries_head_offset, queries_batch_offset,
                           keys_head_offset, keys_batch_offset,
                           logits_head_offset, logits_batch_offset);

              break;
            default:
              break;
          }
        } else {
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

            __m256 q0 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(queries_half + q_head_idx * queries_head_offset +
                            idx * queries_batch_offset + l)));

            __m256 k0 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + k * Dh + l)));
            __m256 k1 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 1) * Dh + l)));
            __m256 k2 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 2) * Dh + l)));
            __m256 k3 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 3) * Dh + l)));
            __m256 k4 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 4) * Dh + l)));
            __m256 k5 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 5) * Dh + l)));
            __m256 k6 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 6) * Dh + l)));
            __m256 k7 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 7) * Dh + l)));
            __m256 k8 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 8) * Dh + l)));
            __m256 k9 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 9) * Dh + l)));
            __m256 k10 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 10) * Dh + l)));
            __m256 k11 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 11) * Dh + l)));
            __m256 k12 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 12) * Dh + l)));
            __m256 k13 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 13) * Dh + l)));
            __m256 k14 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 14) * Dh + l)));
            __m256 k15 = _mm256_cvtph_ps(_mm_load_si128(
                (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                            idx * keys_batch_offset + (k + 15) * Dh + l)));
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
          // Store the accumulated result back into the result array
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k] = __float2half(hsum(c00));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 1] =
              __float2half(hsum(c01));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 2] =
              __float2half(hsum(c02));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 3] =
              __float2half(hsum(c03));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 4] =
              __float2half(hsum(c04));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 5] =
              __float2half(hsum(c05));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 6] =
              __float2half(hsum(c06));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 7] =
              __float2half(hsum(c07));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 8] =
              __float2half(hsum(c08));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 9] =
              __float2half(hsum(c09));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 10] =
              __float2half(hsum(c10));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 11] =
              __float2half(hsum(c11));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 12] =
              __float2half(hsum(c12));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 13] =
              __float2half(hsum(c13));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 14] =
              __float2half(hsum(c14));
          logits_half[q_head_idx * logits_head_offset +
                      idx * logits_batch_offset + k + 15] =
              __float2half(hsum(c15));
        }
      }
    }
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);
  clock_gettime(CLOCK_REALTIME, &end);
  duration->first = thread_id;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
}

void key_bandwidth_test_1(
    half *keys_, half *queries_, half *logits_, int const q_head_num,
    int const kv_head_num, int const batch_size, int const K, int const Dh,
    int const keys_head_offset, int const keys_batch_offset,
    int const queries_head_offset, int const queries_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, pair_tr *duration) {
  struct timespec start, end;
  keys_half = keys_;
  queries_half = queries_;
  logits_half = logits_;
  const int last_case = K % 16;
  // [ ] Reduce memory footprint
  const int q_per_kv = q_head_num / kv_head_num;
  //   printf("q_per_kv: %d\n", q_per_kv);
  // DEBUG
  //   printf("lastcase %d\n", last_case);

  while (!(ready_flag->load(std::memory_order_acquire))) {
  }
  clock_gettime(CLOCK_REALTIME, &start);

  // Multiply and Add
  for (int idx = start_idx; idx < end_idx; ++idx) {  // batch_idx
    for (int k = 0; k < K; k += 16) {
      for (int q_head_idx = 0; q_head_idx < q_head_num;
           ++q_head_idx) {  // head_idx
        bool is_remainder = k + 16 > K;
        int const kv_head_idx = q_head_idx / q_per_kv;

        for (int l = 0; l < Dh; l += 8) {
          volatile __m256 q0 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(queries_half + q_head_idx * queries_head_offset +
                          idx * queries_batch_offset + l)));

          volatile __m256 k0 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + k * Dh + l)));
          volatile __m256 k1 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 1) * Dh + l)));
          volatile __m256 k2 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 2) * Dh + l)));
          volatile __m256 k3 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 3) * Dh + l)));
          volatile __m256 k4 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 4) * Dh + l)));
          volatile __m256 k5 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 5) * Dh + l)));
          volatile __m256 k6 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 6) * Dh + l)));
          volatile __m256 k7 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 7) * Dh + l)));
          volatile __m256 k8 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 8) * Dh + l)));
          volatile __m256 k9 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 9) * Dh + l)));
          volatile __m256 k10 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 10) * Dh + l)));
          volatile __m256 k11 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 11) * Dh + l)));
          volatile __m256 k12 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 12) * Dh + l)));
          volatile __m256 k13 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 13) * Dh + l)));
          volatile __m256 k14 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 14) * Dh + l)));
          volatile __m256 k15 = _mm256_cvtph_ps(_mm_load_si128(
              (__m128i *)(keys_half + kv_head_idx * keys_head_offset +
                          idx * keys_batch_offset + (k + 15) * Dh + l)));
        }
      }
    }
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);
  clock_gettime(CLOCK_REALTIME, &end);
  duration->first = thread_id;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
}

// Function to prepare the threads for Value GEMV
void prepare_value_gemv(float *values, float *logits, float *result,
                        int const head_num, int const batch_size, int const K,
                        int const Dh, int const values_head_offset,
                        int const values_batch_offset,
                        int const logits_head_offset,
                        int const logits_batch_offset,
                        int const result_head_offset,
                        int const result_batch_offset, int const thread_num) {
  // printf("Ready Flag: %p\n", &ready_flag);
  // printf("Done Flag: %p\n", &done_flag);
  // Each thread works on its slice
  int const total_work = head_num * batch_size;
  int const work_per_thread = total_work / thread_num;
  int const work_remained = total_work % thread_num;

  //   int const min_priority = sched_get_priority_min(SCHED_FIFO);
  int const max_priority = sched_get_priority_max(SCHED_FIFO);

  int priority = max_priority;  // Base priority for all threads

  // Init thread pool
  std::vector<std::thread> threads;
  int start_idx = 0, end_idx = 0;
  int acc = 0;
  for (int t = 0; t < thread_num; ++t) {
    start_idx = end_idx;
    end_idx = t < work_remained ? start_idx + work_per_thread + 1
                                : start_idx + work_per_thread;
    int cpu_id = t + acc;
    acc += 1;
    // int cpu_id = t;
    threads.emplace_back(
        value_gemv_threaded, values, logits, result, head_num, batch_size, K,
        Dh, values_head_offset, values_batch_offset, logits_head_offset,
        logits_batch_offset, result_head_offset, result_batch_offset, cpu_id,
        thread_num, start_idx, end_idx, &ready_flag, &finished_flags[t],
        &thread_results[t]);

    // Get the native handle for the created thread
    pthread_t nativeHandle = threads.back().native_handle();

    // Define the scheduling parameters
    struct sched_param param;
    param.sched_priority = priority;  // Set the same priorities for each thread

    // Set the scheduling policy to SCHED_FIFO
    int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
    if (ret != 0) {
      std::cerr << "Failed to set scheduling policy for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
    ///////////////////////////////////////////
    // [x] Set CPU affinity
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // // Method1
    // if (t > 23) {
    //   // int id = 48 + (t - 23);
    //   CPU_SET(48 + (t - 23), &cpuset);  // Bind to specific CPU core
    // } else {
    //   CPU_SET(t, &cpuset);  // Bind to specific CPU core
    // }
    // // Method2
    // CPU_SET(cpu_id, &cpuset);  // Bind to specific CPU core

    // ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
    // if (ret != 0) {
    //   std::cerr << "Failed to set CPU affinity for thread " << t << ": "
    //             << strerror(ret) << std::endl;
    // }
    ///////////////////////////////////////////
  }

  bool all_threads_finished = false;
  bool thread_finished[thread_num];
  for (int i = 0; i < thread_num; ++i) thread_finished[i] = false;

  while (!all_threads_finished) {
    all_threads_finished = true;
    for (int i = 0; i < thread_num; ++i) {
      if (!thread_finished[i]) {
        if (finished_flags[i].load(std::memory_order_acquire)) {
          //   clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
          thread_finished[i] = true;
        } else {
          all_threads_finished = false;
        }
      }
    }
  }
  // clock_gettime(CLOCK_REALTIME, &_end);
  clock_gettime(CLOCK_MONOTONIC, &_end);
  done_flag.store(true, std::memory_order_release);
  // DEBUGGING
  // std::sort(
  //     thread_results, thread_results + thread_num,
  //     [](const pair_tr &i, const pair_tr &j) { return i.second < j.second;
  //     });
  // for (size_t i = 0; i < thread_num; i++)
  //   printf("CPU: %d, duration: %ld\n", thread_results[i].first,
  //          thread_results[i].second);

  // printf("Variance: %ld\n",
  //        thread_results[thread_num - 1].second - thread_results[0].second);
  for (auto &thread : threads) thread.join();
}

void prepare_value_gemv_half(
    half *values, half *logits, half *result, int const q_head_num,
    int const kv_head_num, int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_num) {
  // Each thread works on its slice
  //   int const total_work = q_head_num * batch_size;
  int const total_work = batch_size;
  int const work_per_thread = total_work / thread_num;
  int const work_remained = total_work % thread_num;

  //   int const min_priority = sched_get_priority_min(SCHED_FIFO);
  int const max_priority = sched_get_priority_max(SCHED_FIFO);

  int priority = max_priority;  // Base priority for all threads

  // Init thread pool
  std::vector<std::thread> threads;
  int start_idx = 0, end_idx = 0;
  int acc = 0;
  for (int t = 0; t < thread_num; ++t) {
    start_idx = end_idx;
    end_idx = t < work_remained ? start_idx + work_per_thread + 1
                                : start_idx + work_per_thread;
    int cpu_id = t + acc;
    acc += 1;
    // int cpu_id = t;
    threads.emplace_back(
        value_bandwidth_test, values, logits, result, q_head_num, kv_head_num,
        batch_size, K, Dh, values_head_offset, values_batch_offset,
        logits_head_offset, logits_batch_offset, result_head_offset,
        result_batch_offset, cpu_id, thread_num, start_idx, end_idx,
        &ready_flag, &finished_flags[t], &thread_results[t]);

    // Get the native handle for the created thread
    pthread_t nativeHandle = threads.back().native_handle();

    // Define the scheduling parameters
    struct sched_param param;
    param.sched_priority = priority;  // Set the same priorities for each thread

    // Set the scheduling policy to SCHED_FIFO
    int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
    if (ret != 0) {
      std::cerr << "Failed to set scheduling policy for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
    ///////////////////////////////////////////
    // [x] Set CPU affinity
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // // Method1
    // if (t > 23) {
    //   // int id = 48 + (t - 23);
    //   CPU_SET(48 + (t - 23), &cpuset);  // Bind to specific CPU core
    // } else {
    //   CPU_SET(t, &cpuset);  // Bind to specific CPU core
    // }
    // // Method2
    // CPU_SET(cpu_id, &cpuset);  // Bind to specific CPU core

    // ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
    // if (ret != 0) {
    //   std::cerr << "Failed to set CPU affinity for thread " << t << ": "
    //             << strerror(ret) << std::endl;
    // }
    ///////////////////////////////////////////
  }

  bool all_threads_finished = false;
  bool thread_finished[thread_num];
  for (int i = 0; i < thread_num; ++i) thread_finished[i] = false;

  while (!all_threads_finished) {
    all_threads_finished = true;
    for (int i = 0; i < thread_num; ++i) {
      if (!thread_finished[i]) {
        if (finished_flags[i].load(std::memory_order_acquire)) {
          //   clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
          thread_finished[i] = true;
        } else {
          all_threads_finished = false;
        }
      }
    }
  }
  // clock_gettime(CLOCK_REALTIME, &_end);
  clock_gettime(CLOCK_MONOTONIC, &_end);
  done_flag.store(true, std::memory_order_release);
  // DEBUG
  std::sort(
      thread_results, thread_results + thread_num,
      [](const pair_tr &i, const pair_tr &j) { return i.second < j.second; });
  for (size_t i = 0; i < thread_num; i++)
    printf("CPU: %d, duration: %ld\n", thread_results[i].first,
           thread_results[i].second);

  printf("Variance: %ld\n",
         thread_results[thread_num - 1].second - thread_results[0].second);
  for (auto &thread : threads) thread.join();
}

// Function to prepare the threads for Key GEMV
void prepare_key_gemv(float *keys, float *queries, float *logits,
                      int const head_num, int const batch_size, int const K,
                      int const Dh, int const keys_head_offset,
                      int const keys_batch_offset,
                      int const queries_head_offset,
                      int const queries_batch_offset,
                      int const logits_head_offset,
                      int const logits_batch_offset, int const thread_num) {
  // Each thread works on its slice
  int const total_work = head_num * batch_size;
  int const work_per_thread = total_work / thread_num;
  int const work_remained = total_work % thread_num;

  //   int const min_priority = sched_get_priority_min(SCHED_FIFO);
  int const max_priority = sched_get_priority_max(SCHED_FIFO);

  int priority = max_priority;  // Base priority for all threads

  // Init thread pool
  std::vector<std::thread> threads;
  int start_idx = 0, end_idx = 0;
  int acc = 0;
  for (int t = 0; t < thread_num; ++t) {
    start_idx = end_idx;
    end_idx = t < work_remained ? start_idx + work_per_thread + 1
                                : start_idx + work_per_thread;
    int cpu_id = t + acc;
    acc += 1;
    // int cpu_id = t;
    threads.emplace_back(key_gemv_threaded, keys, queries, logits, head_num,
                         batch_size, K, Dh, keys_head_offset, keys_batch_offset,
                         queries_head_offset, queries_batch_offset,
                         logits_head_offset, logits_batch_offset, t, thread_num,
                         start_idx, end_idx, &ready_flag, &finished_flags[t],
                         &thread_results[t]);

    // Get the native handle for the created thread
    pthread_t nativeHandle = threads.back().native_handle();

    // Define the scheduling parameters
    struct sched_param param;
    param.sched_priority = priority;  // Set the same priorities for each thread

    // Set the scheduling policy to SCHED_FIFO
    int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
    if (ret != 0) {
      std::cerr << "Failed to set scheduling policy for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
    ///////////////////////////////////////////
    // [x] Set CPU affinity
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // // Method1
    // if (t > 23) {
    //   // int id = 48 + (t - 23);
    //   CPU_SET(48 + (t - 23), &cpuset);  // Bind to specific CPU core
    // } else {
    //   CPU_SET(t, &cpuset);  // Bind to specific CPU core
    // }
    // // Method2
    // CPU_SET(cpu_id, &cpuset);  // Bind to specific CPU core

    // ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
    // if (ret != 0) {
    //   std::cerr << "Failed to set CPU affinity for thread " << t << ": "
    //             << strerror(ret) << std::endl;
    // }
    ///////////////////////////////////////////
  }

  bool all_threads_finished = false;
  bool thread_finished[thread_num];
  for (int i = 0; i < thread_num; ++i) thread_finished[i] = false;

  while (!all_threads_finished) {
    all_threads_finished = true;
    for (int i = 0; i < thread_num; ++i) {
      if (!thread_finished[i]) {
        if (finished_flags[i].load(std::memory_order_acquire)) {
          //   clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
          thread_finished[i] = true;
        } else {
          all_threads_finished = false;
        }
      }
    }
  }
  // clock_gettime(CLOCK_REALTIME, &_end);
  clock_gettime(CLOCK_MONOTONIC, &_end);
  done_flag.store(true, std::memory_order_release);
  // DEBUGGING
  // std::sort(
  //     thread_results, thread_results + thread_num,
  //     [](const pair_tr &i, const pair_tr &j) { return i.second < j.second;
  //     });
  // for (size_t i = 0; i < thread_num; i++)
  //   printf("CPU: %d, duration: %ld\n", thread_results[i].first,
  //          thread_results[i].second);

  // printf("Variance: %ld\n",
  //        thread_results[thread_num - 1].second - thread_results[0].second);
  for (auto &thread : threads) thread.join();
}

// Function to prepare the threads for Key GEMV
void prepare_key_gemv_half(
    half *keys, half *queries, half *logits, int const q_head_num,
    int const kv_head_num, int const batch_size, int const K, int const Dh,
    int const keys_head_offset, int const keys_batch_offset,
    int const queries_head_offset, int const queries_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const thread_num) {
  // Each thread works on its slice
  //   int const total_work = q_head_num * batch_size;
  int const total_work = batch_size;
  int const work_per_thread = total_work / thread_num;
  int const work_remained = total_work % thread_num;

  //   int const min_priority = sched_get_priority_min(SCHED_FIFO);
  int const max_priority = sched_get_priority_max(SCHED_FIFO);

  int priority = max_priority;  // Base priority for all threads

  // Init thread pool
  std::vector<std::thread> threads;
  int start_idx = 0, end_idx = 0;
  int acc = 0;
  for (int t = 0; t < thread_num; ++t) {
    start_idx = end_idx;
    end_idx = t < work_remained ? start_idx + work_per_thread + 1
                                : start_idx + work_per_thread;
    int cpu_id = t + acc;
    acc += 1;
    threads.emplace_back(
        key_bandwidth_test_1, keys, queries, logits, q_head_num, kv_head_num,
        batch_size, K, Dh, keys_head_offset, keys_batch_offset,
        queries_head_offset, queries_batch_offset, logits_head_offset,
        logits_batch_offset, t, thread_num, start_idx, end_idx, &ready_flag,
        &finished_flags[t], &thread_results[t]);

    // Get the native handle for the created thread
    pthread_t nativeHandle = threads.back().native_handle();

    // Define the scheduling parameters
    struct sched_param param;
    param.sched_priority = priority;  // Set the same priorities for each thread

    // Set the scheduling policy to SCHED_FIFO
    int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
    if (ret != 0) {
      std::cerr << "Failed to set scheduling policy for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
    ///////////////////////////////////////////
    // [x] Set CPU affinity
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // // Method1
    // cpu_id = (t > 23) ? 48 + (t - 23) : t;
    // // printf("cpu_id: %d", cpu_id);
    // CPU_SET(cpu_id, &cpuset);  // Bind to specific CPU core
    // // Method2
    // // CPU_SET(cpu_id, &cpuset);  // Bind to specific CPU core

    // ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
    // if (ret != 0) {
    //   std::cerr << "Failed to set CPU affinity for thread " << t << ": "
    //             << strerror(ret) << std::endl;
    // }
    ///////////////////////////////////////////
  }

  bool all_threads_finished = false;
  bool thread_finished[thread_num];
  for (int i = 0; i < thread_num; ++i) thread_finished[i] = false;

  while (!all_threads_finished) {
    all_threads_finished = true;
    for (int i = 0; i < thread_num; ++i) {
      if (!thread_finished[i]) {
        if (finished_flags[i].load(std::memory_order_acquire)) {
          //   clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
          thread_finished[i] = true;
        } else {
          all_threads_finished = false;
        }
      }
    }
  }
  // clock_gettime(CLOCK_REALTIME, &_end);
  clock_gettime(CLOCK_MONOTONIC, &_end);
  done_flag.store(true, std::memory_order_release);
  // DEBUGGING
  std::sort(
      thread_results, thread_results + thread_num,
      [](const pair_tr &i, const pair_tr &j) { return i.second < j.second; });
  for (size_t i = 0; i < thread_num; i++)
    printf("CPU: %d, duration: %ld\n", thread_results[i].first,
           thread_results[i].second);

  printf("Variance: %ld\n",
         thread_results[thread_num - 1].second - thread_results[0].second);
  for (auto &thread : threads) thread.join();
}

///////////////////////////////////////////////////////////////////
void softmax_trusted_threads(float *qk, const float *max_values,
                             const float *sums_quant, float *sums_topk,
                             const int seq_len, const int head_num,
                             const int batch_size, const int head_offset,
                             const int batch_offset, const int thread_idx,
                             const int thread_num, const int start_idx,
                             const int end_idx, std::atomic<bool> *ready_flag,
                             std::atomic<bool> *finished_flag,
                             pair_tr *duration) {
  struct timespec end, start;
  while (!(ready_flag->load(std::memory_order_acquire))) {
    // Busy-wait (spinlock) until the main thread signals ready
  }
  clock_gettime(CLOCK_REALTIME, &start);
  for (int idx = start_idx; idx < end_idx; ++idx) {
    const int head_idx = idx / batch_size;
    const int batch_idx = idx % batch_size;
    // printf(
    //     "start_idx: %d, end_idx: %d, head_idx: %d, batch_idx: %d, Index: %d,
    //     " "batch_size: %d\n", start_idx, end_idx, head_idx, batch_idx, idx,
    //     batch_size, idx / static_cast<int>(batch_size));
    const float sum = sums_quant[head_idx * batch_size + batch_idx];
    const float max = max_values[head_idx * batch_size + batch_idx];

    float tot = 0.0;
    for (int i = 0; i < seq_len; i++) {
      qk[head_idx * head_offset + batch_idx * batch_offset + i] =
          expf(qk[head_idx * head_offset + batch_idx * batch_offset + i] - max);
      tot += qk[head_idx * head_offset + batch_idx * batch_offset + i];
    }
    sums_topk[head_idx * batch_size + batch_idx] = tot;
    for (int i = 0; i < seq_len; i++) {
      qk[head_idx * head_offset + batch_idx * batch_offset + i] /= sum;
    }
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);

  clock_gettime(CLOCK_REALTIME, &end);
  duration->first = thread_idx;
  duration->second =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
}

// [x] Function to prepare the threads for Softmax
void prepare_softmax(float *qk, const float *max_values,
                     const float *sums_quant, float *sums_topk,
                     const int seq_len, const int head_num,
                     const int batch_size, const int head_offset,
                     const int batch_offset, const int thread_num) {
  // Each thread works on its slice
  int const total_work = head_num * batch_size;
  int const work_per_thread = total_work / thread_num;
  int const work_remained = total_work % thread_num;

  int const priority = sched_get_priority_max(SCHED_FIFO);

  // Init thread pool
  std::vector<std::thread> threads;
  int start_idx = 0, end_idx = 0;
  // int acc = 0;
  for (int t = 0; t < thread_num; ++t) {
    start_idx = end_idx;
    end_idx = t < work_remained ? start_idx + work_per_thread + 1
                                : start_idx + work_per_thread;

    threads.emplace_back(softmax_trusted_threads, qk, max_values, sums_quant,
                         sums_topk, seq_len, head_num, batch_size, head_offset,
                         batch_offset, t, thread_num, start_idx, end_idx,
                         &ready_flag, &finished_flags[t], &thread_results[t]);

    // Get the native handle for the created thread
    pthread_t nativeHandle = threads.back().native_handle();

    // Define the scheduling parameters
    struct sched_param param;
    param.sched_priority = priority;  // Set the same priorities for each thread

    // Set the scheduling policy to SCHED_FIFO
    int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
    if (ret != 0) {
      std::cerr << "Failed to set scheduling policy for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
    ///////////////////////////////////////////
    // [x] Set CPU affinity
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // // Method1
    // if (t > 23) {
    //   // int id = 48 + (t - 23);
    //   CPU_SET(48 + (t - 23), &cpuset);  // Bind to specific CPU core
    // } else {
    //   CPU_SET(t, &cpuset);  // Bind to specific CPU core
    // }
    // // Method2
    // CPU_SET(t + 8, &cpuset);  // Bind to specific CPU core

    // ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
    // if (ret != 0) {
    //   std::cerr << "Failed to set CPU affinity for thread " << t << ": "
    //             << strerror(ret) << std::endl;
    // }
    ///////////////////////////////////////////
  }

  bool all_threads_finished = false;
  bool thread_finished[thread_num];
  for (int i = 0; i < thread_num; ++i) thread_finished[i] = false;

  while (!all_threads_finished) {
    // for (int i = 0; i < thread_num; ++i)
    //   printf("Thread %d: %d\n", i,
    //          finished_flags[i].load(std::memory_order_acquire));
    all_threads_finished = true;
    for (int i = 0; i < thread_num; ++i) {
      if (!thread_finished[i]) {
        if (finished_flags[i].load(std::memory_order_acquire)) {
          thread_finished[i] = true;
        } else {
          all_threads_finished = false;
        }
      }
    }
  }
  clock_gettime(CLOCK_REALTIME, &_end);
  done_flag.store(true, std::memory_order_release);
  // DEBUG
  // std::sort(
  //     thread_results, thread_results + thread_num,
  //     [](const pair_tr &i, const pair_tr &j) { return i.second < j.second;
  //     });
  // for (size_t i = 0; i < thread_num; i++)
  //   printf("CPU: %d, duration: %ld\n", thread_results[i].first,
  //          thread_results[i].second);

  // printf("Variance: %ld\n",
  //        thread_results[thread_num - 1].second - thread_results[0].second);
  for (auto &thread : threads) thread.join();
}

///////////////////////////////////////////////////////////////////

// Function to set the ready_flag from Python
void set_ready_flag() {
  // usleep(1000000);
  clock_gettime(CLOCK_REALTIME, &_start);
  ready_flag.store(true, std::memory_order_release);
  // printf("ready_flag: %p\n", &ready_flag);
}

// Function to check the all threads are done
long is_finished() {
  while (!done_flag.load(std::memory_order_acquire)) {
  }
  clock_gettime(CLOCK_REALTIME, &_end);
  //   clock_gettime(CLOCK_MONOTONIC, &_end_1);
  long seconds = _end.tv_sec - _start.tv_sec;
  long nanoseconds = _end.tv_nsec - _start.tv_nsec;
  // Handle case where nanoseconds roll over
  if (nanoseconds < 0) {
    --seconds;
    nanoseconds += 1000000000;
  }
  return seconds * 1e9 + nanoseconds;
  // return ((_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec) /
  // 1e9) * 1e6;
}

long wait_finished() {
  while (!done_flag.load(std::memory_order_acquire)) {
  }
  clock_gettime(CLOCK_REALTIME, &_end);
  //   return ((_end.tv_sec - _start.tv_sec) +
  //           (_end.tv_nsec - _start.tv_nsec) / 1e9) *
  //          1e6;
  return (_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec);
  //   //   clock_gettime(CLOCK_MONOTONIC, &_end_1);
  //   long seconds = _end.tv_sec - _start.tv_sec;
  //   long nanoseconds = _end.tv_nsec - _start.tv_nsec;
  //   // Handle case where nanoseconds roll over
  //   if (nanoseconds < 0) {
  //     --seconds;
  //     nanoseconds += 1000000000;
  //   }
  //   return seconds * 1e9 + nanoseconds;
}

double get_duration() {
  clock_gettime(CLOCK_REALTIME, &_end);
  return ((_end.tv_sec - _start.tv_sec) +
          (_end.tv_nsec - _start.tv_nsec) / 1e9) *
         1e6;
}

// Function to clear all flags
void clear_flags() {
  ready_flag.store(false, std::memory_order_release);
  done_flag.store(false, std::memory_order_release);
  for (int i = 0; i < THREAD_NUM; ++i)
    finished_flags[i].store(false, std::memory_order_release);
}
}
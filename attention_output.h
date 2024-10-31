#include <cblas.h>
#include <immintrin.h>

#include <atomic>
#include <vector>

#include "utils.h"

void attn_output_trusted(float* values, const float* logits, float* result,
                         int const num_head, int const batch_size, int const K,
                         int const Dh, int const logits_haed_offset,
                         int const logits_batch_offset,
                         int const values_head_offset,
                         int const values_batch_offset,
                         int const result_head_offset,
                         int const result_batch_offset);

void attn_output_test(float* values, const float* logits, float* result,
                      int const num_head, int const batch_size, int const K,
                      int const Dh, int const values_head_offset,
                      int const values_batch_offset,
                      int const logits_haed_offset,
                      int const logits_batch_offset,
                      int const result_head_offset,
                      int const result_batch_offset);

void attn_output_1(float* values, float* values_t, const float* logits,
                   float* result, int K, int Dh);
void attn_output_2(float* values, const float* logits, float* result,
                   int const num_head, int const batch_size, int const K,
                   int const Dh, int const logits_haed_offset,
                   int const logits_batch_offset, int const values_head_offset,
                   int const values_batch_offset, int const result_head_offset,
                   int const result_batch_offset);
void attn_output_3_threaded(
    float* values, const float* logits, float* result, int const num_head,
    int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool>* ready_flag,
    std::atomic<bool>* finished_flag);
void attn_output_mul_add_threaded(
    float* values, const float* logits, float* result, int const num_head,
    int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset, int thread_id,
    int num_threads, std::atomic<bool>* ready_flag,
    std::atomic<bool>* finished_flag);
void attn_output_threaded(
    float* values, const float* logits, float* result, int const num_head,
    int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool>* ready_flag,
    std::atomic<bool>* finished_flag, std::atomic<bool>* stop_flag);
void attn_output_trusted_threaded(
    float* values, const float* logits, float* result, int const num_head,
    int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset, int thread_id,
    int num_threads, std::atomic<bool>* ready_flag,
    std::atomic<bool>* finished_flag);

void transpose_matrix_avx2_1(const float* src, float* dst, int K, int Dh);
void transpose_matrix_avx2_2(const float* src, float* dst, int K, int Dh);

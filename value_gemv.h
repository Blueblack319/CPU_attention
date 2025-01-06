#include <cblas.h>
#include <cuda_fp16.h>
#include <immintrin.h>
#include <time.h>

#include <atomic>
#include <vector>

void value_gemv_trusted(
    half *values, const half *logits, float *result, const int *topk_indices,
    int const topk_num, int const q_head_num, int const kv_head_num,
    int const batch_size, int const S_len, int const Dh,
    int const logits_head_offset, int const logits_batch_offset,
    int const values_head_offset, int const values_batch_offset,
    int const result_head_offset, int const result_batch_offset);

void value_gemv_trusted(
    float *values, const float *logits, float *result, const int *topk_indices,
    int const topk_num, int const q_head_num, int const kv_head_num,
    int const batch_size, int const S_len, int const Dh,
    int const logits_head_offset, int const logits_batch_offset,
    int const values_head_offset, int const values_batch_offset,
    int const result_head_offset, int const result_batch_offset);

void value_gemv_trusted_threaded(
    float *values, const float *logits, float *result, int const num_head,
    int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset, int thread_id,
    int num_threads, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag);

void value_gemv_threaded(
    float **values_arr, float **logits_arr, float **result_arr,
    int const num_head, int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_haed_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool> *ready_flag,
    std::atomic<bool> *finished_flag, std::atomic<bool> *stop_flag,
    std::atomic<int> *iter_num, double *end_time);

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
    std::atomic<int> *iter_num, double *end_time);

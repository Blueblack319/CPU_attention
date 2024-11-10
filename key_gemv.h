#include <cblas.h>
#include <immintrin.h>

#include <atomic>

#include "utils.h"

inline float hsum(__m128 x);
inline float hsum(__m256 x);

void key_gemv_threaded_1(
    float** keys_arr, float** queries_arr, float** logits_arr,
    int const num_head, int const batch_size, int const K, int const Dh,
    int const keys_head_offset, int const keys_batch_offset,
    int const queries_haed_offset, int const queries_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool>* ready_flag,
    std::atomic<bool>* finished_flag, std::atomic<bool>* stop_flag,
    std::atomic<int>* iter_num);

void key_gemv__threaded_2(float* keys, const float* queries, float* logits,
                          int const num_head, int const batch_size, int const K,
                          int const Dh, int const keys_head_offset,
                          int const keys_batch_offset, int const q_haed_offset,
                          int const q_batch_offset,
                          int const logits_head_offset,
                          int const logits_batch_offset);

void key_gemv_trusted(float* keys, const float* queries, float* logits,
                      int const num_head, int const batch_size, int const K,
                      int const Dh, int const logits_haed_offset,
                      int const logits_batch_offset, int const keys_head_offset,
                      int const keys_batch_offset, int const result_head_offset,
                      int const result_batch_offset);
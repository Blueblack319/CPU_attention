#include <cblas.h>
#include <immintrin.h>

inline float hsum(__m128 x);
inline float hsum(__m256 x);
void attn_score_1(float* keys, const float* queries, float* score,
                  int const num_head, int const batch_size, int const K,
                  int const Dh, int const logits_haed_offset,
                  int const logits_batch_offset, int const keys_head_offset,
                  int const keys_batch_offset, int const result_head_offset,
                  int const result_batch_offset);

void attn_score_2(float* keys, const float* queries, float* score,
                  int const num_head, int const batch_size, int const K,
                  int const Dh, int const logits_haed_offset,
                  int const logits_batch_offset, int const keys_head_offset,
                  int const keys_batch_offset, int const result_head_offset,
                  int const result_batch_offset);

void attn_score_trusted(float* keys, const float* queries, float* score,
                        int const num_head, int const batch_size, int const K,
                        int const Dh, int const logits_haed_offset,
                        int const logits_batch_offset,
                        int const keys_head_offset, int const keys_batch_offset,
                        int const result_head_offset,
                        int const result_batch_offset);
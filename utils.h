#include <cuda_fp16.h>
#include <immintrin.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>

inline float hsum(__m128 x);
inline float hsum(__m256 x);

void flush_cache();

float calculate_mse(const float *C, const float *golden_output, const size_t m);
float calculate_mae(const float *C, const float *golden_output, const size_t m);
float calculate_mse_half(const half *C, const float *golden_output,
                         const size_t m);
float calculate_mae_half(const half *C, const float *golden_output,
                         const size_t m);
float calculate_mse_half_gqa(const half *C, const float *golden_output,
                             const int topk_num, const int q_head_num,
                             const int kv_head_num, const int batch_size,
                             const int Dh, const int result_batch_offset,
                             const int result_head_offset);
void generateUniqueRandomValues(int *arr, int size, int min, int max);
bool fileExists(const char *filename);
void saveBinary(const char *filename, void *data, size_t size);
void loadBinary(const char *filename, void *data, size_t size);
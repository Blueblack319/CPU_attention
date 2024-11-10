#include <cuda_fp16.h>
#include <immintrin.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>

inline float hsum(__m128 x);
inline float hsum(__m256 x);

void flush_cache();

float calculate_mse(const float *C, const float *golden_output, const size_t m);
float calculate_mae(const float *C, const float *golden_output, const size_t m);
float calculate_mse_half(const half *C, const float *golden_output,
                         const size_t m);
float calculate_mae_half(const half *C, const float *golden_output,
                         const size_t m);
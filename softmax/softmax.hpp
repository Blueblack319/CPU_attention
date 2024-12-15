#include <cblas.h>
#include <immintrin.h>
#include <math.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>

#include "avx_mathfun.h"

#define USE_FMA 1
typedef std::pair<int, float> p_iv;

void softmax_trusted(float *x, const int size, const float max_val);
void softmax_trusted_1(float *x, const int size);
std::vector<p_iv> topk(const float *x, int size, int k, bool use_abs);
void softmax_avx(float *input, int size);
void softmax_avx2(float *x, const int size, const float max_val);
void softmax_trusted_threads(float *qk, const float *max_arr,
                             const size_t seq_len, const size_t head_num,
                             const size_t batch_size, const size_t head_offset,
                             const size_t batch_offset, const int thread_idx,
                             const int thread_num, const int start_idx,
                             const int end_idx, std::atomic<bool> *ready_flag,
                             std::atomic<bool> *finished_flag,
                             double *end_time);

inline void flush_cache() {
  size_t cache_flusher_size = 512 * 1024 * 1024;  // 512 MB
  char *cache_flusher = (char *)malloc(cache_flusher_size);

  for (size_t i = 0; i < cache_flusher_size; i += 4096) {
    cache_flusher[i] = 0;
  }

  free(cache_flusher);
}

////////////////////////////////////////////////////////////////////////

inline __m256 exp256_ps_1(__m256 x) {
  __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

  __m256 cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
  __m256 inv_LOG2EF = _mm256_set1_ps(0.693147180559945f);

  __m256 cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
  __m256 cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
  __m256 cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
  __m256 cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
  __m256 cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
  __m256 cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
  __m256 fx;
  __m256i imm0;
  __m256 one = _mm256_set1_ps(1.0f);

  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, cephes_LOG2EF);
  fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 z = _mm256_mul_ps(fx, inv_LOG2EF);
  x = _mm256_sub_ps(x, z);
  z = _mm256_mul_ps(x, x);

  __m256 y = cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}

/* compute exp(x) for x in [-87.33654f, 88.72283]
   maximum relative error: 3.1575e-6 (USE_FMA = 0); 3.1533e-6 (USE_FMA = 1)
*/
inline __m256 faster_more_accurate_exp_avx2(__m256 x) {
  __m256 t, f, p, r;
  __m256i i, j;

  const __m256 l2e = _mm256_set1_ps(1.442695041f);    /* log2(e) */
  const __m256 l2h = _mm256_set1_ps(-6.93145752e-1f); /* -log(2)_hi */
  const __m256 l2l = _mm256_set1_ps(-1.42860677e-6f); /* -log(2)_lo */
  /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
  const __m256 c0 = _mm256_set1_ps(0.041944388f);
  const __m256 c1 = _mm256_set1_ps(0.168006673f);
  const __m256 c2 = _mm256_set1_ps(0.499999940f);
  const __m256 c3 = _mm256_set1_ps(0.999956906f);
  const __m256 c4 = _mm256_set1_ps(0.999999642f);

  /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
  t = _mm256_mul_ps(x, l2e); /* t = log2(e) * x */
  r = _mm256_round_ps(
      t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); /* r = rint (t) */

#if USE_FMA
  f = _mm256_fmadd_ps(r, l2h, x); /* x - log(2)_hi * r */
  f = _mm256_fmadd_ps(r, l2l, f); /* f = x - log(2)_hi * r - log(2)_lo * r */
#else                             // USE_FMA
  p = _mm256_mul_ps(r, l2h); /* log(2)_hi * r */
  f = _mm256_add_ps(x, p);   /* x - log(2)_hi * r */
  p = _mm256_mul_ps(r, l2l); /* log(2)_lo * r */
  f = _mm256_add_ps(f, p);   /* f = x - log(2)_hi * r - log(2)_lo * r */
#endif                            // USE_FMA

  i = _mm256_cvtps_epi32(t); /* i = (int)rint(t) */

  /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
  p = c0; /* c0 */
#if USE_FMA
  p = _mm256_fmadd_ps(p, f, c1); /* c0*f+c1 */
  p = _mm256_fmadd_ps(p, f, c2); /* (c0*f+c1)*f+c2 */
  p = _mm256_fmadd_ps(p, f, c3); /* ((c0*f+c1)*f+c2)*f+c3 */
  p = _mm256_fmadd_ps(p, f, c4); /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
#else                            // USE_FMA
  p = _mm256_mul_ps(p, f);  /* c0*f */
  p = _mm256_add_ps(p, c1); /* c0*f+c1 */
  p = _mm256_mul_ps(p, f);  /* (c0*f+c1)*f */
  p = _mm256_add_ps(p, c2); /* (c0*f+c1)*f+c2 */
  p = _mm256_mul_ps(p, f);  /* ((c0*f+c1)*f+c2)*f */
  p = _mm256_add_ps(p, c3); /* ((c0*f+c1)*f+c2)*f+c3 */
  p = _mm256_mul_ps(p, f);  /* (((c0*f+c1)*f+c2)*f+c3)*f */
  p = _mm256_add_ps(p, c4); /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
#endif                           // USE_FMA

  /* exp(x) = 2^i * p */
  j = _mm256_slli_epi32(i, 23); /* i << 23 */
  r = _mm256_castsi256_ps(
      _mm256_add_epi32(j, _mm256_castps_si256(p))); /* r = p * 2^i */

  return r;
}

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
  //     t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 0, 3, 2)));  // add across lanes
  // return _mm_cvtss_f32(t);                                // get the result
}

inline float hsum(__m256 x) {
  return hsum(
      _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}

// void softmax_threaded(float **qk_arr, const float sum_quant,
//                       const float max_val, int const batch_size, int
//                       const K, int const head_offset, int const
//                       batch_offset, int const thread_id, int const
//                       num_threads, int const start_idx, int const
//                       end_idx, std::atomic<bool> *ready_flag,
//                       std::atomic<bool> *finished_flag,
//                       std::atomic<bool> *stop_flag, std::atomic<int>
//                       *iter_num, double *end_time);

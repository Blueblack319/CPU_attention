#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include "attention_output.h"
#include "attention_score.h"
#include "float.h"

// #define ITERATIONS 30
// #define ALLOC(n) (float *)memalign(4096, sizeof(float) * (n))

// int cpu_get_num_math();

// void llamafile_sgemm_openmp(long m, long n, long k, const void *A, long lda,
//                             const void *B, long ldb, void *C, long ldc,
//                             int Atype, int Btype, int Ctype) {
//   static int nth = cpu_get_num_math();
// #pragma omp parallel for
//   for (int ith = 0; ith < nth; ++ith) {
//     bool res = llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, ith, nth,
//     Atype,
//                                Btype, Ctype);
//     assert(res);
//   }
// }

// int test(void) {
//   int m = 256;
//   //   int n = 500;
//   int k = 260000;
//   //   int lda = ROUNDUP(k, 16);
//   //   int ldb = ROUNDUP(k, 16);
//   //   int ldc = ROUNDUP(m, 16);
//   float *A = ALLOC(lda * m);
//   float *B = ALLOC(ldb * n);
//   float *C = ALLOC(ldc * n);
//   float *G = ALLOC(ldc * n);
//   broadcast(A, lda * m, NAN);
//   broadcast(B, ldb * n, NAN);
//   broadcast(C, ldc * n, NAN);
//   broadcast(G, ldc * n, NAN);
//   randomize(k, m, A, lda);
//   randomize(k, n, B, ldb);

//   BENCH(ansiBLAS::sgemm(m, n, k, A, lda, B, ldb, G, ldc));
//   BENCH(llamafile_sgemm_openmp(m, n, k, A, lda, B, ldb, C, ldc,
//   GGML_TYPE_F32,
//                                GGML_TYPE_F32, GGML_TYPE_F32));

//   /////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   int flips = 0;
//   double err_sum = 0;
//   long long err_worst = 0;
//   for (int i = 0; i < m; ++i)
//     for (int j = 0; j < n; ++j) {
//       float g = G[ldc * j + i];
//       float c = C[ldc * j + i];
//       if (signbit(g) != signbit(c)) ++flips;
//       if (flt::isnan(g)) {
//         fprintf(stderr,
//                 "%s:%err: found nan in reference matrix: i=%err j=%err\n",
//                 __FILE__, __LINE__, i, j);
//         return 3;
//       }
//       if (flt::isnan(c)) {
//         fprintf(stderr, "%s:%err: found nan in output matrix: i=%err
//         j=%err\n",
//                 __FILE__, __LINE__, i, j);
//         return 4;
//       }
//       long long gi = flt::toint(g);
//       long long ci = flt::toint(c);
//       long long err = gi - ci;
//       if (err < 0) err = -err;
//       err_sum += err;
//       if (err > err_worst) err_worst = err;
//     }

//   double err_avg = err_sum / (m * n);
//   fprintf(stderr, "%12g ulp average\n", err_avg);
//   fprintf(stderr, "%12lld ulp worst\n", err_worst);
//   fprintf(stderr, "%12d flips\n", flips);

//   free(G);
//   free(C);
//   free(B);
//   free(A);
//   /////////////////////////////////////////////////////////////////////////////////////////////////////////////

//   return 0;
// }

// static inline float hsum_float_8(const __m256 x) {
//   __m128 res = _mm256_extractf128_ps(x, 1);
//   res = _mm_add_ps(res, _mm256_castps256_ps128(x));
//   res = _mm_add_ps(res, _mm_movehl_ps(res, res));
//   res = _mm_add_ss(res, _mm_movehdup_ps(res));
//   return _mm_cvtss_f32(res);
// }

// void trusted_gemv(float *A, float *B, float *C, const int m, const int n) {
//   // OpenBLAS expects a flat array for matrix A
//   std::vector<float> A_flat(m * n);
//   for (int i = 0; i < m; ++i) {
//     for (int j = 0; j < n; ++j) {
//       A_flat[i * n + j] = A[i][j];
//     }
//   }

//   // Call SGEMV from OpenBLAS
//   cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, A_flat.data(), n,
//               B.data(), 1, 0.0f, C.data(), 1);
// }

float calculate_mse(const float *C, const float *golden_output,
                    const size_t m) {
  float mse = 0.0;
  for (size_t i = 0; i < m; ++i) {
    mse += std::pow(C[i] - golden_output[i], 2);
  }
  return mse / m;
}

float calculate_mae(const float *C, const float *golden_output,
                    const size_t m) {
  float mae = 0.0;
  for (size_t i = 0; i < m; ++i) {
    mae = std::max(mae, std::fabs(C[i] - golden_output[i]));
  }
  return mae;
}

void attn_output_eval(const size_t K, const size_t Dh, const size_t num_head,
                      const size_t batch_size, const int values_head_offset,
                      const int values_batch_offset,
                      int const logits_head_offset,
                      int const logits_batch_offset,
                      int const result_head_offset,
                      int const result_batch_offset) {
  float *values = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * K * Dh * sizeof(float)));
  float *values_trusted = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * K * Dh * sizeof(float)));

  float *logits = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * K * sizeof(float)));

  float *result = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * Dh * sizeof(float)));
  float *result_trusted = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * Dh * sizeof(float)));

  float *values_transposed = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * K * Dh * sizeof(float)));

  // Initialize values (example values for testing)
  for (size_t i = 0; i < num_head; ++i)
    for (size_t j = 0; j < batch_size; ++j) {
      for (size_t k = 0; k < K; ++k)
        for (size_t l = 0; l < Dh; ++l) {
          values[i * values_head_offset + j * values_batch_offset + k * Dh +
                 l] = static_cast<float>(l + 1);
          values_trusted[i * values_head_offset + j * values_batch_offset +
                         k * Dh + l] = static_cast<float>(l + 1);
        }
    }

  for (size_t i = 0; i < num_head; ++i)
    for (size_t j = 0; j < batch_size; ++j)
      for (size_t k = 0; k < K; ++k)
        logits[i * logits_head_offset + j * logits_batch_offset + k] = 0.3f;

  for (size_t i = 0; i < num_head * batch_size * Dh; ++i) {
    result[i] = 0.f;
    result_trusted[i] = 0.f;
  }

  std::chrono::microseconds duration_micro;
  std::chrono::microseconds duration_micro_trusted;
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  // Measure execution time
  start = std::chrono::high_resolution_clock::now();
  attn_output_test(values, logits, result, num_head, batch_size, K, Dh,
                   values_head_offset, values_batch_offset, logits_head_offset,
                   logits_batch_offset, result_head_offset,
                   result_batch_offset);
  end = std::chrono::high_resolution_clock::now();
  duration_micro =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  attn_output_trusted(
      values_trusted, logits, result_trusted, num_head, batch_size, K, Dh,
      logits_head_offset, logits_batch_offset, values_head_offset,
      values_batch_offset, result_head_offset, result_batch_offset);
  end = std::chrono::high_resolution_clock::now();
  duration_micro_trusted =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Calculate FLOPs and GFLOPs
  // size_t K = 48, Dh = 128, num_head = 4, batch_size = 64;
  double flops = 2.0 * Dh * K * num_head * batch_size;
  double gflops = flops / (duration_micro.count() * 1e3);
  double gflops_trusted = flops / (duration_micro_trusted.count() * 1e3);

  std::cout
      << "==========================My attn_output=========================="
      << std::endl;
  std::cout << "Elapsed time: " << 0.000001f * duration_micro.count()
            << " seconds" << std::endl;
  std::cout << "GFLOPs: " << gflops << std::endl;

  std::cout << "==========================Trusted "
               "attn_output=========================="
            << std::endl;
  std::cout << "Elapsed time: " << 0.000001f * duration_micro_trusted.count()
            << " seconds" << std::endl;
  std::cout << "GFLOPs: " << gflops_trusted << std::endl;

  // Calculate MSE and MAE
  float mse = calculate_mse(result, result_trusted, Dh);
  float mae = calculate_mae(result, result_trusted, Dh);

  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Maximum Absolute Error: " << mae << std::endl;

  // std::cout << "Attention Output: ";
  // for (int i = 0; i < Dh; i++) std::cout << result[i] << " ";
  // std::cout << std::endl;

  // std::cout << "Attention Output trusted: ";
  // for (int i = 0; i < Dh; i++) std::cout << result_trusted[i] << " ";
  // std::cout << std::endl;

  // Free the allocated memory
  free(values);
  free(logits);
  free(result);
  free(result_trusted);
}

void attn_score_eval(const size_t K, const size_t Dh, const size_t num_head,
                     const size_t batch_size, const int keys_head_offset,
                     const int keys_batch_offset, int const q_head_offset,
                     int const q_batch_offset, int const result_head_offset,
                     int const result_batch_offset) {
  float *A = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * Dh * K * sizeof(float)));
  float *B = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * Dh * sizeof(float)));
  float *C = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * K * sizeof(float)));
  float *golden_output = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * K * sizeof(float)));

  if (!A || !B || !C) {
    std::cerr << "Memory allocation failed!" << std::endl;
    return;
  }

  // Initialize A and B
  for (int i = 0; i < Dh * K; ++i) {
    A[i] = 1.0f;  // or any value you want to test
  }
  for (int i = 0; i < K; ++i) {
    B[i] = 1.0f;  // or any value you want to test
  }
  for (int i = 0; i < Dh; ++i) {
    C[i] = 0.0f;
    golden_output[i] = 0.0f;
  }

  std::chrono::microseconds duration_micro;
  std::chrono::microseconds duration_micro_trusted;
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  // Measure execution time
  start = std::chrono::high_resolution_clock::now();
  attn_score_2(A, B, C, num_head, batch_size, K, Dh, keys_head_offset,
               keys_batch_offset, q_head_offset, q_batch_offset,
               result_head_offset, result_batch_offset);
  end = std::chrono::high_resolution_clock::now();
  duration_micro =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Measure execution time
  start = std::chrono::high_resolution_clock::now();
  attn_score_trusted(A, B, golden_output, num_head, batch_size, K, Dh,
                     keys_head_offset, keys_batch_offset, q_head_offset,
                     q_batch_offset, result_head_offset, result_batch_offset);
  end = std::chrono::high_resolution_clock::now();
  duration_micro_trusted =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  //   duration_micro_trusted = gemv_trusted(A, B, C);

  // Calculate FLOPs and GFLOPs
  double flops = 2.0 * Dh * K;
  double gflops = flops / (duration_micro.count() * 1e3);
  double gflops_trusted = flops / (duration_micro_trusted.count() * 1e3);

  std::cout
      << "==========================My attn_score=========================="
      << std::endl;
  std::cout << "Elapsed time: " << 0.000001f * duration_micro.count()
            << " seconds" << std::endl;
  std::cout << "GFLOPs: " << gflops << std::endl;

  std::cout << "==========================Trusted "
               "attn_score=========================="
            << std::endl;
  std::cout << "Elapsed time: " << 0.000001f * duration_micro_trusted.count()
            << " seconds" << std::endl;
  std::cout << "GFLOPs: " << gflops_trusted << std::endl;

  // Calculate MSE and MAE
  float mse = calculate_mse(C, golden_output, Dh);
  float mae = calculate_mae(C, golden_output, Dh);

  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Maximum Absolute Error: " << mae << std::endl;

  // std::cout << "Attention Output: ";
  // for (int i = 0; i < K; i++) std::cout << C[i] << " ";
  // std::cout << std::endl;

  // std::cout << "Attention Output trusted: ";
  // for (int i = 0; i < K; i++) std::cout << golden_output[i] << " ";
  // std::cout << std::endl;

  free(C);
  free(B);
  free(A);
}

int main(int argc, char *argv[]) {
  size_t K = 48, Dh = 128, num_head = 4, batch_size = 64;

  int const kv_head_offset = batch_size * K * Dh;
  int const kv_batch_offset = K * Dh;
  int const logits_score_head_offset = batch_size * K;
  int const logits_score_batch_offset = K;
  int const q_out_head_offset = batch_size * Dh;
  int const q_out_batch_offset = Dh;
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  // Attention Score
  // attn_score_eval(K, Dh, num_head, batch_size, kv_head_offset,
  // kv_batch_offset,
  //                 q_out_head_offset, q_out_batch_offset,
  //                 logits_score_head_offset, logits_score_batch_offset);
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  // Attention Output
  attn_output_eval(K, Dh, num_head, batch_size, kv_head_offset, kv_batch_offset,
                   logits_score_head_offset, logits_score_batch_offset,
                   q_out_head_offset, q_out_batch_offset);

  return 0;
}

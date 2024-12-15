#include "softmax.hpp"

std::vector<p_iv> topk(const float* x, int size, int k, bool use_abs) {
  std::vector<p_iv> x_idxs;
  x_idxs.reserve(size);
  for (int i = 0; i < size; i++) {
    x_idxs.emplace_back(i, use_abs ? std::abs(x[i]) : x[i]);
  }

  if (k >= size) {
    return x_idxs;
  }

  std::nth_element(x_idxs.begin(), x_idxs.begin() + k - 1, x_idxs.end(),
                   [](p_iv a, p_iv b) { return a.second > b.second; });
  x_idxs.resize(k);
  return x_idxs;
}

void softmax_trusted_1(float* x, const int size) {
  const float max_val = *std::max_element(x, x + size);
  float tot = 0.0;
  for (int i = 0; i < size; i++) {
    // x[i] = std::exp(x[i] - max_val);
    x[i] = expf(x[i] - max_val);
    tot += x[i];
  }
  for (int i = 0; i < size; i++) {
    x[i] /= tot;
  }
}

void softmax_trusted(float* x, const int size, const float max_val) {
  float tot = 0.0;
  for (int i = 0; i < size; i++) {
    // x[i] = std::exp(x[i] - max_val);
    x[i] = expf(x[i] - max_val);
    tot += x[i];
  }
  for (int i = 0; i < size; i++) {
    x[i] /= tot;
  }
}

// void softmax_trusted_threads(float* qk, const float* max_arr, const int
// seq_len,
//                              const int head_num, const int batch_size,
//                              const int head_offset, const int batch_offset,
//                              const int thread_idx, const int thread_num,
//                              const int start_idx, const int end_idx) {
//   for (int idx = start_idx; idx < end_idx; ++idx) {
//     const int head_idx = idx / batch_size;
//     const int batch_idx = idx % batch_size;

//     float tot = 0.0;
//     for (int i = 0; i < seq_len; i++) {
//       // qk[i] = std::exp(qk[i] - max_val);
//       qk[head_idx * head_offset + batch_idx * batch_offset + i] =
//           expf(qk[head_idx * head_offset + batch_idx * batch_offset + i] -
//                max_arr[head_idx * head_offset + batch_idx * batch_offset]);
//       tot += qk[head_idx * head_offset + batch_idx * batch_offset + i];
//     }
//     for (int i = 0; i < seq_len; i++) {
//       qk[head_idx * head_offset + batch_idx * batch_offset + i] /= tot;
//     }
//   }
// }

void softmax_trusted_threads(float* qk, const float* max_arr,
                             const size_t seq_len, const size_t head_num,
                             const size_t batch_size, const size_t head_offset,
                             const size_t batch_offset, const int thread_idx,
                             const int thread_num, const int start_idx,
                             const int end_idx, std::atomic<bool>* ready_flag,
                             std::atomic<bool>* finished_flag,
                             double* end_time) {
  struct timespec _end, _start;
  while (!ready_flag->load(std::memory_order_acquire)) {
    // Busy-wait (spinlock) until the main thread signals ready
  }
  clock_gettime(CLOCK_REALTIME, &_start);
  for (int idx = start_idx; idx < end_idx; ++idx) {
    const int head_idx = idx / batch_size;
    const int batch_idx = idx % batch_size;

    float tot = 0.0;
    for (int i = 0; i < seq_len; i++) {
      // qk[i] = std::exp(qk[i] - max_val);
      qk[head_idx * head_offset + batch_idx * batch_offset + i] =
          expf(qk[head_idx * head_offset + batch_idx * batch_offset + i] -
               max_arr[head_idx * batch_size + batch_idx]);
      tot += qk[head_idx * head_offset + batch_idx * batch_offset + i];
    }
    for (int i = 0; i < seq_len; i++) {
      qk[head_idx * head_offset + batch_idx * batch_offset + i] /= tot;
    }
  }
  // Mark this thread as finished
  finished_flag->store(true, std::memory_order_release);
  clock_gettime(CLOCK_REALTIME, &_end);
  *end_time =
      ((_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec) / 1e9) *
      1e6;
}

//////////////////////////////////////////////////////////////////////////////////

inline __m256 _mm256_exp_ps(__m256 invec) {
  float* element = (float*)&invec;
  return _mm256_setr_ps(expf(element[0]), expf(element[1]), expf(element[2]),
                        expf(element[3]), expf(element[4]), expf(element[5]),
                        expf(element[6]), expf(element[7]));
}

// Too slow
void softmax_avx2(float* x, const int size, const float max_val) {
  float tot = 0.0;
  __m256 max_broadcast = _mm256_set1_ps(max_val);
  __m256 tot1 = _mm256_set1_ps(0);
  __m256 tot2 = _mm256_set1_ps(0);
  __m256 tot3 = _mm256_set1_ps(0);
  __m256 tot4 = _mm256_set1_ps(0);
  __m256 tot5 = _mm256_set1_ps(0);
  __m256 tot6 = _mm256_set1_ps(0);
  __m256 tot7 = _mm256_set1_ps(0);
  __m256 tot8 = _mm256_set1_ps(0);

  for (int i = 0; i < size; i += 8 * 8) {
    __m256 data1 = _mm256_load_ps(&x[i]);
    __m256 data2 = _mm256_load_ps(&x[i + 8]);
    __m256 data3 = _mm256_load_ps(&x[i + 16]);
    __m256 data4 = _mm256_load_ps(&x[i + 24]);
    __m256 data5 = _mm256_load_ps(&x[i + 32]);
    __m256 data6 = _mm256_load_ps(&x[i + 40]);
    __m256 data7 = _mm256_load_ps(&x[i + 48]);
    __m256 data8 = _mm256_load_ps(&x[i + 56]);
    data1 = _mm256_exp_ps(_mm256_sub_ps(data1, max_broadcast));
    data2 = _mm256_exp_ps(_mm256_sub_ps(data2, max_broadcast));
    data3 = _mm256_exp_ps(_mm256_sub_ps(data3, max_broadcast));
    data4 = _mm256_exp_ps(_mm256_sub_ps(data4, max_broadcast));
    data5 = _mm256_exp_ps(_mm256_sub_ps(data5, max_broadcast));
    data6 = _mm256_exp_ps(_mm256_sub_ps(data6, max_broadcast));
    data7 = _mm256_exp_ps(_mm256_sub_ps(data7, max_broadcast));
    data8 = _mm256_exp_ps(_mm256_sub_ps(data8, max_broadcast));
    tot1 = _mm256_add_ps(tot1, data1);
    tot2 = _mm256_add_ps(tot2, data2);
    tot3 = _mm256_add_ps(tot3, data3);
    tot4 = _mm256_add_ps(tot4, data4);
    tot5 = _mm256_add_ps(tot5, data5);
    tot6 = _mm256_add_ps(tot6, data6);
    tot7 = _mm256_add_ps(tot7, data7);
    tot8 = _mm256_add_ps(tot8, data8);
    _mm256_store_ps(&x[i], data1);
    _mm256_store_ps(&x[i + 8], data2);
    _mm256_store_ps(&x[i + 16], data3);
    _mm256_store_ps(&x[i + 24], data4);
    _mm256_store_ps(&x[i + 32], data1);
    _mm256_store_ps(&x[i + 40], data2);
    _mm256_store_ps(&x[i + 48], data3);
    _mm256_store_ps(&x[i + 56], data4);
  }
  tot += hsum(tot1);
  tot += hsum(tot2);
  tot += hsum(tot3);
  tot += hsum(tot4);
  tot += hsum(tot5);
  tot += hsum(tot6);
  tot += hsum(tot7);
  tot += hsum(tot8);
  for (int i = 0; i < size; i++) {
    x[i] /= tot;
  }
}

// Find the horizontal maximum of an __m256 vector
float horizontal_max_ps(__m256 vec) {
  // Step 1: Permute elements within 128-bit lanes
  __m256 temp1 =
      _mm256_permute_ps(vec, _MM_SHUFFLE(2, 3, 0, 1));  // Swap adjacent pairs
  __m256 max1 = _mm256_max_ps(vec, temp1);              // Max of adjacent pairs

  // Step 2: Permute elements within 128-bit lanes again
  __m256 temp2 = _mm256_permute_ps(
      max1, _MM_SHUFFLE(1, 0, 3, 2));  // Swap nibbles (halves of 128-bit lanes)
  __m256 max2 = _mm256_max_ps(max1, temp2);  // Max of adjacent nibbles

  // Step 3: Permute across 128-bit lanes
  __m256 temp3 = _mm256_permute2f128_ps(
      max2, max2, 1);  // Swap upper and lower 128-bit lanes
  __m256 max3 = _mm256_max_ps(max2, temp3);  // Max between lanes

  // Step 4: Extract the result
  return _mm_cvtss_f32(
      _mm256_castps256_ps128(max3));  // Extract the lowest element
}

void softmax_avx(float* input, int size) {
  __m256 max_val1 = _mm256_set1_ps(-INFINITY);
  __m256 max_val2 = _mm256_set1_ps(-INFINITY);
  __m256 max_val3 = _mm256_set1_ps(-INFINITY);
  __m256 max_val4 = _mm256_set1_ps(-INFINITY);
  __m256 max_val5 = _mm256_set1_ps(-INFINITY);
  __m256 max_val6 = _mm256_set1_ps(-INFINITY);
  __m256 max_val7 = _mm256_set1_ps(-INFINITY);
  __m256 max_val8 = _mm256_set1_ps(-INFINITY);
  const size_t vec_len = 8;

  // Step 1: Find the max value
  for (int i = 0; i < size; i += vec_len * 8) {
    __m256 data1 = _mm256_load_ps(&input[i]);
    __m256 data2 = _mm256_load_ps(&input[i + 8]);
    __m256 data3 = _mm256_load_ps(&input[i + 16]);
    __m256 data4 = _mm256_load_ps(&input[i + 24]);
    __m256 data5 = _mm256_load_ps(&input[i + 32]);
    __m256 data6 = _mm256_load_ps(&input[i + 40]);
    __m256 data7 = _mm256_load_ps(&input[i + 48]);
    __m256 data8 = _mm256_load_ps(&input[i + 56]);
    max_val1 = _mm256_max_ps(max_val1, data1);
    max_val2 = _mm256_max_ps(max_val2, data2);
    max_val3 = _mm256_max_ps(max_val3, data3);
    max_val4 = _mm256_max_ps(max_val4, data4);
    max_val5 = _mm256_max_ps(max_val5, data5);
    max_val6 = _mm256_max_ps(max_val6, data6);
    max_val7 = _mm256_max_ps(max_val7, data7);
    max_val8 = _mm256_max_ps(max_val8, data8);
  }
  // Reduce max_val horizontally
  //   float max_array[8] = {
  //       horizontal_max_ps(max_val1), horizontal_max_ps(max_val2),
  //       horizontal_max_ps(max_val3), horizontal_max_ps(max_val4),
  //       horizontal_max_ps(max_val5), horizontal_max_ps(max_val6),
  //       horizontal_max_ps(max_val7), horizontal_max_ps(max_val8),
  //   };
  __m256 max_array =
      _mm256_set_ps(horizontal_max_ps(max_val1), horizontal_max_ps(max_val2),
                    horizontal_max_ps(max_val3), horizontal_max_ps(max_val4),
                    horizontal_max_ps(max_val5), horizontal_max_ps(max_val6),
                    horizontal_max_ps(max_val7), horizontal_max_ps(max_val8));
  float max_scalar = horizontal_max_ps(max_array);
  __m256 max_broadcast = _mm256_set1_ps(max_scalar);
  __m256 sum;

  // Step 2: Compute exponentials and sum
  for (int i = 0; i < size; i += vec_len * 8) {
    __m256 data1 = _mm256_load_ps(&input[i]);
    __m256 data2 = _mm256_load_ps(&input[i + 8]);
    __m256 data3 = _mm256_load_ps(&input[i + 16]);
    __m256 data4 = _mm256_load_ps(&input[i + 24]);
    __m256 data5 = _mm256_load_ps(&input[i + 32]);
    __m256 data6 = _mm256_load_ps(&input[i + 40]);
    __m256 data7 = _mm256_load_ps(&input[i + 48]);
    __m256 data8 = _mm256_load_ps(&input[i + 56]);
    data1 = _mm256_sub_ps(data1, max_broadcast);  // x - max
    data1 = exp256_ps(data1);                     // e^(x - max), implement
    data2 = _mm256_sub_ps(data2, max_broadcast);  // x -
    data2 = exp256_ps(data2);                     // e^(x - max), implement
    data3 = _mm256_sub_ps(data3, max_broadcast);  // x -
    data3 = exp256_ps(data3);                     // e^(x - max), implement
    data4 = _mm256_sub_ps(data4, max_broadcast);  // x -
    data4 = exp256_ps(data4);                     // e^(x - max), implement
    data5 = _mm256_sub_ps(data5, max_broadcast);  // x -
    data5 = exp256_ps(data5);                     // e^(x - max), implement
    data6 = _mm256_sub_ps(data6, max_broadcast);  // x -
    data6 = exp256_ps(data6);                     // e^(x - max), implement
    data7 = _mm256_sub_ps(data7, max_broadcast);  // x -
    data7 = exp256_ps(data7);                     // e^(x - max), implement
    data8 = _mm256_sub_ps(data8, max_broadcast);  // x -
    data8 = exp256_ps(data8);                     // e^(x - max), implement

    // _mm256_store_ps(&input[i], data1);
    // _mm256_store_ps(&input[i + 8], data2);
    data1 = _mm256_add_ps(data1, data2);
    // _mm256_store_ps(&input[i + 16], data3);
    // _mm256_store_ps(&input[i + 24], data4);
    data3 = _mm256_add_ps(data3, data4);
    // _mm256_store_ps(&input[i + 32], data5);
    // _mm256_store_ps(&input[i + 40], data6);
    data5 = _mm256_add_ps(data5, data6);
    // _mm256_store_ps(&input[i + 48], data7);
    // _mm256_store_ps(&input[i + 56], data8);
    data7 = _mm256_add_ps(data7, data8);

    data1 = _mm256_add_ps(data1, data3);
    data5 = _mm256_add_ps(data5, data7);

    data1 = _mm256_add_ps(data1, data5);
    sum = _mm256_add_ps(sum, data1);
  }

  // Reduce sum horizontally
  float sum_scalar = hsum(sum);

  //   float tot = 0.0;
  //   for (int i = 0; i < size; i++) {
  //     input[i] = std::exp(input[i] - max_scalar);
  //     // input[i] = input[i] - max_scalar;
  //     tot += input[i];
  //   }
  //   for (int i = 0; i < size; i++) {
  //     input[i] /= tot;
  //   }
  //   printf("\n\n%f VS %f\n\n", sum_scalar, tot);

  __m256 sum_broadcast = _mm256_set1_ps(sum_scalar);

  // Step 3: Normalize
  for (int i = 0; i < size; i += vec_len * 8) {
    __m256 data1 = _mm256_load_ps(&input[i]);
    __m256 data2 = _mm256_load_ps(&input[i + 8]);
    __m256 data3 = _mm256_load_ps(&input[i + 16]);
    __m256 data4 = _mm256_load_ps(&input[i + 24]);
    __m256 data5 = _mm256_load_ps(&input[i + 32]);
    __m256 data6 = _mm256_load_ps(&input[i + 40]);
    __m256 data7 = _mm256_load_ps(&input[i + 48]);
    __m256 data8 = _mm256_load_ps(&input[i + 56]);
    data1 = _mm256_div_ps(data1, sum_broadcast);
    data2 = _mm256_div_ps(data2, sum_broadcast);
    data3 = _mm256_div_ps(data3, sum_broadcast);
    data4 = _mm256_div_ps(data4, sum_broadcast);
    data5 = _mm256_div_ps(data5, sum_broadcast);
    data6 = _mm256_div_ps(data6, sum_broadcast);
    data7 = _mm256_div_ps(data7, sum_broadcast);
    data8 = _mm256_div_ps(data8, sum_broadcast);
    _mm256_store_ps(&input[i], data1);
    _mm256_store_ps(&input[i + 8], data2);
    _mm256_store_ps(&input[i + 16], data3);
    _mm256_store_ps(&input[i + 24], data4);
    _mm256_store_ps(&input[i + 32], data5);
    _mm256_store_ps(&input[i + 40], data6);
    _mm256_store_ps(&input[i + 48], data7);
    _mm256_store_ps(&input[i + 56], data8);
  }
}

// void softmax_threaded(float **qk_arr, const float sum_quant,
//                       const float max_val, int const batch_size, int const K,
//                       int const head_offset, int const batch_offset,
//                       int const thread_id, int const num_threads,
//                       int const start_idx, int const end_idx,
//                       std::atomic<bool> *ready_flag,
//                       std::atomic<bool> *finished_flag,
//                       std::atomic<bool> *stop_flag, std::atomic<int>
//                       *iter_num, double *end_time) {
//   struct timespec _end, _start;
//   float *qk;
//   while (!stop_flag->load(std::memory_order_acquire)) {
//     while (!(ready_flag->load(std::memory_order_acquire) &&
//              !finished_flag->load(std::memory_order_acquire) &&
//              !stop_flag->load(std::memory_order_acquire))) {
//       // ready_flag: true
//       // finished_flag: false
//       // stop_flag: false
//       // Busy-wait (spinlock) until the main thread signals ready
//       if (stop_flag->load(std::memory_order_acquire)) return;
//       qk = qk_arr[iter_num->load(std::memory_order_acquire)];
//     }
//     clock_gettime(CLOCK_REALTIME, &_start);
//     if (stop_flag->load(std::memory_order_acquire)) return;

//     // Multiply and Add
//     for (int idx = start_idx; idx < end_idx; ++idx) {
//       int i = idx / batch_size;
//       int j = idx % batch_size;

//       __m256 c00 = _mm256_setzero_ps();
//       __m256 c01 = _mm256_setzero_ps();
//       __m256 c02 = _mm256_setzero_ps();
//       __m256 c03 = _mm256_setzero_ps();
//       __m256 c04 = _mm256_setzero_ps();
//       __m256 c05 = _mm256_setzero_ps();
//       __m256 c06 = _mm256_setzero_ps();
//       __m256 c07 = _mm256_setzero_ps();
//       __m256 c08 = _mm256_setzero_ps();
//       __m256 c09 = _mm256_setzero_ps();
//       __m256 c10 = _mm256_setzero_ps();
//       __m256 c11 = _mm256_setzero_ps();
//       __m256 c12 = _mm256_setzero_ps();
//       __m256 c13 = _mm256_setzero_ps();
//       __m256 c14 = _mm256_setzero_ps();
//       __m256 c15 = _mm256_setzero_ps();

//       for (int k = 0; k < K; ++k) {
//         float logit =
//             logits[i * logits_haed_offset + j * logits_batch_offset + k];
//         __m256 logit_vec = _mm256_set1_ps(logit);

//         if (k + 1 < K) {
//           _mm_prefetch((const char *)(values + i * values_head_offset +
//                                       j * values_batch_offset + (k + 1) *
//                                       Dh),
//                        _MM_HINT_T0);
//         }
//         __m256 v00 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh);
//         __m256 v01 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 8);
//         __m256 v02 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 16);
//         __m256 v03 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 24);
//         __m256 v04 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 32);
//         __m256 v05 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 40);
//         __m256 v06 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 48);
//         __m256 v07 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 56);
//         __m256 v08 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 64);
//         __m256 v09 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 72);
//         __m256 v10 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 80);
//         __m256 v11 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 88);
//         __m256 v12 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 96);
//         __m256 v13 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 104);
//         __m256 v14 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 112);
//         __m256 v15 = _mm256_load_ps(values + i * values_head_offset +
//                                     j * values_batch_offset + k * Dh + 120);
//         c00 = _mm256_fmadd_ps(logit_vec, v00, c00);
//         c01 = _mm256_fmadd_ps(logit_vec, v01, c01);
//         c02 = _mm256_fmadd_ps(logit_vec, v02, c02);
//         c03 = _mm256_fmadd_ps(logit_vec, v03, c03);
//         c04 = _mm256_fmadd_ps(logit_vec, v04, c04);
//         c05 = _mm256_fmadd_ps(logit_vec, v05, c05);
//         c06 = _mm256_fmadd_ps(logit_vec, v06, c06);
//         c07 = _mm256_fmadd_ps(logit_vec, v07, c07);
//         c08 = _mm256_fmadd_ps(logit_vec, v08, c08);
//         c09 = _mm256_fmadd_ps(logit_vec, v09, c09);
//         c10 = _mm256_fmadd_ps(logit_vec, v10, c10);
//         c11 = _mm256_fmadd_ps(logit_vec, v11, c11);
//         c12 = _mm256_fmadd_ps(logit_vec, v12, c12);
//         c13 = _mm256_fmadd_ps(logit_vec, v13, c13);
//         c14 = _mm256_fmadd_ps(logit_vec, v14, c14);
//         c15 = _mm256_fmadd_ps(logit_vec, v15, c15);
//       }
//       // Store the accumulated result back into the result array
//       _mm256_store_ps(result + i * result_head_offset + j *
//       result_batch_offset,
//                       c00);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 8,
//           c01);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 16,
//           c02);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 24,
//           c03);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 32,
//           c04);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 40,
//           c05);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 48,
//           c06);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 56,
//           c07);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 64,
//           c08);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 72,
//           c09);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 80,
//           c10);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 88,
//           c11);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 96,
//           c12);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 104,
//           c13);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 112,
//           c14);
//       _mm256_store_ps(
//           result + i * result_head_offset + j * result_batch_offset + 120,
//           c15);
//     }
//     // Mark this thread as finished
//     finished_flag->store(true, std::memory_order_release);
//     clock_gettime(CLOCK_REALTIME, &_end);
//     *end_time = ((_end.tv_sec - _start.tv_sec) +
//                  (_end.tv_nsec - _start.tv_nsec) / 1e9) *
//                 1e6;
//     while (ready_flag->load(std::memory_order_acquire)) {
//       // Wait until ready_flag is reset
//     }
//   }
// }

// #include "test_softmax.hpp"

// #define ITER 51
// static struct timespec start, end;
// static double acc_time_sec;
// static double cur_time_sec;

// void softmax_eval(const size_t K, const size_t Dh, const size_t num_head,
//                   const size_t batch_size, const int head_offset,
//                   const int batch_offset,
//                   int const num_threads) {  // Total work = 256 / num_threads
//   //////////////////////////////////////////////////////////////////////////////////
//   // Allocate memory
//   float *values[ITER];

//   for (size_t i = 0; i < ITER; ++i) {
//     values[i] = static_cast<float *>(
//         aligned_alloc(64, num_head * batch_size * K * Dh * sizeof(float)));
//     logits[i] = static_cast<float *>(
//         aligned_alloc(64, num_head * batch_size * K * sizeof(float)));
//     result[i] = static_cast<float *>(
//         aligned_alloc(64, num_head * batch_size * Dh * sizeof(float)));
//   }
//   float *values_trusted = static_cast<float *>(
//       aligned_alloc(64, num_head * batch_size * K * Dh * sizeof(float)));
//   float *logits_trusted = static_cast<float *>(
//       aligned_alloc(64, num_head * batch_size * K * sizeof(float)));
//   float *result_trusted = static_cast<float *>(
//       aligned_alloc(64, num_head * batch_size * Dh * sizeof(float)));

//   // random generator
//   std::default_random_engine gen;
//   std::uniform_real_distribution<float> dist(-1.0, 1.0);

//   // Initialize variables with random values
//   for (size_t ii = 0; ii < ITER; ++ii)
//     for (size_t i = 0; i < num_head; ++i)
//       for (size_t j = 0; j < batch_size; ++j)
//         for (size_t k = 0; k < K; ++k)
//           for (size_t l = 0; l < Dh; ++l) {
//             if (ii == 0) {
//               float rand_val = dist(gen);
//               values[ii][i * values_head_offset + j * values_batch_offset +
//                          k * Dh + l] = rand_val;
//               values_trusted[i * values_head_offset + j * values_batch_offset
//               +
//                              k * Dh + l] = rand_val;
//             } else {
//               values[ii][i * values_head_offset + j * values_batch_offset +
//                          k * Dh + l] =
//                   values_trusted[i * values_head_offset +
//                                  j * values_batch_offset + k * Dh + l];
//             }
//           }

//   for (size_t ii = 0; ii < ITER; ++ii)
//     for (size_t i = 0; i < num_head; ++i)
//       for (size_t j = 0; j < batch_size; ++j)
//         for (size_t k = 0; k < K; ++k) {
//           if (ii == 0) {
//             float rand_val = dist(gen);
//             logits[ii][i * logits_head_offset + j * logits_batch_offset + k]
//             =
//                 rand_val;
//             logits_trusted[i * logits_head_offset + j * logits_batch_offset +
//                            k] = rand_val;
//           } else {
//             logits[ii][i * logits_head_offset + j * logits_batch_offset + k]
//             =
//                 logits_trusted[i * logits_head_offset +
//                                j * logits_batch_offset + k];
//           }
//         }

//   for (size_t ii = 0; ii < ITER; ++ii) {
//     for (size_t i = 0; i < num_head * batch_size * Dh; ++i) {
//       result[ii][i] = 0.f;
//       if (ii == 0) result_trusted[i] = 0.f;
//     }
//   }

//   double total_time_sec, total_time_sec_trusted;
//   //////////////////////////////////////////////////////////////////////////////////
// }
#include <immintrin.h>
#include <numa.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "attention_output.h"
#include "attention_score.h"
#include "float.h"
// #include "shared.h"

#define ITER 100

void flush_cache() {
  size_t cache_flusher_size = 512 * 1024 * 1024;  // 512 MB
  char *cache_flusher = (char *)malloc(cache_flusher_size);

  for (size_t i = 0; i < cache_flusher_size; i += 4096) {
    cache_flusher[i] = 0;
  }

  free(cache_flusher);
}

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

void value_gemv_eval(const size_t K, const size_t Dh, const size_t num_head,
                     const size_t batch_size, const size_t iteration,
                     const int values_iter_offset, const int values_head_offset,
                     const int values_batch_offset,
                     int const logits_iter_offset, int const logits_head_offset,
                     int const logits_batch_offset,
                     int const result_head_offset,
                     int const result_batch_offset) {
  // Allocate memory
  float *values[ITER];
  float *logits[ITER];
  float *result[ITER];

  for (size_t i = 0; i < ITER; ++i) {
    values[i] = static_cast<float *>(
        aligned_alloc(64, num_head * batch_size * K * Dh * sizeof(float)));
    logits[i] = static_cast<float *>(
        aligned_alloc(64, num_head * batch_size * K * sizeof(float)));
    result[i] = static_cast<float *>(
        aligned_alloc(64, num_head * batch_size * Dh * sizeof(float)));
  }
  float *values_trusted = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * K * Dh * sizeof(float)));
  float *logits_trusted = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * K * sizeof(float)));
  float *result_trusted = static_cast<float *>(
      aligned_alloc(64, num_head * batch_size * Dh * sizeof(float)));

  // random generator
  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  // Initialize variables with random values
  for (size_t ii = 0; ii < iteration; ++ii)
    for (size_t i = 0; i < num_head; ++i)
      for (size_t j = 0; j < batch_size; ++j)
        for (size_t k = 0; k < K; ++k)
          for (size_t l = 0; l < Dh; ++l) {
            if (ii == 0) {
              float rand_val = dist(gen);
              // values[i * values_head_offset + j * values_batch_offset + k *
              // Dh
              // +
              //        l] = static_cast<float>(l + 1);
              // values_trusted[i * values_head_offset + j * values_batch_offset
              // +
              //                k * Dh + l] = static_cast<float>(l + 1);
              values[ii][i * values_head_offset + j * values_batch_offset +
                         k * Dh + l] = rand_val;
              values_trusted[i * values_head_offset + j * values_batch_offset +
                             k * Dh + l] = rand_val;
            } else {
              values[ii][i * values_head_offset + j * values_batch_offset +
                         k * Dh + l] =
                  values_trusted[i * values_head_offset +
                                 j * values_batch_offset + k * Dh + l];
            }
          }

  for (size_t ii = 0; ii < iteration; ++ii)
    for (size_t i = 0; i < num_head; ++i)
      for (size_t j = 0; j < batch_size; ++j)
        for (size_t k = 0; k < K; ++k) {
          if (ii == 0) {
            float rand_val = dist(gen);
            logits[ii][i * logits_head_offset + j * logits_batch_offset + k] =
                rand_val;
            logits_trusted[i * logits_head_offset + j * logits_batch_offset +
                           k] = rand_val;
          } else {
            logits[ii][i * logits_head_offset + j * logits_batch_offset + k] =
                logits_trusted[i * logits_head_offset +
                               j * logits_batch_offset + k];
          }
        }

  for (size_t ii = 0; ii < iteration; ++ii) {
    for (size_t i = 0; i < num_head * batch_size * Dh; ++i) {
      result[ii][i] = 0.f;
      if (ii == 0) result_trusted[i] = 0.f;
    }
  }

  struct timespec start, end;
  double total_time_sec, total_time_sec_trusted;

  //////////////////////////////////////////////////////////////////////////////////
  // Run attention output with OpenBLAS
  flush_cache();
  clock_gettime(CLOCK_REALTIME, &start);
  attn_output_trusted(
      values_trusted, logits_trusted, result_trusted, num_head, batch_size, K,
      Dh, logits_head_offset, logits_batch_offset, values_head_offset,
      values_batch_offset, result_head_offset, result_batch_offset);
  clock_gettime(CLOCK_REALTIME, &end);
  total_time_sec_trusted =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  //////////////////////////////////////////////////////////////////////////////////
  // Run attention output with AVX2 and thread pool

  // Define the number of threads
  const int num_threads = 32;  // Total work = 256 / num_threads
  // Determine which portion of value is used
  std::atomic<int> iter_num(0);
  // Define synchronization variables
  std::atomic<bool> ready_flag(false);
  std::atomic<bool> stop_flag(false);
  // Define the finished flag for each thread
  std::atomic<bool> finished_flags[num_threads];
  for (int i = 0; i < num_threads; ++i)
    finished_flags[i].store(false, std::memory_order_release);

  // Create array of timespecs to store when each thread finishes
  struct timespec thread_finish_times[num_threads];
  bool thread_finished[num_threads];
  for (int i = 0; i < num_threads; ++i) thread_finished[i] = false;

  // Each thread works on its slice
  int const total_work = num_head * batch_size;
  // int work_per_thread = (total_work + num_threads - 1) / num_threads;
  int const work_per_thread = total_work / num_threads;
  int const min_priority = sched_get_priority_min(SCHED_FIFO);
  int const max_priority = sched_get_priority_max(SCHED_FIFO);
  std::cout << "SCHED_FIFO priority range: " << min_priority << " to "
            << max_priority << std::endl;
  int priority = max_priority;  // Base priority for all threads

  // Init thread pool
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    const int start_idx = t * work_per_thread;
    const int end_idx = std::min(start_idx + work_per_thread, total_work);
    threads.emplace_back(
        attn_output_threaded, values, logits, result, num_head, batch_size, K,
        Dh, values_head_offset, values_batch_offset, logits_head_offset,
        logits_batch_offset, result_head_offset, result_batch_offset, t,
        num_threads, start_idx, end_idx, &ready_flag, &finished_flags[t],
        &stop_flag, &iter_num, total_work);

    // // Get the native handle for the created thread
    pthread_t nativeHandle = threads.back().native_handle();

    // Define the scheduling parameters
    struct sched_param param;
    param.sched_priority = priority;  // Set the same priorities for each thread

    // Set the scheduling policy to SCHED_FIFO
    int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
    if (ret != 0) {
      std::cerr << "Failed to set scheduling policy for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
    // Set CPU affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (t > 23) {
      int id = 48 + (t - 23);
      CPU_SET(48 + (t - 23), &cpuset);  // Bind to specific CPU core
    } else {
      CPU_SET(t, &cpuset);  // Bind to specific CPU core
    }
    // CPU_SET(cpu_ids[t], &cpuset);  // Bind to specific CPU core
    ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
      std::cerr << "Failed to set CPU affinity for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
  }

  usleep(1000000);  // Sleep for 1s to allow threads to start

  // Repeat to measure latency of the kernel
  double acc_time_sec;
  double cur_time_sec;
  for (int ii = 0; ii < iteration; ++ii) {
    // Flush the current data in Cache
    flush_cache();

    // Measure execution time
    clock_gettime(CLOCK_REALTIME, &start);

    // Start the threads by setting the ready flag
    ready_flag.store(true, std::memory_order_release);

    // Busy wait until all threads are finished
    bool all_threads_finished = false;
    while (!all_threads_finished) {
      all_threads_finished = true;
      for (int i = 0; i < num_threads; ++i) {
        if (!thread_finished[i]) {
          if (finished_flags[i].load(std::memory_order_acquire)) {
            clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
            thread_finished[i] = true;
          } else {
            all_threads_finished = false;
          }
        }
      }
    }
    // attn_output_test(values, logits, result, num_head, batch_size, K, Dh,
    //                  values_head_offset, values_batch_offset,
    //                  logits_head_offset, logits_batch_offset,
    //                  result_head_offset, result_batch_offset);
    // Measure execution time
    clock_gettime(CLOCK_REALTIME, &end);

    // Reset flags
    ready_flag.store(false, std::memory_order_release);
    all_threads_finished = false;
    for (int i = 0; i < num_threads; ++i) {
      thread_finished[i] = false;
      finished_flags[i].store(false, std::memory_order_release);
    }
    iter_num.store(ii + 1, std::memory_order_release);

    // usleep(1000);  // Sleep for 1s to allow threads to start

    cur_time_sec =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    // if (ii > 0) {
    acc_time_sec += ii == 0 ? 0 : cur_time_sec;
    printf("Current elapsed time: %f\n", cur_time_sec);

    // Calculate MSE and MAE
    float mse = calculate_mse(result + ii * num_head * batch_size * Dh,
                              result_trusted, num_head * batch_size * Dh);
    float mae = calculate_mae(result + ii * num_head * batch_size * Dh,
                              result_trusted, num_head * batch_size * Dh);

    std::cout << "Mean Squared Error: " << mse << std::endl;
    std::cout << "Maximum Absolute Error: " << mae << std::endl;
    // }
    // Reset the result of Value GEMV
    // for (size_t i = 0; i < num_head * batch_size * Dh; ++i) {
    //   result[i] = 0.f;
    // }
  }
  total_time_sec = acc_time_sec / (iteration - 1);

  // Stop the thread pool
  stop_flag.store(true, std::memory_order_release);

  for (auto &thread : threads) thread.join();

  // Calculate FLOPs and GFLOPs
  double flops = 2.0 * Dh * K * num_head * batch_size;
  double gflops = flops / total_time_sec / 1e9;
  double gflops_trusted = flops / total_time_sec_trusted / 1e9;
  double total_bytes =
      (Dh * K * num_head * batch_size + K * num_head * batch_size) * 4;
  double throughput = total_bytes / total_time_sec / 1e9;

  // With functionality
  std::cout
      << "==========================My attn_output=========================="
      << std::endl;
  std::cout << "Elapsed time: " << total_time_sec * 1e6 << " microseconds"
            << std::endl;
  std::cout << "GFLOPs: " << gflops << std::endl;
  std::cout << "Total Bytes: " << total_bytes << std::endl;
  std::cout << "Throughtput(GB/s): " << throughput << std::endl;

  std::cout << "==========================Trusted "
               "attn_output=========================="
            << std::endl;
  std::cout << "Elapsed time: " << total_time_sec_trusted * 1e6
            << " microseconds" << std::endl;
  std::cout << "GFLOPs: " << gflops_trusted << std::endl;

  // Calculate MSE and MAE
  float mse = calculate_mse(result, result_trusted, num_head * batch_size * Dh);
  float mae = calculate_mae(result, result_trusted, num_head * batch_size * Dh);

  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Maximum Absolute Error: " << mae << std::endl;

  // Free the allocated memory
  for (int i = 0; i < ITER; ++i) {
    free(values[i]);
    free(logits[i]);
    free(result[i]);
  }
  free(logits_trusted);
  free(values_trusted);
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

  free(C);
  free(B);
  free(A);
}

// void attn_output_eval_threaded(
//     const size_t K, const size_t Dh, const size_t num_head,
//     const size_t batch_size, const int values_head_offset,
//     const int values_batch_offset, int const logits_head_offset,
//     int const logits_batch_offset, int const result_head_offset,
//     int const result_batch_offset, int const num_threads) {
//   float *values = static_cast<float *>(
//       aligned_alloc(64, num_head * batch_size * K * Dh * sizeof(float)));

//   float *logits = static_cast<float *>(
//       aligned_alloc(64, num_head * batch_size * K * sizeof(float)));

//   float *result = static_cast<float *>(
//       aligned_alloc(64, num_head * batch_size * Dh * sizeof(float)));

//   // Initialize values (example values for testing)
//   for (size_t i = 0; i < num_head; ++i)
//     for (size_t j = 0; j < batch_size; ++j) {
//       for (size_t k = 0; k < K; ++k)
//         for (size_t l = 0; l < Dh; ++l) {
//           values[i * values_head_offset + j * values_batch_offset + k * Dh +
//                  l] = static_cast<float>(l + 1);
//         }
//     }

//   for (size_t i = 0; i < num_head; ++i)
//     for (size_t j = 0; j < batch_size; ++j)
//       for (size_t k = 0; k < K; ++k)
//         logits[i * logits_head_offset + j * logits_batch_offset + k] = 0.3f;

//   for (size_t i = 0; i < num_head * batch_size * Dh; ++i) {
//     result[i] = 0.f;
//   }

//   struct timespec start, end;
//   double total_time_sec;
//   //////////////////////////////////////////////////////////////////////////////////
//   // Run attention output with AVX2
//   // int cpu_ids[num_threads];
//   // int const cpu_family = 6;
//   // int const cpus_per_family = std::thread::hardware_concurrency() /
//   // cpu_family; int group_num = num_threads / cpu_family + 1; int group_idx
//   =
//   // 0; for (int i = 0; i < num_threads; ++i) {
//   //   if (i > group_num * (group_idx + 1)) group_idx++;
//   //   cpu_ids[i] = cpus_per_family * (group_idx) + (i - group_num *
//   group_idx);
//   // }
//   // for (int i = 0; i < num_threads; ++i) {
//   //   printf("cpu #%d: %d", i, cpu_ids[i]);
//   // }

//   // Define synchronization variables
//   std::atomic<bool> ready_flag(false);
//   std::atomic<bool> stop_flag(false);
//   // Define the finished flag for each thread
//   std::atomic<bool> finished_flags[num_threads];
//   for (int i = 0; i < num_threads; ++i)
//     finished_flags[i].store(false, std::memory_order_release);

//   // Create array of timespecs to store when each thread finishes
//   struct timespec thread_finish_times[num_threads];
//   bool thread_finished[num_threads];
//   for (int i = 0; i < num_threads; ++i) thread_finished[i] = false;

//   // Each thread works on its slice
//   int const total_work = num_head * batch_size;
//   // int work_per_thread = (total_work + num_threads - 1) / num_threads;
//   int const work_per_thread = total_work / num_threads;
//   int const min_priority = sched_get_priority_min(SCHED_FIFO);
//   int const max_priority = sched_get_priority_max(SCHED_FIFO);
//   int const priority = max_priority;  // Base priority for all threads

//   std::vector<std::thread> threads;
//   for (int t = 0; t < num_threads; ++t) {
//     const int start_idx = t * work_per_thread;
//     const int end_idx = std::min(start_idx + work_per_thread, total_work);
//     threads.emplace_back(attn_output_threaded, values, logits, result,
//     num_head,
//                          batch_size, K, Dh, values_head_offset,
//                          values_batch_offset, logits_head_offset,
//                          logits_batch_offset, result_head_offset,
//                          result_batch_offset, t, num_threads, start_idx,
//                          end_idx, &ready_flag, &finished_flags[t],
//                          &stop_flag);

//     // // Get the native handle for the created thread
//     pthread_t nativeHandle = threads.back().native_handle();

//     // Define the scheduling parameters
//     struct sched_param param;
//     param.sched_priority = priority;  // Set the same priorities for each
//     thread

//     // Set the scheduling policy to SCHED_FIFO
//     int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
//     if (ret != 0) {
//       std::cerr << "Failed to set scheduling policy for thread " << t << ": "
//                 << strerror(ret) << std::endl;
//     }
//     // Set CPU affinity
//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     CPU_SET(t % std::thread::hardware_concurrency(),
//             &cpuset);  // Bind to specific CPU core
//     // CPU_SET(cpu_ids[t], &cpuset);  // Bind to specific CPU core
//     ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
//     if (ret != 0) {
//       std::cerr << "Failed to set CPU affinity for thread " << t << ": "
//                 << strerror(ret) << std::endl;
//     }
//   }

//   usleep(800000);  // Sleep for 1s to allow threads to start

//   // Flush the current data in Cache
//   flush_cache();

//   // Measure execution time
//   clock_gettime(CLOCK_REALTIME, &start);

//   // Start the threads by setting the ready flag
//   ready_flag.store(true, std::memory_order_release);

//   // Busy wait until all threads are finished
//   bool all_threads_finished = false;
//   struct timespec current_time;
//   while (!all_threads_finished) {
//     all_threads_finished = true;
//     for (int i = 0; i < num_threads; ++i) {
//       if (!thread_finished[i]) {
//         if (finished_flags[i].load(std::memory_order_acquire)) {
//           clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
//           thread_finished[i] = true;
//         } else {
//           all_threads_finished = false;
//         }
//       }
//     }
//   }

//   clock_gettime(CLOCK_REALTIME, &end);
//   total_time_sec =
//       (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
//   for (auto &thread : threads) thread.join();

//   // for (int i = 0; i < num_threads; ++i) {
//   //   double thread_time_sec =
//   //       (thread_finish_times[i].tv_sec - start.tv_sec) +
//   //       (thread_finish_times[i].tv_nsec - start.tv_nsec) / 1e9;
//   //   std::cout << "Thread #" << i << ": " << thread_time_sec * 1e6 << " us"
//   //             << std::endl;
//   // }

//   // Calculate FLOPs and GFLOPs
//   double flops = 2.0 * Dh * K * num_head * batch_size;
//   double gflops = flops / total_time_sec / 1e9;
//   double total_bytes =
//       (Dh * K * num_head * batch_size + K * num_head * batch_size) * 4;
//   double throughput = total_bytes / total_time_sec / 1e9;
//   // Print the results
//   printf("%-10d %-15.2f %-15.2f\n", num_threads, total_time_sec * 1e6,
//          throughput);

//   // Free the allocated memory
//   free(values);
//   free(logits);
//   free(result);
// }

int main(int argc, char *argv[]) {
  // Check if NUMA is available
  if (numa_available() == -1) {
    std::cerr << "NUMA is not available on this system." << std::endl;
    return 1;
  }

  size_t K = 48, Dh = 128, num_head = 8, batch_size = 64, iteration = 4;

  int const kv_iter_offset = iteration * batch_size * K * Dh;
  int const kv_head_offset = batch_size * K * Dh;
  int const kv_batch_offset = K * Dh;
  int const logits_score_iter_offset = iteration * batch_size * K;
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
  // Set the preferred NUMA node for memory allocation
  numa_set_preferred(0);  // Set preferred memory allocation to NUMA node 0
  value_gemv_eval(K, Dh, num_head, batch_size, iteration, kv_iter_offset,
                  kv_head_offset, kv_batch_offset, logits_score_iter_offset,
                  logits_score_head_offset, logits_score_batch_offset,
                  q_out_head_offset, q_out_batch_offset);
  // Print table header
  // printf("%-10s %-15s %-15s\n", "Threads", "Latency (us)", "Throughput
  // (GB/s)"); printf("--------------------------------------------------\n");
  // for (int i = 1; i < 32; i *= 2)
  //   attn_output_eval_threaded(K, Dh, num_head, batch_size, kv_head_offset,
  //                             kv_batch_offset, logits_score_head_offset,
  //                             logits_score_batch_offset, q_out_head_offset,
  //                             q_out_batch_offset, i);

  return 0;
}

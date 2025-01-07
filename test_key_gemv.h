#ifndef KEY_GEMV_EVAL_H
#define KEY_GEMV_EVAL_H

#include <stdlib.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "key_gemv.h"

#define ITER 10
static struct timespec start, end;
static double acc_time_sec;
static double cur_time_sec;

template <typename T>
void key_gemv_eval(const int K, const int Dh, const int q_head_num,
                   int kv_head_num, const int batch_size,
                   const int keys_head_offset, const int keys_batch_offset,
                   int const queries_head_offset,
                   int const queries_batch_offset, int const logits_head_offset,
                   int const logits_batch_offset,
                   int const num_threads) {  // Total work = 256 /
  num_threads[] if constexpr (std::is_same<T, float>::value) std::cout
      << "Float\n";
  else std::cout << "Half\n";

  // Allocate memory
  T *keys[ITER];
  T *queries[ITER];
  T *logits[ITER];

  for (int i = 0; i < ITER; ++i) {
    keys[i] = static_cast<T *>(
        aligned_alloc(64, head_num * batch_size * K * Dh * sizeof(T)));
    queries[i] = static_cast<T *>(
        aligned_alloc(64, head_num * batch_size * Dh * sizeof(T)));
    logits[i] = static_cast<T *>(
        aligned_alloc(64, head_num * batch_size * K * sizeof(T)));
  }
  float *keys_trusted = static_cast<float *>(
      aligned_alloc(64, head_num * batch_size * K * Dh * sizeof(float)));
  float *queries_trusted = static_cast<float *>(
      aligned_alloc(64, head_num * batch_size * Dh * sizeof(float)));
  float *logits_trusted = static_cast<float *>(
      aligned_alloc(64, head_num * batch_size * K * sizeof(float)));

  // random generator
  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  // Initialize variables with random values
  for (int ii = 0; ii < ITER; ++ii)
    for (int i = 0; i < head_num; ++i)
      for (int j = 0; j < batch_size; ++j)
        for (int k = 0; k < K; ++k)
          for (int l = 0; l < Dh; ++l) {
            if (ii == 0) {
              float rand_val = dist(gen);
              if constexpr (std::is_same<T, float>::value) {
                keys[ii][i * keys_head_offset + j * keys_batch_offset + k * Dh +
                         l] = rand_val;
              } else {
                keys[ii][i * keys_head_offset + j * keys_batch_offset + k * Dh +
                         l] = __float2half(rand_val);
              }
              keys_trusted[i * keys_head_offset + j * keys_batch_offset +
                           k * Dh + l] = rand_val;
            } else {
              keys[ii][i * keys_head_offset + j * keys_batch_offset + k * Dh +
                       l] = keys_trusted[i * keys_head_offset +
                                         j * keys_batch_offset + k * Dh + l];
            }
          }

  for (int ii = 0; ii < ITER; ++ii)
    for (int i = 0; i < head_num; ++i)
      for (int j = 0; j < batch_size; ++j)
        for (int k = 0; k < Dh; ++k) {
          if (ii == 0) {
            float rand_val = dist(gen);
            if constexpr (std::is_same<T, float>::value) {
              queries[ii][i * queries_head_offset + j * queries_batch_offset +
                          k] = rand_val;
            } else {
              queries[ii][i * queries_head_offset + j * queries_batch_offset +
                          k] = __float2half(rand_val);
            }
            queries_trusted[i * queries_head_offset + j * queries_batch_offset +
                            k] = rand_val;
          } else {
            queries[ii][i * queries_head_offset + j * queries_batch_offset +
                        k] = queries_trusted[i * queries_head_offset +
                                             j * queries_batch_offset + k];
          }
        }

  for (int ii = 0; ii < ITER; ++ii) {
    for (int i = 0; i < head_num * batch_size * K; ++i) {
      logits[ii][i] = 0.f;
      if (ii == 0) logits_trusted[i] = 0.f;
    }
  }

  double total_time_sec, total_time_sec_trusted;

  //////////////////////////////////////////////////////////////////////////////////
  // Run Key GEMV with OpenBLAS
  flush_cache();
  clock_gettime(CLOCK_REALTIME, &start);
  key_gemv_trusted(keys_trusted, queries_trusted, logits_trusted, head_num,
                   batch_size, K, Dh, keys_head_offset, keys_batch_offset,
                   queries_head_offset, queries_batch_offset,
                   logits_head_offset, logits_batch_offset);
  clock_gettime(CLOCK_REALTIME, &end);
  total_time_sec_trusted =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  //////////////////////////////////////////////////////////////////////////////////
  // Run Key GEMV with AVX2 and thread pool
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
  int const total_work = head_num * batch_size;
  int const work_per_thread = total_work / num_threads;
  int remains = total_work - (work_per_thread * num_threads);

  int const min_priority = sched_get_priority_min(SCHED_FIFO);
  int const max_priority = sched_get_priority_max(SCHED_FIFO);

  int priority = max_priority;  // Base priority for all threads

  // DEBUGGING
  printf("Total Work: %d\n", total_work);
  printf("Work/Thread: %d\n", work_per_thread);

  // Check for the variance between threads
  double end_times[num_threads];

  // Init thread pool
  std::vector<std::thread> threads;
  int start_idx = 0;
  int end_idx = 0;

  for (int t = 0; t < num_threads; ++t) {
    start_idx = end_idx;
    end_idx = remains > 0 ? start_idx + work_per_thread + 1
                          : start_idx + work_per_thread;
    remains -= 1;

    if constexpr (std::is_same<T, float>::value) {
      threads.emplace_back(
          key_gemv_threaded, keys, queries, logits, head_num, batch_size, K, Dh,
          keys_head_offset, keys_batch_offset, queries_head_offset,
          queries_batch_offset, logits_head_offset, logits_batch_offset, t,
          num_threads, start_idx, end_idx, &ready_flag, &finished_flags[t],
          &stop_flag, &iter_num, &end_times[t]);
    } else {
      threads.emplace_back(
          key_gemv_threaded_half, keys, queries, logits, head_num, batch_size,
          K, Dh, keys_head_offset, keys_batch_offset, queries_head_offset,
          queries_batch_offset, logits_head_offset, logits_batch_offset, t,
          num_threads, start_idx, end_idx, &ready_flag, &finished_flags[t],
          &stop_flag, &iter_num, &end_times[t]);
    }

    // // Get the native handle for the created thread
    pthread_t nativeHandle = threads.back().native_handle();

    // Define the scheduling parameters
    struct sched_param param;
    param.sched_priority = priority;  // Set the same priorities for each
    thread

        // Set the scheduling policy to SCHED_FIFO
        int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
    if (ret != 0) {
      std::cerr << "Failed to set scheduling policy for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
    // Set CPU affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // Method1
    if (t > 23) {
      // int id = 48 + (t - 23);
      CPU_SET(48 + (t - 23), &cpuset);  // Bind to specific CPU core
    } else {
      CPU_SET(t, &cpuset);  // Bind to specific CPU core
    }
    // Method2
    // CPU_SET(t, &cpuset);  // Bind to specific CPU core

    ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
      std::cerr << "Failed to set CPU affinity for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
  }

  usleep(100000);  // Sleep for 1s to allow threads to start

  // Repeat to measure latency of the kernel
  for (int ii = 0; ii < ITER; ++ii) {
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
    // Measure execution time
    clock_gettime(CLOCK_REALTIME, &end);

    // Reset flags
    ready_flag.store(false, std::memory_order_release);
    all_threads_finished = false;
    for (int i = 0; i < num_threads; ++i) {
      thread_finished[i] = false;
      finished_flags[i].store(false, std::memory_order_release);
    }
    // Set the new data to ignore the cache effect
    iter_num.store(ii + 1, std::memory_order_release);

    cur_time_sec =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // acc_time_sec += ii == 0 ? 0 : cur_time_sec;
    acc_time_sec += cur_time_sec;

    usleep(10);
    // DEBUG: Check the duration variance between threads
    // for (int t = 0; t < num_threads; ++t)
    //   printf("CPU %d: %f\n", t, end_times[t]);
    // std::sort(end_times, end_times + num_threads);
    // printf("Variance: %f\n", end_times[num_threads - 1] - end_times[0]);

    // Calculate MSE and MAE
    float mse, mae;
    if constexpr (std::is_same<T, float>::value) {
      mse =
          calculate_mse(logits[ii], logits_trusted, head_num * batch_size * K);
      mae =
          calculate_mae(logits[ii], logits_trusted, head_num * batch_size * K);
    } else {
      mse = calculate_mse_half(logits[ii], logits_trusted,
                               head_num * batch_size * K);
      mae = calculate_mae_half(logits[ii], logits_trusted,
                               head_num * batch_size * K);
    }
    // DEBUGGING
    // printf("Current elapsed time: %f\n", cur_time_sec);
    // std::cout << "Mean Squared Error: " << mse << std::endl;
    // std::cout << "Maximum Absolute Error: " << mae << std::endl;
    // printf("Start elapsed time: %f\n", start.tv_sec + start.tv_nsec /
    1e9);
    // printf("End elapsed time: %f\n", end.tv_sec + end.tv_nsec / 1e9);
    // printf("Acc elapsed time: %f\n", acc_time_sec);
  }
  total_time_sec = acc_time_sec / (ITER - 1);

  // Stop the thread pool
  stop_flag.store(true, std::memory_order_release);

  for (auto &thread : threads) thread.join();

  // Calculate FLOPs and GFLOPs
  double flops = 2.0 * Dh * K * head_num * batch_size;
  double gflops = flops / total_time_sec / 1e9;
  double gflops_trusted = flops / total_time_sec_trusted / 1e9;
  int const num_keys = Dh * K * head_num * batch_size;
  int const num_queries = Dh * head_num * batch_size;
  double total_bytes = (num_keys + num_queries) * sizeof(T);
  double throughput = (total_bytes / total_time_sec) / 1e9;

  printf("Size of each element: %d B", sizeof(T));
  printf("Number of elements in Keys: %d\n", num_keys);
  printf("Number of elements in Logits: %d\n", num_queires);
  std::cout << "Elapsed time: " << total_time_sec * 1e6 << " microseconds"
            << std::endl;
  std::cout << "GFLOPs: " << gflops << std::endl;
  std::cout << "Total Bytes: " << total_bytes << std::endl;
  std::cout << "Throughtput(GB/s): " << throughput << std::endl;
  printf("\n\n");

  // DEBUGGING
  // const int check_head_idx = 0;
  // const int check_batch_idx = 0;
  // for (int k = 0; k < K; ++k) {
  //   printf("logits_trusted[%d][%d][%d]: %f\n", check_head_idx,
  //   check_batch_idx,
  //          k,
  //          logits_trusted[check_head_idx * logits_head_offset +
  //                         check_batch_idx * logits_batch_offset + k]);
  //   printf("logits[%d][%d][%d]: %f\n", check_head_idx, check_batch_idx, k,
  //          logits[0][check_head_idx * logits_head_offset +
  //                    check_batch_idx * logits_batch_offset + k]);
  // }

  // Free the allocated memory
  for (int i = 0; i < ITER; ++i) {
    free(keys[i]);
    free(queries[i]);
    free(logits[i]);
  }
  free(keys_trusted);
  free(queries_trusted);
  free(logits_trusted);
}

template <>
void key_gemv_eval<std::uint16_t>(
    const int K, const int Dh, const int q_head_num, const int kv_head_num,
    const int batch_size, const int keys_head_offset,
    const int keys_batch_offset, int const queries_head_offset,
    int const queries_batch_offset, int const logits_head_offset,
    int const logits_batch_offset,
    int const num_threads) {  // Total work = 256 / num_threads[]
  std::cout << "Half\n";

  // Allocate memory
  half *keys[ITER];
  half *queries[ITER];
  half *logits[ITER];

  for (int i = 0; i < ITER; ++i) {
    keys[i] = static_cast<half *>(
        aligned_alloc(64, kv_head_num * batch_size * K * Dh * sizeof(half)));
    queries[i] = static_cast<half *>(
        aligned_alloc(64, q_head_num * batch_size * Dh * sizeof(half)));
    logits[i] = static_cast<half *>(
        aligned_alloc(64, q_head_num * batch_size * K * sizeof(half)));
  }
  float *keys_trusted = static_cast<float *>(
      aligned_alloc(64, kv_head_num * batch_size * K * Dh * sizeof(float)));
  float *queries_trusted = static_cast<float *>(
      aligned_alloc(64, q_head_num * batch_size * Dh * sizeof(float)));
  float *logits_trusted = static_cast<float *>(
      aligned_alloc(64, q_head_num * batch_size * K * sizeof(float)));

  // random generator
  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  // Initialize variables with random values
  for (int ii = 0; ii < ITER; ++ii)
    for (int i = 0; i < kv_head_num; ++i)
      for (int j = 0; j < batch_size; ++j)
        for (int k = 0; k < K; ++k)
          for (int l = 0; l < Dh; ++l) {
            if (ii == 0) {
              float rand_val = dist(gen);
              keys[ii][i * keys_head_offset + j * keys_batch_offset + k * Dh +
                       l] = __float2half(rand_val);

              keys_trusted[i * keys_head_offset + j * keys_batch_offset +
                           k * Dh + l] = rand_val;
            } else {
              keys[ii][i * keys_head_offset + j * keys_batch_offset + k * Dh +
                       l] = keys_trusted[i * keys_head_offset +
                                         j * keys_batch_offset + k * Dh + l];
            }
          }

  for (int ii = 0; ii < ITER; ++ii)
    for (int i = 0; i < q_head_num; ++i)
      for (int j = 0; j < batch_size; ++j)
        for (int k = 0; k < Dh; ++k) {
          if (ii == 0) {
            float rand_val = dist(gen);
            queries[ii][i * queries_head_offset + j * queries_batch_offset +
                        k] = __float2half(rand_val);
            queries_trusted[i * queries_head_offset + j * queries_batch_offset +
                            k] = rand_val;
          } else {
            queries[ii][i * queries_head_offset + j * queries_batch_offset +
                        k] = queries_trusted[i * queries_head_offset +
                                             j * queries_batch_offset + k];
          }
        }

  for (int ii = 0; ii < ITER; ++ii) {
    for (int i = 0; i < q_head_num * batch_size * K; ++i) {
      logits[ii][i] = 0.f;
      if (ii == 0) logits_trusted[i] = 0.f;
    }
  }

  double total_time_sec, total_time_sec_trusted;

  //////////////////////////////////////////////////////////////////////////////////
  // Run Key GEMV with OpenBLAS
  flush_cache();
  clock_gettime(CLOCK_REALTIME, &start);
  key_gemv_trusted(keys_trusted, queries_trusted, logits_trusted, q_head_num,
                   batch_size, K, Dh, keys_head_offset, keys_batch_offset,
                   queries_head_offset, queries_batch_offset,
                   logits_head_offset, logits_batch_offset);
  clock_gettime(CLOCK_REALTIME, &end);
  total_time_sec_trusted =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  //////////////////////////////////////////////////////////////////////////////////
  // Run Key GEMV with AVX2 and thread pool
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
  int const total_work = q_head_num * batch_size;
  int const work_per_thread = total_work / num_threads;
  int remains = total_work - (work_per_thread * num_threads);

  int const min_priority = sched_get_priority_min(SCHED_FIFO);
  int const max_priority = sched_get_priority_max(SCHED_FIFO);

  int priority = max_priority;  // Base priority for all threads

  // DEBUGGING
  printf("Total Work: %d\n", total_work);
  printf("Work/Thread: %d\n", work_per_thread);

  // Check for the variance between threads
  double end_times[num_threads];

  // Init thread pool
  std::vector<std::thread> threads;
  int start_idx = 0;
  int end_idx = 0;

  for (int t = 0; t < num_threads; ++t) {
    start_idx = end_idx;
    end_idx = remains > 0 ? start_idx + work_per_thread + 1
                          : start_idx + work_per_thread;
    remains -= 1;

    threads.emplace_back(
        key_gemv_threaded_half, keys, queries, logits, q_head_num, batch_size,
        K, Dh, keys_head_offset, keys_batch_offset, queries_head_offset,
        queries_batch_offset, logits_head_offset, logits_batch_offset, t,
        num_threads, start_idx, end_idx, &ready_flag, &finished_flags[t],
        &stop_flag, &iter_num, &end_times[t]);

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
    // Method1
    if (t > 23) {
      // int id = 48 + (t - 23);
      CPU_SET(48 + (t - 23), &cpuset);  // Bind to specific CPU core
    } else {
      CPU_SET(t, &cpuset);  // Bind to specific CPU core
    }
    // Method2
    // CPU_SET(t, &cpuset);  // Bind to specific CPU core

    ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
      std::cerr << "Failed to set CPU affinity for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
  }

  usleep(100000);  // Sleep for 1s to allow threads to start

  // Repeat to measure latency of the kernel
  for (int ii = 0; ii < ITER; ++ii) {
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
    // Measure execution time
    clock_gettime(CLOCK_REALTIME, &end);

    // Reset flags
    ready_flag.store(false, std::memory_order_release);
    all_threads_finished = false;
    for (int i = 0; i < num_threads; ++i) {
      thread_finished[i] = false;
      finished_flags[i].store(false, std::memory_order_release);
    }
    // Set the new data to ignore the cache effect
    iter_num.store(ii + 1, std::memory_order_release);

    cur_time_sec =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // acc_time_sec += ii == 0 ? 0 : cur_time_sec;
    acc_time_sec += cur_time_sec;

    usleep(10);
    // DEBUG: Check the duration variance between threads
    // for (int t = 0; t < num_threads; ++t)
    //   printf("CPU %d: %f\n", t, end_times[t]);
    // std::sort(end_times, end_times + num_threads);
    // printf("Variance: %f\n", end_times[num_threads - 1] - end_times[0]);

    // Calculate MSE and MAE
    float mse, mae;

    mse = calculate_mse_half(logits[ii], logits_trusted,
                             q_head_num * batch_size * K);
    mae = calculate_mae_half(logits[ii], logits_trusted,
                             q_head_num * batch_size * K);

    // DEBUG
    // std::cout << "Mean Squared Error: " << mse << std::endl;
    // std::cout << "Maximum Absolute Error: " << mae << std::endl;
    // printf("Start elapsed time: %f\n", start.tv_sec + start.tv_nsec / 1e9);
    // printf("End elapsed time: %f\n", end.tv_sec + end.tv_nsec / 1e9);
    // printf("Acc elapsed time: %f\n", acc_time_sec);
  }
  total_time_sec = acc_time_sec / (ITER - 1);

  // Stop the thread pool
  stop_flag.store(true, std::memory_order_release);

  for (auto &thread : threads) thread.join();

  // Calculate FLOPs and GFLOPs
  double flops = 2.0 * Dh * K * q_head_num * batch_size;
  double gflops = flops / total_time_sec / 1e9;
  double gflops_trusted = flops / total_time_sec_trusted / 1e9;
  double total_bytes =
      (Dh * K * q_head_num * batch_size + Dh * q_head_num * batch_size) *
      sizeof(half);
  double throughput = (total_bytes / total_time_sec) / 1e9;

  std::cout << "Elapsed time: " << total_time_sec * 1e6 << " microseconds"
            << std::endl;
  std::cout << "GFLOPs: " << gflops << std::endl;
  std::cout << "Total Bytes: " << total_bytes / 1e9 << std::endl;
  std::cout << "Throughtput(GB/s): " << throughput << std::endl;
  printf("\n\n");

  // DEBUGGING
  // const int check_head_idx = 0;
  // const int check_batch_idx = 0;
  // printf("logits_trusted[%d][%d]:\n", check_head_idx, check_batch_idx);
  // for (int k = 0; k < K; ++k) {
  //   printf("%f ", logits_trusted[check_head_idx * logits_head_offset +
  //                                check_batch_idx * logits_batch_offset + k]);
  // }
  // printf("\nlogits[%d][%d]:\n", check_head_idx, check_batch_idx);
  // for (int k = 0; k < K; ++k) {
  //   printf("%f ",
  //          __half2float(logits[0][check_head_idx * logits_head_offset +
  //                                 check_batch_idx * logits_batch_offset +
  //                                 k]));
  // }

  // Free the allocated memory
  for (int i = 0; i < ITER; ++i) {
    free(keys[i]);
    free(queries[i]);
    free(logits[i]);
  }
  free(keys_trusted);
  free(queries_trusted);
  free(logits_trusted);
}

#endif  // KEY_GEMV_EVAL_H
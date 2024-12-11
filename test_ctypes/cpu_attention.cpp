#include <immintrin.h>
#include <sched.h>
#include <string.h>

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
#include <unistd.h>
#include <algorithm>
#define THREAD_NUM 48

extern "C"
{
  // Define the finished flag for each thread
  std::atomic<bool> finished_flags[THREAD_NUM];
  std::atomic<bool> done_flag(false);
  std::atomic<bool> ready_flag(false);

  // Store the time or duration for each thread
  typedef std::pair<int, long> pair_tr;
  pair_tr thread_results[THREAD_NUM];
  // double thread_results[THREAD_NUM];
  static struct timespec _start, _end, _end_1;

  // Value GEMV with multiple threads
  void value_gemv_threaded(float *values, float *logits, float *result,
                           int const head_num, int const batch_size, int const K,
                           int const Dh, int const values_head_offset,
                           int const values_batch_offset,
                           int const logits_head_offset,
                           int const logits_batch_offset,
                           int const result_head_offset,
                           int const result_batch_offset, int const thread_id,
                           int const thread_num, int const start_idx,
                           int const end_idx, std::atomic<bool> *ready_flag,
                           std::atomic<bool> *finished_flag, pair_tr *duration)
  {
    struct timespec start, end;

    while (!(ready_flag->load(std::memory_order_acquire)))
    {
      // while (!(*ready_flag)) {
    }
    clock_gettime(CLOCK_REALTIME, &start);
    // clock_gettime(CLOCK_MONOTONIC, &start);

    // Multiply and Add
    for (int idx = start_idx; idx < end_idx; ++idx)
    {
      int i = idx / batch_size;
      int j = idx % batch_size;

      __m256 c00 = _mm256_setzero_ps();
      __m256 c01 = _mm256_setzero_ps();
      __m256 c02 = _mm256_setzero_ps();
      __m256 c03 = _mm256_setzero_ps();
      __m256 c04 = _mm256_setzero_ps();
      __m256 c05 = _mm256_setzero_ps();
      __m256 c06 = _mm256_setzero_ps();
      __m256 c07 = _mm256_setzero_ps();
      __m256 c08 = _mm256_setzero_ps();
      __m256 c09 = _mm256_setzero_ps();
      __m256 c10 = _mm256_setzero_ps();
      __m256 c11 = _mm256_setzero_ps();
      __m256 c12 = _mm256_setzero_ps();
      __m256 c13 = _mm256_setzero_ps();
      __m256 c14 = _mm256_setzero_ps();
      __m256 c15 = _mm256_setzero_ps();

      for (int k = 0; k < K; ++k)
      {
        float logit =
            logits[i * logits_head_offset + j * logits_batch_offset + k];
        __m256 logit_vec = _mm256_set1_ps(logit);

        if (k + 1 < K)
        {
          _mm_prefetch((const char *)(values + i * values_head_offset +
                                      j * values_batch_offset + (k + 1) * Dh),
                       _MM_HINT_T0);
        }
        __m256 v00 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh);
        __m256 v01 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 8);
        __m256 v02 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 16);
        __m256 v03 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 24);
        __m256 v04 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 32);
        __m256 v05 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 40);
        __m256 v06 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 48);
        __m256 v07 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 56);
        __m256 v08 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 64);
        __m256 v09 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 72);
        __m256 v10 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 80);
        __m256 v11 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 88);
        __m256 v12 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 96);
        __m256 v13 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 104);
        __m256 v14 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 112);
        __m256 v15 = _mm256_load_ps(values + i * values_head_offset +
                                    j * values_batch_offset + k * Dh + 120);
        c00 = _mm256_fmadd_ps(logit_vec, v00, c00);
        c01 = _mm256_fmadd_ps(logit_vec, v01, c01);
        c02 = _mm256_fmadd_ps(logit_vec, v02, c02);
        c03 = _mm256_fmadd_ps(logit_vec, v03, c03);
        c04 = _mm256_fmadd_ps(logit_vec, v04, c04);
        c05 = _mm256_fmadd_ps(logit_vec, v05, c05);
        c06 = _mm256_fmadd_ps(logit_vec, v06, c06);
        c07 = _mm256_fmadd_ps(logit_vec, v07, c07);
        c08 = _mm256_fmadd_ps(logit_vec, v08, c08);
        c09 = _mm256_fmadd_ps(logit_vec, v09, c09);
        c10 = _mm256_fmadd_ps(logit_vec, v10, c10);
        c11 = _mm256_fmadd_ps(logit_vec, v11, c11);
        c12 = _mm256_fmadd_ps(logit_vec, v12, c12);
        c13 = _mm256_fmadd_ps(logit_vec, v13, c13);
        c14 = _mm256_fmadd_ps(logit_vec, v14, c14);
        c15 = _mm256_fmadd_ps(logit_vec, v15, c15);
      }
      // Store the accumulated result back into the result array
      _mm256_store_ps(result + i * result_head_offset + j * result_batch_offset,
                      c00);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 8, c01);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 16, c02);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 24, c03);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 32, c04);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 40, c05);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 48, c06);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 56, c07);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 64, c08);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 72, c09);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 80, c10);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 88, c11);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 96, c12);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 104, c13);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 112, c14);
      _mm256_store_ps(
          result + i * result_head_offset + j * result_batch_offset + 120, c15);
    }
    // Mark this thread as finished
    finished_flag->store(true, std::memory_order_release);
    clock_gettime(CLOCK_REALTIME, &end);
    // clock_gettime(CLOCK_MONOTONIC, &end);
    duration->first = thread_id;
    // duration->second = start.tv_sec * 1e9 + start.tv_nsec;
    // duration->second = (end.tv_nsec) / 1e3;
    duration->second = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
  }

  inline float hsum_128(__m128 x)
  {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
  }

  inline float hsum(__m256 x)
  {
    return hsum_128(
        _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
  }

  // [x] Value GEMV with multiple threads
  void key_gemv_threaded(
      float *keys, float *queries, float *logits, int const num_head,
      int const batch_size, int const K, int const Dh, int const keys_head_offset,
      int const keys_batch_offset, int const queries_haed_offset,
      int const queries_batch_offset, int const logits_head_offset,
      int const logits_batch_offset, int const thread_id, int const num_threads,
      int const start_idx, int const end_idx, std::atomic<bool> *ready_flag,
      std::atomic<bool> *finished_flag, pair_tr *duration)
  {
    struct timespec start, end;

    while (!(ready_flag->load(std::memory_order_acquire)))
    {
      // while (!(*ready_flag)) {
    }
    clock_gettime(CLOCK_REALTIME, &start);
    // clock_gettime(CLOCK_MONOTONIC, &start);

    // Multiply and Add
    for (int idx = start_idx; idx < end_idx; ++idx)
    {
      int i = idx / batch_size;
      int j = idx % batch_size;

      for (int k = 0; k < K; k += 16)
      {
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        __m256 c04 = _mm256_setzero_ps();
        __m256 c05 = _mm256_setzero_ps();
        __m256 c06 = _mm256_setzero_ps();
        __m256 c07 = _mm256_setzero_ps();
        __m256 c08 = _mm256_setzero_ps();
        __m256 c09 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c11 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps();
        __m256 c13 = _mm256_setzero_ps();
        __m256 c14 = _mm256_setzero_ps();
        __m256 c15 = _mm256_setzero_ps();

        for (int l = 0; l < Dh; l += 8)
        {
          // Prefetching the next query and keys for the next iteration
          // if (l + 8 < Dh)
          //   _mm_prefetch((const char*)(queries + i * q_haed_offset +
          //                              j * q_batch_offset + l + 8),
          //                _MM_HINT_T0);
          // if (k + 8 < K)
          //   _mm_prefetch(
          //       (const char*)(keys + i * keys_head_offset +
          //                     j * keys_batch_offset + (k + 8) * Dh + l),
          //       _MM_HINT_T0);

          __m256 q0 = _mm256_loadu_ps(queries + i * queries_haed_offset +
                                      j * queries_batch_offset + l);

          __m256 k0 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + k * Dh + l);
          __m256 k1 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 1) * Dh + l);
          __m256 k2 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 2) * Dh + l);
          __m256 k3 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 3) * Dh + l);
          __m256 k4 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 4) * Dh + l);
          __m256 k5 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 5) * Dh + l);
          __m256 k6 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 6) * Dh + l);
          __m256 k7 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 7) * Dh + l);
          __m256 k8 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 8) * Dh + l);
          __m256 k9 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                      j * keys_batch_offset + (k + 9) * Dh + l);
          __m256 k10 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                       j * keys_batch_offset + (k + 10) * Dh + l);
          __m256 k11 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                       j * keys_batch_offset + (k + 11) * Dh + l);
          __m256 k12 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                       j * keys_batch_offset + (k + 12) * Dh + l);
          __m256 k13 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                       j * keys_batch_offset + (k + 13) * Dh + l);
          __m256 k14 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                       j * keys_batch_offset + (k + 14) * Dh + l);
          __m256 k15 = _mm256_loadu_ps(keys + i * keys_head_offset +
                                       j * keys_batch_offset + (k + 15) * Dh + l);
          c00 = _mm256_fmadd_ps(q0, k0, c00);
          c01 = _mm256_fmadd_ps(q0, k1, c01);
          c02 = _mm256_fmadd_ps(q0, k2, c02);
          c03 = _mm256_fmadd_ps(q0, k3, c03);
          c04 = _mm256_fmadd_ps(q0, k4, c04);
          c05 = _mm256_fmadd_ps(q0, k5, c05);
          c06 = _mm256_fmadd_ps(q0, k6, c06);
          c07 = _mm256_fmadd_ps(q0, k7, c07);
          c08 = _mm256_fmadd_ps(q0, k8, c08);
          c09 = _mm256_fmadd_ps(q0, k9, c09);
          c10 = _mm256_fmadd_ps(q0, k10, c10);
          c11 = _mm256_fmadd_ps(q0, k11, c11);
          c12 = _mm256_fmadd_ps(q0, k12, c12);
          c13 = _mm256_fmadd_ps(q0, k13, c13);
          c14 = _mm256_fmadd_ps(q0, k14, c14);
          c15 = _mm256_fmadd_ps(q0, k15, c15);
        }
        logits[i * logits_head_offset + j * logits_batch_offset + k] = hsum(c00);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 1] =
            hsum(c01);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 2] =
            hsum(c02);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 3] =
            hsum(c03);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 4] =
            hsum(c04);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 5] =
            hsum(c05);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 6] =
            hsum(c06);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 7] =
            hsum(c07);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 8] =
            hsum(c08);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 9] =
            hsum(c09);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 10] =
            hsum(c10);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 11] =
            hsum(c11);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 12] =
            hsum(c12);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 13] =
            hsum(c13);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 14] =
            hsum(c14);
        logits[i * logits_head_offset + j * logits_batch_offset + k + 15] =
            hsum(c15);
      }
    }
    // Mark this thread as finished
    finished_flag->store(true, std::memory_order_release);
    clock_gettime(CLOCK_REALTIME, &end);
    // clock_gettime(CLOCK_MONOTONIC, &end);
    duration->first = thread_id;
    // duration->second = start.tv_sec * 1e9 + start.tv_nsec;
    // duration->second = (end.tv_nsec) / 1e3;
    duration->second = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1e6;
  }

  // Function to prepare the threads for Value GEMV
  void prepare_value_gemv(float *values, float *logits, float *result,
                          int const head_num, int const batch_size, int const K,
                          int const Dh, int const values_head_offset,
                          int const values_batch_offset,
                          int const logits_head_offset,
                          int const logits_batch_offset,
                          int const result_head_offset,
                          int const result_batch_offset, int const thread_num)
  {
    // printf("Ready Flag: %p\n", &ready_flag);
    // printf("Done Flag: %p\n", &done_flag);
    // Each thread works on its slice
    int const total_work = head_num * batch_size;
    int const work_per_thread = total_work / thread_num;
    int const work_remained = total_work % thread_num;

    //   int const min_priority = sched_get_priority_min(SCHED_FIFO);
    int const max_priority = sched_get_priority_max(SCHED_FIFO);

    int priority = max_priority; // Base priority for all threads

    // Init thread pool
    std::vector<std::thread> threads;
    int start_idx = 0, end_idx = 0;
    int acc = 0;
    for (int t = 0; t < thread_num; ++t)
    {
      start_idx = end_idx;
      end_idx = t < work_remained ? start_idx + work_per_thread + 1
                                  : start_idx + work_per_thread;
      int cpu_id = t + acc;
      acc += 1;
      // int cpu_id = t;
      threads.emplace_back(
          value_gemv_threaded, values, logits, result, head_num, batch_size, K,
          Dh, values_head_offset, values_batch_offset, logits_head_offset,
          logits_batch_offset, result_head_offset, result_batch_offset, cpu_id,
          thread_num, start_idx, end_idx, &ready_flag, &finished_flags[t], &thread_results[t]);

      // Get the native handle for the created thread
      pthread_t nativeHandle = threads.back().native_handle();

      // Define the scheduling parameters
      struct sched_param param;
      param.sched_priority = priority; // Set the same priorities for each thread

      // Set the scheduling policy to SCHED_FIFO
      int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
      if (ret != 0)
      {
        std::cerr << "Failed to set scheduling policy for thread " << t << ": "
                  << strerror(ret) << std::endl;
      }
      // Set CPU affinity
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      // Method1
      // if (t > 23)
      // {
      //   // int id = 48 + (t - 23);
      //   CPU_SET(48 + (t - 24), &cpuset); // Bind to specific CPU core
      // }
      // else
      // {
      //   CPU_SET(t, &cpuset); // Bind to specific CPU core
      // }
      // c13
      // if (t > 8)
      // {
      //   CPU_SET(32 + t, &cpuset); // Bind to specific CPU core
      // }
      // else
      // {
      //   CPU_SET(t, &cpuset); // Bind to specific CPU core
      // }
      // if (cpu_id > 28)
      //   CPU_SET(1, &cpuset); // Bind to specific CPU core
      // else
      CPU_SET(cpu_id, &cpuset); // Bind to specific CPU core
      ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
      if (ret != 0)
      {
        std::cerr << "Failed to set CPU affinity for thread " << t << ": "
                  << strerror(ret) << std::endl;
      }
    }

    bool all_threads_finished = false;
    bool thread_finished[thread_num];
    for (int i = 0; i < thread_num; ++i)
      thread_finished[i] = false;

    while (!all_threads_finished)
    {
      all_threads_finished = true;
      for (int i = 0; i < thread_num; ++i)
      {
        if (!thread_finished[i])
        {
          if (finished_flags[i].load(std::memory_order_acquire))
          {
            //   clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
            thread_finished[i] = true;
          }
          else
          {
            all_threads_finished = false;
          }
        }
      }
    }
    // clock_gettime(CLOCK_REALTIME, &_end);
    clock_gettime(CLOCK_MONOTONIC, &_end);
    done_flag.store(true, std::memory_order_release);
    // DEBUGGING
    std::sort(thread_results, thread_results + thread_num, [](const pair_tr &i, const pair_tr &j)
              { return i.second < j.second; });
    for (size_t i = 0; i < thread_num; i++)
      printf("CPU: %d, duration: %ld\n", thread_results[i].first, thread_results[i].second);

    printf("Variance: %ld\n", thread_results[thread_num - 1].second - thread_results[0].second);
    for (auto &thread : threads)
      thread.join();
  }

  // [ ] Function to prepare the threads for Value GEMV
  void prepare_key_gemv(float *keys, float *queries, float *logits,
                        int const head_num, int const batch_size, int const K,
                        int const Dh, int const keys_head_offset,
                        int const keys_batch_offset,
                        int const queries_head_offset,
                        int const queries_batch_offset,
                        int const logits_head_offset,
                        int const logits_batch_offset, int const thread_num)
  {
    // printf("Ready Flag: %p\n", &ready_flag);
    // printf("Done Flag: %p\n", &done_flag);
    // Each thread works on its slice
    int const total_work = head_num * batch_size;
    int const work_per_thread = total_work / thread_num;
    int const work_remained = total_work % thread_num;

    //   int const min_priority = sched_get_priority_min(SCHED_FIFO);
    int const max_priority = sched_get_priority_max(SCHED_FIFO);

    int priority = max_priority; // Base priority for all threads

    // Init thread pool
    std::vector<std::thread> threads;
    int start_idx = 0, end_idx = 0;
    int acc = 0;
    for (int t = 0; t < thread_num; ++t)
    {
      start_idx = end_idx;
      end_idx = t < work_remained ? start_idx + work_per_thread + 1
                                  : start_idx + work_per_thread;
      int cpu_id = t + acc;
      acc += 1;
      // int cpu_id = t;
      threads.emplace_back(key_gemv_threaded, keys, queries, logits, head_num,
                           batch_size, K, Dh, keys_head_offset, keys_batch_offset,
                           queries_head_offset, queries_batch_offset,
                           logits_head_offset, logits_batch_offset, t, thread_num,
                           start_idx, end_idx, &ready_flag, &finished_flags[t], &thread_results[t]);

      // Get the native handle for the created thread
      pthread_t nativeHandle = threads.back().native_handle();

      // Define the scheduling parameters
      struct sched_param param;
      param.sched_priority = priority; // Set the same priorities for each thread

      // Set the scheduling policy to SCHED_FIFO
      int ret = pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
      if (ret != 0)
      {
        std::cerr << "Failed to set scheduling policy for thread " << t << ": "
                  << strerror(ret) << std::endl;
      }
      // Set CPU affinity
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      // Method1
      // if (t > 23)
      // {
      //   // int id = 48 + (t - 23);
      //   CPU_SET(48 + (t - 24), &cpuset); // Bind to specific CPU core
      // }
      // else
      // {
      //   CPU_SET(t, &cpuset); // Bind to specific CPU core
      // }
      // c13
      // if (t > 8)
      // {
      //   CPU_SET(32 + t, &cpuset); // Bind to specific CPU core
      // }
      // else
      // {
      //   CPU_SET(t, &cpuset); // Bind to specific CPU core
      // }
      // if (cpu_id > 28)
      //   CPU_SET(1, &cpuset); // Bind to specific CPU core
      // else
      CPU_SET(cpu_id, &cpuset); // Bind to specific CPU core
      ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
      if (ret != 0)
      {
        std::cerr << "Failed to set CPU affinity for thread " << t << ": "
                  << strerror(ret) << std::endl;
      }
    }

    bool all_threads_finished = false;
    bool thread_finished[thread_num];
    for (int i = 0; i < thread_num; ++i)
      thread_finished[i] = false;

    while (!all_threads_finished)
    {
      all_threads_finished = true;
      for (int i = 0; i < thread_num; ++i)
      {
        if (!thread_finished[i])
        {
          if (finished_flags[i].load(std::memory_order_acquire))
          {
            //   clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
            thread_finished[i] = true;
          }
          else
          {
            all_threads_finished = false;
          }
        }
      }
    }
    // clock_gettime(CLOCK_REALTIME, &_end);
    clock_gettime(CLOCK_MONOTONIC, &_end);
    done_flag.store(true, std::memory_order_release);
    // DEBUGGING
    std::sort(thread_results, thread_results + thread_num, [](const pair_tr &i, const pair_tr &j)
              { return i.second < j.second; });
    for (size_t i = 0; i < thread_num; i++)
      printf("CPU: %d, duration: %ld\n", thread_results[i].first, thread_results[i].second);

    printf("Variance: %ld\n", thread_results[thread_num - 1].second - thread_results[0].second);
    for (auto &thread : threads)
      thread.join();
  }

  // Function to set the ready_flag from Python
  void set_ready_flag()
  {
    // usleep(1000000);
    // clock_gettime(CLOCK_REALTIME, &_start);
    clock_gettime(CLOCK_MONOTONIC, &_start);
    ready_flag.store(true, std::memory_order_release);
  }

  // Function to check the all threads are done
  // bool is_finished() { return done_flag.load(std::memory_order_acquire); }
  // bool is_finished()
  // {
  //   while (!done_flag.load(std::memory_order_acquire))
  //   {
  //   }
  //   return true;
  // }
  long is_finished()
  {
    while (!done_flag.load(std::memory_order_acquire))
    {
    }
    // clock_gettime(CLOCK_REALTIME, &_end);
    clock_gettime(CLOCK_MONOTONIC, &_end_1);
    long seconds = _end.tv_sec - _start.tv_sec;
    long nanoseconds = _end.tv_nsec - _start.tv_nsec;
    // Handle case where nanoseconds roll over
    if (nanoseconds < 0)
    {
      --seconds;
      nanoseconds += 1000000000;
    }
    return seconds * 1e9 + nanoseconds;
    // return ((_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec) / 1e9) * 1e6;
  }

  void wait_finished()
  {
    while (!done_flag.load(std::memory_order_acquire))
    {
    }
  }

  double get_duration()
  {
    clock_gettime(CLOCK_REALTIME, &_end);
    return ((_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec) / 1e9) * 1e6;
  }

  // Function to clear all flags
  void clear_flags()
  {
    ready_flag.store(false, std::memory_order_release);
    done_flag.store(false, std::memory_order_release);
    for (int i = 0; i < THREAD_NUM; ++i)
      finished_flags[i].store(false, std::memory_order_release);
  }
}
#include "test_value_gemv.h"

#define ITER 51
static struct timespec start, end;
static double acc_time_sec;
static double cur_time_sec;

void value_gemv_eval(const size_t K, const size_t Dh, const size_t num_head,
                     const size_t batch_size, const size_t iteration,
                     const int values_head_offset,
                     const int values_batch_offset,
                     int const logits_head_offset,
                     int const logits_batch_offset,
                     int const result_head_offset,
                     int const result_batch_offset,
                     int const num_threads) {  // Total work = 256 / num_threads
  //////////////////////////////////////////////////////////////////////////////////
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

  double total_time_sec, total_time_sec_trusted;
  //////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////
  // Run attention output with OpenBLAS
  flush_cache();
  clock_gettime(CLOCK_REALTIME, &start);
  value_gemv_trusted(values_trusted, logits_trusted, result_trusted, num_head,
                     batch_size, K, Dh, logits_head_offset, logits_batch_offset,
                     values_head_offset, values_batch_offset,
                     result_head_offset, result_batch_offset);
  clock_gettime(CLOCK_REALTIME, &end);
  total_time_sec_trusted =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  //////////////////////////////////////////////////////////////////////////////////
  // Run attention output with AVX2 and thread pool
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

  int priority = max_priority;  // Base priority for all threads

  // DEBUGGING
  printf("Total Work: %d\n", total_work);
  printf("Work/Thread: %d\n", work_per_thread);

  // Check for the variance between threads
  double end_times[num_threads];

  // Init thread pool
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    const int start_idx = t * work_per_thread;
    const int end_idx = std::min(start_idx + work_per_thread, total_work);
    threads.emplace_back(
        value_gemv_threaded, values, logits, result, num_head, batch_size, K,
        Dh, values_head_offset, values_batch_offset, logits_head_offset,
        logits_batch_offset, result_head_offset, result_batch_offset, t,
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
  // double acc_time_sec;
  // double cur_time_sec;
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
    // float mse =
    //     calculate_mse(result[ii], result_trusted, num_head * batch_size *
    //     Dh);
    // float mae =
    //     calculate_mae(result[ii], result_trusted, num_head * batch_size *
    //     Dh);
    // printf("Current elapsed time: %f\n", cur_time_sec);
    // std::cout << "Mean Squared Error: " << mse << std::endl;
    // std::cout << "Maximum Absolute Error: " << mae << std::endl;
    // printf("Start elapsed time: %f\n", start.tv_sec + start.tv_nsec / 1e9);
    // printf("End elapsed time: %f\n", end.tv_sec + end.tv_nsec / 1e9);
    // printf("Acc elapsed time: %f\n", acc_time_sec);
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
      (Dh * K * num_head * batch_size + K * num_head * batch_size) *
      sizeof(float);
  double throughput = (total_bytes / total_time_sec) / 1e9;

  std::cout << "Elapsed time: " << total_time_sec * 1e6 << " microseconds"
            << std::endl;
  std::cout << "GFLOPs: " << gflops << std::endl;
  std::cout << "Total Bytes: " << total_bytes << std::endl;
  std::cout << "Throughput(GB/s): " << throughput << std::endl;
  printf("\n\n");

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

void value_gemv_eval_half(
    const size_t K, const size_t Dh, const size_t num_head,
    const size_t batch_size, const size_t iteration,
    const int values_head_offset, const int values_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const num_threads) {  // Total work = 256 / num_threads
  // Allocate memory
  half *values[ITER];
  half *logits[ITER];
  half *result[ITER];

  for (size_t i = 0; i < ITER; ++i) {
    values[i] = static_cast<half *>(
        aligned_alloc(64, num_head * batch_size * K * Dh * sizeof(half)));
    logits[i] = static_cast<half *>(
        aligned_alloc(64, num_head * batch_size * K * sizeof(half)));
    result[i] = static_cast<half *>(
        aligned_alloc(64, num_head * batch_size * Dh * sizeof(half)));
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
              values[ii][i * values_head_offset + j * values_batch_offset +
                         k * Dh + l] = __float2half(rand_val);
              values_trusted[i * values_head_offset + j * values_batch_offset +
                             k * Dh + l] = rand_val;
            } else {
              values[ii][i * values_head_offset + j * values_batch_offset +
                         k * Dh + l] =
                  values[0][i * values_head_offset + j * values_batch_offset +
                            k * Dh + l];
            }
          }

  for (size_t ii = 0; ii < iteration; ++ii)
    for (size_t i = 0; i < num_head; ++i)
      for (size_t j = 0; j < batch_size; ++j)
        for (size_t k = 0; k < K; ++k) {
          if (ii == 0) {
            float rand_val = dist(gen);
            logits[ii][i * logits_head_offset + j * logits_batch_offset + k] =
                __float2half(rand_val);
            logits_trusted[i * logits_head_offset + j * logits_batch_offset +
                           k] = rand_val;
          } else {
            logits[ii][i * logits_head_offset + j * logits_batch_offset + k] =
                logits[0][i * logits_head_offset + j * logits_batch_offset + k];
          }
        }

  for (size_t ii = 0; ii < iteration; ++ii) {
    for (size_t i = 0; i < num_head * batch_size * Dh; ++i) {
      result[ii][i] = 0.f;
      if (ii == 0) result_trusted[i] = 0.f;
    }
  }

  double total_time_sec, total_time_sec_trusted;

  //////////////////////////////////////////////////////////////////////////////////
  // Run attention output with OpenBLAS
  flush_cache();
  clock_gettime(CLOCK_REALTIME, &start);
  value_gemv_trusted(values_trusted, logits_trusted, result_trusted, num_head,
                     batch_size, K, Dh, logits_head_offset, logits_batch_offset,
                     values_head_offset, values_batch_offset,
                     result_head_offset, result_batch_offset);
  clock_gettime(CLOCK_REALTIME, &end);
  total_time_sec_trusted =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  //////////////////////////////////////////////////////////////////////////////////
  // Run attention output with AVX2 and thread pool
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

  int priority = max_priority;  // Base priority for all threads

  // DEBUGGING
  printf("Total Work: %d\n", total_work);
  printf("Work/Thread: %d\n", work_per_thread);

  // Init thread pool
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    const int start_idx = t * work_per_thread;
    const int end_idx = std::min(start_idx + work_per_thread, total_work);
    // threads.emplace_back(
    //     value_gemv_threaded, cur_values, cur_logits, cur_result, num_head,
    //     batch_size, K, Dh, values_head_offset, values_batch_offset,
    //     logits_head_offset, logits_batch_offset, result_head_offset,
    //     result_batch_offset, t, num_threads, start_idx, end_idx, &ready_flag,
    //     &finished_flags[t], &stop_flag, &iter_num, total_work);
    threads.emplace_back(
        value_gemv_threaded_half, values, logits, result, num_head, batch_size,
        K, Dh, values_head_offset, values_batch_offset, logits_head_offset,
        logits_batch_offset, result_head_offset, result_batch_offset, t,
        num_threads, start_idx, end_idx, &ready_flag, &finished_flags[t],
        &stop_flag, &iter_num);

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
    acc_time_sec += cur_time_sec;

    usleep(10);

    // Calculate MSE and MAE
    float mse = calculate_mse_half(result[ii], result_trusted,
                                   num_head * batch_size * Dh);
    float mae = calculate_mae_half(result[ii], result_trusted,
                                   num_head * batch_size * Dh);
    // [x] Debugging for each iteration
    // printf("Current elapsed time: %f\n", cur_time_sec);
    // std::cout << "Mean Squared Error: " << mse << std::endl;
    // std::cout << "Maximum Absolute Error: " << mae << std::endl;
    // printf("Start elapsed time: %f\n", start.tv_sec + start.tv_nsec / 1e9);
    // printf("End elapsed time: %f\n", end.tv_sec + end.tv_nsec / 1e9);
    // printf("Acc elapsed time: %f\n", acc_time_sec);

    // Reset the result of Value GEMV
    // for (size_t i = 0; i < num_head * batch_size * Dh; ++i) {
    //   result[i] = 0.f;
    // }
  }
  total_time_sec = acc_time_sec / iteration;

  // Stop the thread pool
  stop_flag.store(true, std::memory_order_release);

  for (auto &thread : threads) thread.join();

  // Calculate FLOPs and GFLOPs
  double flops = 2.0 * Dh * K * num_head * batch_size;
  double gflops = flops / total_time_sec / 1e9;
  double gflops_trusted = flops / total_time_sec_trusted / 1e9;
  int const num_keys = Dh * K * num_head * batch_size;
  int const num_logits = K * num_head * batch_size;
  double total_bytes =
      (Dh * K * num_head * batch_size + K * num_head * batch_size) *
      sizeof(half);
  double throughput = (total_bytes / 1e9) / total_time_sec;

  printf("Number of elements in Keys: %d\n", num_keys);
  printf("Number of elements in Logits: %d\n", num_logits);
  std::cout << "Elapsed time: " << total_time_sec * 1e6 << " microseconds"
            << std::endl;
  // std::cout << "GFLOPs: " << gflops << std::endl;
  std::cout << "Total Bytes: " << total_bytes / 1e9 << " GB" << std::endl;
  std::cout << "Throughput(GB/s): " << throughput << std::endl;
  printf("\n\n");

  // std::cout <<
  // "==========================Trusted_value_gemv=================="
  //              "========"
  //           << std::endl;
  // std::cout << "Elapsed time: " << total_time_sec_trusted * 1e6
  //           << " microseconds" << std::endl;
  // std::cout << "GFLOPs: " << gflops_trusted << std::endl;

  // Calculate MSE and MAE
  // float mse =
  //     calculate_mse(result[1], result_trusted, num_head * batch_size * Dh);
  // float mae =
  //     calculate_mae(result[1], result_trusted, num_head * batch_size * Dh);

  // std::cout << "Mean Squared Error: " << mse << std::endl;
  // std::cout << "Maximum Absolute Error: " << mae << std::endl;

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

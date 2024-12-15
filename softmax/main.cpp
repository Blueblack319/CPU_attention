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

#include "float.h"
#include "softmax.hpp"
#define MAX_BATCH_SIZE 128
#define MAX_HEAD_NUM 32
#define MAX_ARR_LEN (MAX_BATCH_SIZE * MAX_HEAD_NUM)
#define ITER 50

/*
  81 =  8k * 0.01 = 4k * 0.02 = 2k * 0.04
  163 =  8k * 0.02 = 4k * 0.04
  245 =  8k * 0.03
  327 =  8k * 0.04
  409 =  8k * 0.05
  40 =  4k * 0.01 = 2k * 0.02 = 1k * 0.04
  122 =  4k * 0.03
  204 =  4k * 0.05
  20 =  2k * 0.01 = 1k * 0.02
  61 =  2k * 0.03
  102 =  2k * 0.05
  10 =  1k * 0.01
  30 =  1k * 0.03
  51 =  1k * 0.05
*/

int main(int argc, char *argv[]) {
  // Check if NUMA is available
  if (numa_available() == -1) {
    std::cerr << "NUMA is not available on this system." << std::endl;
    return 1;
  }

  const size_t batch_size = atoi(argv[1]);
  const size_t seq_len = atoi(argv[2]);
  const size_t K = atoi(argv[3]);
  const size_t thread_num = atoi(argv[4]);

  const size_t head_num = 32;
  const int head_offset = batch_size * seq_len;
  const int batch_offset = seq_len;

  printf("Softmax\n");
  printf("BS: %d, K: %d, head_num: %d, thread_num: %d\n", batch_size, K,
         head_num, thread_num);

  float *qk = static_cast<float *>(
      aligned_alloc(64, head_num * batch_size * seq_len * sizeof(float)));
  std::vector<p_iv> topk_logits[head_num][batch_size];

  float *qk_trusted = static_cast<float *>(
      aligned_alloc(64, head_num * batch_size * seq_len * sizeof(float)));

  // Initialize qk with random values
  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  for (size_t i = 0; i < head_num * batch_size * seq_len; i++) {
    qk[i] = dist(gen);
  }

  // Variables for measuring performance
  struct timespec start_trusted, end_trusted;
  double duration_sec_trusted;
  double total_bytes_trusted = head_num * batch_size * seq_len;

  ////////////////////////////////////////////////////
  // Trusted implementation
  // Max value for each head and batch
  float max_arr[MAX_ARR_LEN];
  for (size_t i = 0; i < head_num; ++i)
    for (size_t j = 0; j < batch_size; ++j) {
      max_arr[i * batch_size + j] =
          *std::max_element(qk + i * head_offset + j * batch_offset,
                            qk + i * head_offset + j * batch_offset + seq_len);
    }
  //////

  // Softmax
  // Single thread
  // clock_gettime(CLOCK_REALTIME, &start_trusted);
  // for (size_t i = 0; i < head_num; ++i)
  //   for (size_t j = 0; j < batch_size; ++j)
  //     softmax_trusted(&qk[i * head_offset + j * batch_offset], seq_len,
  //                     max_arr[i * batch_size + j]);
  // clock_gettime(CLOCK_REALTIME, &end_trusted);
  //////

  // Multiple threads
  // Define synchronization variables
  std::atomic<bool> ready_flag(false);
  // Define the finished flag for each thread
  std::atomic<bool> finished_flags[thread_num];
  for (int i = 0; i < thread_num; ++i)
    finished_flags[i].store(false, std::memory_order_release);

  // Create array of timespecs to store when each thread finishes
  bool thread_finished[thread_num];
  for (int i = 0; i < thread_num; ++i) thread_finished[i] = false;
  // Workload for each thread
  int const total_work = head_num * batch_size;
  int const work_per_thread = total_work / thread_num;
  int const work_remained = total_work % thread_num;
  int const priority = sched_get_priority_max(SCHED_FIFO);

  // Check for the variance between threads
  double end_times[thread_num];

  // Init thread pool
  std::vector<std::thread> threads;
  int start_idx = 0, end_idx = 0;
  int acc = 0;
  for (int t = 0; t < thread_num; ++t) {
    start_idx = end_idx;
    end_idx = t < work_remained ? start_idx + work_per_thread + 1
                                : start_idx + work_per_thread;
    int cpu_id = t + acc;
    acc += 1;
    threads.emplace_back(softmax_trusted_threads, qk, max_arr, seq_len,
                         head_num, batch_size, head_offset, batch_offset, t,
                         thread_num, start_idx, end_idx, &ready_flag,
                         &finished_flags[t], &end_times[t]);

    // Get the native handle for the created thread
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
    CPU_SET(cpu_id, &cpuset);  // Bind to specific CPU core
    ret = pthread_setaffinity_np(nativeHandle, sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
      std::cerr << "Failed to set CPU affinity for thread " << t << ": "
                << strerror(ret) << std::endl;
    }
  }

  usleep(100000);  // Sleep for 1s to allow threads to start

  // Flush the current data in Cache
  flush_cache();

  // Measure execution time
  clock_gettime(CLOCK_REALTIME, &start_trusted);

  // Start the threads by setting the ready flag
  ready_flag.store(true, std::memory_order_release);

  // Busy wait until all threads are finished
  bool all_threads_finished = false;
  while (!all_threads_finished) {
    all_threads_finished = true;
    for (int i = 0; i < thread_num; ++i) {
      if (!thread_finished[i]) {
        if (finished_flags[i].load(std::memory_order_acquire)) {
          thread_finished[i] = true;
        } else {
          all_threads_finished = false;
        }
      }
    }
  }
  // Measure execution time
  clock_gettime(CLOCK_REALTIME, &end_trusted);

  for (size_t t = 0; t < thread_num; ++t) threads[t].join();
  //////

  // Select top-k indices and values
  for (size_t i = 0; i < head_num; ++i) {
    for (size_t j = 0; j < batch_size; ++j) {
      topk_logits[i][j] =
          topk(&qk[i * head_offset + j * batch_offset], seq_len, K, false);
    }
  }

  // printf("After Softmax\n");
  // for (size_t i = 0; i < head_num; ++i) {
  //   for (size_t j = 0; j < batch_size; ++j) {
  //     printf("Head: %d, Batch: %d\n", i, j);
  //     float acc = 0;
  //     for (size_t k = 0; k < K; ++k) {
  //       // acc += qk[i * head_offset + j * batch_offset + k];
  //       // printf("%f ", qk[i * head_offset + j * batch_offset + k]);
  //       acc += topk_logits[i][j][k].second;
  //       // printf("%d ", topk_logits[i][j][k].first);
  //       printf("%d ", topk_logits[i][j][k].first);
  //     }
  //     printf("Acc: %f\n", acc);
  //   }
  // }
  duration_sec_trusted = (end_trusted.tv_sec - start_trusted.tv_sec) +
                         (end_trusted.tv_nsec - start_trusted.tv_nsec) / 1e9;
  double throughput = (total_bytes_trusted / duration_sec_trusted) / 1e9;
  printf("Elapsed time(ms): %lf\n", duration_sec_trusted * 1e6);
  printf("Throughput(GB/s): %lf\n", throughput);
  ////////////////////////////////////////////////////

  return 0;
}

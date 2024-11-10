#include <immintrin.h>
#include <sched.h>
#include <string.h>

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

extern "C" {
std::atomic<bool> ready_flag(false);

// Define the finished flag for each thread
std::atomic<bool> finished_flags[48];
std::atomic<bool> done_flag(false);
bool done = false;

// Declare the function
void attn_output_threaded(float* values, float* logits, float* result,
                          int const head_num, int const batch_size, int const K,
                          int const Dh, int const values_head_offset,
                          int const values_batch_offset,
                          int const logits_head_offset,
                          int const logits_batch_offset,
                          int const result_head_offset,
                          int const result_batch_offset, int const thread_id,
                          int const thread_num, int const start_idx,
                          int const end_idx, std::atomic<bool>* ready_flag,
                          std::atomic<bool>* finished_flag) {
  while (!(ready_flag->load(std::memory_order_acquire))) {
    // while (!(*ready_flag)) {
    // Multiply and Add
  }
  for (int idx = start_idx; idx < end_idx; ++idx) {
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

    for (int k = 0; k < K; ++k) {
      float logit =
          logits[i * logits_head_offset + j * logits_batch_offset + k];
      __m256 logit_vec = _mm256_set1_ps(logit);

      if (k + 1 < K) {
        _mm_prefetch((const char*)(values + i * values_head_offset +
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
}

// Function to prepare the threads
void prepare_threads(float* values, float* logits, float* result,
                     int const head_num, int const batch_size, int const K,
                     int const Dh, int const values_head_offset,
                     int const values_batch_offset,
                     int const logits_head_offset,
                     int const logits_batch_offset,
                     int const result_head_offset,
                     int const result_batch_offset, int const thread_num,
                     bool* done) {
  // Each thread works on its slice
  int const total_work = head_num * batch_size;
  // int work_per_thread = (total_work + thread_num - 1) / thread_num;
  int const work_per_thread = total_work / thread_num;
  //   int const min_priority = sched_get_priority_min(SCHED_FIFO);
  int const max_priority = sched_get_priority_max(SCHED_FIFO);

  int priority = max_priority;  // Base priority for all threads

  // Init thread pool
  std::vector<std::thread> threads;
  for (int t = 0; t < thread_num; ++t) {
    const int start_idx = t * work_per_thread;
    const int end_idx = std::min(start_idx + work_per_thread, total_work);

    threads.emplace_back(
        attn_output_threaded, values, logits, result, head_num, batch_size, K,
        Dh, values_head_offset, values_batch_offset, logits_head_offset,
        logits_batch_offset, result_head_offset, result_batch_offset, t,
        thread_num, start_idx, end_idx, &ready_flag, &finished_flags[t]);

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

  bool all_threads_finished = false;
  bool thread_finished[thread_num];
  for (int i = 0; i < thread_num; ++i) thread_finished[i] = false;

  while (!all_threads_finished) {
    all_threads_finished = true;
    for (int i = 0; i < thread_num; ++i) {
      if (!thread_finished[i]) {
        if (finished_flags[i].load(std::memory_order_acquire)) {
          //   clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
          thread_finished[i] = true;
        } else {
          all_threads_finished = false;
        }
      }
    }
  }
  done_flag.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();
}

// Function to set the ready_flag from Python
void set_ready_flag() { ready_flag.store(true, std::memory_order_release); }

bool is_finished() { return done_flag.load(std::memory_order_acquire); }
}

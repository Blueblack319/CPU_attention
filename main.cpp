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
#include "test_key_gemv.h"
#include "test_value_gemv.h"

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

int main(int argc, char *argv[])
{
  // Check if NUMA is available
  if (numa_available() == -1)
  {
    std::cerr << "NUMA is not available on this system." << std::endl;
    return 1;
  }

  const size_t batch_size = atoi(argv[1]);
  const size_t K = atoi(argv[2]);
  const size_t thread_num = atoi(argv[3]);
  const bool is_key_gemv = bool(atoi(argv[4]));

  size_t Dh = 128, iteration = 51;
  size_t num_head = is_key_gemv ? 4 : 32;
  if (is_key_gemv)
    printf("Key GEMV\n");
  else
    printf("Value GEMV\n");
  printf("BS: %d, K: %d, Dh: %d, num_head: %d, thread_num: %d\n", batch_size, K,
         Dh, num_head, thread_num);
  const int kv_head_offset = batch_size * K * Dh;
  const int kv_batch_offset = K * Dh;
  const int logits_score_head_offset = batch_size * K;
  const int logits_score_batch_offset = K;
  const int q_out_head_offset = batch_size * Dh;
  const int q_out_batch_offset = Dh;

  if (is_key_gemv)
  {
    key_gemv_eval(K, Dh, num_head, batch_size, iteration, kv_head_offset,
                  kv_batch_offset, q_out_head_offset, q_out_batch_offset,
                  logits_score_head_offset, logits_score_batch_offset,
                  thread_num);
  }
  else
  {
    value_gemv_eval(K, Dh, num_head, batch_size, iteration, kv_head_offset,
                    kv_batch_offset, logits_score_head_offset,
                    logits_score_batch_offset, q_out_head_offset,
                    q_out_batch_offset, thread_num);
    // value_gemv_eval_half(K, Dh, num_head, batch_size, iteration, kv_head_offset,
    //                      kv_batch_offset, logits_score_head_offset,
    //                      logits_score_batch_offset, q_out_head_offset,
    //                      q_out_batch_offset, thread_num);
  }

  return 0;
}

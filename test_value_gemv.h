#include <stdlib.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "float.h"
#include "utils.h"
#include "value_gemv.h"

// void value_gemv_eval(const size_t K, const size_t Dh, const size_t num_head,
//                      const size_t batch_size, const size_t iteration,
//                      const int values_head_offset,
//                      const int values_batch_offset,
//                      int const logits_head_offset,
//                      int const logits_batch_offset,
//                      int const result_head_offset,
//                      int const result_batch_offset, int const num_threads);

void value_gemv_eval_half(const size_t S_len, const size_t topk_num,
                          const size_t Dh, const size_t q_head_num,
                          const size_t kv_head_num, const size_t batch_size,
                          const int values_head_offset,
                          const int values_batch_offset,
                          int const logits_head_offset,
                          int const logits_batch_offset,
                          int const result_head_offset,
                          int const result_batch_offset, int const num_threads);
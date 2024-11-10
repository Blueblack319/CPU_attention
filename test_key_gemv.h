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

void key_gemv_eval(const size_t K, const size_t Dh, const size_t num_head,
                   const size_t batch_size, const size_t iteration,
                   const int keys_head_offset, const int keys_batch_offset,
                   int const queries_head_offset,
                   int const queries_batch_offset, int const logits_head_offset,
                   int const logits_batch_offset, int const num_threads);

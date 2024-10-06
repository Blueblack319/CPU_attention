#define _GNU_SOURCE
#include <immintrin.h>  // AVX intrinsics
#include <math.h>       // For sqrt
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>  // For uint32_t
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MATRIX_DTYPE uint64_t
#define MATRIX_DTYPE_BIT_WIDTH (sizeof(MATRIX_DTYPE) * 8)
#define NIBBLES_PER_ELEMENT (MATRIX_DTYPE_BIT_WIDTH / BIT_WIDTH)
#define MATRIX_ROWS 4096
#define BIT_WIDTH 4
#define MATRIX_COLS (4096 / NIBBLES_PER_ELEMENT)
#define NUM_ROWS_TO_FETCH 64
#define LOOKUP_TABLE_SIZE (1 << BIT_WIDTH)

// Struct to store information for each thread
typedef struct {
  int id;
  MATRIX_DTYPE **matrix;
  int *selected_rows;
  float *final_sum;
  int col_start;
  int col_end;
  atomic_int *ready;
  atomic_int *finished;
  float (*lookup_tables)[LOOKUP_TABLE_SIZE];  // Lookup tables for each row
  float *activations;                         // Activations for each row
} thread_info_t;

void flush_cache() {
  size_t cache_flusher_size = 512 * 1024 * 1024;  // 512 MB
  char *cache_flusher = (char *)malloc(cache_flusher_size);

  for (size_t i = 0; i < cache_flusher_size; i += 4096) {
    cache_flusher[i] = 0;
  }

  free(cache_flusher);
}

void *thread_func(void *arg) {
  thread_info_t *info = (thread_info_t *)arg;

  // Spin until the ready flag is set using atomic operations
  while (!atomic_load(info->ready)) {
    // Busy-wait until ready is set to 1
  }

  // Iterate over the columns assigned to this thread
  for (int i = 0; i < NUM_ROWS_TO_FETCH; i++) {
    int selected_row_index = info->selected_rows[i];
    for (int j = info->col_start; j < info->col_end; j++) {
      MATRIX_DTYPE value = info->matrix[selected_row_index][j];

      for (int k = 0; k < NIBBLES_PER_ELEMENT; k++) {
        uint8_t nibble = (value >> (k * BIT_WIDTH)) & ((1 << BIT_WIDTH) - 1);
        info->final_sum[j * NIBBLES_PER_ELEMENT + k] +=
            info->lookup_tables[selected_row_index][nibble] *
            info->activations[selected_row_index];
      }
    }
  }

  // Mark the thread as finished
  atomic_store(info->finished, 1);

  return NULL;
}

void run_test(MATRIX_DTYPE **matrix, int *selected_rows, float *final_sum,
              int num_threads, float (*lookup_tables)[LOOKUP_TABLE_SIZE],
              float *activations, int time_limit) {
  pthread_t threads[num_threads];
  thread_info_t thread_info[num_threads];
  atomic_int ready = 0;
  atomic_int finished_flags[num_threads];
  bool thread_finished[num_threads];
  memset(thread_finished, 0, num_threads * sizeof(bool));

  // Initialize finished flags to false
  for (int i = 0; i < num_threads; i++) {
    atomic_store(&finished_flags[i], 0);
  }

  // Create array of timespecs to store when each thread finishes
  struct timespec thread_finish_times[num_threads];

  // Calculate columns per thread and adjust for uneven distribution
  int base_cols_per_thread = MATRIX_COLS / num_threads;
  int extra_cols = MATRIX_COLS % num_threads;

  int current_col = 0;

  // Create threads
  for (int i = 0; i < num_threads; i++) {
    int cols_for_this_thread = base_cols_per_thread + (i < extra_cols ? 1 : 0);

    thread_info[i].id = i;
    thread_info[i].matrix = matrix;
    thread_info[i].selected_rows = selected_rows;
    thread_info[i].final_sum = final_sum;
    thread_info[i].col_start = current_col;
    thread_info[i].col_end = current_col + cols_for_this_thread;
    thread_info[i].ready = &ready;
    thread_info[i].finished = &finished_flags[i];
    thread_info[i].lookup_tables = lookup_tables;
    thread_info[i].activations = activations;

    pthread_create(&threads[i], NULL, thread_func, &thread_info[i]);
    current_col += cols_for_this_thread;
  }

  usleep(10000);  // Sleep for 10ms to allow threads to start

  struct timespec start, end;
  clock_gettime(CLOCK_REALTIME, &start);

  // Start the threads by setting the ready flag
  atomic_store(&ready, 1);

  // Busy wait until all threads are finished
  int all_threads_finished = 0;
  struct timespec current_time;
  while (!all_threads_finished) {
    all_threads_finished = 1;
    for (int i = 0; i < num_threads; i++) {
      if (!thread_finished[i]) {
        if (atomic_load(&finished_flags[i])) {
          clock_gettime(CLOCK_REALTIME, &thread_finish_times[i]);
          thread_finished[i] = true;
        } else {
          all_threads_finished = 0;
        }
      }
    }
    // Check the time, and if time limit is exceeded in microseconds, break
    clock_gettime(CLOCK_REALTIME, &current_time);
    if ((current_time.tv_sec - start.tv_sec) * 1e6 +
            (current_time.tv_nsec - start.tv_nsec) / 1e3 >
        time_limit) {
      break;
    }
  }

  clock_gettime(CLOCK_REALTIME, &end);
  double total_time_seconds =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  // check the total of the final sum
  float total_sum = 0;
  for (int i = 0; i < MATRIX_COLS * NIBBLES_PER_ELEMENT; i++) {
    total_sum += final_sum[i];
  }

  // Check the progress by dividing the total sum by the expected sum
  float progress =
      total_sum / (MATRIX_COLS * NIBBLES_PER_ELEMENT * NUM_ROWS_TO_FETCH);

  // this is the throughput if progress is 1, in GB/s
  float max_throughput =
      (MATRIX_COLS * NUM_ROWS_TO_FETCH * sizeof(MATRIX_DTYPE)) /
      (total_time_seconds * 1e9);

  // this is the actual throughput
  float real_throughput = max_throughput * progress;

  // Print the results
  printf("%-10d %-15.2f %-15.2f %-15.2f\n", num_threads,
         total_time_seconds * 1e6, progress * 100, real_throughput);

  // Join threads before exiting
  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }
}

void generate_random_rows(int *selected_rows) {
  bool selected[MATRIX_ROWS] = {false};  // Track selected rows

  for (int i = 0; i < NUM_ROWS_TO_FETCH; i++) {
    int rand_row;
    do {
      rand_row = rand() % MATRIX_ROWS;
    } while (selected[rand_row]);  // Ensure the row hasn't been selected before

    selected_rows[i] = rand_row;  // Store the unique row
    selected[rand_row] = true;    // Mark the row as selected
  }
}

int main(int argc, char *argv[]) {
  // Get time limit from arguments
  int time_limit = 8;  // Default time limit is 8 microseconds
  if (argc > 1) {
    time_limit = atoi(argv[1]);
  }

  // Allocate memory for the matrix
  MATRIX_DTYPE **matrix =
      (MATRIX_DTYPE **)malloc(MATRIX_ROWS * sizeof(MATRIX_DTYPE *));
  for (int i = 0; i < MATRIX_ROWS; i++) {
    matrix[i] = (MATRIX_DTYPE *)malloc(MATRIX_COLS * sizeof(MATRIX_DTYPE));
  }

  // Initialize the matrix with random values
  srand(time(NULL));
  for (int i = 0; i < MATRIX_ROWS; i++) {
    for (int j = 0; j < MATRIX_COLS; j++) {
      // Set each nibble to a random value
      matrix[i][j] = 0;
      for (int k = 0; k < NIBBLES_PER_ELEMENT; k++) {
        matrix[i][j] |= ((MATRIX_DTYPE)rand() & ((1 << BIT_WIDTH) - 1))
                        << (k * BIT_WIDTH);
      }
    }
  }

  // Allocate memory for the float lookup tables (one per row)
  float lookup_tables[MATRIX_ROWS][LOOKUP_TABLE_SIZE];
  for (int i = 0; i < MATRIX_ROWS; i++) {
    for (int j = 0; j < LOOKUP_TABLE_SIZE; j++) {
      lookup_tables[i][j] = (float)1.0;
    }
  }

  float activations[MATRIX_ROWS];
  for (int i = 0; i < MATRIX_ROWS; i++) {
    activations[i] = (float)1.0;
  }

  // Array to store the final sum of the selected rows
  float *final_sum =
      (float *)calloc(MATRIX_COLS * NIBBLES_PER_ELEMENT, sizeof(float));

  // Array to store selected random rows
  int selected_rows[NUM_ROWS_TO_FETCH];

  // Print table header
  printf("%-10s %-15s %-15s %-15s\n", "Threads", "Latency (us)", "Progress (%)",
         "Throughput (GB/s)");
  printf("------------------------------------------------------------\n");

  // Run the test with varying thread counts
  for (int num_threads = 1; num_threads <= 32; num_threads += 2) {
    generate_random_rows(selected_rows);
    memset(final_sum, 0,
           MATRIX_COLS * sizeof(float) *
               NIBBLES_PER_ELEMENT);  // Clear the final sum
    flush_cache();
    run_test(matrix, selected_rows, final_sum, num_threads, lookup_tables,
             activations, time_limit);
  }

  // Free the allocated matrix and final sum
  for (int i = 0; i < MATRIX_ROWS; i++) {
    free(matrix[i]);
  }
  free(matrix);
  free(final_sum);

  return 0;
}

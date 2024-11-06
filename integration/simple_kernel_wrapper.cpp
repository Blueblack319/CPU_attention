#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <vector>

namespace py = pybind11;

// Declare the function
void attn_output_threaded(
    float** values_arr, float** logits_arr, float** result_arr,
    int const num_head, int const batch_size, int const K, int const Dh,
    int const values_head_offset, int const values_batch_offset,
    int const logits_head_offset, int const logits_batch_offset,
    int const result_head_offset, int const result_batch_offset,
    int const thread_id, int const num_threads, int const start_idx,
    int const end_idx, std::atomic<bool>* ready_flag,
    std::atomic<bool>* finished_flag, std::atomic<bool>* stop_flag,
    std::atomic<int>* iter_num);

// Wrapper function to convert numpy arrays to raw pointers
void attn_output_threaded_wrapper(
    py::array_t<float> values, py::array_t<float> logits,
    py::array_t<float> result, int num_head, int batch_size, int K, int Dh,
    int values_head_offset, int values_batch_offset, int logits_head_offset,
    int logits_batch_offset, int result_head_offset, int result_batch_offset,
    int thread_id, int num_threads, int start_idx, int end_idx) {
  // Convert numpy arrays to pointers
  float** values_arr = static_cast<float**>(values.mutable_data());
  float** logits_arr = static_cast<float**>(logits.mutable_data());
  float** result_arr = static_cast<float**>(result.mutable_data());

  // Create atomic flags
  std::atomic<bool> ready_flag(false);
  std::atomic<bool> finished_flag(false);
  std::atomic<bool> stop_flag(false);
  std::atomic<int> iter_num(0);

  // Call the actual C++ function
  attn_output_threaded(values_arr, logits_arr, result_arr, num_head, batch_size,
                       K, Dh, values_head_offset, values_batch_offset,
                       logits_head_offset, logits_batch_offset,
                       result_head_offset, result_batch_offset, thread_id,
                       num_threads, start_idx, end_idx, &ready_flag,
                       &finished_flag, &stop_flag, &iter_num);
}

// Expose the function to Python
PYBIND11_MODULE(attn_module, m) {
  m.def("attn_output_threaded", &attn_output_threaded_wrapper,
        "GEMV Attention Output with threading");
}

#include <immintrin.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>

namespace py = pybind11;

// Declare the function
void attn_output_threaded(float* values, float* logits, float* result,
                          int const head_num, int const batch_size, int const K,
                          int const Dh, int const values_head_offset,
                          int const values_batch_offset,
                          int const logits_haed_offset,
                          int const logits_batch_offset,
                          int const result_head_offset,
                          int const result_batch_offset, int const thread_id,
                          int const thread_num, int const start_idx,
                          int const end_idx) {
  //   while (!(ready_flag->load(std::memory_order_acquire))) {
  //     //   while (!(*ready_flag)) {
  //     // Multiply and Add
  //   }
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
          logits[i * logits_haed_offset + j * logits_batch_offset + k];
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
}

// Wrapper function to convert numpy arrays to raw pointers
void attn_output_threaded_pybind(
    py::array_t<float> values, py::array_t<float> logits,
    py::array_t<float> result, int head_num, int batch_size, int K, int Dh,
    int values_head_offset, int values_batch_offset, int logits_head_offset,
    int logits_batch_offset, int result_head_offset, int result_batch_offset,
    int thread_id, int thread_num, int start_idx, int end_idx) {
  // Get C-style pointers from py::array_t
  float* values_ptr = static_cast<float*>(values.mutable_data());
  float* logits_ptr = static_cast<float*>(logits.mutable_data());
  float* result_ptr = static_cast<float*>(result.mutable_data());

  // Call the original function
  attn_output_threaded(values_ptr, logits_ptr, result_ptr, head_num, batch_size,
                       K, Dh, values_head_offset, values_batch_offset,
                       logits_head_offset, logits_batch_offset,
                       result_head_offset, result_batch_offset, thread_id,
                       thread_num, start_idx, end_idx);
}

// Make attn_output_threaded more simple
void attn_output_simple(float* values, float* logits, float* result,
                        int const head_num, int const batch_size, int const K,
                        int const Dh, int const values_head_offset,
                        int const values_batch_offset,
                        int const logits_haed_offset,
                        int const logits_batch_offset,
                        int const result_head_offset,
                        int const result_batch_offset, int const thread_id,
                        int const thread_num, int const start_idx,
                        int const end_idx) {
  // Multiply and Add
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
    float logit = logits[k];
    __m256 logit_vec = _mm256_set1_ps(logit);

    if (k + 1 < K) {
      _mm_prefetch((const char*)(values + (k + 1) * Dh), _MM_HINT_T0);
    }
    __m256 v00 = _mm256_load_ps(values + k * Dh);
    __m256 v01 = _mm256_load_ps(values + k * Dh + 8);
    __m256 v02 = _mm256_load_ps(values + k * Dh + 16);
    __m256 v03 = _mm256_load_ps(values + k * Dh + 24);
    __m256 v04 = _mm256_load_ps(values + k * Dh + 32);
    __m256 v05 = _mm256_load_ps(values + k * Dh + 40);
    __m256 v06 = _mm256_load_ps(values + k * Dh + 48);
    __m256 v07 = _mm256_load_ps(values + k * Dh + 56);
    __m256 v08 = _mm256_load_ps(values + k * Dh + 64);
    __m256 v09 = _mm256_load_ps(values + k * Dh + 72);
    __m256 v10 = _mm256_load_ps(values + k * Dh + 80);
    __m256 v11 = _mm256_load_ps(values + k * Dh + 88);
    __m256 v12 = _mm256_load_ps(values + k * Dh + 96);
    __m256 v13 = _mm256_load_ps(values + k * Dh + 104);
    __m256 v14 = _mm256_load_ps(values + k * Dh + 112);
    __m256 v15 = _mm256_load_ps(values + k * Dh + 120);
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
  _mm256_store_ps(result, c00);
  _mm256_store_ps(result + 8, c01);
  _mm256_store_ps(result + 16, c02);
  _mm256_store_ps(result + 24, c03);
  _mm256_store_ps(result + 32, c04);
  _mm256_store_ps(result + 40, c05);
  _mm256_store_ps(result + 48, c06);
  _mm256_store_ps(result + 56, c07);
  _mm256_store_ps(result + 64, c08);
  _mm256_store_ps(result + 72, c09);
  _mm256_store_ps(result + 80, c10);
  _mm256_store_ps(result + 88, c11);
  _mm256_store_ps(result + 96, c12);
  _mm256_store_ps(result + 104, c13);
  _mm256_store_ps(result + 112, c14);
  _mm256_store_ps(result + 120, c15);
}

void attn_output_simple_pybind(
    py::array_t<float> values, py::array_t<float> logits,
    py::array_t<float> result, int head_num, int batch_size, int K, int Dh,
    int values_head_offset, int values_batch_offset, int logits_head_offset,
    int logits_batch_offset, int result_head_offset, int result_batch_offset,
    int thread_id, int thread_num, int start_idx, int end_idx) {
  // Get C-style pointers from py::array_t
  float* values_ptr = static_cast<float*>(values.mutable_data());
  float* logits_ptr = static_cast<float*>(logits.mutable_data());
  float* result_ptr = static_cast<float*>(result.mutable_data());

  // Call the original function
  attn_output_simple(values_ptr, logits_ptr, result_ptr, head_num, batch_size,
                     K, Dh, values_head_offset, values_batch_offset,
                     logits_head_offset, logits_batch_offset,
                     result_head_offset, result_batch_offset, thread_id,
                     thread_num, start_idx, end_idx);
}

PYBIND11_MODULE(attn_module, m) {
  m.def("attn_output_threaded", &attn_output_threaded_pybind, py::arg("values"),
        py::arg("logits"), py::arg("result"), py::arg("head_num"),
        py::arg("batch_size"), py::arg("K"), py::arg("Dh"),
        py::arg("values_head_offset"), py::arg("values_batch_offset"),
        py::arg("logits_head_offset"), py::arg("logits_batch_offset"),
        py::arg("result_head_offset"), py::arg("result_batch_offset"),
        py::arg("thread_id"), py::arg("thread_num"), py::arg("start_idx"),
        py::arg("end_idx"),
        "Execute threaded attention output with AVX optimizations.");

  //   m.def("attn_output_threaded", &attn_output_threaded_pybind,
  //         "Execute threaded attention output with AVX optimizations.");

  m.def("attn_output_simple", &attn_output_simple_pybind, py::arg("values"),
        py::arg("logits"), py::arg("result"), py::arg("head_num"),
        py::arg("batch_size"), py::arg("K"), py::arg("Dh"),
        py::arg("values_head_offset"), py::arg("values_batch_offset"),
        py::arg("logits_head_offset"), py::arg("logits_batch_offset"),
        py::arg("result_head_offset"), py::arg("result_batch_offset"),
        py::arg("thread_id"), py::arg("thread_num"), py::arg("start_idx"),
        py::arg("end_idx"), "simple");
}
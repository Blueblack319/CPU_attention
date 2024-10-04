void SLLMM3X4(int m, int n, int k, const float *A, int lda, const float *B,
              int ldb, float *C, int ldc) {
#pragma omp parallel for collapse(2) if (m * n * k > 300000)
  for (int i = 0; i < m; i += 3)
    for (int j = 0; j < n; j += 4) {
      __m256 c00 = _mm256_setzero_ps();
      __m256 c01 = _mm256_setzero_ps();
      __m256 c02 = _mm256_setzero_ps();
      __m256 c03 = _mm256_setzero_ps();
      __m256 c10 = _mm256_setzero_ps();
      __m256 c11 = _mm256_setzero_ps();
      __m256 c12 = _mm256_setzero_ps();
      __m256 c13 = _mm256_setzero_ps();
      __m256 c20 = _mm256_setzero_ps();
      __m256 c21 = _mm256_setzero_ps();
      __m256 c22 = _mm256_setzero_ps();
      __m256 c23 = _mm256_setzero_ps();
      for (int l = 0; l < k; l += 8) {
        __m256 k0 = _mm256_loadu_ps(B + ldb * (j + 0) + l);
        __m256 k1 = _mm256_loadu_ps(B + ldb * (j + 1) + l);
        __m256 k2 = _mm256_loadu_ps(B + ldb * (j + 2) + l);
        __m256 k3 = _mm256_loadu_ps(B + ldb * (j + 3) + l);
        __m256 a0 = _mm256_loadu_ps(A + lda * (i + 0) + l);
        c00 = _mm256_fmadd_ps(a0, k0, c00);
        c01 = _mm256_fmadd_ps(a0, k1, c01);
        c02 = _mm256_fmadd_ps(a0, k2, c02);
        c03 = _mm256_fmadd_ps(a0, k3, c03);
        __m256 a1 = _mm256_loadu_ps(A + lda * (i + 1) + l);
        c10 = _mm256_fmadd_ps(a1, k0, c10);
        c11 = _mm256_fmadd_ps(a1, k1, c11);
        c12 = _mm256_fmadd_ps(a1, k2, c12);
        c13 = _mm256_fmadd_ps(a1, k3, c13);
        __m256 a2 = _mm256_loadu_ps(A + lda * (i + 2) + l);
        c20 = _mm256_fmadd_ps(a2, k0, c20);
        c21 = _mm256_fmadd_ps(a2, k1, c21);
        c22 = _mm256_fmadd_ps(a2, k2, c22);
        c23 = _mm256_fmadd_ps(a2, k3, c23);
      }
      C[ldc * (j + 0) + (i + 0)] = hsum(c00);
      C[ldc * (j + 1) + (i + 0)] = hsum(c01);
      C[ldc * (j + 2) + (i + 0)] = hsum(c02);
      C[ldc * (j + 3) + (i + 0)] = hsum(c03);
      C[ldc * (j + 0) + (i + 1)] = hsum(c10);
      C[ldc * (j + 1) + (i + 1)] = hsum(c11);
      C[ldc * (j + 2) + (i + 1)] = hsum(c12);
      C[ldc * (j + 3) + (i + 1)] = hsum(c13);
      C[ldc * (j + 0) + (i + 2)] = hsum(c20);
      C[ldc * (j + 1) + (i + 2)] = hsum(c21);
      C[ldc * (j + 2) + (i + 2)] = hsum(c22);
      C[ldc * (j + 3) + (i + 2)] = hsum(c23);
    }
}
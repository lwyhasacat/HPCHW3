// Jimmy Zhu helped me how to do this

#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <cassert>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
//   int p = omp_get_num_threads();
//   int t = omp_get_thread_num();
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel

    int p = 50;
    long size = ceil(n/double(p));

    #pragma omp parallel for num_threads(p)
    for(int i = 0; i < p; i++){
        long index = i * size;
        prefix_sum[index] = A[index];
        for(int j = index+1; j < std::min(index + size, n); j++){
            prefix_sum[j] = prefix_sum[j-1] + A[j];
        }
    }

    // for computing the offset
    long* offset = (long*) malloc((p-1) * sizeof(long));
    offset[0] = 0;
    for(int i = 1; i < p; i++){
        offset[i] = prefix_sum[i*size-1] + offset[i-1];
    }

    // update the result
    for(int i = 1; i < p; i++){
        long index = i * size;
        #pragma omp parallel for num_threads(p)
        for(int j = index; j < std::min(index + size, n); j++){
            prefix_sum[j] += offset[i];
        }
    }

    free(offset);
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;
  for (long i = 0; i < N; i++) B0[i] = 0;
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
//   for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  for (long i = 0; i < N; i++) {
      err = std::max(err, std::abs(B0[i] - B1[i]));
  }
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
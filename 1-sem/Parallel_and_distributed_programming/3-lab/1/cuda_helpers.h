#pragma once
#include "cuda_runtime.h"

void reduceWithCuda1(dim3 blocks, dim3 threads, int threadSize, int* dev_a, int* dev_c, int size);
void reduceWithCuda2(dim3 blocks, dim3 threads, int threadSize, int* dev_a, int* dev_c, int size);
void reduceWithCuda3(dim3 blocks, dim3 threads, int threadSize, int* dev_a, int* dev_c, int size);
void reduceWithCuda4(dim3 blocks, dim3 threads, int threadSize, int* dev_a, int* dev_c, int size);

cudaError_t reduceWithCuda(int* c, const int* a, unsigned int size, int threadSize, int blockSize, void(*kernel) (dim3, dim3, int, int*, int*, int));
cudaError_t histogramWithCuda(int* c, const int* a, unsigned int size, int threadSize);
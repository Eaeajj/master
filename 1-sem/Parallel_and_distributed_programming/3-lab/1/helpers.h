#pragma once
#include "cuda_runtime.h"

void printArray(int* a, int size);

void checkError(cudaError_t cudaStatus, char* msg, int* dev_a, int* dev_c);

void test_min(int result, int* array, int arraySize);
void test_histogram(int* result, int* array, int arraySize, bool trace);
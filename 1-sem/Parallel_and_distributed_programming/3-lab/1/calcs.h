#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void reduce1(int* inData, int* outData, int arraySize);
__global__ void reduce2(int* inData, int* outData, int arraySize);
__global__ void reduce3(int* inData, int* outData, int arraySize);
__global__ void reduce4(int* inData, int* outData, int arraySize);

__global__ void histogram(const int* in, int size, int NUM_BINS, int* out);
__global__ void histogramMerge(const int* in, int NUM_BINS, int NUM_PARTS, int* out);
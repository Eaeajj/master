#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helpers.h"
#include "calcs.h"
#include <crt/math_functions.hpp>
#include "cuda_helpers.h"
#include <stdio.h>

void reduceWithCuda1(dim3 blocks, dim3 threads, int threadSize, int* dev_a, int* dev_c, int size) {
	reduce1 <<<blocks, threads, threadSize * sizeof(int) >> > (dev_a, dev_c, size);
}

void reduceWithCuda2(dim3 blocks, dim3 threads, int threadSize, int* dev_a, int* dev_c, int size) {
	reduce2 <<<blocks, threads, threadSize * sizeof(int) >> > (dev_a, dev_c, size);
}

void reduceWithCuda3(dim3 blocks, dim3 threads, int threadSize, int* dev_a, int* dev_c, int size) {
	reduce3 <<<blocks, threads, threadSize * sizeof(int) >> > (dev_a, dev_c, size);
}

void reduceWithCuda4(dim3 blocks, dim3 threads, int threadSize, int* dev_a, int* dev_c, int size) {
	reduce4 <<<blocks, threads, threadSize * sizeof(int) >> > (dev_a, dev_c, size);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t reduceWithCuda(int* c, const int* a, unsigned int size, int threadSize, int blockSize, void(*kernel) (dim3, dim3, int, int*, int*, int))
{
	cudaDeviceProp	devProp;
	cudaGetDeviceProperties(&devProp, 0);
	int* dev_a = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkError(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", dev_a, dev_c);

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	checkError(cudaStatus, "cudaMemcpy failed!", dev_a, dev_c);

	cudaStatus = cudaMalloc((void**)&dev_c, blockSize * sizeof(int));
	checkError(cudaStatus, "cudaMemcpy failed!", dev_c, dev_c);

	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus, "cudaMemcpy failed!", dev_a, dev_c);

	// Launch a kernel on the GPU with one thread for each element.
	cudaEvent_t start, stop;		//описываем переменные типа  cudaEvent_t 
	float       gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	dim3 threads(threadSize, 1, 1);
	dim3 blocks(blockSize, 1, 1);
	kernel(blocks, threads, threadSize, dev_a, dev_c, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkError(cudaStatus, "reductionMin4 launch failed: %s\n", dev_a, dev_c);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkError(cudaStatus, "cudaDeviceSynchronize returned error code %d after launching reductionMin4!\n", dev_a, dev_c);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("GPU time: %.5f ms\n", gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, blockSize * sizeof(int), cudaMemcpyDeviceToHost);
	checkError(cudaStatus, "cudaMemcpy failed!", dev_a, dev_c);

	for (size_t i = 1; i < blockSize; i++) {
		if (c[i] < c[0]) {
			c[0] = c[i];
		}
	}
	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t histogramWithCuda(int* c, const int* a, unsigned int size, int threadSize)
{
	int NUM_BINS = 256;
	int PART_SIZE = size < 300000 ? 3000 : 10000;
	int NUM_PARTS = ceil(double(size) / PART_SIZE);
	int* input = 0;
	int* dev_c = 0;
	int* result = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkError(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", input, dev_c);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, 256 * NUM_PARTS * sizeof(int));
	checkError(cudaStatus, "cudaMalloc failed!", input, dev_c);

	cudaStatus = cudaMalloc((void**)&result, 256 * sizeof(int));
	checkError(cudaStatus, "cudaMalloc failed!", input, dev_c);

	cudaStatus = cudaMalloc((void**)&input, size * sizeof(int));
	checkError(cudaStatus, "cudaMalloc failed!", input, dev_c);

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(input, a, size * sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus, "cudaMemcpy failed!", input, dev_c);

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	histogram<< <NUM_PARTS, threadSize, NUM_BINS * sizeof(int) >> > (input, size, NUM_BINS, dev_c);
	cudaStatus = cudaGetLastError();
	checkError(cudaStatus, "histogramSmemAtomics launch failed: %s\n", input, dev_c);

	histogramMerge<< <2, NUM_BINS >> > (dev_c, NUM_BINS, NUM_PARTS, result);
	cudaStatus = cudaGetLastError();
	checkError(cudaStatus, "histogramFinalAccum launch failed: %s\n", input, dev_c);

	cudaStatus = cudaDeviceSynchronize();
	checkError(cudaStatus, "cudaDeviceSynchronize returned error code %d after launching multiplyKernel!\n", input, dev_c);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("GPU time: %.5f ms\n", gpuTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, result, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	checkError(cudaStatus, "cudaMemcpy failed!", input, dev_c);

	return cudaStatus;
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <crt/math_functions.hpp>

#include "cuda_helpers.h"
#include "helpers.h"

#include <random>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

void task_1() {
	int arraySize = 1024;
	int threadSize = 256;
	int blockSize = ceil(arraySize / threadSize);
	int blockSize2 = ceil(arraySize / threadSize / 2);
	int* a = (int*)malloc(arraySize * sizeof(int));
	int* c = (int*)malloc(blockSize * sizeof(int));
	for (size_t i = 0; i < arraySize; i++){
		a[i] = rand() * (rand() % 2 == 0 ? -1 : 1);
	}

	cudaError_t cudaStatus = reduceWithCuda(c, a, arraySize, threadSize, blockSize, reduceWithCuda1);
	test_min(c[0], a, arraySize);
	c = (int*)malloc(blockSize * sizeof(int));

	cudaStatus = reduceWithCuda(c, a, arraySize, threadSize, blockSize, reduceWithCuda2);
	test_min(c[0], a, arraySize);
	c = (int*)malloc(blockSize * sizeof(int));

	cudaStatus = reduceWithCuda(c, a, arraySize, threadSize, blockSize, reduceWithCuda3);
	test_min(c[0], a, arraySize);
	c = (int*)malloc(blockSize * sizeof(int));

	cudaStatus = reduceWithCuda(c, a, arraySize, threadSize, blockSize2, reduceWithCuda4);
	test_min(c[0], a, arraySize);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

void task_2() {
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(128.0, 128.0);

	// draw a sample from the normal distribution and round it to an integer
	//auto random_int = [&d, &gen] { return std::round(d(gen)); };
	

	int arraySize = 1024;
	int threadSize = 256;
	int* a = (int*)malloc(arraySize * sizeof(int));
	int* c = (int*)malloc(256 * sizeof(int));
	for (size_t i = 0; i < arraySize; i++) {
		float number = distribution(generator);
		while ((number <= 0.0) || (number > 256.0)) {
			number = distribution(generator);
		}
	
		a[i] = int(number);
	}

	// Add vectors in parallel.6
	cudaError_t cudaStatus = histogramWithCuda(c, a, arraySize, threadSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "histogramWithCuda failed!");
	}

	test_histogram(c, a, arraySize, true);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

int main()
{
	printf("task_1\n");
	task_1();
	printf("\ntask_2\n");
	task_2();
	return 0;
}
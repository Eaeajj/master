#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "calcs.h"

using namespace std;

__global__ void reduce1(int* inData, int* outData, int arraySize) {
	extern __shared__ int data[];
	int tx = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= arraySize) {
		data[i] = INT_MAX;
	}
	else {
		data[tx] = inData[i]; 	// load into shared memory 
	}
	__syncthreads();
	for (int s = 1; s < blockDim.x; s *= 2) {
		if (tx % (2 * s) == 0) {
			data[tx] = min(data[tx], data[tx + s]);
		}
		__syncthreads();
	}
	if (tx == 0) {
		outData[blockIdx.x] = data[0];
	}
}

__global__ void reduce2(int* inData, int* outData, int arraySize){
	extern __shared__ int data[];
	int tx = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= arraySize) {
		return;
	}

	data[tx] = inData[i]; 	// load into shared memory 
	__syncthreads();
	for (int s = 1; s < blockDim.x; s <<= 1){
		int index = 2 * s * tx;
		if (index < blockDim.x) {
			data[index] = min(data[index], data[index + s]);
		}
		__syncthreads();
	}
	if (tx == 0) {
		outData[blockIdx.x] = data[0];
	}
}

__global__ void reduce3(int* inData, int* outData, int arraySize){
	extern __shared__ int data[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= arraySize)
		return;

	data[tid] = inData[i];
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			data[tid] = min(data[tid], data[tid + s]);
		__syncthreads();
	}
	if (tid == 0)
		outData[blockIdx.x] = data[0];
}

__global__ void reduce4(int* inData, int* outData, int arraySize) {
	extern __shared__ int data[];
	int tx = threadIdx.x;
	int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (i + blockDim.x < arraySize) {
		data[tx] = min(inData[i], inData[i + blockDim.x]);
	}
	else {
		data[tx] = inData[i];
	}
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tx < s) {
			int v = min(data[tx], data[tx + s]);
			data[tx] = v;
		}
		__syncthreads();
	}
	if (tx == 0) {
		outData[blockIdx.x] = data[0];
	}
}

__global__ void histogram(const int* in, int size, int NUM_BINS, int* out){
	// pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	// step in block
	int nx = blockDim.x * gridDim.x;

	// linear thread index within block
	int t = threadIdx.x;

	// total threads in block
	int nt = blockDim.x;

	// block index
	int bx = blockIdx.x;

	// initialize temporary accumulation array in shared memory
	extern __shared__ unsigned int smem[];
	for (int i = t; i < NUM_BINS; i += nt) {
		smem[i] = 0;
	}

	__syncthreads();
	// updates our block's partial histogram in shared memory
	for (int col = x; col < size; col += nx){
		unsigned int r = (unsigned int)(in[col]);
		//printf("%d: %d\n", col, in[col]);
		atomicAdd(&smem[r], 1);
	}
	__syncthreads();

	// write partial histogram into the global memory
	out += bx * NUM_BINS;
	for (int i = t; i < NUM_BINS; i += nt) {
		out[i] = smem[i];
	}
}

__global__ void histogramMerge(const int* in, int NUM_BINS, int NUM_PARTS, int* out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUM_BINS) {
		unsigned int total = 0;
		for (int j = 0; j < NUM_PARTS; j++) {
			total += in[i + NUM_BINS * j];
		}
		out[i] = total;
	}
}
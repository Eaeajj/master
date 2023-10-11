#include <math.h>
#include <iostream>
#include "kernel.h"
#include <stdlib.h>

using namespace std;


__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, size_t N) {
    size_t tx = threadIdx.x;
    size_t rowIdx = (tx + blockIdx.x * blockDim.x) / N;
    size_t colIdx = (tx + blockIdx.x * blockDim.x) % N;
    
    float sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += A[rowIdx + i] * B[colIdx + i * N]; 
    }

    size_t resIdx = colIdx * N + rowIdx;
    C[resIdx] = sum;
}

void matrixMultiplication(float *A, float *B, float *C, size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
    
    
    cudaEventRecord( start, 0 );
    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &gpuTime, start, stop );
        
    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime );
}

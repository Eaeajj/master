#include <iostream>
#include "calcs.h"
#include <stdlib.h>
#include <vector>
#include <chrono>

using namespace std;


// ----------------------------- kernel funcs ---------------------------------------
__global__ void kernelMatrixTranspose(float* inData, float* outData, size_t n) {
    size_t xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    size_t yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    size_t inIndex = xIndex + n * yIndex;
    size_t outIndex = yIndex + n * xIndex;
    const size_t maxIndex = n * n;
    if (inIndex > maxIndex || outIndex > maxIndex) {
        printf("panic!");
    }

    outData[outIndex] = inData[inIndex];
}

__global__ void kernelMultiplyVectors(float *A, float *B, float *C, size_t n) {
    const size_t idx = threadIdx.x + gridDim.x * blockIdx.x;

    C[idx] = A[idx] * B[idx];
}

__global__ void kernelDeoptimizedMultiplyVectors(float *A, float *B, float *C, size_t n) {
    size_t idx = threadIdx.x + gridDim.x * blockIdx.x;

    if (idx % 2 == 0) { idx++; }
    else { idx--; }

    C[idx] = A[idx] * B[idx];
}

// ----------------------------- kernel funcs wrappers---------------------------------------
// hiding cuda syntax to wrap it with timeSpentFuncs wrappers
void matrixTransposeGPU(float *A, float *B, size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    kernelMatrixTranspose<<<blocksPerGrid,threadsPerBlock>>>(A, B, N);
}

void kernelMultiplyVectorsGPU(float *A, float *B, float *C, size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    kernelMultiplyVectors<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}

void kernelDeoptimizedMultiplyVectorsGPU(float *A, float *B, float *C, size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    kernelDeoptimizedMultiplyVectors<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}

// ------------------------------ analogous CPU operations ----------------------------------------

vector<float> matrixTransposeCPU(vector<float> &from, size_t N) {
    vector<float> to(N * N);

    for (size_t row=0; row < N; row++){
        for (size_t col=0; col < N; col++){
                to[N * col + row] = from[row * N + col];
        }
    }

    return to;
}

vector<float> multiplyVectorsCPU(vector<float> a, vector<float> b, size_t n) {
    vector<float> C(n);
    
    for (size_t i = 0; i < n; i++) {
        C[i] = a[i] * b[i];
    }
    
   return C;
}

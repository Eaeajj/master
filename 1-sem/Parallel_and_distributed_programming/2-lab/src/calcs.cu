#include <iostream>
#include "calcs.h"
#include <stdlib.h>
#include <vector>
#include <chrono>

using namespace std;


// ----------------------------- kernel funcs ---------------------------------------
__global__ void mul_vector_with_conflicts(vec3* a, vec3* b, vec3* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    c[i].x = a[i].x * b[i].x;
    c[i].y = a[i].y * b[i].y;
    c[i].z = a[i].z * b[i].z;
}

__global__ void mul_vector(vec3_aligned* a, vec3_aligned* b, float* res21, float* res22, float* res23) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    res21[i] = a[i].x * b[i].x;
    res22[i] = a[i].y * b[i].y;
    res23[i] = a[i].z * b[i].z;
}

__global__ void mul_mat(float* a, float* b, float* c, size_t n) {
    int bx = blockIdx.x;        // block index
    int by = blockIdx.y;

    int tx = threadIdx.x;       // thread index
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * n;
    float sum = 0.0f;           // computed subelement

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        // Shared memory for the sub-matrix of A
        __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
        // Shared memory for the sub-matrix of B
        __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from global memory to shared memory;
        as[ty][tx] = a[ia + n * ty + tx];
        bs[ty][tx] = b[ib + n * ty + tx];

        __syncthreads();    // Synchronize to make sure the matrices are loaded

        // muliply the two matrices together;
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += as[ty][k] * bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory;
    // each thread writes one element
    int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    c[ic + n * ty + tx] = sum;
}

// ----------------------------- kernel funcs wrappers---------------------------------------
// hiding cuda dsl syntax to wrap it with timeSpentFuncs wrappers
void mul_vector_with_conflicts_CUDA_GPU(dim3 threadsPerBlock, dim3 blocksPerGrid, vec3* a, vec3* b, vec3* c) {
    mul_vector_with_conflicts <<<blocksPerGrid, threadsPerBlock>>> (a, b, c);
}

void mul_vector_CUDA_GPU(dim3 threadsPerBlock, dim3 blocksPerGrid, vec3_aligned* a, vec3_aligned* b, float* res1, float* res2, float* res3) {
    mul_vector <<<blocksPerGrid, threadsPerBlock>>> (a, b, res1, res2, res3);
}

void mul_mat_CUDA_GPU(dim3 threadsPerBlock, dim3 blocksPerGrid, float* a, float* b, float* res, size_t n) {
    mul_mat << <blocksPerGrid, threadsPerBlock >> > (a, b, res, n);
}

// ------------------------------ analogous CPU operations ----------------------------------------

void mul_mat(vector<float> a, vector<float> b, vector<float> c, size_t N) {

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0;
            for (size_t k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + j];
            }

            c[i * N + j] = sum;
        }
    }

}

void mul_vector(vector<vec3> a, vector<vec3> b, vector<vec3> c) {
    size_t size = c.size();

    for (size_t i = 0; i < size; i++) {
        c[i].x = a[i].x * b[i].x;
        c[i].y = a[i].y * b[i].y;
        c[i].z = a[i].z * b[i].z;
    }
    
}

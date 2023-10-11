#ifndef KERNEL_CUH_
#define KERNEL_CUH_

void matrixMultiplication(float *A, float *B, float *C, size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid);

#endif
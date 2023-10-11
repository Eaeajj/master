# Lab №1

## Fundamentals of working with CUDA technology. Hybrid programming. Working with global memory

Matrices A and B of NxN natural (non-zero) elements (set randomly) are given. The matrices are located in global memory.
Write a program that performs multiplication of two matrices on GPU.

I wanted to write general function to solve matrix multiplication operation via cuda. But I understood that it is hard to generalize it without many "if" statements. So I decided to place here my 3 solutions.

---

1. Started from the simplest way where: 
* N between [1, 32] -> SIZE = NxN
* dimBlocks(SIZE, 1, 1)
* dimGrids(1,1,1)

kernel func is

```
__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, size_t N) {
    size_t tx = threadIdx.x;
    size_t rowIdx = tx / N;
    size_t colIdx = tx % N;
    
    float sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += A[rowIdx + i] * B[colIdx + i * N]; 
    }

    size_t resIdx = colIdx * N + rowIdx;
    C[resIdx] = sum;
}
```

---

2. Adding blocks to multiply matrices that over 1024 elements in total
* B (Blocks X dim) = [1, 32]               
* N between [1, 32] -> SIZE = NxN          
* P (amount of threads to 1 block) <= 1024
* B * P == N x N 
* dimBlocks(P, 1, 1)
* dimGrids(B, 1, 1)

kernel func is
```
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
```
---

3. Adding second dimension to the Block in first solution
* N between [1, 32] -> N * N = SIZE
* T1 between [1, 32]
* T2 between [1, 32] 
* T1 * T2 == SIZE
* dimBlocks(T1, T2, 1)
* dimGrids(1,1,1)

kernel func is
    
```
__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, size_t N) {
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    
    size_t rowIdx;
    size_t colIdx;

    if (blockDim.x < blockDim.y) {
        rowIdx = (ty * blockDim.x + tx) / N;
        colIdx = (ty * blockDim.x + tx) % N;
    }
    
    if (blockDim.x >= blockDim.y) {
        rowIdx = (tx + ty * blockDim.x) / N;
        colIdx = tx % N;
    }
    

    float sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += A[rowIdx + i] * B[colIdx + i * N]; 
    }

    size_t resIdx = colIdx * N + rowIdx;
    C[resIdx] = sum;
}
```
### Resources:
* [A Simple Makefile Tutorial](https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/)
* [Программирование на CUDA C. Часть 3.](https://cc.dvfu.ru/ru/lesson-7/)
* [Matrix-Matrix Multiplication on the GPU with Nvidia CUDA](https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/)

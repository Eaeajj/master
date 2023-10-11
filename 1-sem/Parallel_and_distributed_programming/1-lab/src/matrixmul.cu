#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "kernel.h"
#include "helpers.cu"
#include "dev_array.h"
#include <math.h>

using namespace std;


void execMatrixComputing(size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    size_t SIZE = N*N;

    // Allocate memory on the host
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);
    
    // Initialize matrices on the host
    for (size_t i=0; i<N; i++){
        for (size_t j=0; j<N; j++){
            h_A[i*N+j] = 1;
            h_B[i*N+j] = 2;
        }
    }

    // Allocate memory on the device
    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);
    dev_array<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N, threadsPerBlock, blocksPerGrid);

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    vector<float> cpu_C = matrixMultiplicationCpu(h_A, h_B, N);

    double err = 0;
    // Check the result and make sure it is correct
    for (size_t ROW=0; ROW < N; ROW++){
        for (size_t COL=0; COL < N; COL++) {
            err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
        }
    }

    cout << "Error: " << err << endl;

    // printVec(h_C, N);
    // cout << "\n --- \n";
    // printVec(cpu_C, N);
}



int main() {

    const size_t N1 = 300;
    dim3 threadsPerBlock1(N1, 1, 1);
    dim3 blocksPerGrid1(N1, 1, 1);

    execMatrixComputing(N1, threadsPerBlock1, blocksPerGrid1);

    return 0;
}


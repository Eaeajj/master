#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "calcs.h"
#include "helpers.cu"
#include "dev_array.h"
#include <math.h>

using namespace std;


#include <iostream>
#include <chrono>
#include <functional>



// Function you want to measure
int myFunction(int param1, float param2) {
    return 5;
}

void execMatrixTransposing(size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    size_t SIZE = N*N;

    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);

    for (size_t i=0; i<N; i++){
        for (size_t j=0; j<N; j++){
            h_A[i*N+j] = rand() % 10;
        }
    }

    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    auto cpuTransposedVec = timeSpentWrapperCPU(matrixTransposeCPU, h_A, N);
    timeSpentWrapperGPU(matrixTransposeGPU, d_A.getData(), d_B.getData(), N, threadsPerBlock, blocksPerGrid);
    
    d_B.get(&h_B[0], SIZE);

    vector<size_t> errorIndexes(0);

    // Check the result for correctness
    for (size_t row = 0; row < N; row++){
        for (size_t col=0; col < N; col++) {
            size_t idx = N * col + row;
            if (h_B[idx] != cpuTransposedVec[idx]) {
                errorIndexes.push_back(idx);
            }
        }
    }
    if (errorIndexes.size() > 0) {
        cout << "Errors amount: " << errorIndexes.size() << "elements aren't placed correctly" << endl;

    }
    
    // // needs to debug
    // printVec(h_A, N);
    // cout << "\n --- \n";
    // printVec(h_B, N);
    // cout << "\n --- \n";
    // printVec(cpuTransposedVec, N);
}



int main() {
    const size_t N1 = 2048;
    dim3 threadsPerBlock1(1, 1, 1);
    dim3 blocksPerGrid1(N1, N1, 1);

    execMatrixTransposing(N1, threadsPerBlock1, blocksPerGrid1);

    return 0;
}


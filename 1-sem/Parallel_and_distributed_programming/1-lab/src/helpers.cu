#include <vector>
#include <iostream>
#include <chrono>

using namespace std;

void printVec(vector<float> vec, size_t n) {
    size_t len = vec.size();

    for (size_t i = 0; i < len; i++) {
        cout << vec[i] << " ";
        if ((i + 1) % n == 0) {
            cout << "\n";
        } 
    }
}

template <typename T>
void printArray(T *arr, size_t n) {
    size_t len = n * n;
    
    for (size_t i = 0; i < len; i++) {
        cout << arr[i] << " ";
        if (i + 1 % n == 0) {
            cout << "\n";
        } 
    }
    cout << "\n";
}

vector<float> matrixMultiplicationCpu(vector<float> A, vector<float> B, size_t N) {
    vector<float> matrix(N * N);

    auto start = std::chrono::high_resolution_clock::now();

    float sum;
    for (size_t row=0; row<N; row++){
        for (size_t col=0; col<N; col++){
            sum = 0.f;
            for (size_t n=0; n<N; n++){
                sum += A[row*N+n] * B[n*N+col];
            }
            matrix[row*N+col] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cout << "time spent executing by the CPU: " << duration / 1000 << " millseconds\n";

    return matrix;

}
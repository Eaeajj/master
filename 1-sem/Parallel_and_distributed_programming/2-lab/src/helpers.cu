#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include "calcs.h"

using namespace std;


// std::random_device rd;
// std::mt19937 gen(rd());
    
//     // Define a distribution for floating-point numbers between 0 and 1
// std::uniform_real_distribution<float> dis(0.0f, 1.0f);

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

template <typename Func, typename... Args>
auto time_spent_wrapper(Func func, Args... args) {
    auto start = std::chrono::high_resolution_clock::now();

    func(args...);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cout << "TASK 1: time spent executing by the CPU: " << duration / 1000 << " milliseconds\n";
}

template <typename Func, typename... Args>
void time_spent_wrapper_GPU(Func func, Args... args) {
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
    
    
    cudaEventRecord( start, 0 );

    func(args...);
    cudaDeviceSynchronize();
    
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &gpuTime, start, stop );
        
    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime );
}

float randomFloat() {

    
    // Generate a random float number
    // return dis(gen);
    return 1.0f;
}
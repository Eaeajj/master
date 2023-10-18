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

template <typename Func, typename... Args>
auto timeSpentWrapperCPU(Func func, Args... args) {
    auto start = std::chrono::high_resolution_clock::now();

    auto res = func(args...);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cout << "time spent executing by the CPU: " << duration << " microseconds | " << duration / 1000 << " milliseconds\n";
    return res;
}

template <typename Func, typename... Args>
void timeSpentWrapperGPU(Func func, Args... args) {
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define	N (32*32)

using namespace std;

__global__ void kernel( float * data ) {
    int   idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x   = 2.0f * 3.1415926f * (float) idx / (float) N;
    data [idx] = sinf ( sqrtf ( x ) );
}

int main(int argc, char *argv[]) {
    float *a = (float*)malloc(N * sizeof(float));
    if (a == NULL) {  __throw_runtime_error("Memory did not allocate on stack");  } 

    float *dev = nullptr;
	// выделить память на GPU
    cudaMalloc ( (void**)&dev, N * sizeof ( float ) );
    
    // конфигурация запуска N нитей
    kernel <<< dim3((N/512),1), dim3(512,1) >>> (dev);
    // скопировать результаты в память CPU
    cudaMemcpy ( a, dev, N * sizeof ( float ), cudaMemcpyDeviceToHost );

    for (int idx = 0; idx < N; idx++) {
        printf("a[%d] = %.5f\n", idx, a[idx]);
    }

    // освободить выделенную память
    cudaFree(dev);
    free(a);

    return 0;
}
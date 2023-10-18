#ifndef CALCS_CUH_
#define CALCS_CUH_
#include <vector>
using namespace std;


void matrixTransposeGPU(float *A, float *B, size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid);
void kernelMultiplyVectorsGPU(float *A, float *B, float *C, size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid);
void kernelDeoptimizedMultiplyVectorsGPU(float *A, float *B, float *C, size_t N, dim3 threadsPerBlock, dim3 blocksPerGrid);
vector<float> matrixTransposeCPU(vector<float> &from, size_t N);
vector<float> multiplyVectorsCPU(vector<float> a, vector<float> b, size_t n);

#endif
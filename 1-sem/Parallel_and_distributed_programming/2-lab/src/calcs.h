#ifndef CALCS_CUH_
#define CALCS_CUH_
#include <vector>
using namespace std;

#define BLOCK_SIZE  16       // submatrix size

struct  vec3 {
    float x;
    float y;
    float z;
};

struct __align__(16) vec3_aligned {
    float x;
    float y;
    float z;
};

void mul_vector_with_conflicts_CUDA_GPU(dim3 threadsPerBlock, dim3 blocksPerGrid, vec3* a, vec3* b, vec3* c);
void mul_vector_CUDA_GPU(dim3 threadsPerBlock, dim3 blocksPerGrid, vec3_aligned* a, vec3_aligned* b, float* res1, float* res2, float* res3);
void mul_mat_CUDA_GPU(dim3 threadsPerBlock, dim3 blocksPerGrid, float* a, float* b, float* res, size_t n);

void mul_vector(vector<vec3> a, vector<vec3> b, vector<vec3> c);
void mul_mat(vector<float> a, vector<float> b, vector<float> c, size_t n);


#endif
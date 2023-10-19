#include <iostream>
#include "dev_array.h"
#include "calcs.h"
#include "helpers.cu"


void task_1_1(size_t n) {
    const size_t VEC_SIZE = n * n;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n / threads.x, n / threads.y);

    vector<vec3> h_a(VEC_SIZE);
    vector<vec3> h_b(VEC_SIZE);
    vector<vec3> h_res(VEC_SIZE);

    for (size_t i = 0; i < VEC_SIZE; i++) {
        vec3 temp1 = { randomFloat(), randomFloat(), randomFloat() };
        vec3 temp2 = { randomFloat(), randomFloat(), randomFloat() };
        h_a[i] = temp1;
        h_b[i] = temp2;
    }

    dev_array<vec3> d_a(VEC_SIZE);
    dev_array<vec3> d_b(VEC_SIZE);
    dev_array<vec3> d_res(VEC_SIZE);

    d_a.set(&h_a[0], n);
    d_b.set(&h_b[0], n);
    d_res.set(&h_res[0], n);

    printf("TASK 1: with global memory conflicts - ");
    time_spent_wrapper_GPU(mul_vector_with_conflicts_CUDA_GPU, threads, blocks, d_b.getData(), d_a.getData(), d_res.getData());
}

void task_1_2(size_t n) {
    const size_t VEC_SIZE = n * n;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n / threads.x, n / threads.y);

    vector<vec3_aligned> h_a(VEC_SIZE);
    vector<vec3_aligned> h_b(VEC_SIZE);

    for (size_t i = 0; i < VEC_SIZE; i++) {
        vec3_aligned temp1 = { randomFloat(), randomFloat(), randomFloat() };
        vec3_aligned temp2 = { randomFloat(), randomFloat(), randomFloat() };
        h_a[i] = temp1;
        h_b[i] = temp2;
    }

    dev_array<vec3_aligned> d_a(VEC_SIZE);
    dev_array<vec3_aligned> d_b(VEC_SIZE);


    vector<float> h_res_x(VEC_SIZE);
    vector<float> h_res_y(VEC_SIZE);
    vector<float> h_res_z(VEC_SIZE);
    
    dev_array<float> d_res_y(VEC_SIZE);
    dev_array<float> d_res_x(VEC_SIZE);
    dev_array<float> d_res_z(VEC_SIZE);

    d_a.set(&h_a[0], n);
    d_b.set(&h_b[0], n);

    d_res_x.set(&h_res_x[0], n);
    d_res_y.set(&h_res_y[0], n);
    d_res_z.set(&h_res_z[0], n);


    printf("TASK 1: without global memory conflicts - ");
    time_spent_wrapper_GPU(mul_vector_CUDA_GPU, threads, blocks, d_a.getData(), d_b.getData(), d_res_x.getData(), d_res_y.getData(), d_res_z.getData());
}

void task_1_cpu(size_t n) {
    vector<vec3> h_a(n*n);
    vector<vec3> h_b(n*n);
    vector<vec3> h_c(n*n);

    for (size_t i = 0; i < n*n; i++) {
        vec3 temp1 = { randomFloat(), randomFloat(), randomFloat() };
        vec3 temp2 = { randomFloat(), randomFloat(), randomFloat() };
        h_a[i] = temp1;
        h_b[i] = temp2;
    }

    time_spent_wrapper(mul_vector, h_a, h_b, h_c);

}

void task_1() {
    task_1_1(32); 
    cout << "\t ----------- warmed up -----------\t\n\n";
    const size_t n = 1024;
    
    cout << "VEC size = NxN\n";
    cout << "\nN is " << n << '\n';
    task_1_1(n);
    task_1_2(n);
    task_1_cpu(n);

    cout << "\nN is " << n * 4 << '\n';
    task_1_1(n * 4);
    task_1_2(n * 4);
    task_1_cpu(n * 4);

    cout << "\nN is " << n * 8 << '\n';
    task_1_1(n * 8);
    task_1_2(n * 8);
    task_1_cpu(n * 8);

    cout << "\nN is " << n * 9 << '\n';
    task_1_1(n * 9);
    task_1_2(n * 9);

    cout << "\nN is " << n * 10 << '\n';
    task_1_1(n * 10);
    task_1_2(n * 10);
}

void task_2_1(size_t n) {
    size_t size = n * n;

    vector<float> h_a(size);
    vector<float> h_b(size);
    vector<float> h_c(size);

    for (size_t i = 0; i < size; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }

    time_spent_wrapper(mul_mat, h_a, h_b, h_c, n);
}

void task_2_2(size_t n) {
    size_t size = n * n;
    vector<float> h_a(size);
    vector<float> h_b(size);
    vector<float> h_c(size);

    for (size_t i = 0; i < size; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }

    dev_array<float> d_a(size);
    dev_array<float> d_b(size);
    dev_array<float> d_c(size);

    d_a.set(&h_a[0], size);
    d_b.set(&h_b[0], size);
    d_c.set(&h_c[0], size);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n / threads.x, n / threads.y);

    time_spent_wrapper_GPU(mul_mat_CUDA_GPU, threads, blocks, d_a.getData(), d_b.getData(), d_c.getData(), n);
}

int main(int argc, char* argv[]) {
    task_2_2(2048);
}



#include <iostream>

#define N 10000
#define N_BLOCK 10

__global__ void simple_vector_add(const float *a, const float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    using namespace std;
    cudaError_t errorId;

    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];

    for (int i = 0; i < N; i++) {
        a[i] = (float) i;
        b[i] = (float) (N - i);
        c[i] = 0.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(float) * N);
    cudaMalloc(&d_b, sizeof(float) * N);
    cudaMalloc(&d_c, sizeof(float) * N);

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    simple_vector_add<<< N_BLOCK, N/N_BLOCK >>>(d_a, d_b, d_c, N);

    errorId = cudaGetLastError();
    if (errorId != cudaSuccess) {
        std::cerr << "failed : " << cudaGetErrorString(errorId) << std::endl;
        exit(-1);
    }

    cudaMemcpy(c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (c[i] != N) {
            std::cerr << "not accurate, c[i] = " << c[i] << std::endl;
            exit(-1);
        }
    }

    std::cout << "accurate" << std::endl;

    return 0;
}
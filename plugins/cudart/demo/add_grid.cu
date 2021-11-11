#include <stdio.h>
#include <iostream>
#include <math.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;
    float *x, *y;

    printf("Allocate Unified Memory -- accessible from CPU or GPU\n");
    gpuErrchk(cudaMallocManaged(&x, N*sizeof(float)));
    gpuErrchk(cudaMallocManaged(&y, N*sizeof(float)));

    printf("Initialize x and y arrays on the host\n");
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    printf("Launch kernel on 1M elements on the GPU\n");
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    printf("Wait for GPU to finish before accessing on host\n");
    gpuErrchk(cudaDeviceSynchronize());

    printf("Check for errors (all values should be 3.0f)\n");
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    printf("Free memory\n");
    gpuErrchk(cudaFree(x));
    gpuErrchk(cudaFree(y));

    return 0;
}
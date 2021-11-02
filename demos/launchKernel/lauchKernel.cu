#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;
    float *hx, *hy, *dx, *dy;
    hx = (float*)malloc(N * sizeof(float));
    hy = (float*)malloc(N * sizeof(float));

    cudaMalloc(&dx, N * sizeof(float));
    cudaMalloc(&dy, N * sizeof(float));

    for (int idx = 0; idx < N; idx++)
    {
        hx[idx] = 1.0f;
        hy[idx] = 2.0f;
    }

    cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, N * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int threads = 256;
    unsigned int blocks = (N + 255) / threads;

    float ratio = 2.0f;

    //saxpy<<<blocks, threads>>>(N, ratio, dx, dy);

    void *args[] = { &N, &ratio, &dx, &dy };
    cudaLaunchKernel((void*)saxpy, dim3(blocks), dim3(threads), args, 0, NULL);

    cudaMemcpy(hy, dy, N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int jdx = 0; jdx < N; jdx++)
    {
        max_error = max(max_error, abs(hy[jdx] - 4.0f));
    }

    printf("Max Error: %f\n", max_error);

    cudaFree(dx);
    cudaFree(dy);
    free(hx);
    free(hy);

    return 0;
}

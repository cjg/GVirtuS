#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__
void dummy0()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
}


int main(void)
{
    int N=1 << 20;
    unsigned int threads = 256;
    unsigned int blocks = (N + 255) / threads;

//    dummy0<<<blocks, threads>>>();


    void *args[] = {};

    printf("cudaLaunchKernel with 0 arguments\n");
    cudaError_t cudaError = cudaLaunchKernel((void*)dummy0, dim3(blocks), dim3(threads), args, 0, NULL);

    printf("cudaError:%d\n",cudaError);
    return 0;
}

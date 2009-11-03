#include <cuda_runtime_api.h>
#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(GetDeviceCount) {
    /* cudaError_t cudaGetDeviceCount(int *count) */

    int *count = input_buffer->Assign<int>();

    cudaError_t exit_code = cudaGetDeviceCount(count);

    return new Result(exit_code, new Buffer(*input_buffer));
}

CUDA_ROUTINE_HANDLER(GetDeviceProperties) {
    /* cudaError_t cudaGetDeviceCount(struct cudaDeviceProp *prop,
       int device) */
    struct cudaDeviceProp *prop = input_buffer->Assign<struct cudaDeviceProp>();
    int device = input_buffer->Get<int>();

    cudaError_t exit_code = cudaGetDeviceProperties(prop, device);

    return new Result(exit_code, new Buffer(*input_buffer));
}

CUDA_ROUTINE_HANDLER(SetDevice) {
    /* cudaError_t cudaSetDevice(int device) */
    int device = input_buffer->Get<int>();

    cudaError_t exit_code = cudaSetDevice(device);

    return new Result(exit_code);
}
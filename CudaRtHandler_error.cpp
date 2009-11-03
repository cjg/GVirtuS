#include <cuda_runtime_api.h>
#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(GetErrorString) {
    /* const char* cudaGetErrorString(cudaError_t error) */
    cudaError_t error = input_buffer->Get<cudaError_t>();
    const char *error_string = cudaGetErrorString(error);
    cout << error_string << endl;
    Buffer * output_buffer = new Buffer();
    output_buffer->AddString(error_string);
    return new Result(cudaSuccess, output_buffer);
}

CUDA_ROUTINE_HANDLER(GetLastError) {
    /* cudaError_t cudaGetLastError(void) */
    return new Result(cudaGetLastError());
}


#include <cuda_runtime_api.h>
#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(ChooseDevice) {
    int *device = input_buffer->Assign<int>();
    const cudaDeviceProp *prop = input_buffer->Assign<cudaDeviceProp>();

    cudaError_t exit_code = cudaChooseDevice(device, prop);

    Buffer *out = new Buffer();
    out->Add(device);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetDevice) {
    int *device = input_buffer->Assign<int>();

    cudaError_t exit_code = cudaGetDevice(device);

    Buffer *out = new Buffer();
    out->Add(device);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetDeviceCount) {
    int *count = input_buffer->Assign<int>();

    cudaError_t exit_code = cudaGetDeviceCount(count);

    Buffer *out = new Buffer();
    out->Add(count);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetDeviceProperties) {
    struct cudaDeviceProp *prop = input_buffer->Assign<struct cudaDeviceProp>();
    int device = input_buffer->Get<int>();

    cudaError_t exit_code = cudaGetDeviceProperties(prop, device);
#if 0 
// FIXME: this should be conditioned on cuda version
    prop->canMapHostMemory = 0;
#endif

    Buffer *out = new Buffer();
    out->Add(prop);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(SetDevice) {
    int device = input_buffer->Get<int>();

    cudaError_t exit_code = cudaSetDevice(device);

    return new Result(exit_code);
}

#if 0 
// FIXME: this should be conditioned on cuda version
CUDA_ROUTINE_HANDLER(SetDeviceFlags) {
    int flags = input_buffer->Get<int>();

    cudaError_t exit_code = cudaSetDeviceFlags(flags);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(SetValidDevices) {
    int len = input_buffer->BackGet<int>();
    int *device_arr = input_buffer->Assign<int>(len);

    cudaError_t exit_code = cudaSetValidDevices(device_arr, len);

    Buffer *out = new Buffer();
    out->Add(device_arr, len);

    return new Result(exit_code, out);
}
#endif


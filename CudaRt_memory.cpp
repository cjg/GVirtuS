#include <cstring>
#include "Frontend.h"
#include "CudaRt.h"

extern cudaError_t cudaFree(void *devPtr) {
    char *in_buffer, *out_buffer;
    size_t in_buffer_size, out_buffer_size;
    cudaError_t result;

    in_buffer_size = CudaRt::DevicePointerSize;
    in_buffer = CudaRt::MarshalDevicePointer(devPtr);

    result = Frontend::GetFrontend().Execute("cudaFree",
            in_buffer, in_buffer_size, &out_buffer, &out_buffer_size);

    delete[] in_buffer;

    return result;
}

extern cudaError_t cudaMalloc(void **devPtr, size_t size) {
    char *in_buffer, *out_buffer;
    size_t in_buffer_size, out_buffer_size;
    cudaError_t result;

    in_buffer_size = CudaRt::DevicePointerSize + sizeof(size_t);
    in_buffer = new char[in_buffer_size];

    *devPtr = new char[1];
    char *marshal = CudaRt::MarshalDevicePointer(*devPtr);
    memmove(in_buffer, marshal, CudaRt::DevicePointerSize);
    delete[] marshal;
    memmove(in_buffer + CudaRt::DevicePointerSize, &size, sizeof(size_t));

    result = Frontend::GetFrontend().Execute("cudaMalloc",
            in_buffer, in_buffer_size, &out_buffer, &out_buffer_size);

    if (result == cudaSuccess) {
        delete[] out_buffer;
    }

    delete[] in_buffer;

    return result;
}


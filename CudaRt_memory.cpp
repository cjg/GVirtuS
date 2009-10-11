#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaFree(void *devPtr) {
    char *in_buffer, *out_buffer;
    size_t in_buffer_size, out_buffer_size;
    cudaError_t result;

    in_buffer_size = CudaUtil::MarshaledDevicePointerSize;
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

    in_buffer_size = CudaUtil::MarshaledDevicePointerSize + sizeof(size_t);
    in_buffer = new char[in_buffer_size];

    *devPtr = new char[1];
    char *marshal = CudaRt::MarshalDevicePointer(*devPtr);
    memmove(in_buffer, marshal, CudaUtil::MarshaledDevicePointerSize);
    delete[] marshal;
    memmove(in_buffer + CudaUtil::MarshaledDevicePointerSize, &size, sizeof(size_t));

    result = Frontend::GetFrontend().Execute("cudaMalloc",
            in_buffer, in_buffer_size, &out_buffer, &out_buffer_size);

    if (result == cudaSuccess) {
        delete[] out_buffer;
    }

    delete[] in_buffer;

    return result;
}

extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
    char *in_buffer, *out_buffer;
    size_t in_buffer_size, out_buffer_size;
    cudaError_t result;

    switch(kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead here */
            if(memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            in_buffer_size = CudaUtil::MarshaledDevicePointerSize + count + sizeof(size_t)
                    + sizeof(cudaMemcpyKind);
            in_buffer = new char[in_buffer_size];
            CudaRt::MarshalDevicePointer(dst, in_buffer);
            memmove(in_buffer + CudaUtil::MarshaledDevicePointerSize, (char *) src, count);
            memmove(in_buffer + CudaUtil::MarshaledDevicePointerSize + count, &count,
                    sizeof(size_t));
            memmove(in_buffer + CudaUtil::MarshaledDevicePointerSize + count
                    + sizeof(size_t), &kind, sizeof(cudaMemcpyKind));
            result = Frontend::GetFrontend().Execute("cudaMemcpy", in_buffer,
                    in_buffer_size, &out_buffer, &out_buffer_size);
            delete[] in_buffer;
            return result;
            break;
        case cudaMemcpyDeviceToHost:
            in_buffer_size = CudaUtil::MarshaledDevicePointerSize + sizeof(size_t)
                    + sizeof(cudaMemcpyKind);
            in_buffer = new char[in_buffer_size];
            CudaRt::MarshalDevicePointer(const_cast<void *>(src), in_buffer);
            memmove(in_buffer + CudaUtil::MarshaledDevicePointerSize, &count,
                    sizeof(size_t));
            memmove(in_buffer + CudaUtil::MarshaledDevicePointerSize
                    + sizeof(size_t), &kind, sizeof(cudaMemcpyKind));
            result = Frontend::GetFrontend().Execute("cudaMemcpy", in_buffer,
                    in_buffer_size, &out_buffer, &out_buffer_size);
            delete[] in_buffer;
            if(result == cudaSuccess) {
                memmove(dst, out_buffer, out_buffer_size);
                delete[] out_buffer;
            }
            return result;
            break;
        case cudaMemcpyDeviceToDevice:
            in_buffer_size = CudaUtil::MarshaledDevicePointerSize * 2 + sizeof(size_t)
                    + sizeof(cudaMemcpyKind);
            in_buffer = new char[in_buffer_size];
            CudaRt::MarshalDevicePointer(dst, in_buffer);
            CudaRt::MarshalDevicePointer(const_cast<void *>(src), in_buffer
                + CudaUtil::MarshaledDevicePointerSize);
            memmove(in_buffer + CudaUtil::MarshaledDevicePointerSize * 2, &count,
                    sizeof(size_t));
            memmove(in_buffer + CudaUtil::MarshaledDevicePointerSize * 2
                    + sizeof(size_t), &kind, sizeof(cudaMemcpyKind));
            result = Frontend::GetFrontend().Execute("cudaMemcpy", in_buffer,
                    in_buffer_size, &out_buffer, &out_buffer_size);
            delete[] in_buffer;
            return result;
            break;
    }

    return cudaErrorUnknown;
}

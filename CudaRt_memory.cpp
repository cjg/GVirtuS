#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaFree(void *devPtr) {
    Buffer * input_buffer = new Buffer();

    input_buffer->Add(CudaRt::MarshalDevicePointer(devPtr),
            CudaUtil::MarshaledDevicePointerSize);

    Result * result = Frontend::GetFrontend().Execute("cudaFree", input_buffer);

    delete input_buffer;

    cudaError_t exit_code = result->GetExitCode();

    delete result;

    return exit_code;
}

extern cudaError_t cudaMalloc(void **devPtr, size_t size) {
    Buffer * input_buffer = new Buffer();

    *devPtr = new char[1];
    input_buffer->Add(CudaRt::MarshalDevicePointer(*devPtr),
            CudaUtil::MarshaledDevicePointerSize);
    input_buffer->Add(size);

    Result * result = Frontend::GetFrontend().Execute("cudaMalloc",
            input_buffer);

    delete input_buffer;

    cudaError_t exit_code = result->GetExitCode();

    delete result;

    return exit_code;
}

extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
    Buffer * input_buffer;
    Result * result;
    cudaError_t exit_code;


    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead here */
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            input_buffer = new Buffer();
            input_buffer->Add(CudaRt::MarshalDevicePointer(dst),
                    CudaUtil::MarshaledDevicePointerSize);
            input_buffer->Add<char>(static_cast<char *>(const_cast<void *>(src)), count);
            input_buffer->Add(count);
            input_buffer->Add(kind);

            result = Frontend::GetFrontend().Execute("cudaMemcpy",
                    input_buffer);
            delete input_buffer;

            exit_code = result->GetExitCode();

            delete result;

            return exit_code;
            break;
        case cudaMemcpyDeviceToHost:
            input_buffer = new Buffer();
            input_buffer->Add(dst, count);
            input_buffer->Add(CudaRt::MarshalDevicePointer(dst),
                    CudaUtil::MarshaledDevicePointerSize);
            input_buffer->Add(count);
            input_buffer->Add(kind);

            result = Frontend::GetFrontend().Execute("cudaMemcpy",
                    input_buffer);
            delete input_buffer;

            exit_code = result->GetExitCode();

            if (exit_code == cudaSuccess) {
                Buffer *output_buffer = const_cast<Buffer * >(result->GetOutputBufffer());
                memmove(dst, output_buffer->GetBuffer(),
                        output_buffer->GetBufferSize());
            }

            delete result;

            return exit_code;
            break;
        case cudaMemcpyDeviceToDevice:
            input_buffer = new Buffer();
            input_buffer->Add(CudaRt::MarshalDevicePointer(dst),
                    CudaUtil::MarshaledDevicePointerSize);
            input_buffer->Add(CudaRt::MarshalDevicePointer(const_cast<void *> (src)),
                    CudaUtil::MarshaledDevicePointerSize);
            input_buffer->Add(count);
            input_buffer->Add(kind);

            result = Frontend::GetFrontend().Execute("cudaMemcpy",
                    input_buffer);
            delete input_buffer;

            exit_code = result->GetExitCode();

            delete result;

            return exit_code;
            break;
    }

    return cudaErrorUnknown;
}

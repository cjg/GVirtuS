#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaFree(void *devPtr) {
    CudaRt *c = new CudaRt("cudaFree");
    c->AddDevicePointerForArguments(devPtr);
    c->Execute();
    return CudaRt::Finalize(c);
}

extern cudaError_t cudaMalloc(void **devPtr, size_t size) {
    CudaRt *c = new CudaRt("cudaMalloc");

    /* Fake device pointer */
    *devPtr = new char[1];
    c->AddDevicePointerForArguments(*devPtr);
    c->AddVariableForArguments(size);
    c->Execute();
    return CudaRt::Finalize(c);
}

extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
    CudaRt *c = new CudaRt("cudaMemcpy");
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead here */
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            delete c;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            c->AddDevicePointerForArguments(dst);
            c->AddHostPointerForArguments<char>(static_cast<char *>(const_cast<void *>(src)), count);
            c->AddVariableForArguments(count);
            c->AddVariableForArguments(kind);
            c->Execute();
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            c->AddHostPointerForArguments("");
            c->AddDevicePointerForArguments(src);
            c->AddVariableForArguments(count);
            c->AddVariableForArguments(kind);
            c->Execute();
            if (c->Success())
               memmove(dst, c->GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            c->AddDevicePointerForArguments(dst);
            c->AddDevicePointerForArguments(src);
            c->AddVariableForArguments(count);
            c->AddVariableForArguments(kind);
            c->Execute();
            break;
    }

    return CudaRt::Finalize(c);
}

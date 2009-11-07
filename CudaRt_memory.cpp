#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaFree(void *devPtr) {
    Frontend *f = Frontend::GetFrontend();
    f->AddDevicePointerForArguments(devPtr);
    f->Execute("cudaFree");
    return f->GetExitCode();
}

extern cudaError_t cudaMalloc(void **devPtr, size_t size) {
    Frontend *f = Frontend::GetFrontend();

    /* Fake device pointer */
    *devPtr = new char[1];
    f->AddDevicePointerForArguments(*devPtr);
    f->AddVariableForArguments(size);
    f->Execute("cudaMalloc");
    return f->GetExitCode();
}

extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
    Frontend *f = Frontend::GetFrontend();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead here */
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            f->AddDevicePointerForArguments(dst);
            f->AddHostPointerForArguments<char>(static_cast<char *>(const_cast<void *>(src)), count);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpy");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            f->AddHostPointerForArguments("");
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpy");
            if (f->Success())
               memmove(dst, f->GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            f->AddDevicePointerForArguments(dst);
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpy");
            break;
    }

    return f->GetExitCode();
}

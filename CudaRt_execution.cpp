#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(gridDim);
    f->AddVariableForArguments(blockDim);
    f->AddVariableForArguments(sharedMem);
    f->AddVariableForArguments(stream);
    f->Execute("cudaConfigureCall");
    return f->GetExitCode();
}

extern cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
        const char *func) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(attr);
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(func));
    f->Execute("cudaFuncGetAttributes");
    if(f->Success())
        memmove(attr, f->GetOutputHostPointer<cudaFuncAttributes>(),
                sizeof(cudaFuncAttributes));
    return f->GetExitCode();
}

extern cudaError_t cudaLaunch(const char *entry) {
    Frontend *f = Frontend::GetFrontend();
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(entry));
    f->Execute("cudaLaunch");
    return f->GetExitCode();
}

extern cudaError_t cudaSetDoubleForDevice(double *d) {
    // FIXME: implement
    cerr << "*** Error: cudaSetDoubleForDevice() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaSetDoubleForHost(double *d) {
    // FIXME: implement
    cerr << "*** Error: cudaSetDoubleForHost() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaSetupArgument(const void *arg, size_t size,
        size_t offset) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(static_cast<const char *> (arg), size);
    f->AddVariableForArguments(size);
    f->AddVariableForArguments(offset);
    f->Execute("cudaSetupArgument");
    return f->GetExitCode();
}



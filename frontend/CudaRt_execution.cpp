#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
#if 0
    f->AddVariableForArguments(gridDim);
    f->AddVariableForArguments(blockDim);
    f->AddVariableForArguments(sharedMem);
    f->AddVariableForArguments(stream);
    f->Execute("cudaConfigureCall");
    return f->GetExitCode();
#endif
    Buffer *launch = f->GetLaunchBuffer();
    launch->Reset();
    // CNCL
    launch->Add<int>(0x434e34c);
    launch->Add(gridDim);
    launch->Add(blockDim);
    launch->Add(sharedMem);
    launch->Add(stream);
    return cudaSuccess;
}

#if CUDART_VERSION >= 2030
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
#endif

extern cudaError_t cudaLaunch(const char *entry) {
    Frontend *f = Frontend::GetFrontend();
#if 0
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(entry));
    f->Execute("cudaLaunch");
    return f->GetExitCode();
#endif
    Buffer *launch = f->GetLaunchBuffer();
    // LAUN
    launch->Add<int>(0x4c41554e);
    launch->AddString(CudaUtil::MarshalHostPointer(entry));
    f->Execute("cudaLaunch", launch);
    return f->GetExitCode();
}

extern cudaError_t cudaSetDoubleForDevice(double *d) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(d);
    f->Execute("cudaSetDoubleForDevice");
    if(f->Success())
        *d = *(f->GetOutputHostPointer<double >());
    return f->GetExitCode();
}

extern cudaError_t cudaSetDoubleForHost(double *d) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(d);
    f->Execute("cudaSetDoubleForHost");
    if(f->Success())
        *d = *(f->GetOutputHostPointer<double >());
    return f->GetExitCode();
}

extern cudaError_t cudaSetupArgument(const void *arg, size_t size,
        size_t offset) {
    Frontend *f = Frontend::GetFrontend();
#if 0
    f->AddHostPointerForArguments(static_cast<const char *> (arg), size);
    f->AddVariableForArguments(size);
    f->AddVariableForArguments(offset);
    f->Execute("cudaSetupArgument");
    return f->GetExitCode();
#endif
    Buffer *launch = f->GetLaunchBuffer();
    // STAG
    launch->Add<int>(0x53544147);
    launch->Add<char>(static_cast<char *>(const_cast<void *>(arg)), size);
    launch->Add(size);
    launch->Add(offset);
    return cudaSuccess;
}



#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaChooseDevice(int *device, const cudaDeviceProp *prop) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(device);
    f->AddHostPointerForArguments(prop);
    f->Execute("cudaChooseDevice");
    if(f->Success())
        *device = *(f->GetOutputHostPointer<int>());
    return f->GetExitCode();
}

extern cudaError_t cudaGetDevice(int *device) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(device);
    f->Execute("cudaGetDevice");
    if(f->Success())
        *device = *(f->GetOutputHostPointer<int>());
    return f->GetExitCode();
}

extern cudaError_t cudaGetDeviceCount(int *count) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(count);
    f->Execute("cudaGetDeviceCount");
    if(f->Success())
        *count = *(f->GetOutputHostPointer<int>());
    return f->GetExitCode();
}

extern cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(prop);
    f->AddVariableForArguments(device);
    f->Execute("cudaGetDeviceProperties");
    if(f->Success()) {
        memmove(prop, f->GetOutputHostPointer<cudaDeviceProp>(),
                sizeof(cudaDeviceProp));
#if CUDA_VERSION
        prop->canMapHostMemory = 0;
#endif
    }
    return f->GetExitCode();
}

extern cudaError_t cudaSetDevice(int device) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(device);
    f->Execute("cudaSetDevice");
    return f->GetExitCode();
}

extern cudaError_t cudaSetDeviceFlags(int flags) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(flags);
    f->Execute("cudaSetDeviceFlags");
    return f->GetExitCode();
}

extern cudaError_t cudaSetValidDevices(int *device_arr, int len) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(device_arr, len);
    f->AddVariableForArguments(len);
    f->Execute("cudaSetValidDevices");
    if(f->Success()) {
        int *out_device_arr = f->GetOutputHostPointer<int>();
        memmove(device_arr, out_device_arr, sizeof(int) * len);
    }
    return f->GetExitCode();
}

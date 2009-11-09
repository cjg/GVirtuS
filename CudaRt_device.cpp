#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaChooseDevice(int *device, const cudaDeviceProp *prop) {
    /* FIXME: implement */
    cerr << "*** Error: cudaChooseDevice() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGetDevice(int *device) {
    /* FIXME: implement */
    cerr << "*** Error: cudaGetDevice() not yet implemented!" << endl;
    return cudaErrorUnknown;
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
    if(f->Success())
        *prop = *(f->GetOutputHostPointer<struct cudaDeviceProp>());
    return f->GetExitCode();
}

extern cudaError_t cudaSetDevice(int device) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(device);
    f->Execute("cudaSetDevice");
    return f->GetExitCode();
}

extern cudaError_t cudaSetDeviceFlags(int flags) {
    /* FIXME: implement */
    cerr << "*** Error: cudaSetDeviceFlags() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaSetValidDevices(int *device_arr, int len) {
    /* FIXME: implement */
    cerr << "*** Error: cudaSetValidDevices() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

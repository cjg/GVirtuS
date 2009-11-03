#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
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
    CudaRt * c = new CudaRt("cudaGetDeviceCount");
    c->AddHostPointerForArguments(count);
    c->Execute();
    if(c->Success())
        *count = *(c->GetOutputHostPointer<int>());
    return CudaRt::Finalize(c);
}

extern cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,
    int device) {
    CudaRt *c = new CudaRt("cudaGetDeviceProperties");
    c->AddHostPointerForArguments(prop);
    c->AddVariableForArguments(device);
    c->Execute();
    if(c->Success())
        *prop = *(c->GetOutputHostPointer<struct cudaDeviceProp>());
    return CudaRt::Finalize(c);
}

extern cudaError_t cudaSetDevice(int device) {
    CudaRt *c = new CudaRt("cudaSetDevice");
    c->AddVariableForArguments(device);
    c->Execute();
    return CudaRt::Finalize(c);
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

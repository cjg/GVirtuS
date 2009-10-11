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
    char *in_buffer, *out_buffer;
    size_t in_buffer_size, out_buffer_size;
    cudaError_t result;

    in_buffer_size = sizeof (int);
    in_buffer = new char[in_buffer_size];

    memmove(in_buffer, count, sizeof (int));

    result = Frontend::GetFrontend().Execute("cudaGetDeviceCount",
            in_buffer, in_buffer_size, &out_buffer, &out_buffer_size);

    if (result == cudaSuccess) {
        memmove(count, out_buffer, sizeof (int));
        delete[] out_buffer;
    }

    delete[] in_buffer;

    return result;
}

extern cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,
    int device) {
    char *in_buffer, *out_buffer;
    size_t in_buffer_size, out_buffer_size;
    cudaError_t result;

    in_buffer_size = sizeof (struct cudaDeviceProp) + sizeof (int);
    in_buffer = new char[in_buffer_size];

    memmove(in_buffer, prop, sizeof (struct cudaDeviceProp));
    memmove(in_buffer + sizeof (struct cudaDeviceProp), &device,
            sizeof (int));

    result = Frontend::GetFrontend().Execute("cudaGetDeviceProperties",
            in_buffer, in_buffer_size, &out_buffer, &out_buffer_size);

    if (result == cudaSuccess) {
        memmove(prop, out_buffer, sizeof (struct cudaDeviceProp));
        delete[] out_buffer;
    }

    delete[] in_buffer;

    return result;
}

extern cudaError_t cudaSetDevice(int device) {
    /* FIXME: implement */
    cerr << "*** Error: cudaSetDevice() not yet implemented!" << endl;
    return cudaErrorUnknown;
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

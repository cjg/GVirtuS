#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

cudaError_t cudaThreadSynchronize() {
    char *out_buffer;
    size_t out_buffer_size;
    cudaError_t result;

    result = Frontend::GetFrontend().Execute("cudaThreadSynchronize",
            NULL, 0, &out_buffer, &out_buffer_size);

    return result;
}

cudaError_t cudaThreadExit() {
    /* FIXME: implement */
    cerr << "*** Error: cudaThreadExit() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

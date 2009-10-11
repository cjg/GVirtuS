#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern const char* cudaGetErrorString(cudaError_t error) {
    /* FIXME: implement */
    cerr << "*** Error: cudaGetErrorString() not yet implemented!" << endl;
    return "";
}

extern cudaError_t cudaGetLastError(void) {
    char *out_buffer;
    size_t out_buffer_size;
    cudaError_t result;

    result = Frontend::GetFrontend().Execute("cudaGetLastError",
            NULL, 0, &out_buffer, &out_buffer_size);

    return result;
}

#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    // FIXME: implement
    cerr << "*** Error: cudaStreamCreate() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaStreamDestroy() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaStreamQuery(cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaStreamQuery() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaStreamSynchronize() not yet implemented!" << endl;
    return cudaErrorUnknown;
}
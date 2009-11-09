#include "Frontend.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaEventCreate(cudaEvent_t *event) {
    // FIXME: implement
    cerr << "*** Error: cudaEventCreate() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
    // FIXME: implement
    cerr << "*** Error: cudaEventCreateWithFlags() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaEventDestroy(cudaEvent_t event) {
    // FIXME: implement
    cerr << "*** Error: cudaEventDestroy() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    // FIXME: implement
    cerr << "*** Error: cudaEventElapsedTime() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaEventQuery(cudaEvent_t event) {
    // FIXME: implement
    cerr << "*** Error: cudaEventQuery() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaEventRecord() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    // FIXME: implement
    cerr << "*** Error: cudaEventSynchronize() not yet implemented!" << endl;
    return cudaErrorUnknown;
}
#include "Frontend.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaDriverGetVersion(int *driverVersion) {
    // FIXME: implement
    cerr << "*** Error: cudaDriverGetVersion() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    // FIXME: implement
    cerr << "*** Error: cudaRuntimeGetVersion() not yet implemented!" << endl;
    return cudaErrorUnknown;
}
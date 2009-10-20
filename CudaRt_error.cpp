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
    CudaRt *c = new CudaRt("cudaGetLastError");
    c->Execute();
    return CudaRt::Finalize(c);
}

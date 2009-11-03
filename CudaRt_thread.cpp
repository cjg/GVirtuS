#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

cudaError_t cudaThreadSynchronize() {
    CudaRt * c = new CudaRt("cudaThreadSynchronize");
    c->Execute();
    return CudaRt::Finalize(c);
}

cudaError_t cudaThreadExit() {
    CudaRt * c = new CudaRt("cudaThreadExit");
    c->Execute();
    return CudaRt::Finalize(c);
}

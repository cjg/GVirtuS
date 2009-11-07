#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

cudaError_t cudaThreadSynchronize() {
    Frontend *f = Frontend::GetFrontend();
    f->Execute("cudaThreadSynchronize");
    return f->GetExitCode();
}

cudaError_t cudaThreadExit() {
    Frontend *f = Frontend::GetFrontend();
    f->Execute("cudaThreadExit");
    return f->GetExitCode();
}
